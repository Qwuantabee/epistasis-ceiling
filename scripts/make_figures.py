#!/usr/bin/env python3
"""Figures from committed artefacts. Every number read from a CSV, none typed."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FIGS = ROOT / "figures"


def _footer(fig, text: str) -> None:
    """Grey wrapping note; wrapped by measured width so it stays on the page."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    probe = fig.text(0, 0, "", fontsize=7)
    limit = 0.975 * fig.get_figwidth() * fig.dpi
    lines: list[str] = []
    for para in text.split("\n"):
        line = ""
        for word in para.split(" "):
            trial = (line + " " + word).strip()
            probe.set_text(trial)
            if line and probe.get_window_extent(renderer).width > limit:
                lines.append(line)
                line = word
            else:
                line = trial
        lines.append(line)
    probe.remove()
    fig.text(0.012, 0.004, "\n".join(lines), fontsize=7, color="#555555",
             ha="left", va="bottom", linespacing=1.35)


def fig_ceiling() -> None:
    """Two panels: the residual distribution, and the prior that predicts it."""
    summ = pd.read_csv(RES / "assay_summary.csv")
    prior = RES / "prior_pooled.json"
    if not prior.exists():
        print("  (skipping prior panel: run scripts/prior.py)")
        pj = {}
    else:
        pj = json.loads(prior.read_text())

    # pooled residuals for the histogram
    frames = []
    for p in sorted((RES / "pairs").glob("*.csv")):
        d = pd.read_csv(p, usecols=["residual"])
        if len(d) >= 500:
            frames.append(d)
    if not frames:
        print("  (skipping ceiling: no pair files)")
        return
    allres = pd.concat(frames, ignore_index=True).residual.to_numpy()

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.4),
                                  gridspec_kw=dict(width_ratios=[1.1, 1]))
    ax.hist(np.clip(allres, -6, 6), bins=90, color="#4c72b0",
            edgecolor="none", alpha=0.9)
    for x in (-2, 2):
        ax.axvline(x, ls="--", lw=1.2, color="#d62728")
    ax.set_xlabel("additivity residual (assay standard deviations)")
    ax.set_ylabel("double mutants")
    ax.set_title(f"The additive shortcut's error\n"
                 f"{len(allres):,} pairs, {np.mean(np.abs(allres) > 2):.1%} POOLED "
                 f"beyond ±2 sd", fontsize=9.5)
    ax.set_xlim(-6, 6)
    ax.text(2.1, ax.get_ylim()[1] * 0.92, "±2 sd", color="#d62728", fontsize=7.5)
    ax.grid(axis="y", alpha=0.25)

    terc = pj.get("median_abs_residual_by_additive_tercile", {})
    if terc:
        keys = ["small", "mid", "large"]
        vals = [terc.get(k, np.nan) for k in keys]
        ax2.bar(keys, vals, color=["#d62728", "#dd8452", "#55a868"],
                edgecolor="black", linewidth=0.5)
        for i, v in enumerate(vals):
            ax2.text(i, v + 0.03, f"{v:.2f}", ha="center", fontsize=9)
        ax2.set_ylabel("median |residual| (assay sd)")
        ax2.set_xlabel("how dramatic the singles look\n(|additive prediction| tercile)")
        ax2.set_title("Additivity fails where the singles look MILD\n"
                      f"rho(|residual|, |prediction|) = "
                      f"{pj.get('rho_abspred_absres_median', float('nan')):+.2f}",
                      fontsize=9.5)
        ax2.set_ylim(0, max(vals) * 1.18)
        ax2.grid(axis="y", alpha=0.25)

    _footer(fig, "Left: pooled additivity residuals across all assays with >=500 "
                 "pairs, each residual divided by its own assay's residual sd "
                 "(ProteinGym assays are not on a common scale). Right: pairs "
                 "binned by |additive prediction| into terciles; a NEGATIVE "
                 "correlation means the quietest singles produce the largest "
                 "epistatic surprises. Simulation null for the left-panel width: "
                 "rho 0.001 +- 0.002 when observed is independent of predicted.")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    FIGS.mkdir(exist_ok=True)
    fig.savefig(FIGS / "ceiling.png", dpi=160)
    plt.close(fig)
    print("  wrote figures/ceiling.png")


def main() -> int:
    global FIGS          # must precede any read of FIGS (SyntaxError otherwise)
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(FIGS))
    args = ap.parse_args()
    FIGS = Path(args.out)
    FIGS.mkdir(parents=True, exist_ok=True)
    fig_ceiling()
    return 0


if __name__ == "__main__":
    sys.exit(main())
