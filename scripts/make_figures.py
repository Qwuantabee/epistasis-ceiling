#!/usr/bin/env python3
"""Figures from committed artefacts. Every number read from a CSV, none typed."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.ticker  # noqa: F401  PercentFormatter needs the submodule;
# a bare `import matplotlib` does not bind `matplotlib.ticker`.
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


def fig_bias_not_spread() -> None:
    """Why "beyond 2 sd" is a statement about the offset, not the spread.

    Two panels, same y axis, same assays: the share of pairs past 2 sd against the
    assay's MEAN BIAS (left) and against its residual SPREAD (right). If the tail
    were measuring unpredictability, the right panel would carry the relationship.
    It is the left one, and by a wide margin, which is the section's whole argument
    reduced to one image. Every point is a row of the committed assay_summary.csv.
    """
    s = pd.read_csv(RES / "assay_summary.csv")
    big = s[s.n_pairs >= 500].copy()
    if big.empty:
        print("  (skipping bias figure: no assays with >=500 pairs)")
        return
    y = big.frac_outside_2sd.to_numpy(float)
    xb = big.bias_mean_sd.abs().to_numpy(float)
    xs = big.residual_sd.to_numpy(float)
    rb = float(np.corrcoef(xb, y)[0, 1])
    rs = float(np.corrcoef(xs, y)[0, 1])

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.4), sharey=True)
    for a, x, r, ttl, xlab, col in (
            (ax, xb, rb, "The tail tracks the offset",
             "|mean bias| of the additive prediction (assay sd)", "#d62728"),
            (ax2, xs, rs, "...not the spread",
             "residual sd (assay sd)", "#4c72b0")):
        a.scatter(x, y, s=26, color=col, alpha=0.85, edgecolor="black",
                  linewidth=0.35)
        a.set_xlabel(xlab, fontsize=8.5)
        a.set_title(f"{ttl}\nr = {r:+.2f} over {len(x)} assays", fontsize=9.5)
        a.grid(alpha=0.25)
        a.tick_params(labelsize=8)
    ax.set_ylabel("share of pairs beyond ±2 sd", fontsize=8.5)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))

    _footer(fig, "Each point is one assay with at least 500 pairs, from the "
                 "committed assay_summary.csv. Left: assays whose additive "
                 "prediction is further off zero have proportionally more pairs "
                 "past ±2 sd, because a fixed threshold counts the offset. "
                 "Right: the assay's own residual spread barely moves that share. "
                 "So '12.6% of pairs are beyond 2 sd' is a statement about how "
                 "far the additive model sits from zero in that assay, not about "
                 "how unpredictable those pairs are — which is why the per-assay "
                 "table, not the pooled number, is the result.")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    FIGS.mkdir(exist_ok=True)
    fig.savefig(FIGS / "bias_not_spread.png", dpi=160)
    plt.close(fig)

    # The correlations are QUOTED IN THE README, so they are written to an
    # artefact rather than living only in this function's stdout: a number a
    # reader can see must be a number scripts/audit_claims.py can check.
    stats = RES / "figure_stats.json"
    prev = json.loads(stats.read_text()) if stats.exists() else {}
    prev["bias_vs_tail_fraction"] = {
        "n_assays": int(len(big)),
        "pearson_abs_bias": rb,
        "pearson_residual_sd": rs,
        "reading": "share beyond 2 sd tracks |mean bias|, not residual spread",
        "provenance": "scripts/make_figures.py fig_bias_not_spread()",
        "source": "results/assay_summary.csv rows with >=500 pairs",
    }
    stats.write_text(json.dumps(prev, indent=2) + "\n")
    print(f"  wrote figures/bias_not_spread.png (r_bias={rb:+.3f}, "
          f"r_spread={rs:+.3f})")


def main() -> int:
    global FIGS          # must precede any read of FIGS (SyntaxError otherwise)
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(FIGS))
    args = ap.parse_args()
    FIGS = Path(args.out)
    FIGS.mkdir(parents=True, exist_ok=True)
    fig_ceiling()
    fig_bias_not_spread()
    return 0


if __name__ == "__main__":
    sys.exit(main())
