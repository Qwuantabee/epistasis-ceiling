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
    limit = 0.95 * fig.get_figwidth() * fig.dpi   # text starts at x=0.012
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


PAIRS_HINT = ("needs the gitignored pair tables in results/pairs/: run "
              "scripts/build_pairs.py first (README, Reproducing)")


def fig_ceiling() -> bool:
    """Two panels, both on statistics that survive review round 2.

    The earlier right panel ("Additivity fails where the singles look MILD",
    rho -0.26) carried a claim the README withdrew because it fails its own
    construction-preserving null, and the left panel's title headlined the pooled
    40.9% under the median-of-all-variants centring. That centring puts the
    assay's zero at an arbitrary place (results/reference.json: the bias flips
    sign under wild type = 0), so both panels now show only quantities that are
    invariant to it:

      left  - the residual after removing each assay's own mean offset (the
              centred residual), pooled over the 48 assays with >= 500 pairs;
      right - each assay's threshold-free AUC, which no per-assay constant moves.

    Returns False when the gitignored pair tables are absent, so main() can fail
    loudly instead of silently skipping the headline figure.
    """
    import sys as _sys
    _sys.path.insert(0, str(ROOT / "src"))
    from epsceil.pairs import add_centred_columns

    summ = pd.read_csv(RES / "assay_summary.csv")
    big = summ[summ.n_pairs >= 500]
    frames = []
    for p in sorted((RES / "pairs").glob("*.csv")):
        d = pd.read_csv(p, usecols=["residual", "predicted"])
        if len(d) >= 500:
            frames.append(add_centred_columns(d)[["residual_c"]])
    if not frames:
        return False
    res_c = pd.concat(frames, ignore_index=True).residual_c.to_numpy()
    frac_c = float(np.mean(np.abs(res_c) > 2))
    auc = np.sort(big.auc_direction.to_numpy(float))
    med_auc = float(np.median(auc))

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.4),
                                  gridspec_kw=dict(width_ratios=[1.1, 1]))
    ax.hist(np.clip(res_c, -6, 6), bins=90, color="#4c72b0",
            edgecolor="none", alpha=0.9)
    for x in (-2, 2):
        ax.axvline(x, ls="--", lw=1.2, color="#d62728")
    ax.set_xlabel("centred additivity residual (assay standard deviations)")
    ax.set_ylabel("double mutants")
    ax.set_title(f"Residual after removing each assay's own offset\n"
                 f"{len(frames)} assays, {len(res_c):,} pairs; {frac_c:.1%} "
                 f"pooled beyond ±2 sd", fontsize=9.5)
    ax.set_xlim(-6, 6)
    ax.text(2.1, ax.get_ylim()[1] * 0.92, "±2 sd", color="#d62728", fontsize=7.5)
    ax.grid(axis="y", alpha=0.25)

    ax2.scatter(np.arange(len(auc)), auc, s=22, color="#55a868",
                edgecolor="black", linewidth=0.35, zorder=3)
    ax2.axhline(med_auc, ls="--", lw=1.1, color="#333333")
    ax2.axhline(0.5, ls=":", lw=1.0, color="#d62728")
    ax2.text(0.5, med_auc + 0.012, f"median {med_auc:.3f}", fontsize=8)
    ax2.text(0.5, 0.512, "chance", fontsize=7.5, color="#d62728")
    ax2.set_ylim(0.45, 1.0)
    ax2.set_xlabel(f"assays, sorted by AUC ({len(auc)} with >= 500 pairs)")
    ax2.set_ylabel("AUC, additive prediction vs observed direction")
    ax2.set_title(f"Additivity ranks doubles well\nmedian AUC {med_auc:.3f} "
                  f"over {len(auc)} assays", fontsize=9.5)
    ax2.grid(alpha=0.25)

    _footer(fig, "Left: additivity residuals pooled over the assays with >=500 "
                 "pairs, each divided by its own assay's residual sd and with that "
                 "assay's mean offset removed. The offset itself is not shown "
                 "because its size and sign depend on where each assay's zero is "
                 "put (median of all variants vs wild type = 0; "
                 "results/reference.json). Right: each assay's threshold-free AUC, "
                 "which no per-assay constant can move. The earlier 'fails where "
                 "the singles look mild' panel is withdrawn: it failed its own "
                 "construction-preserving null.")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    FIGS.mkdir(exist_ok=True)
    fig.savefig(FIGS / "ceiling.png", dpi=160)
    plt.close(fig)

    stats = RES / "figure_stats.json"
    prev = json.loads(stats.read_text()) if stats.exists() else {}
    prev["ceiling"] = {
        "n_assays": int(len(frames)),
        "n_pairs": int(len(res_c)),
        "pooled_frac_beyond_2sd_centred": frac_c,
        "median_auc": med_auc,
        "provenance": "scripts/make_figures.py fig_ceiling()",
        "source": "results/pairs/ (centred via epsceil.pairs.add_centred_columns) "
                  "and results/assay_summary.csv, assays with >=500 pairs",
    }
    stats.write_text(json.dumps(prev, indent=2) + "\n")
    print(f"  wrote figures/ceiling.png (centred beyond 2 sd {frac_c:.1%}, "
          f"median AUC {med_auc:.3f})")
    return True


def fig_bias_not_spread() -> None:
    """Why "beyond 2 sd" is a statement about the offset.

    One panel: the share of pairs past 2 sd against the assay's MEAN BIAS. Every
    point is a row of the committed assay_summary.csv.

    An earlier version had a second panel against the assay's residual sd, titled
    "r = +0.12". That x variable is 1 for every assay by construction (each
    residual is divided by its own assay's sd), so the panel was a vertical line
    and the correlation was with floating-point rounding. It is withdrawn, and
    figure_stats.json no longer carries it.
    """
    s = pd.read_csv(RES / "assay_summary.csv")
    big = s[s.n_pairs >= 500].copy()
    if big.empty:
        print("  (skipping bias figure: no assays with >=500 pairs)")
        return
    y = big.frac_outside_2sd.to_numpy(float)
    xb = big.bias_mean_sd.abs().to_numpy(float)
    rb = float(np.corrcoef(xb, y)[0, 1])

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.scatter(xb, y, s=30, color="#d62728", alpha=0.85, edgecolor="black",
               linewidth=0.35)
    ax.set_xlabel("|mean bias| of the additive prediction (assay sd)",
                  fontsize=10)
    ax.set_ylabel("share of pairs beyond ±2 sd", fontsize=10)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title(f"The tail tracks the offset\nr = {rb:+.2f} over {len(xb)} "
                 f"assays (median centring)", fontsize=11)
    ax.grid(alpha=0.25)
    ax.tick_params(labelsize=9)

    _footer(fig, "Each point is one assay with at least 500 pairs, from the "
                 "committed assay_summary.csv, under the median-of-all-variants "
                 "centring. Assays whose additive prediction sits further from "
                 "zero have more pairs past ±2 sd, because a fixed threshold "
                 "counts the offset. So '12.6% of pairs are beyond 2 sd' is "
                 "mostly a statement about the offset, not about how "
                 "unpredictable those pairs are.")
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    FIGS.mkdir(exist_ok=True)
    fig.savefig(FIGS / "bias_not_spread.png", dpi=160)
    plt.close(fig)

    # The correlation is QUOTED IN THE README, so it is written to an artefact
    # rather than living only in this function's stdout: a number a reader can
    # see must be a number scripts/audit_claims.py can check.
    stats = RES / "figure_stats.json"
    prev = json.loads(stats.read_text()) if stats.exists() else {}
    prev["bias_vs_tail_fraction"] = {
        "n_assays": int(len(big)),
        "pearson_abs_bias": rb,
        "reading": "share beyond 2 sd tracks |mean bias| under the median centring",
        "provenance": "scripts/make_figures.py fig_bias_not_spread()",
        "source": "results/assay_summary.csv rows with >=500 pairs",
    }
    stats.write_text(json.dumps(prev, indent=2) + "\n")
    print(f"  wrote figures/bias_not_spread.png (r_bias={rb:+.3f})")


def main() -> int:
    global FIGS          # must precede any read of FIGS (SyntaxError otherwise)
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(FIGS))
    args = ap.parse_args()
    FIGS = Path(args.out)
    FIGS.mkdir(parents=True, exist_ok=True)
    # the summary-only figure first, so a clone without pair tables still gets it
    fig_bias_not_spread()
    if not fig_ceiling():
        print(f"figures/ceiling.png NOT rebuilt: make_figures.py {PAIRS_HINT}",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
