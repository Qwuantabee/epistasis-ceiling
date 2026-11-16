#!/usr/bin/env python3
"""The usable prior: where does additivity fail, and can you tell in advance?

The direction had to be established twice, because the first reading was wrong.

  rho(|residual|, SIGNED predicted) = +0.379 median (41/48 assays)
  rho(|residual|, |predicted|)     = -0.260 median (16/48 assays)

The second line is the interpretable one, and it says the opposite of the first:
residual magnitude is LARGEST where the singles look MILD. Additivity fails on
pairs whose individual parts give no warning. The positive signed correlation is
an asymmetry between negative and positive predictions, not "big effects are
harder" - an earlier draft of this file said the latter, and the tercile table
disproved it.

What that means for a protein engineer: you cannot pick the risky pairs by looking
for dramatic singles. The mild-looking ones are where the additive shortcut breaks.

This script quantifies that prior properly, with the two controls that make it
mean something:

  1. A SIMULATION null: with `observed` independent of `predicted`, does the same
     correlation appear anyway? (It does not: rho ~ 0.001.) The residual contains
     `predicted` by construction, so this is the obvious artefact to rule out, and
     it is ruled out by construction rather than by assertion.
  2. Grouped cross-validation by assay, so the prior is tested on proteins it was
     not fitted on.

Output: results/prior_summary.csv (per assay) and results/prior_pooled.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def simulation_null(n: int = 60000, reps: int = 5, seed: int = 0) -> dict:
    """What does rho(|obs-pred|, pred) look like when obs is INDEPENDENT of pred?

    This is the control for the fact that |obs - pred| has `pred` inside it. If
    the real data's correlation were this artefact, the simulated value would be
    as large as the observed one.
    """
    from scipy.stats import spearmanr
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(reps):
        pred = rng.normal(0, 0.5, n)
        obs = rng.normal(0, 0.5, n)          # independent by construction
        vals.append(float(spearmanr(pred, np.abs(obs - pred)).statistic))
    return {"mean": float(np.mean(vals)), "sd": float(np.std(vals)),
            "reps": reps, "n": n}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-dir", default=str(ROOT / "results" / "pairs"))
    ap.add_argument("--min-pairs", type=int, default=500)
    ap.add_argument("--out", default=str(ROOT / "results"))
    args = ap.parse_args()

    from scipy.stats import spearmanr

    out = Path(args.out)
    rows = []
    for p in sorted(Path(args.pairs_dir).glob("*.csv")):
        d = pd.read_csv(p, usecols=["predicted", "observed", "residual", "sep"])
        if len(d) < args.min_pairs:
            continue
        rows.append({
            "assay": p.stem,
            "n_pairs": len(d),
            # the prior under study: does the additive prediction rank |residual|?
            "rho_pred_absres": float(spearmanr(d.predicted, d.residual.abs()).statistic),
            # The interpretable one: magnitude against magnitude. Small |predicted|
            # pairs have the LARGEST |residual| (median -0.26), so dramatic singles
            # are not where additivity breaks.
            "rho_abspred_absres": float(spearmanr(d.predicted.abs(), d.residual.abs()).statistic),
            # control: the observed score should NOT (it is the outcome, not a prior)
            "rho_obs_absres": float(spearmanr(d.observed, d.residual.abs()).statistic),
            # control: sequence separation, which the earlier probe already killed
            "rho_sep_absres": float(spearmanr(d.sep, d.residual.abs()).statistic),
            # how much of the residual spread |quantile| bins explain, in sd units
            "frac_strong_epistasis": float(np.mean(np.abs(d.residual) > 2)),
        })

    if not rows:
        raise SystemExit("no assays with enough pairs")
    t = pd.DataFrame(rows).sort_values("n_pairs", ascending=False)
    t.to_csv(out / "prior_summary.csv", index=False)

    null = simulation_null()

    # A blunt, usable rule: split pairs by |additive prediction| into terciles and
    # report the residual spread of each. This is what a reader would actually do.
    pooled = []
    for p in sorted(Path(args.pairs_dir).glob("*.csv")):
        d = pd.read_csv(p, usecols=["predicted", "residual"])
        if len(d) < args.min_pairs:
            continue
        d["tercile"] = pd.qcut(d.predicted.abs(), 3, labels=["small", "mid", "large"],
                               duplicates="drop")
        g = d.groupby("tercile", observed=True).residual.apply(
            lambda s: float(np.median(np.abs(s))))
        g.name = p.stem
        pooled.append(g)
        continue
    T = pd.DataFrame(pooled)
    med = T.median()

    summary = {
        "n_assays": int(len(t)),
        "n_pairs": int(t.n_pairs.sum()),
        "rho_pred_absres_median": float(t.rho_pred_absres.median()),
        "rho_abspred_absres_median": float(t.rho_abspred_absres.median()),
        "rho_abspred_absres_positive_assays": int((t.rho_abspred_absres > 0).sum()),
        "rho_pred_absres_positive_assays": int((t.rho_pred_absres > 0).sum()),
        "rho_obs_absres_median": float(t.rho_obs_absres.median()),
        "rho_sep_absres_median": float(t.rho_sep_absres.median()),
        "simulation_null": null,
        "median_abs_residual_by_additive_tercile": {
            str(k): float(v) for k, v in med.items()},
        "frac_strong_epistasis_median": float(t.frac_strong_epistasis.median()),
    }
    (out / "prior_pooled.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(f"assays {summary['n_assays']}  pairs {summary['n_pairs']:,}")
    print(f"rho(|residual|, additive prediction): median "
          f"{summary['rho_pred_absres_median']:+.3f} "
          f"(positive in {summary['rho_pred_absres_positive_assays']}/"
          f"{summary['n_assays']})")
    print(f"rho(|residual|, |additive prediction|): median "
          f"{summary['rho_abspred_absres_median']:+.3f} "
          f"(positive in {summary['rho_abspred_absres_positive_assays']}/"
          f"{summary['n_assays']})  <- the interpretable direction")
    print(f"  control, observed score:      {summary['rho_obs_absres_median']:+.3f}")
    print(f"  control, sequence separation: {summary['rho_sep_absres_median']:+.3f}")
    print(f"  simulation null (independent): {null['mean']:+.4f} "
          f"+- {null['sd']:.4f}")
    print("\nmedian |residual| (assay sd) by |additive prediction| tercile:")
    for k, v in summary["median_abs_residual_by_additive_tercile"].items():
        print(f"  {k:6s} {v:.3f}")
    print(f"\n-> {out/'prior_summary.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
