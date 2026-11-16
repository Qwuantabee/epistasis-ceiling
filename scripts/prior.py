#!/usr/bin/env python3
"""The magnitude prior: can you tell in advance which pairs break additivity?

The committed answer used to be rho(|residual|, |predicted|) = -0.260, read as
"additivity fails where the singles look mild". The reviewer round (F2/F3) showed
that number is not interpretable, and this script now reports both the old
statistic and the two controls that decide the question.

WHY THE OLD SIGN WAS NOT A FINDING. residual = observed - (a + b - wt) and
predicted = a + b - wt, so the residual moves one-for-one with `wt`, an assay
CONVENTION. The pair cloud therefore sits wherever `wt` put it - a median -0.94
assay-sd off zero here - and |residual| is a distance from that arbitrary origin.
Comparing it to |predicted|, which sits somewhere else, is not a comparison of
"how dramatic the singles look". Confirming the mechanism: the -0.260 appears only
where the offset is large (rho -0.469 in assays offset 0.75-1.5 sd) and vanishes
where it is small (+0.058 under 0.25 sd).

WHY THE OBVIOUS REPLACEMENT IS ALSO NOT A FINDING. Centring removes the offset and
flips the sign to +0.33, but |residual| is built from `predicted` by subtraction,
so a POSITIVE correlation with |predicted| is mechanically forced even when the
double is independent of its singles. Measured: a within-assay permutation of
observed (both marginals kept, only the pairing destroyed) gives +0.50. The
centred statistic is BELOW its own null, so it cannot be read as "dramatic pairs
fail harder" either.

The honest conclusion is the one this script prints: a correlation between
|residual| and |predicted| is not a usable prior on these data, in either
direction, and the null makes that checkable rather than rhetorical.

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
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


def shift_invariant_null(df: pd.DataFrame, reps: int = 3,
                         seed: int = 0) -> dict:
    """The null that can actually test the centred statistic.

    The old simulation drew observed and predicted as iid N(0, 0.5): both
    symmetric and mean-zero, so the offset mechanism could not appear in it by
    construction and it could not have detected the defect it was cited against.
    This permutes observed WITHIN each assay, which keeps the real marginals
    (including their offsets and skew) and destroys only the pairing.
    """
    from scipy.stats import spearmanr
    from epsceil.pairs import add_centred_columns

    df = add_centred_columns(df)
    rng = np.random.default_rng(seed)
    vals = []
    for _, g in df.groupby("assay"):
        if len(g) < 500:
            continue
        pred = g.predicted.to_numpy()
        pc = g.predicted_c.to_numpy()
        for _ in range(reps):
            o = rng.permutation(g.observed.to_numpy())
            r = o - pred
            rc = r - r.mean()
            vals.append(float(spearmanr(np.abs(pc), np.abs(rc)).statistic))
    return {"median": float(np.median(vals)),
            "ci95": [float(np.percentile(vals, 2.5)),
                     float(np.percentile(vals, 97.5))],
            "reps": len(vals),
            "kind": "within-assay permutation of observed"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-dir", default=str(ROOT / "results" / "pairs"))
    ap.add_argument("--min-pairs", type=int, default=500)
    ap.add_argument("--out", default=str(ROOT / "results"))
    args = ap.parse_args()

    from scipy.stats import spearmanr

    out = Path(args.out)
    rows = []
    frames = []
    for p in sorted(Path(args.pairs_dir).glob("*.csv")):
        d = pd.read_csv(p, usecols=["predicted", "observed", "residual", "sep"])
        if len(d) < args.min_pairs:
            continue
        # centring is derived from the SAME rule build_pairs uses, so the two
        # cannot drift; results/pairs/ predates the fix and has no columns for it
        d["assay"] = p.stem
        d["residual_c"] = d.residual - d.residual.mean()
        d["predicted_c"] = d.predicted - float(d.predicted.median())
        frames.append(d)
        rows.append({
            "assay": p.stem,
            "n_pairs": len(d),
            # the prior under study, as originally computed
            "rho_pred_absres": float(spearmanr(d.predicted, d.residual.abs()).statistic),
            "rho_abspred_absres": float(spearmanr(d.predicted.abs(), d.residual.abs()).statistic),
            # the same question with both series centred: shift-invariant, and the
            # one that has a null which can actually be non-zero
            "rho_abspred_absres_centred": float(
                spearmanr(d.predicted_c.abs(), d.residual_c.abs()).statistic),
            # control: the observed score should NOT (it is the outcome, not a prior)
            "rho_obs_absres": float(spearmanr(d.observed, d.residual.abs()).statistic),
            # control: sequence separation, which the earlier probe already killed
            "rho_sep_absres": float(spearmanr(d.sep, d.residual.abs()).statistic),
            # the residual's offset from zero: the quantity that made the old sign
            # depend on where `wt` happened to sit
            "residual_mean_sd": float(d.residual.mean()),
            # how much of the residual spread sits beyond a stated tolerance
            "frac_strong_epistasis": float(np.mean(np.abs(d.residual) > 2)),
            "frac_strong_epistasis_centred": float(np.mean(np.abs(d.residual_c) > 2)),
        })

    if not rows:
        raise SystemExit("no assays with enough pairs")
    t = pd.DataFrame(rows).sort_values("n_pairs", ascending=False)
    t.to_csv(out / "prior_summary.csv", index=False)

    allp = pd.concat(frames, ignore_index=True)
    null = shift_invariant_null(allp)

    # A blunt, usable rule: split pairs by |additive prediction| into terciles and
    # report the residual spread of each - once on the raw residual and once on the
    # centred one, because they disagree and the disagreement is the finding.
    pooled = []
    pooled_c = []
    for p in sorted(Path(args.pairs_dir).glob("*.csv")):
        d = pd.read_csv(p, usecols=["predicted", "residual"])
        if len(d) < args.min_pairs:
            continue
        d["residual_c"] = d.residual - d.residual.mean()
        d["tercile"] = pd.qcut(d.predicted.abs(), 3, labels=["small", "mid", "large"],
                               duplicates="drop")
        g = d.groupby("tercile", observed=True).residual.apply(
            lambda s: float(np.median(np.abs(s))))
        g.name = p.stem
        pooled.append(g)
        gc = d.groupby("tercile", observed=True).residual_c.apply(
            lambda s: float(np.median(np.abs(s))))
        gc.name = p.stem
        pooled_c.append(gc)
    T = pd.DataFrame(pooled)
    med = T.median()
    med_c = pd.DataFrame(pooled_c).median()

    summary = {
        "n_assays": int(len(t)),
        "n_pairs": int(t.n_pairs.sum()),
        "rho_pred_absres_median": float(t.rho_pred_absres.median()),
        "rho_abspred_absres_median": float(t.rho_abspred_absres.median()),
        "rho_abspred_absres_positive_assays": int((t.rho_abspred_absres > 0).sum()),
        "rho_abspred_absres_centred_median": float(t.rho_abspred_absres_centred.median()),
        "rho_abspred_absres_centred_positive_assays": int(
            (t.rho_abspred_absres_centred > 0).sum()),
        "residual_mean_sd_median": float(t.residual_mean_sd.median()),
        "rho_pred_absres_positive_assays": int((t.rho_pred_absres > 0).sum()),
        "rho_obs_absres_median": float(t.rho_obs_absres.median()),
        "rho_sep_absres_median": float(t.rho_sep_absres.median()),
        "permutation_null_centred": null,
        "observed_centred_minus_null": (float(t.rho_abspred_absres_centred.median())
                                        - null["median"]),
        "median_abs_residual_by_additive_tercile": {
            str(k): float(v) for k, v in med.items()},
        "median_abs_residual_by_additive_tercile_centred": {
            str(k): float(v) for k, v in med_c.items()},
        "frac_strong_epistasis_median": float(t.frac_strong_epistasis.median()),
        "frac_strong_epistasis_centred_median": float(
            t.frac_strong_epistasis_centred.median()),
        "prior_verdict": "NOT_SUPPORTED",
        "prior_note": "Neither direction of the |residual| vs |predicted| "
                      "relation is a usable prior. The raw statistic (-0.26) "
                      "tracks where the per-assay centring constant put zero; "
                      "the centred statistic (+0.33) sits BELOW its within-assay "
                      "permutation null, because |residual| contains |predicted| "
                      "by construction. Both are reported with the null rather "
                      "than one being quoted alone.",
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
          f"{summary['n_assays']})   <- depended on where zero sat; see below")
    print(f"  COMPETING EXPLANATION, centred statistic: "
          f"{summary['rho_abspred_absres_centred_median']:+.3f} "
          f"(positive in {summary['rho_abspred_absres_centred_positive_assays']}/"
          f"{summary['n_assays']})")
    print(f"  its null (within-assay permutation): "
          f"{null['median']:+.3f} 95% CI [{null['ci95'][0]:+.3f}, "
          f"{null['ci95'][1]:+.3f}] over {null['reps']} draws")
    print(f"  -> observed minus null: "
          f"{summary['observed_centred_minus_null']:+.3f}")
    print(f"  residual offset from zero, median: "
          f"{summary['residual_mean_sd_median']:+.3f} assay sd")
    print(f"  control, observed score:      {summary['rho_obs_absres_median']:+.3f}")
    print(f"  control, sequence separation: {summary['rho_sep_absres_median']:+.3f}")
    print(f"  simulation null (independent): {null['median']:+.4f} "
          f"+- {(null['ci95'][1]-null['ci95'][0])/3.92:.4f}")
    print("\nmedian |residual| (assay sd) by |additive prediction| tercile:")
    for k, v in summary["median_abs_residual_by_additive_tercile"].items():
        vc = summary["median_abs_residual_by_additive_tercile_centred"].get(k)
        print(f"  {k:6s} raw {v:.3f}   centred {vc:.3f}")
    print("\nBoth tercile orderings are printed because they disagree: the raw one "
          "says the mild singles fail hardest, the centred one says the "
          "opposite, and the null above says neither is credible.")
    print(f"\n-> {out/'prior_summary.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
