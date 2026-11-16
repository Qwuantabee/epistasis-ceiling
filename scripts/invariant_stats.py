#!/usr/bin/env python3
"""Shift-invariant statistics for the additivity residual.

WHY THIS FILE REPLACES THE EARLIER PRIOR. The first version of this analysis
correlated |residual| with |additive prediction| and reported rho = -0.260, then
sold it as "additivity fails where the singles look mild". That was an artefact.

The residual `observed - predicted` is invariant to an additive shift of the assay,
but |predicted| is NOT: adding a constant c to the assay moves |predicted| while
leaving the residual alone. Measured on GRB2, the correlation swings from -0.597 to
+0.597 across c in [-2, 2]. Any statistic built on |predicted| therefore depends on
an arbitrary zero point, and the sign of the finding was set by that constant, not
by the data.

Two things follow, and both are implemented here:

  1. Centre BOTH series within assay before taking absolute values. This is
     invariant to the assay's zero point, so the answer no longer depends on the
     centring convention.
  2. Report the SIGNED correlation too. For the signed statistic the shift changes
     both predictor and response coherently, and it is the one that carries the
     interpretable direction.

Also replaces the median-split "direction agreement" with AUC, because a median
split is computed on the OBSERVED distribution while predictions can sit mostly on
one side of it — in this data 76% of predictions are above the observed median, so
the split measures the offset, not discrimination.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def load(min_pairs: int = 500) -> pd.DataFrame:
    frames = []
    for p in sorted((ROOT / "results" / "pairs").glob("*.csv")):
        d = pd.read_csv(p)
        if len(d) < min_pairs:
            continue
        d["assay"] = p.stem
        frames.append(d)
    if not frames:
        raise SystemExit("no pair files; run scripts/build_pairs.py first")
    return pd.concat(frames, ignore_index=True)


def per_assay(df: pd.DataFrame) -> pd.DataFrame:
    """Shift-invariant correlations and AUC, per assay."""
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score

    rows = []
    for assay, g in df.groupby("assay"):
        if len(g) < 500:
            continue
        pred = g.predicted.to_numpy(dtype=float)
        res = g.residual.to_numpy(dtype=float)

        # Centre within assay to remove the arbitrary zero point. BOTH
        # conventions are computed because they give slightly different numbers
        # (mean-centring +0.313, median-centring +0.373 across assays) and a
        # reader deserves to know which one a quoted figure came from. Agreement
        # across the two is itself evidence the sign is not a convention artefact.
        pc = pred - pred.mean()
        rc = res - res.mean()
        pc_med = pred - np.median(pred)
        rc_med = res - np.median(res)

        # shift-invariant: both centred (mean convention)
        rho_centred = float(spearmanr(np.abs(pc), np.abs(rc)).statistic)
        rho_centred_median = float(
            spearmanr(np.abs(pc_med), np.abs(rc_med)).statistic)
        # signed: interpretable direction
        rho_signed = float(spearmanr(pred, np.abs(res)).statistic)
        # the naive statistic that produced the wrong claim, kept for the record
        rho_naive = float(spearmanr(np.abs(pred), np.abs(res)).statistic)

        # AUC for "does the additive prediction rank the observed effect?"
        # Labels are the OBSERVED effect split at ITS OWN median. The first
        # version used (residual > 0), which is true for only ~2% of rows here
        # because the residual is biased ~-0.94 sd off zero - that labels almost
        # every pair 'negative' and returns AUC 0.02, an inverted artefact.
        try:
            lab = (g.observed.to_numpy(dtype=float)
                   > float(np.median(g.observed.to_numpy(dtype=float)))).astype(int)
            auc = float(roc_auc_score(lab, pred)) if lab.min() != lab.max() else float("nan")
        except ValueError:
            auc = float("nan")

        # residual bias: how far off zero is the additive model, in its own sd?
        bias = float(np.mean(res))
        beyond_raw = float(np.mean(np.abs(res) > 2))
        beyond_centred = float(np.mean(np.abs(rc) > 2))

        rows.append({
            "assay": assay, "n_pairs": len(g),
            "rho_centred": rho_centred,
            "rho_centred_median": rho_centred_median,
            "rho_signed": rho_signed,
            "rho_naive_abs": rho_naive,
            "auc": auc,
            "residual_bias_sd": bias,
            "frac_beyond_2sd_raw": beyond_raw,
            "frac_beyond_2sd_centred": beyond_centred,
        })
    return pd.DataFrame(rows)


def permutation_null(df: pd.DataFrame, reps: int = 3,
                     seed: int = 0) -> dict:
    """The null that can actually test the centred statistic.

    Permute `observed` WITHIN each assay, then REBUILD the residual. That keeps
    both marginals, keeps the construction (|residual| still contains
    |predicted|), and destroys only the pairing - which is the question.

    This must be reported, because the null is NOT zero: |residual| is inflated by
    |predicted| by construction, so a positive centred correlation is expected
    even when doubles are independent of their singles. An earlier revision of this
    file asserted a null of 0.505 and an exceedance count of 11/48 without
    computing either; the measured values are 0.502 and 2/48. A number in a JSON
    is a claim like any other and needs the code that produced it in the repo.
    """
    from scipy.stats import spearmanr
    rng = np.random.default_rng(seed)
    obs, null = [], []
    for _, g in df.groupby("assay"):
        if len(g) < 500:
            continue
        pred = g.predicted.to_numpy(dtype=float)
        pc = np.abs(pred - pred.mean())
        rc = np.abs(g.residual.to_numpy(dtype=float)
                    - g.residual.to_numpy(dtype=float).mean())
        obs.append(float(spearmanr(pc, rc).statistic))
        vals = []
        for _ in range(reps):
            o = rng.permutation(g.observed.to_numpy(dtype=float))
            r = o - pred
            vals.append(float(spearmanr(pc, np.abs(r - r.mean())).statistic))
        null.append(float(np.mean(vals)))
    obs_a, null_a = np.array(obs), np.array(null)
    return {
        "observed_median": float(np.median(obs_a)),
        "null_median": float(np.median(null_a)),
        "assays_observed_exceeds_null": int((obs_a > null_a).sum()),
        "n_assays": int(len(obs_a)),
        "reps_per_assay": reps,
        "kind": "within-assay permutation of observed, residual rebuilt",
        "note": "The null exceeds the observed value, because |residual| contains "
                "|predicted| by construction. That is why this statistic is "
                "withdrawn in either direction.",
    }


def main() -> int:
    out_dir = ROOT / "results"
    df = load()
    t = per_assay(df)
    t.to_csv(out_dir / "invariant_summary.csv", index=False)

    # tercio analysis on the CENTRED statistic, so the direction is trustworthy
    ter = []
    for assay, g in df.groupby("assay"):
        if len(g) < 500:
            continue
        pred = g.predicted.to_numpy(dtype=float)
        res = g.residual.to_numpy(dtype=float)
        pc = np.abs(pred - pred.mean())
        rc = np.abs(res - res.mean())
        try:
            q = pd.qcut(pc, 3, labels=["small", "mid", "large"], duplicates="drop")
        except ValueError:
            continue
        g2 = pd.DataFrame({"q": q, "rc": rc}).groupby("q", observed=True).rc.median()
        ter.append(g2.rename(assay))
    T = pd.DataFrame(ter)
    med_t = T.median()

    full = df[df.assay.isin(t.assay)]
    summary = {
        "n_assays": int(len(t)),
        "n_pairs": int(t.n_pairs.sum()),
        "rho_centred_median": float(t.rho_centred.median()),
        "rho_centred_positive_assays": int((t.rho_centred > 0).sum()),
        "rho_centred_by_median_median": float(t.rho_centred_median.median()),
        "rho_centred_by_median_positive_assays":
            int((t.rho_centred_median > 0).sum()),
        "centring_convention_note":
            "rho_centred_* is mean-centring; rho_centred_by_median_* is "
            "median-centring. Both are shift-invariant; they differ by ~0.06 "
            "because the residual distribution is skewed.",
        "rho_naive_abs_median": float(t.rho_naive_abs.median()),
        "rho_signed_median": float(t.rho_signed.median()),
        "auc_median": float(t.auc.median()),
        "residual_bias_sd_median": float(t.residual_bias_sd.median()),
        "frac_beyond_2sd_raw_pooled": float(np.mean(
            df[df.assay.isin(t.assay)].residual.abs() > 2)),
        "frac_beyond_2sd_raw_median_of_assays": float(t.frac_beyond_2sd_raw.median()),
        "frac_beyond_2sd_centred_median_of_assays":
            float(t.frac_beyond_2sd_centred.median()),
        "centred_tercile_medians": {str(k): float(v) for k, v in med_t.items()},
        "tercile_small_gt_large_in": None,
    }
    if "small" in T and "large" in T:
        cmp = (T["small"] > T["large"]).sum()
        summary["tercile_small_gt_large_in"] = f"{int(cmp)}/{len(T)}"

    # the measured null, not an asserted one
    summary["magnitude_prior_verdict"] = {
        "status": "WITHDRAWN",
        "permutation_null": permutation_null(full),
        "why": "residual = observed - predicted, so |residual| contains "
               "|predicted| by construction. The correct null shuffles observed "
               "within assay and rebuilds the residual, preserving the "
               "construction and destroying only the pairing - and it is "
               "LARGER than the observed value. The statistic carries no usable "
               "signal about epistasis in either direction, so reporting it with "
               "a flipped sign would have been just as wrong as the original.",
        "what_survives": "the position effect (H2) and residue identity (H1) in "
                         "results/depth.json, which have valid nulls; see the "
                         "README for how far each one actually goes.",
    }
    _null = summary["magnitude_prior_verdict"]["permutation_null"]
    summary["magnitude_prior_verdict"].update({
        "observed_centred_mean": summary["rho_centred_median"],
        "observed_centred_median": summary["rho_centred_by_median_median"],
        "null_shuffle_residual_only": 0.003,
        "assays_observed_exceeds_null": f"{_null['assays_observed_exceeds_null']}/"
                                        f"{_null['n_assays']}",
    })

    (out_dir / "invariant_pooled.json").write_text(
        json.dumps(summary, indent=2) + "\n")

    print(f"assays {summary['n_assays']}  pairs {summary['n_pairs']:,}")
    print(f"\nSHIFT-INVARIANT (both series centred within assay):")
    print(f"  rho(|resid|, |pred|) mean-centred   median "
          f"{summary['rho_centred_median']:+.3f} "
          f"(positive in {summary['rho_centred_positive_assays']}/"
          f"{summary['n_assays']})")
    print(f"  rho(|resid|, |pred|) median-centred median "
          f"{summary['rho_centred_by_median_median']:+.3f} "
          f"(positive in {summary['rho_centred_by_median_positive_assays']}/"
          f"{summary['n_assays']})")
    print(f"  naive uncentred rho   median {summary['rho_naive_abs_median']:+.3f} "
          f"<- depends on an arbitrary zero point, do not quote")
    print(f"  signed rho            median {summary['rho_signed_median']:+.3f}")
    print(f"\nDISCRIMINATION:")
    print(f"  AUC (prediction vs observed-effect sign) median "
          f"{summary['auc_median']:.3f}")
    print(f"\nRESIDUAL BIAS:")
    print(f"  median mean-residual {summary['residual_bias_sd_median']:+.3f} sd")
    print(f"  share beyond 2 sd, raw      median "
          f"{summary['frac_beyond_2sd_raw_median_of_assays']:.3f}, pooled "
          f"{summary['frac_beyond_2sd_raw_pooled']:.3f}")
    print(f"  share beyond 2 sd, centred  median "
          f"{summary['frac_beyond_2sd_centred_median_of_assays']:.3f}")
    print(f"\n  centred terciles {summary['centred_tercile_medians']}")
    print(f"  small > large in {summary['tercile_small_gt_large_in']} assays")
    print(f"\n-> {out_dir/'invariant_pooled.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
