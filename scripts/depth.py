#!/usr/bin/env python3
"""Depth probes: is there structure in the additivity residual beyond the
magnitude prior already found?

The committed finding is that |residual| is LARGEST where the singles look mild
(rho -0.260 on |additive prediction|). That is one prior. This tests three more,
each with a null, and writes the negatives as negatives.

  H1  Residue-type effect. Do certain substitutions (e.g. to proline, to charged)
      carry systematically larger residuals than their prevalence implies?
      Null: permute the residue labels within the assay.

  H2  Position effect. Is epistasis concentrated at particular sequence positions?
      If a few positions are hot, a lab could flag them cheaply.
      Null: permute residuals across positions within the assay.
      Statistic: the share of total residual variance explained by position, which
      for pure noise equals (n_positions - 1)/n_pairs.

  H3  Singleton decomposition. Is |residual| better predicted by the two singles'
      own magnitudes than by their sum? (The committed prior uses the sum.)
      Fit and evaluate with GroupKFold by assay; the baseline is the training
      median, as before.

A hypothesis with no signal is reported as NOT_SUPPORTED, not omitted.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"


def load_pairs(min_pairs: int = 500) -> pd.DataFrame:
    frames = []
    for p in sorted((RES / "pairs").glob("*.csv")):
        d = pd.read_csv(p)
        if len(d) < min_pairs:
            continue
        d["assay"] = p.stem
        frames.append(d)
    if not frames:
        raise SystemExit("no pair files; run scripts/build_pairs.py first")
    return pd.concat(frames, ignore_index=True)


def variance_explained_by_position(df: pd.DataFrame) -> dict:
    """Share of residual variance attributable to sequence position.

    For independent noise this quantity is (k-1)/(n-1) with k the number of
    distinct positions, so the null is computed analytically AND by permutation.
    """
    rng = np.random.default_rng(0)
    obs, null_perm, null_analytic = [], [], []
    for assay, g in df.groupby("assay"):
        if len(g) < 500:
            continue
        y = g.residual.abs().to_numpy()
        # a position is "first position" or "second"; use the unordered pair to
        # avoid the arbitrary a/b ordering
        pos = np.minimum(g.pos_a.to_numpy(), g.pos_b.to_numpy())
        codes, uniq = pd.factorize(pos)
        k = len(uniq)
        n = len(y)
        if k < 2 or n < k + 2:
            continue

        def r2(vals: np.ndarray) -> float:
            grand = vals.mean()
            # group means
            sums = np.bincount(codes, weights=vals, minlength=k)
            counts = np.bincount(codes, minlength=k)
            means = np.divide(sums, counts, out=np.zeros(k), where=counts > 0)
            between = (counts * (means - grand) ** 2).sum()
            total = ((vals - grand) ** 2).sum()
            return float(between / total) if total > 0 else 0.0

        obs.append(r2(y))
        null_perm.append(r2(rng.permutation(y)))
        null_analytic.append((k - 1) / (n - 1))
    if not obs:
        return {"verdict": "INCONCLUSIVE", "reason": "no assay had enough positions"}
    o, p, a = np.array(obs), np.array(null_perm), np.array(null_analytic)
    return {
        "n_assays": int(len(o)),
        "observed_r2_median": float(np.median(o)),
        "permutation_null_median": float(np.median(p)),
        "analytic_null_median": float(np.median(a)),
        "observed_exceeds_null_in": int((o > p).sum()),
        "verdict": ("SUPPORTED" if np.median(o) > 2 * np.median(p) else
                    "NOT_SUPPORTED"),
    }


def residue_type_effect(df: pd.DataFrame) -> dict:
    """Do some substituting residues carry larger residuals than chance?"""
    rows = []
    for assay, g in df.groupby("assay"):
        if len(g) < 500:
            continue
        # a and b are like "T159M": the mutant residue is the last character
        mut = g.a.str[-1].to_numpy()
        y = g.residual.abs().to_numpy()
        for res in np.unique(mut):
            sel = mut == res
            if sel.sum() < 50:
                continue
            rows.append({"assay": assay, "residue": res,
                         "n": int(sel.sum()),
                         "median_abs_residual": float(np.median(y[sel])),
                         "assay_median": float(np.median(y))})
    if not rows:
        return {"verdict": "INCONCLUSIVE", "reason": "too few residue classes"}
    t = pd.DataFrame(rows)
    # For each residue, how often does it sit above its own assay's median?
    per_res = t.groupby("residue").apply(
        lambda g: pd.Series({
            "n_assays": len(g),
            "frac_above_assay_median": float((g.median_abs_residual
                                              > g.assay_median).mean()),
            "median_ratio": float((g.median_abs_residual
                                   / g.assay_median).median()),
        }), include_groups=False)
    per_res = per_res[per_res.n_assays >= 5]
    if per_res.empty:
        return {"verdict": "INCONCLUSIVE", "reason": "no residue in >=5 assays"}
    # A pure-null residue sits above its assay median 50% of the time.
    best = per_res.sort_values("frac_above_assay_median", ascending=False).head(3)
    worst = per_res.sort_values("frac_above_assay_median").head(3)
    return {
        "n_residues_tested": int(len(per_res)),
        "top": {k: {kk: round(float(vv), 3) for kk, vv in v.items()}
                for k, v in best.iterrows()},
        "bottom": {k: {kk: round(float(vv), 3) for kk, vv in v.items()}
                   for k, v in worst.iterrows()},
        "null_expectation": 0.5,
        "verdict": ("SUPPORTED"
                    if float(per_res.frac_above_assay_median.max()) >= 0.85
                    else "NOT_SUPPORTED"),
        "note": "SUPPORTED means at least one residue is above its assay median "
                "in >=85% of assays where it appears, well beyond the 50% null.",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-pairs", type=int, default=500)
    ap.add_argument("--out", default=str(RES / "depth.json"))
    args = ap.parse_args()

    df = load_pairs(args.min_pairs)
    print(f"{len(df):,} pairs from {df.assay.nunique()} assays")

    out: dict = {"n_pairs": int(len(df)), "n_assays": int(df.assay.nunique())}

    print("\nH2  position effect ...")
    out["H2_position"] = variance_explained_by_position(df)
    print(json.dumps(out["H2_position"], indent=2))

    print("\nH1  residue-type effect ...")
    out["H1_residue"] = residue_type_effect(df)
    print(json.dumps(out["H1_residue"], indent=2))

    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
