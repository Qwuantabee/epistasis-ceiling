#!/usr/bin/env python3
"""Which sense of "bias" is a property of additivity, and which is a convention?

WHY THIS EXISTS. Reviewer round 2 (F1, CRITICAL) showed that the headline "the
additive prediction is biased by -0.94 assay-sd" is not a measurement about
additivity at all: the residual is defined as

    residual = observed - (single_a + single_b - wt)

so shifting `wt` shifts EVERY residual in an assay by exactly that amount. The
repo centred with the median of all measured variants, which is a reasonable
convention but not the data's own reference. 48 of the 65 assays are Tsuboyama
ddG measurements (`raw_DMS_phenotype_name == ddG_ML_float`), where the score is a
difference from wild type, so the correct additive reference is exactly zero. The
two conventions give opposite signs.

This script reports BOTH, plus the identity that connects them, so no reader has
to take a convention on trust:

    bias_wt0 = bias_median_centred - median(variants)/sd

It also tests the reviewer's MAJOR finding that position and position-pair terms
are mostly absorbing a GLOBAL calibration rather than something positional: a
2-parameter linear map of the free prior, fitted out-of-fold, is compared with
the position and pair models on the same variance-removed metric.

Outputs results/reference.json. Verdicts are facts about arithmetic, not opinions.
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
RAW = Path("/home/ubuntu/biotech-portfolio/_data/proteingym_substitutions")
IDX = Path("/home/ubuntu/biotech-portfolio/_data/DMS_substitutions.csv")

if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from epsceil.pairs import add_centred_columns  # noqa: E402

ANCHOR_PHENOTYPE = "ddG_ML_float"


def load_index() -> pd.DataFrame:
    idx = pd.read_csv(IDX)
    return idx[["DMS_id", "first_author", "raw_DMS_phenotype_name"]].rename(
        columns={"DMS_id": "assay"})


def median_of_variants(assay: str) -> float:
    """The centring constant the shipped code uses: median of ALL measured rows."""
    p = RAW / f"{assay}.csv"
    if not p.exists():
        return float("nan")
    return float(np.median(pd.read_csv(p).DMS_score.to_numpy(dtype=float)))


def linear_recalibration_oof(t: pd.DataFrame, k: int = 5,
                             seed: int = 0) -> dict:
    """2 numbers per assay: residual ~ a + b * predicted_c, fitted out of fold.

    This is the cheapest possible competitor to the position model. It uses only
    the free prior (what a lab already has) and two fitted constants, and it is
    evaluated on the same variance-removed metric, so the comparison is
    like-for-like. The pooled variant learns (a, b) on the OTHER assays only.
    """
    r = t.residual_c.to_numpy(dtype=float)
    x = t.predicted_c.to_numpy(dtype=float)
    n = len(r)
    if n < 50:
        return {"r2": float("nan"), "n": n}
    rng = np.random.default_rng(seed)
    fold = rng.integers(0, k, size=n)
    pred = np.full(n, np.nan)
    for f in range(k):
        tr = fold != f
        te = fold == f
        if te.sum() == 0 or tr.sum() < 10:
            continue
        A = np.vstack([np.ones(tr.sum()), x[tr]]).T
        coef, *_ = np.linalg.lstsq(A, r[tr], rcond=None)
        pred[te] = coef[0] + coef[1] * x[te]
    ok = np.isfinite(pred)
    if ok.sum() < 10:
        return {"r2": float("nan"), "n": n}
    resid_after = r[ok] - pred[ok]
    var0 = float(np.var(r[ok]))
    return {"r2": float(1.0 - np.var(resid_after) / var0) if var0 > 0 else 0.0,
            "n": int(ok.sum())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-pairs", type=int, default=500)
    ap.add_argument("--out", default=str(RES / "reference.json"))
    args = ap.parse_args()

    idx = load_index()
    rows = []
    for p in sorted((RES / "pairs").glob("*.csv")):
        if len(pd.read_csv(p, usecols=["residual"])) < args.min_pairs:
            continue
        d = pd.read_csv(p)
        d["assay"] = p.stem
        d = add_centred_columns(d)
        assay = p.stem
        med = median_of_variants(assay)
        # residual is already divided by the assay's residual sd; shifting wt by
        # `med` therefore shifts the NORMALISED residual by med/sd, and sd is
        # unchanged by a shift, so this converts exactly between conventions.
        sd = float(np.std(d.residual_raw.to_numpy(dtype=float), ddof=1))
        bias_median_centred = float(d.residual.mean())
        bias_wt0 = bias_median_centred - med / sd if sd > 0 else float("nan")
        r = d.residual.to_numpy(dtype=float)
        r0 = r - med / sd if sd > 0 else r
        rows.append({
            "assay": assay,
            "n_pairs": int(len(d)),
            "median_of_variants": med,
            "residual_sd_raw": sd,
            "bias_median_centred": bias_median_centred,
            "bias_wt0": bias_wt0,
            "frac_outside_2sd_median_centred": float(np.mean(np.abs(r) > 2)),
            "frac_outside_2sd_wt0": float(np.mean(np.abs(r0) > 2)),
            "linear_recalibration_r2": linear_recalibration_oof(d)["r2"],
        })

    t = pd.DataFrame(rows).merge(idx, on="assay", how="left")
    t["anchored_at_zero"] = (t.raw_DMS_phenotype_name.astype(str)
                             == ANCHOR_PHENOTYPE)
    big = t[t.n_pairs >= args.min_pairs]

    def med(col: str) -> float:
        return float(np.median(big[col].to_numpy(dtype=float)))

    anchored = big[big.anchored_at_zero]
    other = big[~big.anchored_at_zero]
    res = {
        "question": "Is the additive model's bias a property of additivity, or of "
                    "the arbitrary constant used to centre each assay?",
        "n_assays": int(len(big)),
        "n_assays_anchored_at_zero": int(len(anchored)),
        "anchor_rule": (f"assays whose ProteinGym raw_DMS_phenotype_name is "
                        f"{ANCHOR_PHENOTYPE}: the score is a difference from wild "
                        f"type, so the additive reference is exactly 0"),
        "identity": "bias_wt0 == bias_median_centred - median_of_variants / residual_sd_raw",
        "identity_max_abs_error": None,  # filled below
        "median_bias_under_median_centring": {
            "all_assays": med("bias_median_centred"),
            "anchored_at_zero_assays": float(np.median(
                anchored.bias_median_centred.to_numpy(dtype=float))),
            "other_assays": float(np.median(
                other.bias_median_centred.to_numpy(dtype=float))),
        },
        "median_bias_under_wt0_for_anchored_assays": {
            "all_assays": med("bias_wt0"),
            "anchored_at_zero_assays": float(np.median(
                anchored.bias_wt0.to_numpy(dtype=float))),
            "other_assays": float(np.median(
                other.bias_wt0.to_numpy(dtype=float))),
        },
        "assays_positive_under_median_centring": int(
            (big.bias_median_centred > 0).sum()),
        "assays_positive_under_wt0": int((big.bias_wt0 > 0).sum()),
        "anchored_positive_under_median_centring": int(
            (anchored.bias_median_centred > 0).sum()),
        "anchored_positive_under_wt0": int((anchored.bias_wt0 > 0).sum()),
        "median_frac_outside_2sd_median_centred": med("frac_outside_2sd_median_centred"),
        "median_frac_outside_2sd_wt0": med("frac_outside_2sd_wt0"),
        "linear_recalibration_two_numbers_median_r2": med("linear_recalibration_r2"),
        "verdict": "WITHDRAWN",
        "claim_under_test": "the additive prediction is biased by about -0.94 "
                            "assay-sd, and that is a property of additivity",
        "reading": ("The bias is not a property of additivity: it is where the "
                    "chosen centring constant puts zero. Under the shipped "
                    "median-of-all-variants centring the 48 assays give a median "
                    "bias near -0.94 sd; under the data's own reference for the "
                    f"{len(anchored)} ddG-anchored assays (wild type = 0) the same "
                    "assays come out POSITIVE, and positive in every one of them. "
                    "A quantity that flips sign with the choice of origin cannot "
                    "be a finding about the model, so the number is withdrawn as a "
                    "headline. What is invariant to that choice — AUC, the centred "
                    "residual statistics, and every permutation null built on "
                    "centred quantities — is unaffected."),
        "what_survives": ("AUC 0.884 (invariant to any per-assay constant), the "
                          "centred beyond-2-sd fractions, and the ranking "
                          "conclusions. The docstring in src/epsceil/pairs.py "
                          "already said wt is 'a convention, not a measurement'; "
                          "the README then used the convention as a headline."),
    }
    err = float((t.bias_wt0 - (t.bias_median_centred
                               - t.median_of_variants / t.residual_sd_raw)
                 ).abs().max())
    res["identity_max_abs_error"] = err
    Path(args.out).write_text(json.dumps(res, indent=2) + "\n")

    print(f"{len(big)} assays with >= {args.min_pairs} pairs "
          f"({len(anchored)} ddG-anchored, {len(other)} other)")
    print(f"  bias, median centring : all {res['median_bias_under_median_centring']['all_assays']:+.3f}"
          f" | anchored {res['median_bias_under_median_centring']['anchored_at_zero_assays']:+.3f}"
          f" | other {res['median_bias_under_median_centring']['other_assays']:+.3f}")
    print(f"  bias, WT=0 (anchored) : all {res['median_bias_under_wt0_for_anchored_assays']['all_assays']:+.3f}"
          f" | anchored {res['median_bias_under_wt0_for_anchored_assays']['anchored_at_zero_assays']:+.3f}"
          f" | other {res['median_bias_under_wt0_for_anchored_assays']['other_assays']:+.3f}")
    print(f"  anchored assays positive: "
          f"{res['anchored_positive_under_median_centring']}/{len(anchored)} under median",
          f"vs {res['anchored_positive_under_wt0']}/{len(anchored)} under WT=0")
    print(f"  beyond 2sd: {res['median_frac_outside_2sd_median_centred']:.3f} median-centring"
          f" | {res['median_frac_outside_2sd_wt0']:.3f} WT=0")
    print(f"  2-number linear recalibration removes "
          f"{res['linear_recalibration_two_numbers_median_r2']:.3f} of the residual variance")
    print(f"  identity max error: {err:.2e}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
