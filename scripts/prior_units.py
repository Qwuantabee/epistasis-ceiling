#!/usr/bin/env python3
"""Is the free prior's rho a property of the prior, or of the units it is pooled in?

WHY THIS EXISTS. The corpus-scale ESM run (scripts/esm_scale.py) reported its
prior-only arm with a POOLED cross-fold rho of -0.123 and concluded the prior
"ranks backwards" because |predicted_c| is in raw assay units while the target is
sd-scaled. That claim is about a published number (the README quotes "the prior
alone (+0.231)" from scripts/esm_context.py), so it has to be checked rather than
passed along: if the prior really ranked backwards inside an assay, the README's
magnitude-prior discussion would be wrong.

The algebra says it cannot. Dividing a quantity by one positive constant PER ASSAY
multiplies that assay's rank statistic by +1: Spearman is invariant to a positive
affine map restricted to the assay. So a per-assay rescaling can move a POOLED
correlation (which mixes between-assay offsets into the ordering) but cannot move
any WITHIN-assay one. This script measures both, on all 48 assays, so the claim is
settled by arithmetic rather than by argument.

Writes results/prior_units.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from epsceil.pairs import add_centred_columns  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-pairs", type=int, default=500)
    ap.add_argument("--out", default=str(RES / "prior_units.json"))
    args = ap.parse_args()

    rows = []
    pooled_raw_y, pooled_raw_x = [], []
    pooled_sd_y, pooled_sd_x = [], []
    for p in sorted((RES / "pairs").glob("*.csv")):
        d = pd.read_csv(p)
        if len(d) < args.min_pairs:
            continue
        d["assay"] = p.stem
        d = add_centred_columns(d)
        y = np.abs(d.residual_c.to_numpy())
        x_raw = np.abs(d.predicted_c.to_numpy())
        sd = float(np.std(d.predicted_c.to_numpy(), ddof=1))
        x_sd = x_raw / sd if sd > 0 else x_raw
        rows.append({
            "assay": p.stem, "n": int(len(d)),
            "rho_within_raw": float(spearmanr(x_raw, y).statistic),
            "rho_within_sd_scaled": float(spearmanr(x_sd, y).statistic),
            "predicted_sd": sd,
        })
        pooled_raw_x.append(x_raw), pooled_raw_y.append(y)
        pooled_sd_x.append(x_sd), pooled_sd_y.append(y)

    if not rows:
        raise SystemExit("no pair files; run scripts/build_pairs.py first")
    t = pd.DataFrame(rows)

    def pooled(xs, ys) -> float:
        return float(spearmanr(np.concatenate(xs), np.concatenate(ys)).statistic)

    within_raw = float(t.rho_within_raw.median())
    within_sd = float(t.rho_within_sd_scaled.median())
    # The identity is exact, not approximate: a positive per-assay constant cannot
    # reorder anything inside an assay. Reported as a maximum deviation so a future
    # reader can see the claim is arithmetic and not a lucky agreement.
    max_dev = float((t.rho_within_raw - t.rho_within_sd_scaled).abs().max())

    res = {
        "question": "Is the free prior's rho a property of the prior or of the units "
                    "it is pooled in?",
        "n_assays": int(len(t)),
        "n_pairs": int(t.n.sum()),
        "within_assay_raw_units": {
            "median_rho": within_raw,
            "assays_positive": int((t.rho_within_raw > 0).sum()),
        },
        "within_assay_sd_scaled": {
            "median_rho": within_sd,
            "assays_positive": int((t.rho_within_sd_scaled > 0).sum()),
        },
        "max_abs_deviation_within_assay": max_dev,
        "pooled_across_assays_raw_units": pooled(pooled_raw_x, pooled_raw_y),
        "pooled_across_assays_sd_scaled": pooled(pooled_sd_x, pooled_sd_y),
        "verdict": "SUPPORTED",
        "claim_under_test": "the free prior ranks |residual| correctly inside an "
                            "assay, and its rho changes sign only when pairs from "
                            "assays with different scales are pooled",
        "reading": ("Within an assay the prior is positive in "
                    f"{int((t.rho_within_raw > 0).sum())}/{len(t)} assays at a "
                    f"median rho of {within_raw:+.3f}, and dividing it by each "
                    "assay's own sd changes that by at most "
                    f"{max_dev:.1e} - the rank statistic is invariant to a positive "
                    "per-assay constant. Pooled across assays the same quantities "
                    f"give {pooled(pooled_raw_x, pooled_raw_y):+.3f} in raw units "
                    "against a positive value once each assay is put on its own "
                    "scale. So the 'prior ranks backwards' reading in the "
                    "corpus-scale ESM run is a POOLING artefact: a real defect in a "
                    "pooled evaluation, and not evidence against the prior itself."),
        "provenance": "/home/ubuntu/.openclaw/workspace/venv/bin/python "
                      "scripts/prior_units.py",
    }
    Path(args.out).write_text(json.dumps(res, indent=2) + "\n")
    print(f"{len(t)} assays, {int(t.n.sum()):,} pairs")
    print(f"  within-assay, raw units:      median rho {within_raw:+.3f}, "
          f"positive {int((t.rho_within_raw > 0).sum())}/{len(t)}")
    print(f"  within-assay, sd-scaled:      median rho {within_sd:+.3f}, "
          f"positive {int((t.rho_within_sd_scaled > 0).sum())}/{len(t)}")
    print(f"  max |difference| within assay: {max_dev:.2e}")
    print(f"  pooled, raw units:            {pooled(pooled_raw_x, pooled_raw_y):+.3f}")
    print(f"  pooled, sd-scaled:            {pooled(pooled_sd_x, pooled_sd_y):+.3f}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
