#!/usr/bin/env python3
"""Build the additivity-residual table for every ProteinGym assay with doubles.

Writes one row per assay to results/assay_summary.csv, plus a per-pair table per
assay under results/pairs/. Nothing is modelled; this is measurement.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from epsceil.pairs import build_pairs, summarise  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True,
                    help="directory of ProteinGym <assay>.csv files")
    ap.add_argument("--index", default=None,
                    help="DMS_substitutions.csv; used to preselect assays")
    ap.add_argument("--min-multimutants", type=int, default=200)
    ap.add_argument("--min-singles", type=int, default=50)
    ap.add_argument("--out", default=str(ROOT / "results"))
    args = ap.parse_args()

    data = Path(args.data)
    if not data.exists():
        raise SystemExit(f"missing {data}")

    if args.index:
        idx = pd.read_csv(args.index)
        ids = idx[idx.DMS_number_multiple_mutants > args.min_multimutants].DMS_id
        ids = [a for a in ids if (data / f"{a}.csv").exists()]
    else:
        ids = sorted(p.stem for p in data.glob("*.csv") if p.stem != "DMS_substitutions")

    out = Path(args.out)
    (out / "pairs").mkdir(parents=True, exist_ok=True)

    rows, skipped = [], []
    for i, assay in enumerate(ids, 1):
        try:
            df = pd.read_csv(data / f"{assay}.csv")
        except Exception as exc:  # noqa: BLE001
            skipped.append((assay, f"read error: {exc}"))
            continue
        ap_ = build_pairs(df, assay, min_singles=args.min_singles)
        if ap_ is None:
            skipped.append((assay, "insufficient singles or doubles"))
            continue
        ap_.doubles.to_csv(out / "pairs" / f"{assay}.csv", index=False)
        s = summarise(ap_)
        rows.append(s)
        if i % 10 == 0:
            print(f"  {i}/{len(ids)} {assay}: {s['n_pairs']:,} pairs, "
                  f"spearman {s['spearman_additive']:.3f}", flush=True)

    if not rows:
        raise SystemExit("no assays produced pairs")
    summary = pd.DataFrame(rows).sort_values("n_pairs", ascending=False)
    summary.to_csv(out / "assay_summary.csv", index=False)
    (out / "skipped.txt").write_text("\n".join(f"{a}\t{r}" for a, r in skipped) + "\n")

    total = int(summary.n_pairs.sum())
    print(f"\nassays used : {len(summary)}  (skipped {len(skipped)})")
    print(f"total pairs : {total:,}")
    print(f"median assay spearman (additivity vs observed): "
          f"{summary.spearman_additive.median():.3f}")
    print(f"median fraction of pairs within 1 residual sd: "
          f"{summary.frac_within_1sd.median():.3f}")
    print(f"-> {out/'assay_summary.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
