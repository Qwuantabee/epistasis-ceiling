#!/usr/bin/env python3
"""Rebuild results/assay_summary.csv from the committed pair tables.

WHY THIS EXISTS. assay_summary.csv predated the reviewer fixes: it had no AUC, no
bias column and no centred-column statistics, so the README could not quote them
and audit_claims.py could not check them. Rebuilding needs the raw ProteinGym CSVs
via build_pairs.py - and those are ~1.8 GB and live outside the repo. The derived
pair tables ARE committed-adjacent artefacts with everything required, so this
recomputes the summary from them through the same summarise_table() that
build_pairs.py uses. One definition, two entry points.

Coverage is not recomputable from a pair table (it needs the count of doubles that
were never measured), so the existing coverage/singles columns are carried over
from the old file rather than invented.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from epsceil.pairs import summarise_table  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-dir", default=str(ROOT / "results" / "pairs"))
    ap.add_argument("--out", default=str(ROOT / "results" / "assay_summary.csv"))
    args = ap.parse_args()

    out = Path(args.out)
    old = {}
    if out.exists():
        prev = pd.read_csv(out)
        old = {r.assay: r for r in prev.itertuples()}

    rows = []
    for p in sorted(Path(args.pairs_dir).glob("*.csv")):
        t = pd.read_csv(p)
        prev = old.get(p.stem)
        rows.append(summarise_table(
            t, p.stem,
            coverage=float(prev.coverage) if prev is not None else float("nan"),
            n_singles=int(prev.n_singles_measured) if prev is not None else 0))
    if not rows:
        raise SystemExit("no pair files; run scripts/build_pairs.py first")
    df = pd.DataFrame(rows).sort_values("n_pairs", ascending=False)
    df.to_csv(out, index=False)
    print(f"wrote {out} with {len(df)} assays, {int(df.n_pairs.sum()):,} pairs")
    print(f"columns: {', '.join(df.columns)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
