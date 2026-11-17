#!/usr/bin/env python3
"""Seed-aggregated verdict for the other labs' AUC against the corpus median.

Applies the rule in DESIGN.md ("Seed-aggregated verdict"), written and committed
before this script. The per-seed interval and verdict come from the unmodified
heldout.boot_median and heldout.verdict; results/heldout.json is not touched.

Usage: python scripts/seed_sweep.py [--summary CSV] [--seeds TXT] [--out JSON]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
sys.path.insert(0, str(Path(__file__).resolve().parent))
import heldout  # noqa: E402

REPS = 20000
MIN_PAIRS = 500
STABLE_SEEDS = 190          # of 200: DESIGN.md rule 1
RULE = ("One verdict on at least 190 of the 200 seeds is the seed-aggregated "
        "verdict; otherwise INCONCLUSIVE (seed-dependent), reported as the share "
        "of seeds behind each verdict.")


def load_seeds(path: Path) -> list[int]:
    seeds = [int(t) for t in path.read_text().split()]
    if seeds != list(range(200)):
        raise SystemExit(f"{path} is not the committed list 0..199")
    return seeds


def groups(summary: Path):
    s = pd.read_csv(summary)
    big = s[s.n_pairs >= MIN_PAIRS]
    other = big[~big.assay.str.contains("Tsuboyama")].auc_direction.to_numpy(float)
    corpus = float(np.median(big.auc_direction.to_numpy(float)))
    return big, other, corpus


def per_seed(other: np.ndarray, corpus: float, seed: int) -> dict:
    lo, hi = heldout.boot_median(other, reps=REPS, seed=seed)
    med = float(np.median(other))
    v = heldout.verdict(corpus, lo, hi, len(other), med)
    clearance = lo - corpus if med > corpus else corpus - hi
    return {"seed": seed, "lo": lo, "hi": hi, "verdict": v,
            "clearance": clearance}


def spread(x) -> dict:
    x = np.asarray(x, float)
    return {"median": float(np.median(x)),
            "p2_5": float(np.percentile(x, 2.5)),
            "p97_5": float(np.percentile(x, 97.5))}


def aggregate(rows: list[dict]) -> dict:
    n = len(rows)
    counts = Counter(r["verdict"] for r in rows)
    top, top_n = counts.most_common(1)[0]
    verdict = top if top_n >= STABLE_SEEDS else "INCONCLUSIVE (seed-dependent)"
    return {"n_seeds": n,
            "verdict_counts": dict(sorted(counts.items())),
            "verdict_share": {k: v / n for k, v in sorted(counts.items())},
            "clearance": spread([r["clearance"] for r in rows]),
            "lo": spread([r["lo"] for r in rows]),
            "hi": spread([r["hi"] for r in rows]),
            "n_seeds_excluding_corpus": sum(r["clearance"] > 0 for r in rows),
            "verdict": verdict}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default=str(RES / "assay_summary.csv"))
    ap.add_argument("--seeds", default=str(RES / "seed_sweep_seeds.txt"))
    ap.add_argument("--out", default=str(RES / "seed_sweep.json"))
    a = ap.parse_args()
    t0 = time.time()
    seeds = load_seeds(Path(a.seeds))
    big, other, corpus = groups(Path(a.summary))
    if len(big) != 48 or len(other) != 15:
        raise SystemExit(f"expected 48 corpus / 15 other-labs assays, got "
                         f"{len(big)} / {len(other)}")
    rows = [per_seed(other, corpus, s) for s in seeds]
    agg = aggregate(rows)
    out = {
        "question": "Does the other labs' median AUC differ from the corpus "
                    "median, judged over 200 bootstrap seeds instead of one?",
        "rule": RULE,
        "rule_source": "DESIGN.md, 'Seed-aggregated verdict'",
        "corpus_median_auc": corpus,
        "other_labs": {"n_assays": int(len(other)),
                       "median_auc": float(np.median(other))},
        "bootstrap": {"reps": REPS, "method": "percentile",
                      "function": "heldout.boot_median"},
        "single_seed_in_heldout_json": heldout.RNG_SEED,
        "aggregate": agg,
        "per_seed": rows,
        "provenance": {
            "command": "python scripts/seed_sweep.py",
            "script_sha256": hashlib.sha256(
                Path(__file__).read_bytes()).hexdigest(),
            "seeds_sha256": hashlib.sha256(
                Path(a.seeds).read_bytes()).hexdigest(),
            "runtime_s": round(time.time() - t0, 1)},
    }
    Path(a.out).write_text(json.dumps(out, indent=2) + "\n")
    c = agg["clearance"]
    print(f"corpus {corpus:.4f}; other labs {np.median(other):.4f} (n={len(other)})")
    print(f"verdicts {agg['verdict_counts']} -> {agg['verdict']}")
    print(f"clearance median {c['median']:+.4f} [{c['p2_5']:+.4f}, {c['p97_5']:+.4f}]")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
