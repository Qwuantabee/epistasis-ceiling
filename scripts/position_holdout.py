#!/usr/bin/env python3
"""Q1: does the per-assay position R2 survive holding out WHOLE position pairs?

scripts/alt_baseline.py fits the position models per assay with an inner RANDOM
split over pairs, so a held-out double almost always shares both of its
positions - and its position pair - with training doubles. That is the most
favourable split a position term can get. This script refits the SAME models
(epsceil.baseline: fit_position, cell_sums, lowrank_pair_term, scored by
r2_resid_var; nothing about the model or the metric is redefined here) under two
inner splits, on the SAME 48 assays (>= 500 pairs):

  random_pair          KFold(5, shuffle, random_state=seed) over pairs: this IS
                       epsceil.baseline.crossfit_assay, called directly, so it
                       must reproduce results/alt_baseline.json (checked below)
  whole_position_pair  GroupKFold(min(5, n_cells)) with one group per unordered
                       position pair {pos_a, pos_b} ("cell"): no position pair in
                       a test fold has ANY double in that fold's training data

Ridge lambda is chosen exactly as alt_baseline.py chooses it, by calling its
outer_selection (GroupKFold(5) over assays, argmax of the median inner-CV R2
over the TRAINING assays), separately for each split, for the real fits and for
every permutation.

THE NULL is alt_baseline's: the identical pipeline with residuals permuted among
the training pairs of each inner fold (same per-assay RNG), scored against the
real held-out residuals.

WHY GROUP COUNT MATTERS. Many assays (mostly Tsuboyama) have doubles at only a
handful of position pairs. With 2 disjoint cells, a whole-pair holdout leaves a
test cell whose positions never occur in training, so the position model can
only predict a constant for it. Every assay therefore also reports, for its
held-out pairs, how many of their two positions occur in that fold's training
pairs, and R2 is reported by n_cells bin rather than as one median.

THE VERDICT RULE (RULE below) was written into this file before any R2 under
either split was computed, and is copied verbatim into the artefact.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from epsceil.baseline import (LAMBDAS, RANKS, cell_sums,  # noqa: E402
                              crossfit_assay, fit_position, lowrank_pair_term,
                              position_codes, r2_resid_var)

MIN_PAIRS = 500
N_SPLITS = 5
MODELS = ["position"] + [f"pair_rank{k}" for k in RANKS]
REPORT = ("position", "pair_rank3")
SPLITS = ("random_pair", "whole_position_pair")
TIE = 1e-9
BOOT = 5000
BOOT_SEED = 20261116
# n_cells bins, fixed from the DESIGN (count of distinct position pairs per
# assay, read from results/alt_baseline.json per_assay.n_cells) before any R2
# under either split was computed: 9 assays have 2 cells, 23 have 3-9, 16 >= 10.
CELL_BINS = (("2", 2, 2), ("3-9", 3, 9), (">=10", 10, None))
RULE = {
    "statistic": (
        "per-assay out-of-fold r2_resid_var (epsceil.baseline.r2_resid_var) at the "
        "outer-selected ridge lambda, under the whole_position_pair split, compared "
        "with the same assay under the random_pair split"),
    "evaluated_for": "position and pair_rank3, on all assays and within each n_cells bin (2, 3-9, >=10)",
    "ratio": "median over the stratum's assays of whole-pair R2 / median of random-pair R2 (same assays)",
    "beats_null": "share of the stratum's assays whose whole-pair R2 exceeds the mean of its own permutation nulls by > 1e-9",
    "INCONCLUSIVE": "fewer than 6 assays in the stratum, or median random-pair R2 <= 0 (ratio undefined)",
    "HOLDS": "ratio >= 0.75 AND beats_null >= 0.75",
    "COLLAPSES": "ratio < 0.25 OR beats_null < 0.50",
    "PARTIAL": "anything else",
    "order": "INCONCLUSIVE, then HOLDS, then COLLAPSES, else PARTIAL (HOLDS and COLLAPSES are mutually exclusive)",
    "paired_difference": (
        "per-assay (whole-pair R2 - random-pair R2); median with a 95% percentile "
        f"bootstrap interval over assays ({BOOT} resamples, seed {BOOT_SEED}); "
        "reported, not used by the rule"),
    "written": "before any R2 under either split was computed",
}


def _load_alt():
    """scripts/alt_baseline.py as a module: its outer_selection and _rng ARE the
    lambda-selection rule and the null's RNG, so they are called, not copied."""
    spec = importlib.util.spec_from_file_location(
        "alt_baseline", ROOT / "scripts" / "alt_baseline.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ALT = _load_alt()
_key = ALT._key
_rng = ALT._rng


def cell_codes(pos_a: np.ndarray, pos_b: np.ndarray) -> tuple[np.ndarray, int]:
    """One integer per unordered position pair {pos_a, pos_b}."""
    lo = np.minimum(pos_a, pos_b)
    hi = np.maximum(pos_a, pos_b)
    _, inv = np.unique(np.stack([lo, hi], axis=1), axis=0, return_inverse=True)
    inv = np.asarray(inv).ravel()
    return inv, int(inv.max()) + 1


def random_pair_splits(n: int, seed: int = 0) -> list:
    """The split crossfit_assay uses, materialised (for the equivalence test)."""
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    return list(kf.split(np.arange(n)))


def whole_pair_splits(cells: np.ndarray) -> list:
    """GroupKFold over position pairs; leave-one-cell-out below 5 cells."""
    from sklearn.model_selection import GroupKFold
    k = min(N_SPLITS, int(len(np.unique(cells))))
    if k < 2:
        raise ValueError("a whole-position-pair holdout needs >= 2 position pairs")
    gkf = GroupKFold(n_splits=k)
    return list(gkf.split(np.zeros((len(cells), 1)), groups=cells))


def crossfit_splits(t: pd.DataFrame, splits, lambdas=LAMBDAS, ranks=RANKS,
                    permute_rng: np.random.Generator | None = None) -> dict:
    """crossfit_assay with the inner split passed in instead of KFold.

    The body is crossfit_assay's, line for line, over the same model functions;
    tests/test_position_holdout.py pins that with random_pair_splits() this
    returns crossfit_assay's predictions exactly (real and permuted), so the
    only thing that differs between the two splits is which pairs are held out.
    """
    r = t["residual"].to_numpy(dtype=float)
    ia, ib, n_pos = position_codes(t["pos_a"].to_numpy(), t["pos_b"].to_numpy())
    n = len(r)
    out = {("assay_offset", None): np.full(n, np.nan)}
    for lam in lambdas:
        out[("position", lam)] = np.full(n, np.nan)
        for k in ranks:
            out[(f"pair_rank{k}", lam)] = np.full(n, np.nan)
    for tr, te in splits:
        r_tr = r[tr] if permute_rng is None else permute_rng.permutation(r[tr])
        out[("assay_offset", None)][te] = float(r_tr.mean())
        for lam in lambdas:
            mu, alpha = fit_position(ia[tr], ib[tr], r_tr, n_pos, lam)
            base_tr = mu + alpha[ia[tr]] + alpha[ib[tr]]
            base_te = mu + alpha[ia[te]] + alpha[ib[te]]
            out[("position", lam)][te] = base_te
            s, c = cell_sums(ia[tr], ib[tr], r_tr - base_tr, n_pos)
            for k in ranks:
                low = lowrank_pair_term(s, c, lam, k)
                out[(f"pair_rank{k}", lam)][te] = base_te + low[ia[te], ib[te]]
    for key, v in out.items():
        assert np.isfinite(v).all(), f"unfilled out-of-fold prediction {key}"
    return out


def positions_seen(ia, ib, n_pos, splits) -> np.ndarray:
    """For each held-out pair: how many of its 2 positions occur in training."""
    seen = np.full(len(ia), -1)
    for tr, te in splits:
        m = np.zeros(n_pos, dtype=bool)
        m[ia[tr]] = True
        m[ib[tr]] = True
        seen[te] = m[ia[te]].astype(int) + m[ib[te]].astype(int)
    assert (seen >= 0).all()
    return seen


# ---------------------------------------------------------------- per assay
def run_assay(job: tuple) -> dict:
    path, n_perm, seed = job
    assay = Path(path).stem
    t = pd.read_csv(path)
    r = t["residual"].to_numpy(dtype=float)
    ia, ib, n_pos = position_codes(t["pos_a"].to_numpy(), t["pos_b"].to_numpy())
    cells, n_cells = cell_codes(t["pos_a"].to_numpy(), t["pos_b"].to_numpy())
    wsplits = whole_pair_splits(cells)
    for tr, te in wsplits:          # the defining property, checked on real data
        assert not np.intersect1d(cells[tr], cells[te]).size
    seen = positions_seen(ia, ib, n_pos, wsplits)

    def r2s(preds):
        return {_key(m, lam): r2_resid_var(r, v)
                for (m, lam), v in preds.items() if m != "assay_offset"}

    res = {"random_pair": {
               "real": r2s(crossfit_assay(t, seed=seed)),
               "null": [r2s(crossfit_assay(t, seed=seed,
                                           permute_rng=_rng(assay, j)))
                        for j in range(n_perm)]},
           "whole_position_pair": {
               "real": r2s(crossfit_splits(t, wsplits)),
               "null": [r2s(crossfit_splits(t, wsplits,
                                            permute_rng=_rng(assay, j)))
                        for j in range(n_perm)]}}
    return {"assay": assay, "path": str(path), "n_pairs": int(len(t)),
            "n_positions": int(n_pos), "n_cells": int(n_cells),
            "whole_pair_n_folds": len(wsplits),
            "whole_pair_test_positions_seen": {
                "both": float(np.mean(seen == 2)),
                "one": float(np.mean(seen == 1)),
                "none": float(np.mean(seen == 0))},
            "bias": float(r.mean()), "r2": res}


# ---------------------------------------------------------------- summary
def select(per: list[dict], split: str, n_perm: int) -> tuple[dict, list]:
    """alt_baseline.outer_selection on this split's inner-CV R2."""
    p1 = [{"assay": d["assay"], "bias": d["bias"],
           "null_bias": [d["bias"]] * n_perm,
           "r2_real": d["r2"][split]["real"],
           "r2_null": d["r2"][split]["null"]} for d in per]
    saved = ALT.FITTED
    ALT.FITTED = list(MODELS)
    try:
        plan, folds = ALT.outer_selection(p1, n_perm)
    finally:
        ALT.FITTED = saved
    for f in folds:                       # offsets are not part of this question
        f.pop("pooled_offset", None)
        f.pop("pooled_offset_null", None)
    return plan, folds


def chosen_r2(d: dict, split: str, plan: dict, n_perm: int) -> dict:
    pa = plan[d["assay"]]
    out = {}
    for m in MODELS:
        real = d["r2"][split]["real"][_key(m, pa["real"][m])]
        null = [d["r2"][split]["null"][j][_key(m, pa["null"][j][m])]
                for j in range(n_perm)]
        out[m] = {"r2": float(real), "lambda": pa["real"][m],
                  "null_mean": float(np.mean(null)) if null else float("nan"),
                  "null_by_perm": [float(x) for x in null]}
    return out


def boot_median_ci(x: np.ndarray) -> list[float]:
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(BOOT_SEED)
    meds = np.median(x[rng.integers(0, len(x), size=(BOOT, len(x)))], axis=1)
    return [float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))]


def verdict_for(rand: np.ndarray, whole: np.ndarray, whole_null: np.ndarray
                ) -> dict:
    """RULE, applied mechanically."""
    n = len(rand)
    med_r, med_w = float(np.median(rand)), float(np.median(whole))
    beats = float(np.mean(whole - whole_null > TIE))
    if n < 6 or med_r <= 0:
        v, ratio = "INCONCLUSIVE", (med_w / med_r if med_r > 0 else None)
    else:
        ratio = med_w / med_r
        if ratio >= 0.75 and beats >= 0.75:
            v = "HOLDS"
        elif ratio < 0.25 or beats < 0.50:
            v = "COLLAPSES"
        else:
            v = "PARTIAL"
    return {"verdict": v, "n_assays": n, "ratio": ratio,
            "beats_null_share": beats,
            "median_random_pair_r2": med_r, "median_whole_pair_r2": med_w}


def _bin(n_cells: int) -> str:
    for name, lo, hi in CELL_BINS:
        if n_cells >= lo and (hi is None or n_cells <= hi):
            return name
    raise ValueError(n_cells)


def summarise(rows: list[dict]) -> dict:
    out = {"models": {}, "verdict": {}}
    strata = [("all", rows)] + [(name, [r for r in rows if r["bin"] == name])
                                for name, _, _ in CELL_BINS]
    for m in MODELS:
        out["models"][m] = {}
        for sname, rs in strata:
            if not rs:
                continue
            rand = np.array([r["random_pair"][m]["r2"] for r in rs])
            whole = np.array([r["whole_position_pair"][m]["r2"] for r in rs])
            rnull = np.array([r["random_pair"][m]["null_mean"] for r in rs])
            wnull = np.array([r["whole_position_pair"][m]["null_mean"]
                              for r in rs])
            diff = whole - rand
            out["models"][m][sname] = {
                "n_assays": len(rs),
                "median_n_cells": float(np.median([r["n_cells"] for r in rs])),
                "random_pair": {
                    "median_r2": float(np.median(rand)),
                    "null_median_r2": float(np.median(rnull)),
                    "assays_above_null": f"{int((rand - rnull > TIE).sum())}/{len(rs)}"},
                "whole_position_pair": {
                    "median_r2": float(np.median(whole)),
                    "null_median_r2": float(np.median(wnull)),
                    "assays_above_null": f"{int((whole - wnull > TIE).sum())}/{len(rs)}",
                    "median_share_test_pairs_both_positions_seen": float(np.median(
                        [r["seen"]["both"] for r in rs])),
                    "median_share_test_pairs_no_position_seen": float(np.median(
                        [r["seen"]["none"] for r in rs]))},
                "paired_whole_minus_random": {
                    "median": float(np.median(diff)),
                    "ci95_bootstrap": boot_median_ci(diff),
                    "assays_whole_lower": f"{int((diff < -TIE).sum())}/{len(rs)}"},
            }
            if m in REPORT:
                out["verdict"].setdefault(m, {})[sname] = verdict_for(
                    rand, whole, wnull)
    return out


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(*args) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception as e:  # pragma: no cover
        return f"unavailable: {e}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pairs-dir", default="results/pairs")
    ap.add_argument("--out", default="results/position_holdout.json")
    ap.add_argument("--committed", default="results/alt_baseline.json",
                    help="read-only: the random-split numbers to reproduce")
    ap.add_argument("--n-perm", type=int, default=10)
    ap.add_argument("--min-pairs", type=int, default=MIN_PAIRS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--only", nargs="*", default=None,
                    help="restrict to these assay stems (smoke tests)")
    args = ap.parse_args()
    t0 = time.time()
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")

    paths = []
    for p in sorted((ROOT / args.pairs_dir).resolve().glob("*.csv")):
        if args.only and p.stem not in args.only:
            continue
        with open(p) as f:
            n = sum(1 for _ in f) - 1
        if n >= args.min_pairs:
            paths.append(p)
    if not paths:
        raise SystemExit("no pair files; run scripts/build_pairs.py first")
    paths.sort(key=lambda p: -p.stat().st_size)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        per = list(ex.map(run_assay, [(str(p), args.n_perm, args.seed)
                                      for p in paths]))
    per.sort(key=lambda d: d["assay"])
    print(f"fits done: {len(per)} assays, {time.time() - t0:.0f}s", flush=True)

    plans, folds = {}, {}
    for s in SPLITS:
        plans[s], folds[s] = select(per, s, args.n_perm)
    rows = []
    for d in per:
        rows.append({"assay": d["assay"], "n_pairs": d["n_pairs"],
                     "n_positions": d["n_positions"], "n_cells": d["n_cells"],
                     "bin": _bin(d["n_cells"]),
                     "whole_pair_n_folds": d["whole_pair_n_folds"],
                     "seen": d["whole_pair_test_positions_seen"],
                     **{s: chosen_r2(d, s, plans[s], args.n_perm)
                        for s in SPLITS}})
    summary = summarise(rows)

    # reproduction of the committed random-split numbers (read-only)
    repro = {}
    cpath = ROOT / args.committed
    if cpath.exists() and not args.only:
        c = json.loads(cpath.read_text())
        cmed = {m["model"]: m["median"]["r2_resid_var"] for m in c["models"]}
        cper = {a["assay"]: a["models"] for a in c["per_assay"]}
        for m in MODELS:
            name = "additive+position" + ("" if m == "position" else f"+{m}")
            ours = summary["models"][m]["all"]["random_pair"]["median_r2"]
            worst = max(abs(r["random_pair"][m]["r2"]
                            - cper[r["assay"]][name]["r2_resid_var"])
                        for r in rows)
            repro[m] = {"ours_median": ours, "committed_median": cmed[name],
                        "max_abs_per_assay_diff": worst,
                        "exact": bool(worst <= 1e-12)}

    cells = np.array([r["n_cells"] for r in rows])
    code = {p: _sha256(ROOT / p) for p in ("scripts/position_holdout.py",
                                           "scripts/alt_baseline.py",
                                           "src/epsceil/baseline.py",
                                           "src/epsceil/pairs.py")}
    pairs_digest = hashlib.sha256("".join(
        f"{p.name}:{_sha256(p)}\n" for p in sorted(paths)).encode()).hexdigest()
    out = {
        "question": ("Does the per-assay position R2 survive when whole position "
                     "pairs, not random pairs, are held out?"),
        "rule": RULE,
        "provenance": {
            "command": ("python scripts/position_holdout.py "
                        + " ".join(sys.argv[1:])).strip(),
            "cwd": "repository root",
            "interpreter": sys.executable,
            "python": platform.python_version(),
            "versions": {m.__name__: m.__version__ for m in (np, pd)} | {
                "sklearn": __import__("sklearn").__version__},
            "git_head": _git("rev-parse", "HEAD"),
            "code_uncommitted_at_run": _git("status", "--porcelain", "--",
                                            *code.keys()),
            "code_sha256": code,
            "pairs_sha256_of_file_hashes": pairs_digest,
            "started": started,
            "runtime_s": round(time.time() - t0, 1),
        },
        "design": {
            "assays": f"{len(rows)} assays with >= {args.min_pairs} pairs",
            "models": ("epsceil.baseline position and pair_rank{1,2,3}, unchanged; "
                       "random_pair predictions come from crossfit_assay itself"),
            "random_pair": "KFold(5, shuffle, random_state=seed) over pairs (crossfit_assay)",
            "whole_position_pair": ("GroupKFold(min(5, n_cells)) with one group per "
                                    "unordered position pair; leave-one-cell-out when "
                                    "an assay has < 5 cells"),
            "lambda": ("alt_baseline.outer_selection, run separately per split, for "
                       "the real fits and for every permutation"),
            "metric": "epsceil.baseline.r2_resid_var, out of fold, pooled over the assay's folds",
            "null": (f"{args.n_perm} permutations per assay: residuals permuted among "
                     "the training pairs of each inner fold (alt_baseline._rng), "
                     "scored against the real held-out residuals"),
            "positions_seen": ("for each held-out pair under whole_position_pair: "
                               "how many of its two positions occur in that fold's "
                               "training pairs (a position never seen in training "
                               "has alpha = 0, so the model can only fall back to "
                               "the intercept)"),
            "cell_bins": [list(b) for b in CELL_BINS],
        },
        "n_assays": len(rows),
        "n_pairs": int(sum(r["n_pairs"] for r in rows)),
        "position_pairs_per_assay": {
            "median": float(np.median(cells)),
            "quantiles_0_25_50_75_100": [float(q) for q in
                                         np.percentile(cells, [0, 25, 50, 75, 100])],
            "assays_per_bin": {name: int(sum(r["bin"] == name for r in rows))
                               for name, _, _ in CELL_BINS},
            "note": ("bimodal: most assays with < 10 cells have doubles at only a "
                     "few position pairs (2 disjoint cells in the smallest bin), so "
                     "a whole-pair holdout there predicts cells whose positions are "
                     "largely unseen; read R2 by bin, not as one median"),
        },
        "reproduces_committed_random_split": repro,
        "summary": summary["models"],
        "verdict": summary["verdict"],
        "outer_folds": folds,
        "per_assay": rows,
    }
    out_path = ROOT / args.out
    out_path.write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(f"wrote {out_path} in {time.time() - t0:.0f}s")
    for m in REPORT:
        for sname, v in summary["models"][m].items():
            print(f"{m:11s} {sname:5s} n={v['n_assays']:2d} random "
                  f"{v['random_pair']['median_r2']:+.3f} "
                  f"(null {v['random_pair']['null_median_r2']:+.3f}, "
                  f"{v['random_pair']['assays_above_null']}) whole "
                  f"{v['whole_position_pair']['median_r2']:+.3f} "
                  f"(null {v['whole_position_pair']['null_median_r2']:+.3f}, "
                  f"{v['whole_position_pair']['assays_above_null']}) "
                  f"verdict {summary['verdict'][m][sname]['verdict']}")
    for m, v in repro.items():
        print("repro", m, v)
    return 0


if __name__ == "__main__":
    sys.exit(main())
