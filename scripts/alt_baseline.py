#!/usr/bin/env python3
"""Does a stronger baseline than plain additivity close the gap?

The README measures plain additivity - predicted = s_a + s_b - wt, nothing fitted
- and finds AUC 0.884, 12.6% of pairs beyond 2 sd (per-assay median) and a
residual bias of -0.94 sd. Its own Limits section concedes that this is the
crudest baseline a lab could use. This script measures whether a stronger one
shrinks those numbers, like-for-like: the SAME 48 assays (>= 500 pairs), the SAME
per-assay scaling (plain additivity's residual sd), and the SAME summary function
(epsceil.pairs.summarise_table via epsceil.baseline.score_correction).

Models (all corrections to the additivity residual, see src/epsceil/baseline.py):

  additive                 nothing fitted (recomputed, not quoted)
  additive+pooled_offset   one constant learned on OTHER assays (GroupKFold(5)
                           by assay); no residual of the scored assay enters it,
                           but it is applied in that assay's residual-sd units,
                           and that sd itself comes from the assay's doubles
  additive+assay_offset    the assay's own mean residual, inner 5-fold split
  additive+position        + alpha_p + alpha_q per assay, inner 5-fold split
  additive+position+pair_rank{1,2,3}
                           + rank-k position x position term, inner 5-fold split

Position terms are protein-specific, so they cannot be carried from one assay to
another; they are fitted per assay on 4/5 of its pairs and scored on the held-out
1/5. The pooled ridge strength is the only thing learned across assays, and it is
chosen by GroupKFold(5) over assays: for every outer fold, lambda maximises the
median inner-CV R2 over the TRAINING assays and is then applied to the test
assays. No protein both chooses lambda and is scored with it.

Every fitted model has a null: the identical pipeline (including the outer lambda
selection) with the residuals permuted among the training pairs of each inner
fold, scored against the real held-out residuals. A permutation keeps the mean,
so an offset survives its own null by construction; what the null removes is
positional structure.

Two-phase so the expensive summary runs only for the chosen lambda: phase 1
computes inner-CV R2 for every (model, lambda); the outer GroupKFold picks lambda;
phase 2 recomputes the chosen fits (deterministic seeds, identical predictions)
and scores them through summarise_table.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from epsceil.baseline import (LAMBDAS, RANKS, crossfit_assay,  # noqa: E402
                              position_codes, r2_resid_var, score_correction)
from epsceil.pairs import add_centred_columns, summarise_table  # noqa: E402

MIN_PAIRS = 500          # the README's correlation subset: 48 assays
METRICS = ("r2_resid_var", "mse_skill", "auc_direction", "frac_outside_2sd",
           "frac_outside_2sd_centred", "bias_mean_sd", "spearman_additive")
FITTED = ["position"] + [f"pair_rank{k}" for k in RANKS]
ROWS = (["additive", "additive+pooled_offset", "additive+assay_offset",
         "additive+position"]
        + [f"additive+position+pair_rank{k}" for k in RANKS])
README_PLAIN = {"auc_direction": 0.884, "frac_outside_2sd": 0.126,
                "bias_mean_sd": -0.94}
TIE = 1e-9               # "beats" means by more than float noise
_OWN = ("fitted on 4/5 of the scored assay's own doubles (inner KFold(5)); "
        "ridge lambda chosen by GroupKFold(5) over the other assays")
USES = {
    "additive": "nothing fitted",
    "additive+pooled_offset": (
        "one constant from the OTHER assays (GroupKFold(5) by assay); no residual "
        "of the scored assay enters it, but it is applied in that assay's "
        "residual-sd units, which come from its doubles"),
    "additive+assay_offset": ("fitted on 4/5 of the scored assay's own doubles "
                              "(inner KFold(5))"),
    "additive+position": _OWN,
    **{f"additive+position+pair_rank{k}": _OWN for k in RANKS},
}
CAVEATS = [
    "Every per-assay row is fitted on 4/5 of the scored protein's own measured "
    "doubles. A lab that has only singles has none of them, so these rows bound "
    "what a stronger baseline can do; they are not a singles-only predictor.",
    "The inner split is random over pairs, so a held-out pair usually shares both "
    "positions (and often a single mutation) with training pairs. That is the "
    "most favourable split for position terms; holding out whole position pairs "
    "was not run.",
    "Position terms can absorb a global nonlinearity (for example a floor or "
    "saturation of the assay), because single-effect size clusters by position. "
    "A monotone global-epistasis fit was not run, so the position R2 is not "
    "evidence of position-SPECIFIC epistasis.",
    "r2_resid_var is out-of-fold R2 of the SIGNED residual. The README's position "
    "figure (0.140) is in-sample R2 of |residual| grouped by position pair; the two "
    "are different quantities and are not comparable.",
    "The pooled offset is expressed in each scored assay's residual-sd units; "
    "turning it into raw units needs that assay's residual sd, which comes from "
    "its doubles.",
]


def _rng(assay: str, rep: int) -> np.random.Generator:
    return np.random.default_rng([zlib.crc32(assay.encode()), rep])


def _key(model: str, lam) -> str:
    return f"{model}|{lam}"


# ---------------------------------------------------------------- phase 1
def phase1(job: tuple) -> dict:
    """Inner-CV R2 for every (model, lambda), real and per permutation."""
    path, n_perm, seed = job
    assay = Path(path).stem
    t = pd.read_csv(path)
    r = t["residual"].to_numpy(dtype=float)
    _, _, n_pos = position_codes(t["pos_a"].to_numpy(), t["pos_b"].to_numpy())
    cells = int(pd.MultiIndex.from_arrays(
        [np.minimum(t.pos_a, t.pos_b), np.maximum(t.pos_a, t.pos_b)]).nunique())

    def r2s(preds):
        return {_key(m, lam): r2_resid_var(r, v) for (m, lam), v in preds.items()}

    real = r2s(crossfit_assay(t, seed=seed))
    null = [r2s(crossfit_assay(t, seed=seed, permute_rng=_rng(assay, j)))
            for j in range(n_perm)]
    # the pooled offset's null input: the assay mean of PERMUTED residuals,
    # computed rather than assumed (it equals the real mean - that is the point)
    null_bias = [float(_rng(assay, 10_000 + j).permutation(r).mean())
                 for j in range(n_perm)]
    return {"assay": assay, "path": str(path), "n_pairs": int(len(t)),
            "n_positions": n_pos, "n_cells": cells, "bias": float(r.mean()),
            "null_bias": null_bias, "r2_real": real, "r2_null": null}


# ---------------------------------------------------------------- selection
def outer_selection(p1: list[dict], n_perm: int) -> tuple[dict, list]:
    """GroupKFold(5) by assay: pick lambda and the pooled offset on train assays."""
    from sklearn.model_selection import GroupKFold

    assays = [d["assay"] for d in p1]
    by = {d["assay"]: d for d in p1}
    gkf = GroupKFold(n_splits=5)
    plan = {a: {"fold": None, "real": {}, "null": [{} for _ in range(n_perm)]}
            for a in assays}
    folds = []
    for fold, (tr, te) in enumerate(gkf.split(np.zeros((len(assays), 1)),
                                              groups=np.arange(len(assays)))):
        train = [assays[i] for i in tr]
        test = [assays[i] for i in te]
        rec = {"fold": fold, "test_assays": test, "lambda": {}, "null_lambda": {}}

        def pick(model, src):
            med = {lam: float(np.median([src(a)[_key(model, lam)]
                                         for a in train])) for lam in LAMBDAS}
            # ties go to the larger lambda (more shrinkage = more conservative)
            return max(LAMBDAS, key=lambda lam: (med[lam], lam)), med

        for m in FITTED:
            lam, med = pick(m, lambda a: by[a]["r2_real"])
            rec["lambda"][m] = {"chosen": lam, "train_median_r2": med}
            rec["null_lambda"][m] = []
            for j in range(n_perm):
                lj, _ = pick(m, lambda a, j=j: by[a]["r2_null"][j])
                rec["null_lambda"][m].append(lj)
                for a in test:
                    plan[a]["null"][j][m] = lj
            for a in test:
                plan[a]["real"][m] = lam
        c = float(np.median([by[a]["bias"] for a in train]))
        c_null = [float(np.median([by[a]["null_bias"][j] for a in train]))
                  for j in range(n_perm)]
        rec["pooled_offset"] = c
        rec["pooled_offset_null"] = c_null
        for a in test:
            plan[a]["fold"] = fold
            plan[a]["real"]["pooled_offset"] = c
            for j in range(n_perm):
                plan[a]["null"][j]["pooled_offset"] = c_null[j]
        folds.append(rec)
    assert all(plan[a]["fold"] is not None for a in assays)
    return plan, folds


# ---------------------------------------------------------------- phase 2
def _score_plan(t, assay, preds, choice) -> dict:
    out = {"additive+pooled_offset": score_correction(
               t, np.full(len(t), choice["pooled_offset"]), assay),
           "additive+assay_offset": score_correction(
               t, preds[("assay_offset", None)], assay),
           "additive+position": score_correction(
               t, preds[("position", choice["position"])], assay)}
    for k in RANKS:
        m = f"pair_rank{k}"
        out[f"additive+position+{m}"] = score_correction(
            t, preds[(m, choice[m])], assay)
    return out


def phase2(job: tuple) -> dict:
    path, plan, seed = job
    assay = Path(path).stem
    t = pd.read_csv(path)
    plain = score_correction(t, np.zeros(len(t)), assay)
    ref = summarise_table(add_centred_columns(t), assay)
    lams = sorted({plan["real"][m] for m in FITTED})
    real = _score_plan(t, assay, crossfit_assay(t, lambdas=lams, seed=seed),
                       plan["real"])
    real["additive"] = plain
    null = []
    for j, choice in enumerate(plan["null"]):
        lams_j = sorted({choice[m] for m in FITTED})
        preds = crossfit_assay(t, lambdas=lams_j, seed=seed,
                               permute_rng=_rng(assay, j))
        null.append(_score_plan(t, assay, preds, choice))
    return {"assay": assay, "fold": plan["fold"], "real": real, "null": null,
            "plain_ref": {k: ref[k] for k in ("auc_direction",
                                              "frac_outside_2sd",
                                              "bias_mean_sd")}}


# ---------------------------------------------------------------- report
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


def summarise_models(p2, n_perm) -> list[dict]:
    rows = []
    for name in ROWS:
        real = {mt: np.array([d["real"][name][mt] for d in p2]) for mt in METRICS}
        plain = {mt: np.array([d["real"]["additive"][mt] for d in p2])
                 for mt in METRICS}
        row = {"model": name,
               "median": {mt: float(np.median(real[mt])) for mt in METRICS},
               "median_delta_vs_additive": {
                   mt: float(np.median(real[mt] - plain[mt])) for mt in METRICS}}
        if name != "additive":
            nul = {mt: np.array([[d["null"][j][name][mt] for d in p2]
                                 for j in range(n_perm)]) for mt in METRICS}
            null_mean = {mt: nul[mt].mean(axis=0) for mt in METRICS}
            row["null_median"] = {mt: float(np.median(null_mean[mt]))
                                  for mt in METRICS}
            row["null_median_by_perm"] = {
                mt: [float(np.median(nul[mt][j])) for j in range(n_perm)]
                for mt in METRICS}
            row["null_median_delta_vs_additive"] = {
                mt: float(np.median(null_mean[mt] - plain[mt])) for mt in METRICS}
            def n_above(x, y):
                return f"{int((x - y > TIE).sum())}/{len(p2)}"
            row["assays_real_beats_own_null"] = {
                "r2_resid_var": n_above(real["r2_resid_var"],
                                        null_mean["r2_resid_var"]),
                "auc_direction": n_above(real["auc_direction"],
                                         null_mean["auc_direction"]),
                "frac_outside_2sd_lower": n_above(null_mean["frac_outside_2sd"],
                                                  real["frac_outside_2sd"]),
            }
            row["assays_auc_above_additive"] = n_above(real["auc_direction"],
                                                       plain["auc_direction"])
        row["uses"] = USES[name]
        rows.append(row)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pairs-dir", default="results/pairs")
    ap.add_argument("--out", default="results/alt_baseline.json")
    ap.add_argument("--n-perm", type=int, default=5)
    ap.add_argument("--min-pairs", type=int, default=MIN_PAIRS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6,
                    help="processes; results do not depend on it")
    args = ap.parse_args()
    t0 = time.time()
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")

    pairs_dir = (ROOT / args.pairs_dir).resolve()
    paths = []
    for p in sorted(pairs_dir.glob("*.csv")):
        with open(p) as f:
            n = sum(1 for _ in f) - 1
        if n >= args.min_pairs:
            paths.append(p)
    if not paths:
        raise SystemExit("no pair files; run scripts/build_pairs.py first")
    # largest first so the long assays start immediately
    paths.sort(key=lambda p: -p.stat().st_size)

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        p1 = list(ex.map(phase1, [(str(p), args.n_perm, args.seed)
                                  for p in paths]))
    p1.sort(key=lambda d: d["assay"])
    print(f"phase 1 done: {len(p1)} assays, {time.time() - t0:.0f}s", flush=True)
    plan, folds = outer_selection(p1, args.n_perm)
    jobs = [(d["path"], plan[d["assay"]], args.seed)
            for d in sorted(p1, key=lambda d: -d["n_pairs"])]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        p2 = list(ex.map(phase2, jobs))
    p2.sort(key=lambda d: d["assay"])
    print(f"phase 2 done: {time.time() - t0:.0f}s", flush=True)

    # like-for-like guard: the recomputed plain row must BE summarise_table's
    worst = max(abs(d["real"]["additive"][k] - d["plain_ref"][k])
                for d in p2 for k in d["plain_ref"])
    if worst > 1e-12:
        raise SystemExit(f"plain additivity drifted from summarise_table: {worst}")

    models = summarise_models(p2, args.n_perm)
    med = {m["model"]: m["median"] for m in models}
    nul = {m["model"]: m.get("null_median") for m in models}
    by1 = {d["assay"]: d for d in p1}
    fitted_rows = [r for r in ROWS if r not in ("additive",
                                                "additive+pooled_offset")]
    best = max(fitted_rows, key=lambda r: med[r]["r2_resid_var"])

    verdict = {
        "plain_additivity_recomputed": {
            k: med["additive"][k] for k in ("auc_direction", "frac_outside_2sd",
                                            "frac_outside_2sd_centred",
                                            "bias_mean_sd")},
        "readme_quotes_for_comparison": README_PLAIN,
        "best_model_by_median_r2": best,
        "best_model_median": med[best],
        "best_model_null_median": nul[best],
        "bias": (
            f"plain additivity {med['additive']['bias_mean_sd']:+.3f} sd; "
            f"pooled offset from other assays {med['additive+pooled_offset']['bias_mean_sd']:+.3f}; "
            f"own-assay offset {med['additive+assay_offset']['bias_mean_sd']:+.3f} "
            f"(null {nul['additive+assay_offset']['bias_mean_sd']:+.3f}); "
            f"{best} {med[best]['bias_mean_sd']:+.3f} (null {nul[best]['bias_mean_sd']:+.3f})"),
        "beyond_2sd": (
            f"plain {med['additive']['frac_outside_2sd']:.3f} -> pooled offset "
            f"{med['additive+pooled_offset']['frac_outside_2sd']:.3f} -> own offset "
            f"{med['additive+assay_offset']['frac_outside_2sd']:.3f} -> "
            f"{best} {med[best]['frac_outside_2sd']:.3f} "
            f"(its null {nul[best]['frac_outside_2sd']:.3f}); plain centred "
            f"{med['additive']['frac_outside_2sd_centred']:.3f}"),
        "spread": (
            f"{best} removes a median {med[best]['r2_resid_var']:.3f} of the "
            f"residual variance (null {nul[best]['r2_resid_var']:+.3f}); "
            f"additive+position alone {med['additive+position']['r2_resid_var']:.3f} "
            f"(null {nul['additive+position']['r2_resid_var']:+.3f})"),
        "ranking": (
            f"AUC plain {med['additive']['auc_direction']:.3f}; "
            f"{best} {med[best]['auc_direction']:.3f} "
            f"(null {nul[best]['auc_direction']:.3f})"),
    }
    rows_by = {m["model"]: m for m in models}
    b = rows_by[best]
    verdict["gap_survives"] = {
        "bias": (
            f"median bias {med['additive']['bias_mean_sd']:+.3f} sd -> "
            f"{med['additive+assay_offset']['bias_mean_sd']:+.3f} with the assay's "
            f"own offset, and {nul['additive+assay_offset']['bias_mean_sd']:+.3f} "
            f"under that offset's permutation null; "
            f"{med['additive+pooled_offset']['bias_mean_sd']:+.3f} with one "
            "offset learned on the other assays"),
        "beyond_2sd": (
            f"{med['additive']['frac_outside_2sd']:.3f} plain -> "
            f"{med['additive+pooled_offset']['frac_outside_2sd']:.3f} pooled "
            f"offset -> {med['additive+assay_offset']['frac_outside_2sd']:.3f} own "
            f"offset (plain centred {med['additive']['frac_outside_2sd_centred']:.3f})"
            f" -> {med[best]['frac_outside_2sd']:.3f} {best}; the position "
            f"model's null {nul['additive+position']['frac_outside_2sd']:.3f}"),
        "spread": (
            f"the strongest baseline ({best}) leaves "
            f"{1 - med[best]['r2_resid_var']:.3f} of the residual variance: R2 "
            f"{med[best]['r2_resid_var']:.3f} against its null "
            f"{nul[best]['r2_resid_var']:+.3f}, above the null in "
            f"{b['assays_real_beats_own_null']['r2_resid_var']} assays"),
        "ranking": (
            f"median AUC {med['additive']['auc_direction']:.3f} -> "
            f"{med[best]['auc_direction']:.3f} (null "
            f"{nul[best]['auc_direction']:.3f}); per-assay median gain "
            f"{b['median_delta_vs_additive']['auc_direction']:+.4f}, above "
            f"additive in {b['assays_auc_above_additive']} assays"),
        "needs_doubles": USES[best],
    }

    per_assay = []
    for d in p2:
        a = by1[d["assay"]]
        row = {"assay": d["assay"], "n_pairs": a["n_pairs"],
               "n_positions": a["n_positions"], "n_cells": a["n_cells"],
               "outer_fold": d["fold"], "models": d["real"],
               "null_mean_over_perms": {
                   name: {mt: float(np.mean([d["null"][j][name][mt]
                                             for j in range(args.n_perm)]))
                          for mt in METRICS}
                   for name in ROWS if name != "additive"}}
        per_assay.append(row)

    code = {p: _sha256(ROOT / p) for p in ("scripts/alt_baseline.py",
                                           "src/epsceil/baseline.py",
                                           "src/epsceil/pairs.py")}
    pairs_digest = hashlib.sha256("".join(
        f"{p.name}:{_sha256(p)}\n" for p in sorted(paths)).encode()).hexdigest()
    cmd = "python scripts/alt_baseline.py " + " ".join(sys.argv[1:])
    out = {
        "question": ("Does a stronger baseline than plain additivity shrink the "
                     "gap (bias, share beyond 2 sd, AUC), or is the gap real?"),
        "provenance": {
            "command": cmd.strip(),
            "cwd": "repository root",
            "interpreter": sys.executable,
            "python": platform.python_version(),
            "versions": {m.__name__: m.__version__ for m in (np, pd)} | {
                "scipy": __import__("scipy").__version__,
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
            "assays": (f"{len(p2)} assays with >= {args.min_pairs} pairs "
                       f"({sum(d['n_pairs'] for d in p1):,} pairs), the README's "
                       "correlation subset"),
            "scale": ("every residual in units of plain additivity's own "
                      "per-assay residual sd (residual_scale_sd); the 2-sd "
                      "threshold is therefore the same for every model"),
            "metrics": {
                "r2_resid_var": "1 - var(r - r_hat)/var(r): share of the additivity residual's variance removed; constant shifts score 0",
                "mse_skill": "1 - mean((r - r_hat)^2)/mean(r^2): share of additivity's squared error removed, bias included",
                "auc_direction": "epsceil.pairs.direction_auc of the corrected prediction vs observed, via summarise_table",
                "frac_outside_2sd": "share of pairs with |r - r_hat| > 2, via summarise_table",
                "frac_outside_2sd_centred": "the same after mean-centring within assay (add_centred_columns)",
                "bias_mean_sd": "mean(r - r_hat), via summarise_table",
                "spearman_additive": "summarise_table's Spearman of the (corrected) prediction vs observed; the key keeps summarise_table's name",
                "aggregation": "median over assays of per-assay values; null = mean over permutations per assay, then median",
            },
            "inner_split": ("KFold(5, shuffle, random_state=seed) over each assay's "
                            "pairs; every pair is scored by parameters fitted "
                            "without it. Needs measured doubles from the same "
                            "protein, which plain additivity does not."),
            "outer_split": ("GroupKFold(5) over assays (one group per assay): ridge "
                            "lambda per model = argmax of the median inner-CV "
                            "r2_resid_var over the TRAINING assays; the pooled "
                            "offset = median bias over the TRAINING assays"),
            "lambda_grid": list(LAMBDAS),
            "ranks": list(RANKS),
            "pair_term": ("rank-k symmetric approximation (largest |eigenvalues|) "
                          "of the position x position matrix of the position "
                          "model's training residuals; cell means weighted "
                          "n/(n+lambda), unobserved cells and diagonal completed "
                          "by EM for weighted low-rank approximation"),
            "null": (f"{args.n_perm} permutations per assay: the identical "
                     "pipeline, including the outer lambda selection, with "
                     "residuals permuted among the training pairs of each inner "
                     "fold; scored against the REAL held-out residuals. The mean "
                     "survives permutation, so offsets survive their nulls by "
                     "construction."),
        },
        "n_assays": len(p2),
        "n_pairs": int(sum(d["n_pairs"] for d in p1)),
        "caveats": CAVEATS,
        "models": models,
        "verdict": verdict,
        "outer_folds": folds,
        "per_assay": per_assay,
    }
    out_path = (ROOT / args.out)
    out_path.write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(f"wrote {out_path} in {time.time() - t0:.0f}s")
    hdr = f"{'model':38s} {'R2var':>7s} {'AUC':>6s} {'>2sd':>6s} {'bias':>7s}"
    print(hdr)
    for m in models:
        md, nm = m["median"], m.get("null_median")
        print(f"{m['model']:38s} {md['r2_resid_var']:7.3f} "
              f"{md['auc_direction']:6.3f} {md['frac_outside_2sd']:6.3f} "
              f"{md['bias_mean_sd']:+7.3f}")
        if nm:
            print(f"{'   null':38s} {nm['r2_resid_var']:7.3f} "
                  f"{nm['auc_direction']:6.3f} {nm['frac_outside_2sd']:6.3f} "
                  f"{nm['bias_mean_sd']:+7.3f}")
    for k, v in verdict["gap_survives"].items():
        print(f"{k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
