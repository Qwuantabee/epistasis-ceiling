#!/usr/bin/env python3
"""Q2: is the per-assay "position" term a global monotone nonlinearity in disguise?

The reviewer's alternative explanation for the position result: single-effect
size clusters by position, so a GLOBAL monotone nonlinearity of the assay
(saturation, a floor, regression to the mean) leaves position-clustered
additivity residuals without any position-specific epistasis. This script fits,
per assay and out of fold, on the SAME 48 assays (>= 500 pairs), the SAME inner
split as epsceil.baseline.crossfit_assay (KFold(5, shuffle, random_state=seed)
over pairs) and the SAME metric (epsceil.baseline.r2_resid_var):

  (1) additive              r_hat = 0
  (2) isotonic              observed ~ g(predicted), g non-decreasing
                            (sklearn isotonic regression, clipped outside the
                            training range); r_hat = (g(pred) - pred) / sd
      linear / quadratic / cubic   least-squares polynomial g, for reference
                            (NOT constrained to be monotone)
  (3) position              epsceil.baseline.fit_position, unchanged
  (4) position+isotonic     g and the position term fitted JOINTLY by
                            backfitting (alternate: g on what position leaves,
                            position on what g leaves) to convergence
      position+isotonic_seq the first backfitting step only: g first, position
                            on what g leaves (monotone gets all shared variance)
      position+cubic        joint backfit with the cubic, for reference

Everything is in units of the assay's additivity-residual sd: x = predicted/sd,
z = observed/sd = x + r, and a fitted g gives r_hat = g(x) - x. The isotonic
fit is computed by sklearn.isotonic.isotonic_regression on x's unique values
(weights = multiplicities) and evaluated by linear interpolation clipped at the
ends, which IS IsotonicRegression(increasing=True, out_of_bounds="clip");
tests/test_position_holdout.py pins the two equal. Doing it this way lets the
backfit refit g many times on a fixed x without re-sorting 430k points.

Ridge lambda for every position-containing model (real and null) is the value
scripts/alt_baseline.py's outer GroupKFold over assays chose for the position
model in that assay's outer fold, read from results/alt_baseline.json (no assay
chose its own lambda). (3) is therefore the committed position model exactly.

NULLS. Every fitted model: residuals permuted among the training pairs of each
inner fold (alt_baseline._rng stream, one permutation per fold, as
crossfit_assay draws it), scored against the real held-out residuals. The
decision statistic gets its own CONDITIONAL null: position LABELS permuted among
the training pairs (x and observed untouched, so the monotone term still sees
real data), the same joint fit, scored on the real held-out pairs.

THE VERDICT RULE (RULE below) was written into this file before any Q2 R2 was
computed, and is copied verbatim into the artefact.
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

from epsceil.baseline import (fit_position, position_codes,  # noqa: E402
                              r2_resid_var)

MIN_PAIRS = 500
N_SPLITS = 5
MONO = ("isotonic", "linear", "quadratic", "cubic")
POLY_DEG = {"linear": 1, "quadratic": 2, "cubic": 3}
JOINT = ("isotonic", "cubic")
MODELS = (["additive", *MONO, "position", "position+isotonic",
           "position+isotonic_seq", "position+cubic"])
POSITIONAL = ("position", "position+isotonic", "position+isotonic_seq",
              "position+cubic")
BF_MAX_ITER = 200      # linear convergence; ~50-80 iterations on collinear synthetic data
BF_TOL = 1e-4          # max change of the fitted position term, residual-sd units
TIE = 1e-9
EPS = 0.02
LABEL_NULL_OFFSET = 50_000     # separate RNG stream for the label permutations
RULE = {
    "decision_statistic": (
        "D42 = per-assay R2(position+isotonic, joint backfit) - R2(isotonic), R2 "
        "= epsceil.baseline.r2_resid_var out of fold: what the position term adds "
        "once a monotone global nonlinearity of the additive prediction is allowed"),
    "why_not_3_minus_2": (
        "(3 - 2) = R2(position) - R2(isotonic) compares the two terms head to head. "
        "They can explain different variance, so (3 - 2) ~ 0 does not by itself show "
        "that position IS the calibration curve, and (3 - 2) < 0 does not show that "
        "position adds nothing; the nested (4 - 2) does. (3 - 2) is reported with its "
        "interval and with the verdict the same thresholds would give it "
        "(verdict_if_decided_by_3_minus_2), but it is not the headline"),
    "eps": EPS,
    "interval": "95% percentile bootstrap of the median over assays (position_holdout.boot_median_ci: 5000 resamples)",
    "null_of_D42": (
        "position labels permuted among the training pairs of each inner fold "
        "(isotonic still sees the real data), the same joint fit, mean over "
        "permutations, minus R2(isotonic)"),
    "beats_null": "share of assays whose D42 exceeds its own null by > 1e-9",
    "SUPPORTED": "CI lower bound of median D42 > eps AND beats_null >= 0.75",
    "NOT_SUPPORTED": "CI upper bound of median D42 < eps OR beats_null < 0.50",
    "INCONCLUSIVE": "anything else",
    "order": "SUPPORTED, then NOT_SUPPORTED, else INCONCLUSIVE (mutually exclusive)",
    "brief_statistic_reading": (
        "D32 = R2(position) - R2(isotonic): SUPPORTED if its CI lower bound > eps, "
        "NOT_SUPPORTED if its CI upper bound < eps, else INCONCLUSIVE"),
    "written": "before any Q2 R2 was computed",
}


def _load(name: str):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PH = _load("position_holdout")     # boot_median_ci, CELL_BINS, _rng (alt_baseline's)


class IsoOnX:
    """IsotonicRegression(increasing=True, out_of_bounds="clip") on a FIXED x.

    Ties in x are averaged with their multiplicities as weights (as sklearn's
    IsotonicRegression does), PAVA runs on the unique sorted x, and prediction
    is linear interpolation clipped at the ends (as its predict does)."""

    def __init__(self, x: np.ndarray):
        self.xu, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
        self.inv = np.asarray(inv).ravel()
        self.w = cnt.astype(float)
        self.yu = None

    def fit(self, y: np.ndarray) -> "IsoOnX":
        from sklearn.isotonic import isotonic_regression
        ym = np.bincount(self.inv, weights=y, minlength=len(self.xu)) / self.w
        self.yu = isotonic_regression(ym, sample_weight=self.w, increasing=True)
        return self

    def __call__(self, xq: np.ndarray) -> np.ndarray:
        return np.interp(xq, self.xu, self.yu)


class PolyOnX:
    def __init__(self, x: np.ndarray, deg: int):
        self.x, self.deg, self.p = x, deg, None

    def fit(self, y: np.ndarray) -> "PolyOnX":
        self.p = np.polynomial.Polynomial.fit(self.x, y, self.deg)
        return self

    def __call__(self, xq: np.ndarray) -> np.ndarray:
        return self.p(xq)


def mono_fitter(kind: str, x_tr: np.ndarray):
    return IsoOnX(x_tr) if kind == "isotonic" else PolyOnX(x_tr, POLY_DEG[kind])


def _freeze(g):
    """Snapshot a fitted transform (the fitter object is refitted in place)."""
    if isinstance(g, IsoOnX):
        xu, yu = g.xu, g.yu.copy()
        return lambda xq: np.interp(xq, xu, yu)
    p = g.p
    return lambda xq: p(xq)


def backfit(kind, x_tr, r_tr, ia_tr, ib_tr, n_pos, lam):
    """Joint fit of r ~ (g(x) - x) + mu + alpha_p + alpha_q by backfitting.

    Returns (first, final, iterations, last_change); first is the sequential
    fit (g on the data alone, then position on what g leaves)."""
    fitter = mono_fitter(kind, x_tr)
    z_tr = x_tr + r_tr
    pos = np.zeros(len(r_tr))
    first = None
    delta = float("inf")
    it = 0
    g = None
    for it in range(1, BF_MAX_ITER + 1):
        g = fitter.fit(z_tr - pos)
        m = g(x_tr) - x_tr
        mu, alpha = fit_position(ia_tr, ib_tr, r_tr - m, n_pos, lam)
        if first is None:
            first = (_freeze(g), mu, alpha)
        new = mu + alpha[ia_tr] + alpha[ib_tr]
        delta = float(np.max(np.abs(new - pos)))
        pos = new
        if delta <= BF_TOL:
            break
    return first, (_freeze(g), mu, alpha), it, delta


def inner_splits(n: int, seed: int) -> list:
    """crossfit_assay's inner split, materialised."""
    from sklearn.model_selection import KFold
    return list(KFold(n_splits=N_SPLITS, shuffle=True,
                      random_state=seed).split(np.arange(n)))


def crossfit_models(x, r, ia, ib, n_pos, lam, splits, r_rng=None,
                    label_rng=None) -> tuple[dict, dict]:
    """Out-of-fold r_hat for every model in MODELS, plus backfit diagnostics.

    r_rng permutes residuals among each fold's training pairs (the full null,
    one permutation per fold, as crossfit_assay draws it); label_rng permutes
    position labels among them instead (the conditional null of D42)."""
    n = len(r)
    out = {m: np.full(n, np.nan) for m in MODELS}
    diag = {"max_iter": 0, "max_last_change": 0.0}
    for tr, te in splits:
        r_tr = r[tr] if r_rng is None else r_rng.permutation(r[tr])
        ia_tr, ib_tr = ia[tr], ib[tr]
        if label_rng is not None:
            p = label_rng.permutation(len(tr))
            ia_tr, ib_tr = ia_tr[p], ib_tr[p]
        x_tr, x_te = x[tr], x[te]
        out["additive"][te] = 0.0
        for kind in MONO:
            g = mono_fitter(kind, x_tr).fit(x_tr + r_tr)
            out[kind][te] = g(x_te) - x_te
        mu, alpha = fit_position(ia_tr, ib_tr, r_tr, n_pos, lam)
        out["position"][te] = mu + alpha[ia[te]] + alpha[ib[te]]
        for kind in JOINT:
            first, (g, mu, alpha), it, delta = backfit(
                kind, x_tr, r_tr, ia_tr, ib_tr, n_pos, lam)
            out[f"position+{kind}"][te] = (g(x_te) - x_te + mu
                                           + alpha[ia[te]] + alpha[ib[te]])
            if kind == "isotonic":
                g1, mu1, a1 = first
                out["position+isotonic_seq"][te] = (g1(x_te) - x_te + mu1
                                                    + a1[ia[te]] + a1[ib[te]])
                diag["max_iter"] = max(diag["max_iter"], it)
                diag["max_last_change"] = max(diag["max_last_change"], delta)
    for k, v in out.items():
        assert np.isfinite(v).all(), f"unfilled out-of-fold prediction {k}"
    return out, diag


def run_assay(job: tuple) -> dict:
    path, lam, n_perm, seed = job
    assay = Path(path).stem
    t = pd.read_csv(path)
    r = t["residual"].to_numpy(dtype=float)
    sd = float(t["residual_scale_sd"].iloc[0])
    x = t["predicted"].to_numpy(dtype=float) / sd
    ia, ib, n_pos = position_codes(t["pos_a"].to_numpy(), t["pos_b"].to_numpy())
    cells, n_cells = PH.cell_codes(t["pos_a"].to_numpy(), t["pos_b"].to_numpy())
    splits = inner_splits(len(t), seed)

    def r2s(preds):
        return {m: r2_resid_var(r, v) for m, v in preds.items()}

    real, diag = crossfit_models(x, r, ia, ib, n_pos, lam, splits)
    full_null, label_null = [], []
    for j in range(n_perm):
        pj, _ = crossfit_models(x, r, ia, ib, n_pos, lam, splits,
                                r_rng=PH._rng(assay, j))
        full_null.append(r2s(pj))
        pj, _ = crossfit_models(x, r, ia, ib, n_pos, lam, splits,
                                label_rng=PH._rng(assay, LABEL_NULL_OFFSET + j))
        label_null.append({m: r2_resid_var(r, pj[m]) for m in POSITIONAL})
    return {"assay": assay, "n_pairs": int(len(t)), "n_positions": int(n_pos),
            "n_cells": int(n_cells), "lambda": lam, "r2": r2s(real),
            "full_null": full_null, "label_null": label_null,
            "backfit": diag}


# ---------------------------------------------------------------- summary
def decide(ci: list, beats) -> str:
    """RULE. beats=None applies the interval part only (the D32 reading)."""
    if ci[0] > EPS and (beats is None or beats >= 0.75):
        return "SUPPORTED"
    if ci[1] < EPS or (beats is not None and beats < 0.50):
        return "NOT_SUPPORTED"
    return "INCONCLUSIVE"


def _paired(per, a, b, null_key=None) -> dict:
    d = np.array([p["r2"][a] - p["r2"][b] for p in per])
    out = {"median": float(np.median(d)), "ci95_bootstrap": PH.boot_median_ci(d),
           "assays_positive": f"{int((d > TIE).sum())}/{len(d)}"}
    if null_key:
        nd = np.array([np.mean([nl[a] for nl in p[null_key]]) - p["r2"][b]
                       for p in per])
        out["null_median"] = float(np.median(nd))
        out["assays_above_null"] = f"{int((d - nd > TIE).sum())}/{len(d)}"
        out["beats_null_share"] = float(np.mean(d - nd > TIE))
    out["by_n_cells_bin"] = {}
    for name, lo, hi in PH.CELL_BINS:
        sel = np.array([p["n_cells"] >= lo and (hi is None or p["n_cells"] <= hi)
                        for p in per])
        if sel.any():
            out["by_n_cells_bin"][name] = {
                "n_assays": int(sel.sum()), "median": float(np.median(d[sel])),
                "ci95_bootstrap": PH.boot_median_ci(d[sel])}
    return out


def summarise(per: list) -> dict:
    models = []
    for m in MODELS:
        real = np.array([p["r2"][m] for p in per])
        row = {"model": m, "median_r2": float(np.median(real))}
        if m != "additive":
            nm = np.array([np.mean([nl[m] for nl in p["full_null"]]) for p in per])
            row["null_median_r2"] = float(np.median(nm))
            row["assays_above_null"] = f"{int((real - nm > TIE).sum())}/{len(per)}"
        if m in POSITIONAL:
            lm = np.array([np.mean([nl[m] for nl in p["label_null"]])
                           for p in per])
            row["label_null_median_r2"] = float(np.median(lm))
        models.append(row)
    paired = {
        "position_minus_isotonic (3-2)": _paired(per, "position", "isotonic"),
        "position+isotonic_minus_isotonic (4-2)": _paired(
            per, "position+isotonic", "isotonic", null_key="label_null"),
        "position+isotonic_minus_position (4-3)": _paired(
            per, "position+isotonic", "position"),
        "position+isotonic_seq_minus_isotonic": _paired(
            per, "position+isotonic_seq", "isotonic", null_key="label_null"),
        "position_minus_cubic": _paired(per, "position", "cubic"),
        "position+cubic_minus_cubic": _paired(per, "position+cubic", "cubic",
                                              null_key="label_null"),
        "position_minus_linear": _paired(per, "position", "linear"),
    }
    d42 = paired["position+isotonic_minus_isotonic (4-2)"]
    d32 = paired["position_minus_isotonic (3-2)"]
    med_pos = float(np.median([p["r2"]["position"] for p in per]))
    return {
        "models": models, "paired": paired,
        "verdict": decide(d42["ci95_bootstrap"], d42["beats_null_share"]),
        "verdict_inputs": {
            "D42_median": d42["median"], "D42_ci95": d42["ci95_bootstrap"],
            "D42_beats_label_null_share": d42["beats_null_share"],
            "eps": EPS},
        "verdict_if_decided_by_3_minus_2": decide(d32["ci95_bootstrap"], None),
        "position_share_surviving": {
            "median_D42_over_median_position_r2": (d42["median"] / med_pos
                                                   if med_pos > 0 else None),
            "median_position_r2": med_pos},
    }


def committed_lambda(alt_path: Path) -> dict:
    a = json.loads(alt_path.read_text())
    lam = {f["fold"]: f["lambda"]["position"]["chosen"] for f in a["outer_folds"]}
    return {p["assay"]: float(lam[p["outer_fold"]]) for p in a["per_assay"]}


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
    ap.add_argument("--out", default="results/monotone_fit.json")
    ap.add_argument("--committed", default="results/alt_baseline.json",
                    help="read-only: per-assay ridge lambda of the position model")
    ap.add_argument("--n-perm", type=int, default=5)
    ap.add_argument("--min-pairs", type=int, default=MIN_PAIRS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    t0 = time.time()
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    lam_of = committed_lambda(ROOT / args.committed)

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
    missing = [p.stem for p in paths if p.stem not in lam_of]
    if missing:
        raise SystemExit(f"no committed lambda for {missing}")
    paths.sort(key=lambda p: -p.stat().st_size)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        per = list(ex.map(run_assay, [(str(p), lam_of[p.stem], args.n_perm,
                                       args.seed) for p in paths]))
    per.sort(key=lambda d: d["assay"])
    print(f"fits done: {len(per)} assays, {time.time() - t0:.0f}s", flush=True)

    # (3) must BE the committed position model: same split, lambda, code
    repro = {}
    if not args.only:
        a = json.loads((ROOT / args.committed).read_text())
        cper = {p["assay"]: p["models"]["additive+position"]["r2_resid_var"]
                for p in a["per_assay"]}
        worst = max(abs(p["r2"]["position"] - cper[p["assay"]]) for p in per)
        repro = {"max_abs_per_assay_diff_vs_alt_baseline_position": worst,
                 "exact": bool(worst <= 1e-12)}
    s = summarise(per)
    code = {p: _sha256(ROOT / p) for p in ("scripts/monotone_fit.py",
                                           "scripts/position_holdout.py",
                                           "scripts/alt_baseline.py",
                                           "src/epsceil/baseline.py",
                                           "src/epsceil/pairs.py")}
    pairs_digest = hashlib.sha256("".join(
        f"{p.name}:{_sha256(p)}\n" for p in sorted(paths)).encode()).hexdigest()
    out = {
        "question": ("Does the position term's contribution survive once a "
                     "monotone global nonlinearity of the additive prediction "
                     "is allowed?"),
        "rule": RULE,
        "verdict": s["verdict"],
        "verdict_inputs": s["verdict_inputs"],
        "verdict_if_decided_by_3_minus_2": s["verdict_if_decided_by_3_minus_2"],
        "provenance": {
            "command": ("python scripts/monotone_fit.py "
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
            "assays": f"{len(per)} assays with >= {args.min_pairs} pairs",
            "inner_split": "KFold(5, shuffle, random_state=seed) over pairs: crossfit_assay's split",
            "units": "x = predicted/sd, z = observed/sd = x + r; a fitted g gives r_hat = g(x) - x",
            "lambda": ("per assay, the ridge lambda alt_baseline's outer "
                       "GroupKFold chose for the position model in that assay's "
                       "outer fold (results/alt_baseline.json), used for every "
                       "position-containing model, real and null"),
            "backfit": f"alternate g and position to max change <= {BF_TOL} sd or {BF_MAX_ITER} iterations",
            "metric": "epsceil.baseline.r2_resid_var, out of fold, pooled over the assay's folds",
            "null": (f"{args.n_perm} permutations per assay: residuals permuted "
                     "among each fold's training pairs (alt_baseline._rng)"),
            "label_null": (f"{args.n_perm} permutations per assay: position labels "
                           "permuted among each fold's training pairs, x and z "
                           "untouched; the null of D42"),
            "polynomials": "linear/quadratic/cubic are references, not constrained monotone",
        },
        "n_assays": len(per),
        "n_pairs": int(sum(p["n_pairs"] for p in per)),
        "reproduces_committed_position": repro,
        "models": s["models"],
        "paired": s["paired"],
        "position_share_surviving": s["position_share_surviving"],
        "backfit": {
            "max_iterations_used": int(max(p["backfit"]["max_iter"] for p in per)),
            "assays_hitting_iteration_cap": int(sum(
                p["backfit"]["max_iter"] >= BF_MAX_ITER
                and p["backfit"]["max_last_change"] > BF_TOL for p in per)),
            "max_last_change": float(max(p["backfit"]["max_last_change"]
                                         for p in per))},
        "per_assay": [{
            "assay": p["assay"], "n_pairs": p["n_pairs"],
            "n_positions": p["n_positions"], "n_cells": p["n_cells"],
            "lambda": p["lambda"], "r2": p["r2"],
            "null_mean": {m: float(np.mean([nl[m] for nl in p["full_null"]]))
                          for m in MODELS if m != "additive"},
            "label_null_mean": {m: float(np.mean([nl[m] for nl in p["label_null"]]))
                                for m in POSITIONAL},
            "backfit": p["backfit"]} for p in per],
    }
    out_path = ROOT / args.out
    out_path.write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(f"wrote {out_path} in {time.time() - t0:.0f}s")
    for m in s["models"]:
        print(f"{m['model']:24s} {m['median_r2']:+.3f} null "
              f"{m.get('null_median_r2', float('nan')):+.3f} "
              f"{m.get('assays_above_null', '')}")
    for k, v in s["paired"].items():
        print(f"{k:42s} {v['median']:+.3f} CI {v['ci95_bootstrap'][0]:+.3f}.."
              f"{v['ci95_bootstrap'][1]:+.3f} pos {v['assays_positive']} "
              f"{v.get('assays_above_null', '')}")
    print("verdict", s["verdict"], "| by (3-2):", s["verdict_if_decided_by_3_minus_2"])
    print("repro", repro)
    return 0


if __name__ == "__main__":
    sys.exit(main())
