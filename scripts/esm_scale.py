#!/usr/bin/env python3
"""ESM-2 additivity probe at corpus scale: three heads, two controls, a rule fixed first.

The shipped probe (scripts/esm_context.py -> results/esm_context.json) found that
ESM-2 window context does not rank epistasis (rho -0.003 +- 0.049 against a
+0.007 noise floor). Two caveats kept that from being a defensible null:

  1. it used 36,000 pairs from 24 assays - 4.4% of the 817,120-pair corpus;
  2. it was one feature set and one head (PCA 100 + RidgeCV), and its
     "ESM + prior" row sent the free prior through the same PCA, where one
     column among 5,122 is discarded - so it never tested ESM *over* the prior.

Kept from esm_context.py because they were right: wild-type-only windows,
GroupKFold by assay, scaler and PCA fitted inside the training fold, the log1p
target, the clip-to-training-range guard. Changed:

  * coverage - every assay with a parent sequence and >= 300 pairs, with a
    per-assay cap whose size and resulting corpus fraction go in the artefact;
  * embeddings once - a WT window depends only on (sequence, position), so the
    corpus needs a few thousand unique windows, not 2 x 817k. They are embedded
    once into a gitignored .npy cache and every head gathers from it;
  * three heads, each against two controls through its own identical pipeline,
    on the same folds, judged by VERDICT_RULE - written to the artefact by
    --preregister and committed before any head is fitted.

The 5,121-wide pair matrix is never materialised: the in-fold scaler and PCA
come from streamed first and second moments (exact covariance eigenvectors -
the subspace sklearn's randomized PCA approximates), and head (b)'s gradient
descent path is evaluated in closed form from the same moments.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import shlex
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INDEX = ROOT.parent / "_data" / "DMS_substitutions.csv"
DEFAULT_PAIRS = ROOT / "results" / "pairs"
DEFAULT_CACHE = ROOT / "results" / "emb_cache"
DEFAULT_OUT = ROOT / "results" / "esm_scale.json"

MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
POOLING = ("mean of last_hidden_state over attention-masked tokens, <cls>/<eos> "
           "included (the esm_context.py pooling)")
CACHE_STEM = "esm2_t33_650M_UR50D_window_meanpool"
ALPHAS = np.logspace(-3, 3, 13)          # the shipped RidgeCV grid
N_COMPONENTS = 100                        # the shipped PCA width
MIN_GAIN = 0.02                           # smallest rho gain worth reporting
GD_STEPS = np.unique(np.round(np.logspace(0, 6, 61)).astype(np.int64))
MODES = ("esm", "noise", "shuffled")
NOISE_SEED = {"full": 11, "pooled": 13}

VERDICT_RULE = {
    "fixed_before_run": ("written to results/esm_scale.json by --preregister and "
                         "committed before any head was fitted"),
    "statistic": ("Spearman rho between a head's out-of-fold prediction and "
                  "|centred additivity residual|, pooled over the pairs of one "
                  "GroupKFold(5)-by-assay test fold (the esm_context.py "
                  "definition); one value per fold, K = 5; the same folds for "
                  "every head and control"),
    "controls": {
        "noise": ("N(0,1) features of the head's own width, one fixed draw per "
                  "pair, through the identical pipeline"),
        "shuffled": ("the head's own ESM feature rows permuted within assay "
                     "(marginals kept, pair-to-feature link destroyed); prior, "
                     "target and folds untouched"),
    },
    "comparison": ("C* = the control with the larger fold-mean rho; "
                   "d_k = rho_head,k - rho_C*,k on the same fold k; "
                   "Delta = mean(d), SE = sd(d, ddof=1) / sqrt(K)"),
    "SUPPORTED": "Delta - 2 SE > 0: the head beats the larger control by more "
                 "than 2 standard errors",
    "NOT_SUPPORTED": ("not SUPPORTED and Delta + 2 SE < 0.02: the comparison rules "
                      "out a gain of 0.02 rho over the larger control"),
    "INCONCLUSIVE": ("neither (Delta +- 2 SE spans 0 and reaches 0.02), or any fold "
                     "gave a non-finite rho"),
    "head_c": ("head (c) keeps the free prior in every arm: rho(prior + ESM) is "
               "judged against rho(prior + control block), so its verdict is the "
               "verdict on the incremental rho of ESM over the free prior"),
    "min_gain": MIN_GAIN,
}

HEADS = {
    "a": ("PCA(100) + RidgeCV on [Ea, Eb, |Ea-Eb|, Ea*Eb, log1p(sep)] - the shipped "
          "esm_context.py estimator: scaler and PCA fitted in the training fold, "
          "alpha by RidgeCV, log1p target, prediction clipped to the training range"),
    "b": ("validated early-stopping linear model on the mean-pooled window "
          "embeddings only, [Ea, Eb] (2,560 wide, no product, no abs-diff, no sep): "
          "least squares on standardised features by full-batch gradient descent "
          "from w = 0, stopped at the step count (log grid 1..1e6) with the lowest "
          "MSE on 20% of the training assays held out (GroupShuffleSplit); same "
          "log1p target and clip as (a)"),
    "c": ("supervised residual head with the free prior always present: RidgeCV on "
          "[log1p|predicted_c| (standardised, outside the PCA), PCA(100) of the (a) "
          "block]; the prior-only arm is RidgeCV on the prior alone; incremental "
          "rho = rho(prior + block) - rho(prior), per fold"),
}


# ---------------------------------------------------------------- data ----------
def load_parent_sequences(index_csv: Path) -> dict[str, str]:
    idx = pd.read_csv(index_csv)
    return dict(zip(idx.DMS_id, idx.target_seq))


def load_pairs(pairs_dir: Path, seqs: dict[str, str], max_per_assay: int,
               seed: int = 0, min_pairs: int = 300, max_assays: int = 0):
    """Every assay with a parent sequence and >= min_pairs pairs, capped per assay.

    The esm_context.py selection rule (ranked by size, seeded draw per assay)
    with the assay limit removed. max_per_assay <= 0 means no cap. max_assays
    exists for smoke runs only and is recorded when used.
    """
    rng = np.random.default_rng(seed)
    cands, corpus_pairs, corpus_files = [], 0, 0
    for p in pairs_dir.glob("*.csv"):
        n = sum(1 for _ in p.open()) - 1
        corpus_pairs += n
        corpus_files += 1
        if p.stem in seqs and isinstance(seqs[p.stem], str) and n >= min_pairs:
            cands.append((n, p))
    cands.sort(reverse=True)
    use = cands[:max_assays] if max_assays and max_assays > 0 else cands
    cols = ["a", "b", "residual", "predicted", "pos_a", "pos_b", "sep"]
    frames, capped = [], {}
    for n, p in use:
        d = pd.read_csv(p, usecols=cols)
        if max_per_assay and max_per_assay > 0 and len(d) > max_per_assay:
            keep = rng.choice(len(d), max_per_assay, replace=False)
            d = d.iloc[np.sort(keep)]
            capped[p.stem] = int(n)
        d = d.copy()
        d["assay"] = p.stem
        frames.append(d)
    if not frames:
        raise SystemExit("no usable assays (missing parent sequences?)")
    out = pd.concat(frames, ignore_index=True)
    avail = int(sum(n for n, _ in cands))
    selection = {
        "rule": ("every assay with a parent sequence in DMS_substitutions.csv and "
                 f">= {min_pairs} pairs; a seeded random draw of at most "
                 f"{max_per_assay if max_per_assay > 0 else 'all'} pairs per assay"),
        "max_per_assay": int(max_per_assay) if max_per_assay > 0 else None,
        "seed": seed,
        "assays_eligible": len(cands),
        "assays_used": len(use),
        "pairs_eligible": avail,
        "pairs_used": int(len(out)),
        "fraction_of_eligible_pairs": float(len(out) / avail) if avail else None,
        "pairs_in_all_pair_tables": int(corpus_pairs),
        "pair_tables": corpus_files,
        "fraction_of_all_pairs": float(len(out) / corpus_pairs) if corpus_pairs else None,
        "capped_assays_original_sizes": capped,
        "max_assays_smoke_limit": int(max_assays) if max_assays else None,
        "per_assay_pairs_used": {k: int(v) for k, v in
                                 out.assay.value_counts().sort_index().items()},
    }
    return out, selection


def window(seq: str, pos: int, w: int) -> str:
    """The esm_context.py window: 1-based pos, w residues each side, clipped."""
    lo, hi = max(0, pos - 1 - w), min(len(seq), pos + w)
    return seq[lo:hi]


def window_ids(df: pd.DataFrame, seqs: dict[str, str], w: int):
    """Map every pair to two rows of a table of UNIQUE window strings."""
    vocab: dict[str, int] = {}
    ia = np.empty(len(df), np.int64)
    ib = np.empty(len(df), np.int64)
    pa_all, pb_all = df.pos_a.to_numpy(), df.pos_b.to_numpy()
    for assay, idx in df.groupby("assay", sort=False).indices.items():
        seq = seqs[assay]
        pa, pb = pa_all[idx], pb_all[idx]
        upos = np.unique(np.concatenate([pa, pb]))
        ids = np.array([vocab.setdefault(window(seq, int(p), w), len(vocab))
                        for p in upos], dtype=np.int64)
        ia[idx] = ids[np.searchsorted(upos, pa)]
        ib[idx] = ids[np.searchsorted(upos, pb)]
    windows = [""] * len(vocab)
    for s, i in vocab.items():
        windows[i] = s
    return windows, ia, ib


def wt_agreement(df: pd.DataFrame, seqs: dict[str, str]) -> dict:
    """Share of mutations whose wild-type letter equals target_seq at pos.

    A window is only the mutation's context if the numbering agrees with the
    parent sequence; the shipped probe never checked.
    """
    per = {}
    for assay, d in df.groupby("assay", sort=False):
        s = np.array(list(seqs[assay]))
        ok = []
        for col, pcol in (("a", "pos_a"), ("b", "pos_b")):
            pos = d[pcol].to_numpy() - 1
            inside = (pos >= 0) & (pos < len(s))
            wt = d[col].str[0].to_numpy()
            got = np.where(inside, s[np.clip(pos, 0, len(s) - 1)], "")
            ok.append(wt == got)
        per[assay] = float(np.concatenate(ok).mean())
    vals = np.array(list(per.values()))
    return {"min_over_assays": float(vals.min()), "assays_below_0.99":
            {k: v for k, v in per.items() if v < 0.99}}


def within_group_permutation(groups: np.ndarray, seed: int) -> np.ndarray:
    """perm[i] is a row of the same assay; feature row i becomes row perm[i]."""
    rng = np.random.default_rng(seed)
    codes = pd.factorize(groups)[0]
    perm = np.arange(len(groups))
    order = np.argsort(codes, kind="stable")
    bounds = np.flatnonzero(np.diff(codes[order])) + 1
    for idx in np.split(order, bounds):
        perm[idx] = rng.permutation(idx)
    return perm


# ------------------------------------------------------------- backends ---------
class NumpyBackend:
    """CPU arrays; used by the tests and to re-run heads from the cache."""
    name = "numpy"

    def put(self, a):
        return np.ascontiguousarray(a, dtype=np.float32)

    def take(self, W, idx):
        return W[idx]

    def hstack(self, xs):
        return np.hstack(xs)

    def randn(self, seed, shape):
        return np.random.default_rng(seed).standard_normal(shape, dtype=np.float32)

    def rows(self, X, mask):
        return X[mask]

    def colsum(self, X):
        return X.sum(0, dtype=np.float64)

    def gram(self, X):
        return (X.T @ X).astype(np.float64)

    def xty(self, X, r):
        return (X.T @ r).astype(np.float64)

    def host(self, a):
        return np.asarray(a, dtype=np.float64)

    def host32(self, a):
        return np.asarray(a, dtype=np.float32)

    def eigh(self, A):
        return np.linalg.eigh(A)


class TorchBackend:
    """GPU arrays: fp32 per-chunk products, fp64 accumulation."""

    def __init__(self, device: str):
        import torch
        self.torch = torch
        self.device = torch.device(device)
        self.name = f"torch:{device}"
        torch.set_float32_matmul_precision("highest")

    def put(self, a):
        return self.torch.as_tensor(np.ascontiguousarray(a, dtype=np.float32),
                                    device=self.device)

    def take(self, W, idx):
        return W.index_select(0, self.torch.as_tensor(
            np.asarray(idx, dtype=np.int64), device=self.device))

    def hstack(self, xs):
        return self.torch.cat(xs, dim=1)

    def randn(self, seed, shape):
        g = self.torch.Generator(device=self.device)
        g.manual_seed(int(seed))
        return self.torch.randn(shape, generator=g, device=self.device,
                                dtype=self.torch.float32)

    def rows(self, X, mask):
        return X[self.torch.as_tensor(np.asarray(mask), device=self.device)]

    def colsum(self, X):
        return X.double().sum(0)

    def gram(self, X):
        return (X.T @ X).double()

    def xty(self, X, r):
        return (X.T @ r).double()

    def host(self, a):
        return a.double().cpu().numpy()

    def host32(self, a):
        return a.float().cpu().numpy()

    def eigh(self, A):
        """fp64 symmetric eigensolver on the GPU (1.2 s at 5,121 wide; the CPU
        LAPACK path took minutes on the shared box)."""
        w, V = self.torch.linalg.eigh(self.torch.as_tensor(
            np.asarray(A, dtype=np.float64), device=self.device))
        return w.cpu().numpy(), V.cpu().numpy()


# ------------------------------------------------------------- features ---------
class PairFeatures:
    """Pair feature rows built per chunk from the window table, never stored.

    layout "full":   [Ea, Eb, |Ea-Eb|, Ea*Eb, log1p(sep)]  (5,121 = shipped)
    layout "pooled": [Ea, Eb]                              (2,560, head b)
    mode   "esm" | "shuffled" (rows permuted within assay) | "noise"

    Noise is drawn per fixed chunk from seed + chunk id, so a pair sees the same
    noise in the mean, scatter and projection passes whatever else is selected.
    """

    def __init__(self, be, W, ia, ib, lsep, layout, mode, perm=None, seed=0,
                 chunk=4096):
        assert layout in ("full", "pooled") and mode in MODES
        self.be, self.W, self.ia, self.ib = be, W, ia, ib
        self.lsep = np.asarray(lsep, dtype=np.float32)
        self.layout, self.mode, self.perm = layout, mode, perm
        self.seed, self.chunk = int(seed), int(chunk)
        emb = int(W.shape[1])
        self.dim = 4 * emb + 1 if layout == "full" else 2 * emb
        self.n = len(ia)
        self.n_chunks = int(math.ceil(self.n / self.chunk))
        if mode == "shuffled" and perm is None:
            raise ValueError("shuffled mode needs a within-assay permutation")

    def bounds(self, c):
        return c * self.chunk, min(self.n, (c + 1) * self.chunk)

    def get(self, c, mask):
        lo, hi = self.bounds(c)
        if self.mode == "noise":
            X = self.be.randn(self.seed * 1_000_003 + c, (hi - lo, self.dim))
            return self.be.rows(X, mask)
        rows = np.arange(lo, hi)[mask]
        if self.mode == "shuffled":
            rows = self.perm[rows]
        Ea = self.be.take(self.W, self.ia[rows])
        Eb = self.be.take(self.W, self.ib[rows])
        if self.layout == "pooled":
            return self.be.hstack([Ea, Eb])
        return self.be.hstack([Ea, Eb, abs(Ea - Eb), Ea * Eb,
                               self.be.put(self.lsep[rows][:, None])])


def _chunks(src, sel):
    for c in range(src.n_chunks):
        lo, hi = src.bounds(c)
        m = sel[lo:hi]
        if m.any():
            yield c, lo, hi, m


def stream_mean(src, sel):
    acc, n = None, 0
    for c, lo, hi, m in _chunks(src, sel):
        part = src.be.colsum(src.get(c, m))
        acc = part if acc is None else acc + part
        n += int(m.sum())
    return n, src.be.host(acc) / n


def stream_scatter(src, sel, mu, y=None, ybar=0.0):
    """Sum of (x-mu)(x-mu)^T over selected rows, and (x-mu)(y-ybar) if y given."""
    be = src.be
    mu_d = be.put(mu)
    S = sxy = None
    syy, n = 0.0, 0
    for c, lo, hi, m in _chunks(src, sel):
        X = src.get(c, m) - mu_d
        g = be.gram(X)
        S = g if S is None else S + g
        if y is not None:
            r = y[lo:hi][m].astype(np.float64) - ybar
            v = be.xty(X, be.put(r))
            sxy = v if sxy is None else sxy + v
            syy += float(r @ r)
        n += int(m.sum())
    return n, be.host(S), (be.host(sxy) if sxy is not None else None), syy


def stream_project(src, sel, mu, P):
    """(x - mu) @ P for the selected rows, in row order."""
    be = src.be
    mu_d, P_d = be.put(mu), be.put(P)
    out = [be.host32((src.get(c, m) - mu_d) @ P_d) for c, lo, hi, m in
           _chunks(src, sel)]
    return np.vstack(out) if out else np.zeros((0, P.shape[1]), np.float32)


# --------------------------------------------------------------- heads ----------
def standard_scale(n, S):
    """Population sd per feature, zero-variance columns left unscaled (sklearn)."""
    sigma = np.sqrt(np.clip(np.diag(S) / n, 0.0, None))
    sigma[sigma < 10 * np.finfo(np.float64).eps] = 1.0
    return sigma


def pca_from_scatter(n, S, k, eigh=np.linalg.eigh):
    """StandardScaler + PCA(k) of the training rows from their scatter matrix."""
    sigma = standard_scale(n, S)
    R = S / n / np.outer(sigma, sigma)
    d = R.shape[0]
    k = int(min(k, d, n - 1))
    w, V = eigh(R)
    order = np.argsort(w)[::-1][:k]
    return sigma, V[:, order], w[order]


def ridge_predict(Ztr, ytr, Zte):
    """The shipped head: RidgeCV on log1p(y), expm1, clip to the training range."""
    from sklearn.linear_model import RidgeCV
    m = RidgeCV(alphas=ALPHAS).fit(Ztr, np.log1p(ytr))
    pred = np.clip(np.expm1(m.predict(Zte)), 0.0, float(ytr.max()))
    return pred, float(m.alpha_)


def gd_filter(lam, eta, t):
    """Spectral filter of t full-batch GD steps from 0: (1 - (1 - eta*lam)^t)/lam."""
    lam = np.asarray(lam, dtype=np.float64)
    out = np.empty_like(lam)
    small = lam <= 1e-12 * max(float(lam.max()), 1e-300)
    x = np.clip(eta * lam[~small], 0.0, 1.0)
    with np.errstate(divide="ignore"):
        out[~small] = -np.expm1(t * np.log1p(-x)) / lam[~small]
    out[small] = eta * t
    return out


def early_stopped_linear(n_t, S_t, sxy_t, n_v, S_v, sxy_v, syy_v, steps=GD_STEPS,
                         eigh=np.linalg.eigh):
    """Least squares by gradient descent, stopped on held-out-assay MSE.

    S_t, sxy_t: training scatter and cross-moment about the training means;
    S_v, sxy_v, syy_v: the validation rows' moments about the SAME training
    means. GD on 0.5*mean(r^2) in standardised coordinates with step 1/lam_max
    has w_t = U diag(gd_filter(lam, eta, t)) U^T h, so the whole path is scored
    exactly without iterating. Returns weights on the raw (x - mu) scale.
    """
    sigma = standard_scale(n_t, S_t)
    D = np.outer(sigma, sigma)
    G, h = S_t / n_t / D, sxy_t / n_t / sigma
    Gv, hv, cv = S_v / n_v / D, sxy_v / n_v / sigma, syy_v / n_v
    lam, U = eigh(G)
    lam = np.clip(lam, 0.0, None)
    eta = 1.0 / float(lam.max())
    a = U.T @ h
    M, b = U.T @ Gv @ U, U.T @ hv
    path = []
    for t in steps:
        c = gd_filter(lam, eta, t) * a
        path.append(float(c @ M @ c - 2.0 * c @ b + cv))
    i = int(np.argmin(path))
    w_std = U @ (gd_filter(lam, eta, steps[i]) * a)
    info = {"steps": int(steps[i]), "val_mse": path[i],
            "val_mse_constant": float(cv),
            "beats_constant_on_validation": bool(path[i] < cv)}
    return w_std / sigma, info


def spearman(pred, y):
    from scipy.stats import spearmanr
    if np.ptp(pred) == 0 or np.ptp(y) == 0:
        return float("nan")
    return float(spearmanr(pred, y).statistic)


def fold_metrics(pred, y_te, y_tr, groups_te):
    within = [spearman(pred[groups_te == g], y_te[groups_te == g])
              for g in np.unique(groups_te)]
    within = [w for w in within if np.isfinite(w)]
    return {"rho": spearman(pred, y_te),
            "rho_within_assay_mean": float(np.mean(within)) if within else float("nan"),
            "mae_model": float(np.abs(pred - y_te).mean()),
            "mae_constant": float(np.abs(np.median(y_tr) - y_te).mean())}


def run_heads(y, prior_abs, groups, W, ia, ib, lsep, be, folds, seed=0,
              chunk=4096, n_components=N_COMPONENTS, log=print):
    """All three heads x (ESM, noise, shuffled) on the same folds.

    y: |centred residual|; prior_abs: |predicted_c|; groups: assay per pair.
    """
    N = len(y)
    perm = within_group_permutation(groups, seed)
    Wd = be.put(W)
    src = {(lay, mode): PairFeatures(be, Wd, ia, ib, lsep, lay, mode, perm,
                                     seed=NOISE_SEED[lay] + seed, chunk=chunk)
           for lay in ("full", "pooled") for mode in MODES}
    prior = np.log1p(np.abs(prior_abs))
    everything = np.ones(N, bool)
    ylog = np.log1p(y)
    rec = {"a": {m: [] for m in MODES}, "b": {m: [] for m in MODES},
           "c": {m: [] for m in MODES}, "prior_only": []}
    from sklearn.model_selection import GroupShuffleSplit
    for k, (tr, te) in enumerate(folds):
        t0 = time.time()
        trm = np.zeros(N, bool)
        trm[tr] = True
        y_tr, y_te, g_te = y[tr], y[te], groups[te]
        sd = prior[tr].std()
        pc = (prior - prior[tr].mean()) / (sd if sd > 0 else 1.0)
        pred, alpha = ridge_predict(pc[tr, None], y_tr, pc[te, None])
        rec["prior_only"].append({**fold_metrics(pred, y_te, y_tr, g_te),
                                  "alpha": alpha})
        # inner validation split for head (b): 20% of the TRAINING assays
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed + k)
        itr, iva = next(gss.split(tr, groups=groups[tr]))
        inner = np.zeros(N, bool)
        inner[tr[itr]] = True
        val = np.zeros(N, bool)
        val[tr[iva]] = True
        tem = np.zeros(N, bool)
        tem[te] = True
        for mode in MODES:
            # heads (a) and (c): in-fold scaler + PCA of the 5,121-wide block
            s = src[("full", mode)]
            n_tr, mu = stream_mean(s, trm)
            _, S, _, _ = stream_scatter(s, trm, mu)
            sigma, V, ev = pca_from_scatter(n_tr, S, n_components, eigh=be.eigh)
            del S
            Z = stream_project(s, everything, mu, V / sigma[:, None])
            pred, alpha = ridge_predict(Z[tr], y_tr, Z[te])
            rec["a"][mode].append({**fold_metrics(pred, y_te, y_tr, g_te),
                                   "alpha": alpha,
                                   "pca_var_share_top": float(ev.sum() / s.dim)})
            Zc_tr = np.hstack([pc[tr, None], Z[tr]])
            Zc_te = np.hstack([pc[te, None], Z[te]])
            pred, alpha = ridge_predict(Zc_tr, y_tr, Zc_te)
            rec["c"][mode].append({**fold_metrics(pred, y_te, y_tr, g_te),
                                   "alpha": alpha})
            del Z, Zc_tr, Zc_te
            # head (b): early-stopped linear on [Ea, Eb]
            s = src[("pooled", mode)]
            n_t, mu = stream_mean(s, inner)
            ybar = float(ylog[inner].mean())
            n_t, S_t, sxy_t, _ = stream_scatter(s, inner, mu, ylog, ybar)
            n_v, S_v, sxy_v, syy_v = stream_scatter(s, val, mu, ylog, ybar)
            w, info = early_stopped_linear(n_t, S_t, sxy_t, n_v, S_v, sxy_v, syy_v,
                                           eigh=be.eigh)
            del S_t, S_v
            plog = ybar + stream_project(s, tem, mu, w[:, None])[:, 0]
            pred = np.clip(np.expm1(plog), 0.0, float(y_tr.max()))
            rec["b"][mode].append({**fold_metrics(pred, y_te, y_tr, g_te), **info})
        log(f"  fold {k}: {len(te):,} test pairs / {len(np.unique(g_te))} assays, "
            f"{time.time() - t0:.0f}s | a " +
            " ".join(f"{m}={rec['a'][m][-1]['rho']:+.3f}" for m in MODES) +
            " | b " + " ".join(f"{m}={rec['b'][m][-1]['rho']:+.3f}" for m in MODES) +
            " | c " + " ".join(f"{m}={rec['c'][m][-1]['rho']:+.3f}" for m in MODES) +
            f" | prior={rec['prior_only'][-1]['rho']:+.3f}")
    return rec


# -------------------------------------------------------------- verdict ---------
def verdict(head, controls: dict, min_gain: float = MIN_GAIN) -> dict:
    """VERDICT_RULE, mechanically: head vs the larger control, paired by fold."""
    h = np.asarray(head, dtype=np.float64)
    cs = {k: np.asarray(v, dtype=np.float64) for k, v in controls.items()}
    if not np.isfinite(h).all() or not all(np.isfinite(v).all() for v in cs.values()):
        return {"verdict": "INCONCLUSIVE", "reason": "a fold gave a non-finite rho"}
    means = {k: float(v.mean()) for k, v in cs.items()}
    cstar = max(means, key=lambda k: means[k])
    d = h - cs[cstar]
    delta = float(d.mean())
    se = float(d.std(ddof=1) / math.sqrt(len(d)))
    lo, hi = delta - 2 * se, delta + 2 * se
    if lo > 0:
        v = "SUPPORTED"
    elif hi < min_gain:
        v = "NOT_SUPPORTED"
    else:
        v = "INCONCLUSIVE"
    return {"verdict": v, "head_mean": float(h.mean()), "control": cstar,
            "control_mean": means[cstar], "control_means": means,
            "delta": delta, "se": se, "delta_minus_2se": lo, "delta_plus_2se": hi,
            "min_gain": min_gain}


def _mean_se(x):
    x = np.asarray(x, dtype=np.float64)
    return {"folds": [float(v) for v in x], "mean": float(x.mean()),
            "se": float(x.std(ddof=1) / math.sqrt(len(x)))}


def summarise(rec: dict) -> dict:
    out = {}
    for h in ("a", "b", "c"):
        rho = {m: [f["rho"] for f in rec[h][m]] for m in MODES}
        within = {m: float(np.mean([f["rho_within_assay_mean"] for f in rec[h][m]]))
                  for m in MODES}
        out[h] = {
            "head": HEADS[h],
            "rho": rho,
            "rho_mean": {m: float(np.mean(v)) for m, v in rho.items()},
            "rho_within_assay_mean": within,
            "verdict": verdict(rho["esm"], {"noise": rho["noise"],
                                            "shuffled": rho["shuffled"]}),
            "per_fold": rec[h],
        }
    po = [f["rho"] for f in rec["prior_only"]]
    out["c"]["rho_prior_only"] = _mean_se(po)
    out["c"]["incremental_rho_over_prior"] = {
        m: _mean_se(np.asarray(out["c"]["rho"][m]) - np.asarray(po)) for m in MODES}
    out["c"]["prior_only_per_fold"] = rec["prior_only"]
    return out


# ------------------------------------------------------------ embedding ---------
def embed_windows(windows, batch, device, log=print):
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    try:
        model = AutoModel.from_pretrained(MODEL_NAME, add_pooling_layer=False,
                                          dtype=torch.float32)
    except TypeError:
        model = AutoModel.from_pretrained(MODEL_NAME, add_pooling_layer=False,
                                          torch_dtype=torch.float32)
    model = model.to(device).eval()
    order = np.argsort([len(s) for s in windows], kind="stable")
    out = np.zeros((len(windows), model.config.hidden_size), np.float32)
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(order), batch):
            idx = order[i:i + batch]
            enc = tok([windows[j] for j in idx], return_tensors="pt", padding=True,
                      truncation=True, max_length=64).to(device)
            h = model(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).float()
            out[idx] = ((h * m).sum(1) / m.sum(1).clamp(min=1)).float().cpu().numpy()
    peak = (torch.cuda.max_memory_allocated() / 2**20
            if str(device).startswith("cuda") else None)
    log(f"embedded {len(windows):,} windows in {time.time() - t0:.1f}s "
        f"(batch {batch}, peak {peak or 0:.0f} MiB)")
    del model
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return out, peak


def load_or_embed(windows, cache_dir: Path, batch: int, device: str, log=print):
    """Window table for windows, embedding only strings not already cached."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    npy = cache_dir / f"{CACHE_STEM}.npy"
    meta_p = cache_dir / f"{CACHE_STEM}.json"
    known, E = [], None
    if npy.exists() and meta_p.exists():
        meta = json.loads(meta_p.read_text())
        if meta.get("model") == MODEL_NAME and meta.get("pooling") == POOLING:
            E = np.load(npy)
            known = list(meta["windows"])
            if len(known) != len(E):
                known, E = [], None
    pos = {s: i for i, s in enumerate(known)}
    missing = sorted({s for s in windows if s not in pos})
    peak = None
    if missing:
        E_new, peak = embed_windows(missing, batch, device, log)
        E = E_new if E is None else np.vstack([E, E_new])
        known = known + missing
        tmp = cache_dir / f"{CACHE_STEM}.tmp.npy"
        np.save(tmp, E)
        tmp.replace(npy)
        meta_p.write_text(json.dumps({"model": MODEL_NAME, "pooling": POOLING,
                                      "windows": known}))
        pos = {s: i for i, s in enumerate(known)}
    table = E[np.array([pos[s] for s in windows], dtype=np.int64)]
    rel = npy.resolve()
    info = {"cache_npy": str(rel.relative_to(ROOT)) if rel.is_relative_to(ROOT)
            else str(rel),
            "cache_rows": int(len(known)), "cache_bytes": int(npy.stat().st_size),
            "windows_needed": len(windows), "windows_embedded_this_run": len(missing),
            "embed_batch": batch, "embed_peak_mib": peak}
    return table, info


# ------------------------------------------------------------------ main --------
def _git(*a):
    try:
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True,
                              text=True, timeout=30).stdout.strip()
    except Exception:
        return ""


def _jsonable(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def design(args) -> dict:
    return {
        "question": ("does ESM-2 wild-type window context rank |additivity "
                     "residual| beyond what the free prior already gives, at "
                     "corpus scale and with more than one head?"),
        "model": MODEL_NAME, "window_residues_each_side": args.window,
        "pooling": POOLING,
        "target": "|residual - assay mean(residual)|; heads fit log1p of it",
        "free_prior": "|predicted - assay median(predicted)|, entered as log1p",
        "cv": f"GroupKFold({args.folds}) by assay, one fold set for all arms",
        "heads": HEADS,
        "controls": VERDICT_RULE["controls"],
        "max_per_assay": args.max_per_assay, "seed": args.seed,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--index", default=str(DEFAULT_INDEX))
    ap.add_argument("--pairs-dir", default=str(DEFAULT_PAIRS))
    ap.add_argument("--max-per-assay", type=int, default=50000,
                    help="per-assay cap (<= 0: no cap)")
    ap.add_argument("--sensitivity-caps", default="",
                    help="comma list of further caps run after the primary (0 = none)")
    ap.add_argument("--max-assays", type=int, default=0, help="smoke runs only")
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--batch", type=int, default=256, help="windows per ESM batch, <= 512")
    ap.add_argument("--chunk", type=int, default=4096, help="pairs per feature chunk")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-components", type=int, default=N_COMPONENTS)
    ap.add_argument("--backend", choices=("auto", "torch", "numpy"), default="auto")
    ap.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--preregister", action="store_true",
                    help="write the verdict rule and design to --out and stop")
    args = ap.parse_args(argv)
    if args.batch > 512:
        raise SystemExit("--batch above 512 is not allowed on the shared GPU")
    out_p = Path(args.out)
    cmd = " ".join(shlex.quote(x) for x in [sys.executable, *sys.argv])

    if args.preregister:
        doc = {"status": "preregistered",
               "note": ("verdict rule and design fixed before any head was fitted; "
                        "the run overwrites this file and records whether the rule "
                        "it applied is identical to this one"),
               "preregistered_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               "verdict_rule": VERDICT_RULE, "design": design(args)}
        out_p.write_text(json.dumps(doc, indent=2) + "\n")
        print(f"wrote preregistration to {out_p}")
        return 0

    os.environ.setdefault("HF_HUB_OFFLINE", "1")   # weights are cached; never fetch
    prereg = None
    if out_p.exists():
        try:
            prev = json.loads(out_p.read_text())
            prereg = prev if prev.get("status") == "preregistered" else prev.get(
                "preregistration", {}).get("document")
        except Exception:
            prereg = None
    t_start = time.time()
    use_torch = args.backend in ("torch", "auto")
    device = "cpu"
    if use_torch:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            use_torch = False
    be = TorchBackend(device) if (use_torch and device == "cuda") else NumpyBackend()
    print(f"backend {be.name}, device {device}", flush=True)

    seqs = load_parent_sequences(Path(args.index))
    caps = [args.max_per_assay] + [int(c) for c in args.sensitivity_caps.split(",")
                                   if c.strip()]
    runs = []
    for cap in caps:
        t0 = time.time()
        df, sel = load_pairs(Path(args.pairs_dir), seqs, cap, seed=args.seed,
                             max_assays=args.max_assays)
        df["residual_c"] = df.residual - df.groupby("assay").residual.transform("mean")
        df["predicted_c"] = (df.predicted
                             - df.groupby("assay").predicted.transform("median"))
        y = np.abs(df.residual_c.to_numpy(np.float64))
        prior_abs = np.abs(df.predicted_c.to_numpy(np.float64))
        groups = df.assay.to_numpy()
        lsep = np.log1p(df.sep.to_numpy(np.float64))
        print(f"[cap {cap}] {len(df):,} pairs / {df.assay.nunique()} assays = "
              f"{sel['fraction_of_eligible_pairs']:.1%} of eligible pairs", flush=True)
        windows, ia, ib = window_ids(df, seqs, args.window)
        wt = wt_agreement(df, seqs)
        print(f"  {len(windows):,} unique windows; WT letter agreement min "
              f"{wt['min_over_assays']:.3f}", flush=True)
        W, emb = load_or_embed(windows, Path(args.cache_dir), args.batch,
                               device if use_torch else "cpu",
                               log=lambda s: print("  " + s, flush=True))
        from sklearn.model_selection import GroupKFold
        folds = list(GroupKFold(n_splits=args.folds).split(np.zeros(len(df)),
                                                           groups=groups))
        rec = run_heads(y, prior_abs, groups, W, ia, ib, lsep, be, folds,
                        seed=args.seed, chunk=args.chunk,
                        n_components=args.n_components,
                        log=lambda s: print(s, flush=True))
        summ = summarise(rec)
        runs.append({
            "max_per_assay": cap if cap > 0 else None,
            "selection": sel,
            "wt_letter_agreement": wt,
            "embedding": {**emb, "unique_windows": len(windows),
                          "pairs_per_unique_window": float(2 * len(df) / len(windows))},
            "folds": [{"k": k, "n_train": int(len(tr)), "n_test": int(len(te)),
                       "test_assays": sorted(set(groups[te].tolist()))}
                      for k, (tr, te) in enumerate(folds)],
            "heads": summ,
            "seconds": round(time.time() - t0, 1),
        })
        for h in ("a", "b", "c"):
            v = summ[h]["verdict"]
            print(f"  head {h}: rho {summ[h]['rho_mean']['esm']:+.4f} vs "
                  f"{v.get('control')} {v.get('control_mean', float('nan')):+.4f} -> "
                  f"{v['verdict']} (delta {v.get('delta', float('nan')):+.4f} "
                  f"+- 2x{v.get('se', float('nan')):.4f})", flush=True)
        inc = summ["c"]["incremental_rho_over_prior"]
        print("  incremental rho over prior: " + " ".join(
            f"{m}={inc[m]['mean']:+.4f}+-{inc[m]['se']:.4f}" for m in MODES), flush=True)

    primary = runs[0]
    shipped = None
    sp = ROOT / "results" / "esm_context.json"
    if sp.exists():
        s = json.loads(sp.read_text())
        ctl = s.get("controls", {})
        shipped = {"n_pairs": s.get("n_pairs"), "n_assays": s.get("n_assays"),
                   "fraction_of_available": s.get("selection", {}).get(
                       "fraction_of_available"),
                   "esm_rho": s.get("spearman_mean"),
                   "noise_rho": ctl.get("noise", {}).get("spearman_mean"),
                   "prior_only_rho": ctl.get("prior_only", {}).get("spearman_mean"),
                   "esm_plus_prior_rho": ctl.get("esm_plus_prior", {}).get(
                       "spearman_mean")}
    vers = {"python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__}
    try:
        import scipy
        import sklearn
        vers.update(sklearn=sklearn.__version__, scipy=scipy.__version__)
        if use_torch:
            import torch
            import transformers
            vers.update(torch=torch.__version__, transformers=transformers.__version__)
            if device == "cuda":
                vers["gpu"] = torch.cuda.get_device_name(0)
                vers["gpu_peak_mib_this_process"] = round(
                    torch.cuda.max_memory_allocated() / 2**20)
    except Exception:
        pass
    rel_out = out_p.resolve()
    rel_out = str(rel_out.relative_to(ROOT)) if rel_out.is_relative_to(ROOT) else str(rel_out)
    doc = {
        "status": "complete",
        "verdict_rule": VERDICT_RULE,
        "preregistration": {
            "rule_identical_to_preregistered": bool(
                prereg is not None and prereg.get("verdict_rule") == json.loads(
                    json.dumps(VERDICT_RULE))),
            "preregistered_at": prereg.get("preregistered_at") if prereg else None,
            "preregistration_commit": _git("log", "-1", "--format=%H", "--",
                                           rel_out) or None,
            "document": prereg,
        },
        "design": design(args),
        "headline": {
            "pairs_used": primary["selection"]["pairs_used"],
            "assays_used": primary["selection"]["assays_used"],
            "fraction_of_eligible_pairs": primary["selection"][
                "fraction_of_eligible_pairs"],
            "max_per_assay": primary["max_per_assay"],
            "verdicts": {h: primary["heads"][h]["verdict"]["verdict"]
                         for h in ("a", "b", "c")},
            "incremental_rho_of_esm_over_prior":
                primary["heads"]["c"]["incremental_rho_over_prior"]["esm"]["mean"],
        },
        "primary": primary,
        "sensitivity": runs[1:],
        "shipped_probe_for_comparison": shipped,
        "caveats": [
            "WT-window features are identical for every substitution pair at the "
            "same two positions, so they can rank position pairs but not the "
            "substitutions within one; the free prior is substitution-specific.",
            "Fold rho pools the pairs of the test fold (shipped definition), so an "
            "assay-level offset can move it; the within-assay shuffle control "
            "keeps assay-level marginals and is there to catch exactly that.",
            "K = 5 folds of whole assays: the SE is over 5 paired fold values and "
            "the 2-SE rule uses no t-correction, as written before the run.",
        ],
        "cache": {
            "path": "results/emb_cache/ (gitignored; never committed)",
            "regenerate": ("the same command re-embeds any window missing from the "
                           "cache; deleting the directory rebuilds it from the HF "
                           f"cache of {MODEL_NAME} (HF_HUB_OFFLINE=1)"),
        },
        "provenance": {
            "command": cmd,
            "git_head_at_run": _git("rev-parse", "HEAD") or None,
            "git_dirty_at_run": bool(_git("status", "--porcelain", "--",
                                          "scripts/esm_scale.py")),
            "backend": be.name, "versions": vers,
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "seconds_total": round(time.time() - t_start, 1),
        },
    }
    out_p.write_text(json.dumps(doc, indent=2, default=_jsonable) + "\n")
    print(f"wrote {out_p}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
