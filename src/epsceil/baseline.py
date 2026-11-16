"""Stronger baselines than plain additivity, fitted without leakage.

Plain additivity predicts a double as ``s_a + s_b - wt`` and fits nothing. The
README calls it the crudest baseline a lab could use. This module asks whether
a less crude one closes the gap, on the SAME pairs, the SAME per-assay scaling
(the plain-additivity residual sd, ``residual_scale_sd``) and the SAME summary
function (``epsceil.pairs.summarise_table``), so every number is like-for-like.

Every model below predicts a correction ``r_hat`` to the additivity residual
``r`` (in assay-sd units). The corrected double prediction is
``predicted + r_hat * residual_scale_sd`` and the corrected residual is
``r - r_hat``. Plain additivity is ``r_hat = 0``.

  assay_offset          r_hat = mu
  position              r_hat = mu + alpha[p] + alpha[q]          (ridge on alpha)
  pair_rank{k}          r_hat = position + L_k[p, q]

where ``L_k`` is a rank-k symmetric approximation to the position x position
matrix of the position model's residuals: cell means weighted by
``n / (n + lam)`` and completed by EM for weighted low-rank approximation, so a
cell with few pairs leans on the low-rank structure and a cell with no pairs (or
the diagonal, which no double occupies) is filled by it.

WHY EVERY TERM IS FITTED PER ASSAY, AND HOW LEAKAGE IS PREVENTED. Position 159
of one protein has nothing to do with position 159 of another, so a position or
position-pair parameter cannot be learned on one protein and applied to the
next. These terms are therefore fitted per assay with an INNER split
(``crossfit_assay``): every pair's correction comes from parameters fitted on the
other 4/5 of that assay's pairs, never on itself. That needs measured doubles
from the same protein, which plain additivity does not, so the comparison
favours the stronger baselines. The only pooled quantity, the ridge strength
``lam``, is chosen by GroupKFold over assays in scripts/alt_baseline.py, so no
protein both picks ``lam`` and is scored with it.

THE NULL is the identical fit on residuals permuted among the training pairs of
the same fold, scored against the real held-out residuals. Permutation keeps
the residual distribution (including its mean, so an intercept survives it) and
destroys only which positions a residual belongs to.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .pairs import add_centred_columns, summarise_table

LAMBDAS = (1.0, 10.0, 100.0, 1000.0)   # ridge strength in pseudo-pairs
RANKS = (1, 2, 3)
EM_MAX_ITER = 100
EM_TOL = 1e-6


def position_codes(pos_a: np.ndarray, pos_b: np.ndarray
                   ) -> tuple[np.ndarray, np.ndarray, int]:
    """Map sequence positions to 0..P-1. Order within a pair does not matter."""
    pos = np.union1d(pos_a, pos_b)
    return (np.searchsorted(pos, pos_a), np.searchsorted(pos, pos_b),
            int(len(pos)))


def fit_position(ia: np.ndarray, ib: np.ndarray, r: np.ndarray, n_pos: int,
                 lam: float) -> tuple[float, np.ndarray]:
    """Ridge fit of r ~ mu + alpha[ia] + alpha[ib], mu unpenalised.

    Solved through the (P+1)x(P+1) normal equations, assembled from counts so the
    cost is linear in pairs. A position absent from the training pairs gets
    alpha = 0 exactly (its row is lam * I), i.e. it falls back to the offset.
    """
    n = len(r)
    cnt = (np.bincount(ia, minlength=n_pos)
           + np.bincount(ib, minlength=n_pos)).astype(float)
    cell = np.bincount(ia * n_pos + ib, minlength=n_pos * n_pos).reshape(
        n_pos, n_pos).astype(float)
    xtx = cell + cell.T
    xtx[np.diag_indices(n_pos)] += cnt + lam
    xtr = (np.bincount(ia, weights=r, minlength=n_pos)
           + np.bincount(ib, weights=r, minlength=n_pos))
    m = np.zeros((n_pos + 1, n_pos + 1))
    m[0, 0] = n
    m[0, 1:] = cnt
    m[1:, 0] = cnt
    m[1:, 1:] = xtx
    rhs = np.concatenate([[float(r.sum())], xtr])
    try:
        sol = np.linalg.solve(m, rhs)
    except np.linalg.LinAlgError:          # pragma: no cover - lam > 0 is PD
        sol = np.linalg.lstsq(m, rhs, rcond=None)[0]
    return float(sol[0]), sol[1:]


def cell_sums(ia: np.ndarray, ib: np.ndarray, e: np.ndarray, n_pos: int
              ) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric position x position sums and counts of e (both orientations)."""
    s = np.bincount(ia * n_pos + ib, weights=e,
                    minlength=n_pos * n_pos).reshape(n_pos, n_pos)
    c = np.bincount(ia * n_pos + ib,
                    minlength=n_pos * n_pos).reshape(n_pos, n_pos).astype(float)
    return s + s.T, c + c.T


def _rank_k_sym(z: np.ndarray, k: int) -> np.ndarray:
    w, v = np.linalg.eigh(z)
    idx = np.argsort(-np.abs(w), kind="stable")[:k]
    return (v[:, idx] * w[idx]) @ v[:, idx].T


def lowrank_pair_term(s: np.ndarray, c: np.ndarray, lam: float, k: int,
                      max_iter: int = EM_MAX_ITER, tol: float = EM_TOL
                      ) -> np.ndarray:
    """Rank-k symmetric approximation of the position x position residual matrix.

    Cell means x = s / c are trusted with weight w = c / (c + lam); the EM
    update for weighted low-rank approximation, z = w*x + (1-w)*L, then takes the
    rank-k part of z (largest |eigenvalue|, so negative interaction structure is
    kept). The first iterate is the ridge-shrunk cell mean s / (c + lam).
    Unobserved cells and the diagonal have w = 0 and are filled by the low-rank
    structure rather than read as zeros.
    """
    n_pos = s.shape[0]
    if n_pos < 2:
        return np.zeros_like(s)
    w = c / (c + lam)
    x = np.divide(s, c, out=np.zeros_like(s), where=c > 0)
    wx = w * x
    low = np.zeros_like(s)
    for _ in range(max_iter):
        new = _rank_k_sym(wx + (1.0 - w) * low, k)
        delta = float(np.linalg.norm(new - low))
        scale = float(np.linalg.norm(new)) + 1e-12
        low = new
        if delta <= tol * scale:
            break
    return low


def model_names(ranks=RANKS) -> list[str]:
    return ["assay_offset", "position"] + [f"pair_rank{k}" for k in ranks]


def crossfit_assay(t: pd.DataFrame, lambdas=LAMBDAS, ranks=RANKS,
                   n_splits: int = 5, seed: int = 0,
                   permute_rng: np.random.Generator | None = None
                   ) -> dict[tuple[str, float | None], np.ndarray]:
    """Out-of-fold corrections r_hat for every pair of one assay.

    Inner KFold over the assay's pairs: each pair is predicted by parameters
    fitted on the other folds only. With permute_rng the training residuals of
    each fold are permuted among the training pairs before fitting (the null);
    the returned predictions are still for the real held-out pairs.

    Keys are (model, lam); assay_offset has lam None.
    """
    from sklearn.model_selection import KFold

    r = t["residual"].to_numpy(dtype=float)
    ia, ib, n_pos = position_codes(t["pos_a"].to_numpy(), t["pos_b"].to_numpy())
    n = len(r)
    out: dict[tuple[str, float | None], np.ndarray] = {
        ("assay_offset", None): np.full(n, np.nan)}
    for lam in lambdas:
        out[("position", lam)] = np.full(n, np.nan)
        for k in ranks:
            out[(f"pair_rank{k}", lam)] = np.full(n, np.nan)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in kf.split(np.arange(n)):
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


def r2_resid_var(r: np.ndarray, r_hat: np.ndarray) -> float:
    """Share of the additivity residual's VARIANCE removed: 1 - var(r-r_hat)/var(r).

    Invariant to a constant in r_hat, so an offset scores exactly 0 here and the
    bias is reported separately. Plain additivity (r_hat = 0) scores 0.
    """
    v = float(np.var(r))
    return 1.0 - float(np.var(r - r_hat)) / v if v > 0 else float("nan")


def mse_skill(r: np.ndarray, r_hat: np.ndarray) -> float:
    """Share of plain additivity's squared error removed (bias included)."""
    m = float(np.mean(r ** 2))
    return 1.0 - float(np.mean((r - r_hat) ** 2)) / m if m > 0 else float("nan")


SUMMARY_KEYS = ("auc_direction", "frac_outside_2sd", "frac_outside_2sd_centred",
                "bias_mean_sd", "spearman_additive", "residual_sd")


def score_correction(t: pd.DataFrame, r_hat: np.ndarray, assay: str) -> dict:
    """Score a corrected prediction through the repo's ONE summary definition.

    The corrected table keeps observed and the per-assay scale; it replaces
    predicted by predicted + r_hat*sd and residual by residual - r_hat, then
    goes through add_centred_columns and summarise_table exactly as plain
    additivity does. With r_hat = 0 this returns plain additivity's own row.
    """
    r = t["residual"].to_numpy(dtype=float)
    sd = t["residual_scale_sd"].to_numpy(dtype=float)
    r_hat = np.asarray(r_hat, dtype=float) * np.ones(len(r))
    t2 = pd.DataFrame({
        "observed": t["observed"].to_numpy(dtype=float),
        "predicted": t["predicted"].to_numpy(dtype=float) + r_hat * sd,
        "residual": r - r_hat,
        "residual_raw": t["residual_raw"].to_numpy(dtype=float) - r_hat * sd,
        "residual_scale_sd": sd,
    })
    s = summarise_table(add_centred_columns(t2), assay)
    out = {k: s[k] for k in SUMMARY_KEYS}
    out["r2_resid_var"] = r2_resid_var(r, r_hat)
    out["mse_skill"] = mse_skill(r, r_hat)
    return out
