"""Tests for the whole-position-pair holdout (scripts/position_holdout.py) and
the monotone global-epistasis comparison (scripts/monotone_fit.py).

Unit tests plant KNOWN structure in synthetic assays; artefact tests tie the
committed results/position_holdout.json and results/monotone_fit.json to the
code that produced them and to their pre-registered verdict rules.
"""
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from epsceil.baseline import crossfit_assay, r2_resid_var  # noqa: E402

PH_ART = ROOT / "results" / "position_holdout.json"
MF_ART = ROOT / "results" / "monotone_fit.json"


def _load(name):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def ph():
    return _load("position_holdout")


def _assay(n_pos=8, reps=20, alpha_sd=0.0, noise=1.0, seed=0, cells=None):
    """Synthetic pair table (as tests/test_alt_baseline.py builds it):
    residual = alpha_p + alpha_q + noise, scaled by its own sd. cells=None
    means every unordered position pair; otherwise the listed (p, q) pairs."""
    rng = np.random.default_rng(seed)
    alpha = rng.normal(0, alpha_sd, n_pos) if alpha_sd else np.zeros(n_pos)
    if cells is None:
        cells = [(p, q) for p in range(n_pos) for q in range(p + 1, n_pos)]
    rows = []
    for p, q in cells:
        for _ in range(reps):
            rows.append((p + 10, q + 10, alpha[p] + alpha[q]
                         + rng.normal(0, noise)))
    t = pd.DataFrame(rows, columns=["pos_a", "pos_b", "res"])
    sd = float(t.res.std(ddof=1))
    t["residual_raw"] = t.res
    t["residual"] = t.res / sd
    t["residual_scale_sd"] = sd
    t["predicted"] = rng.normal(0, 1, len(t))
    t["observed"] = t.predicted + t.residual_raw
    return t.drop(columns="res")


# ------------------------------------------------------------ Q1 unit tests
def test_random_split_harness_is_crossfit_assay_exactly(ph):
    """The split is the ONLY thing allowed to differ between the two splits."""
    t = _assay(alpha_sd=0.7, seed=11)
    a = crossfit_assay(t, seed=0)
    b = ph.crossfit_splits(t, ph.random_pair_splits(len(t), 0))
    assert a.keys() == b.keys()
    for k in a:
        assert np.array_equal(a[k], b[k]), k
    a = crossfit_assay(t, seed=0, permute_rng=np.random.default_rng(3))
    b = ph.crossfit_splits(t, ph.random_pair_splits(len(t), 0),
                           permute_rng=np.random.default_rng(3))
    for k in a:
        assert np.array_equal(a[k], b[k]), k


def test_whole_pair_split_never_shares_a_position_pair(ph):
    t = _assay(n_pos=7, reps=5, seed=12)
    cells, n_cells = ph.cell_codes(t.pos_a.to_numpy(), t.pos_b.to_numpy())
    assert n_cells == 21
    # orientation does not matter: (q, p) is the same cell as (p, q)
    c2, _ = ph.cell_codes(t.pos_b.to_numpy(), t.pos_a.to_numpy())
    assert np.array_equal(cells, c2)
    splits = ph.whole_pair_splits(cells)
    assert len(splits) == 5
    tested = np.concatenate([te for _, te in splits])
    assert np.array_equal(np.sort(tested), np.arange(len(t)))
    for tr, te in splits:
        assert not np.intersect1d(cells[tr], cells[te]).size


def test_few_cells_fall_back_to_leave_one_cell_out(ph):
    t = _assay(n_pos=4, reps=30, cells=[(0, 1), (2, 3)], seed=13)
    cells, n_cells = ph.cell_codes(t.pos_a.to_numpy(), t.pos_b.to_numpy())
    splits = ph.whole_pair_splits(cells)
    assert n_cells == 2 and len(splits) == 2
    ia, ib, n_pos = ph.position_codes(t.pos_a.to_numpy(), t.pos_b.to_numpy())
    seen = ph.positions_seen(ia, ib, n_pos, splits)
    assert (seen == 0).all()          # two disjoint cells: nothing transfers
    with pytest.raises(ValueError):
        ph.whole_pair_splits(np.zeros(10, dtype=int))


def test_a_held_out_cell_never_informs_its_own_prediction(ph):
    """Shifting every residual of one cell must not move that cell's
    out-of-fold predictions under the whole-pair split (it does move others)."""
    t = _assay(alpha_sd=0.5, reps=10, seed=14)
    cells, _ = ph.cell_codes(t.pos_a.to_numpy(), t.pos_b.to_numpy())
    splits = ph.whole_pair_splits(cells)
    before = ph.crossfit_splits(t, splits, lambdas=(10.0,))
    t2 = t.copy()
    hit = cells == cells[0]
    t2.loc[hit, "residual"] += 25.0
    after = ph.crossfit_splits(t2, splits, lambdas=(10.0,))
    for k in before:
        assert np.allclose(after[k][hit], before[k][hit], atol=1e-9), k
        assert np.abs(after[k] - before[k]).max() > 1e-3, k


def test_position_effects_transfer_to_unseen_cells(ph):
    """Planted per-position effects ARE learnable from other cells, so a
    whole-pair holdout keeps most of the R2 when every position recurs; the
    permutation null does not."""
    t = _assay(n_pos=10, reps=15, alpha_sd=1.0, seed=15)
    cells, _ = ph.cell_codes(t.pos_a.to_numpy(), t.pos_b.to_numpy())
    splits = ph.whole_pair_splits(cells)
    r = t.residual.to_numpy()
    real = ph.crossfit_splits(t, splits, lambdas=(1.0,))
    null = ph.crossfit_splits(t, splits, lambdas=(1.0,),
                              permute_rng=np.random.default_rng(9))
    assert r2_resid_var(r, real[("position", 1.0)]) > 0.3
    assert r2_resid_var(r, null[("position", 1.0)]) < 0.02


def test_q1_rule_is_applied_mechanically(ph):
    one = np.ones(10)
    v = ph.verdict_for(0.4 * one, 0.35 * one, 0.0 * one)
    assert v["verdict"] == "HOLDS"
    v = ph.verdict_for(0.4 * one, 0.05 * one, 0.0 * one)
    assert v["verdict"] == "COLLAPSES"            # ratio < 0.25
    v = ph.verdict_for(0.4 * one, 0.35 * one, 0.5 * one)
    assert v["verdict"] == "COLLAPSES"            # never above its null
    v = ph.verdict_for(0.4 * one, 0.2 * one, 0.0 * one)
    assert v["verdict"] == "PARTIAL"
    v = ph.verdict_for(0.4 * one[:5], 0.4 * one[:5], 0.0 * one[:5])
    assert v["verdict"] == "INCONCLUSIVE"          # < 6 assays
    v = ph.verdict_for(-0.1 * one, 0.1 * one, 0.0 * one)
    assert v["verdict"] == "INCONCLUSIVE"          # ratio undefined
