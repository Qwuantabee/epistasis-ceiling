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


# ------------------------------------------------------------ Q2 unit tests
@pytest.fixture(scope="module")
def mf():
    return _load("monotone_fit")


def _global_assay(kind, n_pos=10, per_pos=6, noise=0.3, seed=0):
    """Singles cluster by position (position means m_p); a double's additive
    prediction is A = s_a + s_b. kind="saturation": observed = 2 tanh(A/2) +
    noise (a global monotone curve, NO position-specific term).
    kind="position": observed = A + alpha_p + alpha_q + noise with alpha
    independent of m_p (position-specific, NO nonlinearity)."""
    rng = np.random.default_rng(seed)
    m = rng.normal(0, 1.0, n_pos)
    alpha = rng.normal(0, 0.6, n_pos)
    s = {p: rng.normal(m[p], 0.5, per_pos) for p in range(n_pos)}
    rows = []
    for p in range(n_pos):
        for q in range(p + 1, n_pos):
            for i in range(per_pos):
                for j in range(per_pos):
                    a = s[p][i] + s[q][j]
                    if kind == "saturation":
                        obs = 2.0 * np.tanh(a / 2.0)
                    else:
                        obs = a + alpha[p] + alpha[q]
                    rows.append((p + 10, q + 10, a, obs + rng.normal(0, noise)))
    t = pd.DataFrame(rows, columns=["pos_a", "pos_b", "predicted", "observed"])
    t["residual_raw"] = t.observed - t.predicted
    sd = float(t.residual_raw.std(ddof=1))
    t["residual"] = t.residual_raw / sd
    t["residual_scale_sd"] = sd
    return t


def _fit_q2(mf, t, lam=1.0, **kw):
    r = t.residual.to_numpy()
    x = t.predicted.to_numpy() / float(t.residual_scale_sd.iloc[0])
    ia, ib, n_pos = mf.position_codes(t.pos_a.to_numpy(), t.pos_b.to_numpy())
    preds, diag = mf.crossfit_models(x, r, ia, ib, n_pos, lam,
                                     mf.inner_splits(len(t), 0), **kw)
    return {k: r2_resid_var(r, v) for k, v in preds.items()}, preds, diag


def test_isotonic_helper_is_sklearn_isotonic_regression(mf):
    from sklearn.isotonic import IsotonicRegression
    rng = np.random.default_rng(21)
    x = np.round(rng.normal(size=3000), 2)          # many ties
    y = np.tanh(x) + rng.normal(0, 0.5, size=3000)
    xq = np.concatenate([rng.normal(size=500) * 2, [-9.0, 9.0]])   # beyond range
    ref = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(x, y)
    got = mf.IsoOnX(x).fit(y)
    assert np.allclose(got(xq), ref.predict(xq), atol=1e-10)
    assert np.allclose(got(x), ref.predict(x), atol=1e-10)


def test_position_model_is_the_committed_position_model(mf):
    """(3) must be crossfit_assay's position prediction, real and null."""
    t = _global_assay("position", seed=22)
    _, preds, _ = _fit_q2(mf, t, lam=10.0)
    ref = crossfit_assay(t, lambdas=(10.0,), seed=0)
    assert np.array_equal(preds["position"], ref[("position", 10.0)])
    _, preds, _ = _fit_q2(mf, t, lam=10.0, r_rng=np.random.default_rng(4))
    ref = crossfit_assay(t, lambdas=(10.0,), seed=0,
                         permute_rng=np.random.default_rng(4))
    assert np.array_equal(preds["position"], ref[("position", 10.0)])


def test_global_saturation_is_absorbed_by_the_monotone_term(mf):
    """No position-specific term planted: position alone scores, because single
    effects cluster by position, but adds ~nothing once isotonic is allowed."""
    t = _global_assay("saturation", seed=23)
    r2, _, _ = _fit_q2(mf, t)
    assert r2["position"] > 0.1                       # the confound is real
    assert r2["isotonic"] > r2["position"]
    assert r2["position+isotonic"] - r2["isotonic"] < 0.02


def test_position_specific_effects_survive_the_monotone_term(mf):
    t = _global_assay("position", seed=24)
    r2, _, _ = _fit_q2(mf, t)
    assert r2["isotonic"] < 0.1
    assert r2["position+isotonic"] - r2["isotonic"] > 0.3
    lab, _, _ = _fit_q2(mf, t, label_rng=np.random.default_rng(5))
    assert lab["position+isotonic"] - r2["isotonic"] < 0.02   # its null
    full, _, _ = _fit_q2(mf, t, r_rng=np.random.default_rng(6))
    for k in ("isotonic", "position", "position+isotonic"):
        assert full[k] < 0.02, k


def test_backfit_first_step_is_the_sequential_fit(mf):
    t = _global_assay("saturation", seed=25)
    r = t.residual.to_numpy()
    x = t.predicted.to_numpy() / float(t.residual_scale_sd.iloc[0])
    ia, ib, n_pos = mf.position_codes(t.pos_a.to_numpy(), t.pos_b.to_numpy())
    (g1, mu1, a1), _, it, delta = mf.backfit("isotonic", x, r, ia, ib, n_pos, 1.0)
    g = mf.IsoOnX(x).fit(x + r)
    mu, a = mf.fit_position(ia, ib, r - (g(x) - x), n_pos, 1.0)
    assert np.allclose(g1(x), g(x)) and mu1 == pytest.approx(mu)
    assert np.allclose(a1, a)
    assert it >= 1 and delta <= mf.BF_TOL


def test_q2_rule_is_applied_mechanically(mf):
    e = mf.EPS
    assert mf.decide([e + 0.01, 0.3], 0.8) == "SUPPORTED"
    assert mf.decide([e + 0.01, 0.3], 0.6) == "INCONCLUSIVE"
    assert mf.decide([e + 0.01, 0.3], 0.4) == "NOT_SUPPORTED"
    assert mf.decide([-0.05, e - 0.001], 0.9) == "NOT_SUPPORTED"
    assert mf.decide([0.0, 0.1], 0.9) == "INCONCLUSIVE"
    assert mf.decide([e + 0.01, 0.3], None) == "SUPPORTED"
    assert mf.decide([-0.2, -0.1], None) == "NOT_SUPPORTED"



# ------------------------------------------------------------ artefacts
SUMMARY = ROOT / "results" / "assay_summary.csv"


@pytest.fixture(scope="module")
def ph_art():
    return json.loads(PH_ART.read_text())


@pytest.fixture(scope="module")
def mf_art():
    return json.loads(MF_ART.read_text())


def _check_provenance(art, script):
    import hashlib
    prov = art["provenance"]
    assert prov["command"].startswith(f"python scripts/{script}.py")
    assert re.fullmatch(r"[0-9a-f]{40}", prov["git_head"])
    assert prov["code_uncommitted_at_run"] == "", \
        "artefact was produced by uncommitted code"
    for rel, digest in prov["code_sha256"].items():
        if rel == "src/epsceil/pairs.py":
            continue        # owned elsewhere; recorded, not pinned
        got = hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
        assert got == digest, f"{rel} changed since results were produced"


def _check_corpus(art):
    s = pd.read_csv(SUMMARY)
    s = s[s.n_pairs >= 500]
    assert art["n_assays"] == len(s) == 48
    assert art["n_pairs"] == int(s.n_pairs.sum())


def test_holdout_artefact_provenance_and_corpus(ph_art):
    _check_provenance(ph_art, "position_holdout")
    _check_corpus(ph_art)


def test_holdout_rule_was_fixed_in_code(ph, ph_art):
    assert ph_art["rule"] == json.loads(json.dumps(ph.RULE))


def test_holdout_random_split_reproduces_alt_baseline(ph_art):
    """The random-split arm IS the committed result, per assay, exactly."""
    rep = ph_art["reproduces_committed_random_split"]
    assert set(rep) == {"position", "pair_rank1", "pair_rank2", "pair_rank3"}
    for m, v in rep.items():
        assert v["exact"], (m, v)
        assert v["ours_median"] == pytest.approx(v["committed_median"],
                                                 abs=1e-12)


def test_holdout_every_model_has_a_null_at_or_below_zero(ph_art):
    for m, strata in ph_art["summary"].items():
        for split in ("random_pair", "whole_position_pair"):
            assert strata["all"][split]["null_median_r2"] < 0.01, (m, split)


def test_holdout_verdicts_follow_the_rule(ph, ph_art):
    rows = ph_art["per_assay"]
    cells = [r["n_cells"] for r in rows]
    assert ph_art["position_pairs_per_assay"]["median"] == float(np.median(cells))
    strata = {"all": rows}
    for name, lo, hi in ph.CELL_BINS:
        strata[name] = [r for r in rows if r["bin"] == name]
    for m in ph.REPORT:
        for sname, rs in strata.items():
            got = ph.verdict_for(
                np.array([r["random_pair"][m]["r2"] for r in rs]),
                np.array([r["whole_position_pair"][m]["r2"] for r in rs]),
                np.array([r["whole_position_pair"][m]["null_mean"] for r in rs]))
            assert got["verdict"] == ph_art["verdict"][m][sname]["verdict"]


def test_monotone_artefact_provenance_and_corpus(mf_art):
    _check_provenance(mf_art, "monotone_fit")
    _check_corpus(mf_art)


def test_monotone_rule_was_fixed_in_code(mf, mf_art):
    assert mf_art["rule"] == json.loads(json.dumps(mf.RULE))


def test_monotone_position_model_is_alt_baseline_position(mf_art):
    assert mf_art["reproduces_committed_position"]["exact"]


def test_monotone_every_fitted_model_has_a_null(mf, mf_art):
    for m in mf_art["models"]:
        if m["model"] == "additive":
            assert "null_median_r2" not in m
            continue
        assert m["null_median_r2"] < 0.01, m["model"]


def test_monotone_verdict_follows_the_rule(mf, ph, mf_art):
    """Recompute D42, its label null, the interval and the verdict from the
    per-assay rows; the stored verdict must be what the rule says."""
    per = mf_art["per_assay"]
    d42 = np.array([p["r2"]["position+isotonic"] - p["r2"]["isotonic"]
                    for p in per])
    n42 = np.array([p["label_null_mean"]["position+isotonic"]
                    - p["r2"]["isotonic"] for p in per])
    ci = ph.boot_median_ci(d42)
    beats = float(np.mean(d42 - n42 > mf.TIE))
    assert ci == pytest.approx(mf_art["verdict_inputs"]["D42_ci95"], abs=1e-12)
    assert mf_art["verdict"] == mf.decide(ci, beats)
    d32 = np.array([p["r2"]["position"] - p["r2"]["isotonic"] for p in per])
    assert mf_art["verdict_if_decided_by_3_minus_2"] == mf.decide(
        ph.boot_median_ci(d32), None)
    cap = sum(p["backfit"]["max_iter"] >= mf.BF_MAX_ITER
              and p["backfit"]["max_last_change"] > mf.BF_TOL for p in per)
    assert mf_art["backfit"]["assays_hitting_iteration_cap"] == cap
