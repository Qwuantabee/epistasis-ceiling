"""Tests for the stronger-baseline comparison (scripts/alt_baseline.py).

Two kinds. Unit tests plant KNOWN structure in a synthetic assay and check that
each model finds what was planted, that its permutation null does not, and that
no pair is ever scored by a fit that saw it. Artefact tests tie the committed
results/alt_baseline.json to the committed results/assay_summary.csv and to the
code that produced it, so a fresh clone without results/pairs/ can check it.
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

from epsceil.baseline import (crossfit_assay, lowrank_pair_term,  # noqa: E402
                              r2_resid_var, score_correction)
from epsceil.pairs import add_centred_columns, summarise_table  # noqa: E402

ART = ROOT / "results" / "alt_baseline.json"
SUMMARY = ROOT / "results" / "assay_summary.csv"


def _assay(n_pos=8, reps=20, alpha_sd=0.0, pair_u=None, noise=1.0,
           offset=0.0, seed=0):
    """Synthetic pair table: every unordered position pair, reps doubles each,
    residual = offset + alpha_p + alpha_q + u_p*u_q + noise, scaled like
    build_pairs does (raw residual divided by its own sd)."""
    rng = np.random.default_rng(seed)
    alpha = rng.normal(0, alpha_sd, n_pos) if alpha_sd else np.zeros(n_pos)
    rows = []
    for p in range(n_pos):
        for q in range(p + 1, n_pos):
            inter = 0.0 if pair_u is None else pair_u[p] * pair_u[q]
            for _ in range(reps):
                rows.append((p + 10, q + 10, offset + alpha[p] + alpha[q]
                             + inter + rng.normal(0, noise)))
    t = pd.DataFrame(rows, columns=["pos_a", "pos_b", "res"])
    sd = float(t.res.std(ddof=1))
    t["residual_raw"] = t.res
    t["residual"] = t.res / sd
    t["residual_scale_sd"] = sd
    t["predicted"] = rng.normal(0, 1, len(t))
    t["observed"] = t.predicted + t.residual_raw
    return t.drop(columns="res")


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "alt_baseline", ROOT / "scripts" / "alt_baseline.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------ unit
def test_plain_row_is_summarise_table():
    """r_hat = 0 must reproduce plain additivity through the ONE definition."""
    t = _assay(offset=-1.0)
    got = score_correction(t, np.zeros(len(t)), "toy")
    ref = summarise_table(add_centred_columns(t), "toy")
    for k in ("auc_direction", "frac_outside_2sd", "frac_outside_2sd_centred",
              "bias_mean_sd", "spearman_additive"):
        assert got[k] == pytest.approx(ref[k], abs=1e-12), k
    assert got["r2_resid_var"] == 0.0
    assert got["mse_skill"] == 0.0


def test_offset_removes_bias_but_explains_no_variance():
    """A constant is a constant: bias goes, spread and ranking do not move."""
    t = _assay(offset=-1.0)
    pr = crossfit_assay(t, lambdas=(1.0,))
    s = score_correction(t, pr[("assay_offset", None)], "toy")
    plain = score_correction(t, np.zeros(len(t)), "toy")
    assert abs(plain["bias_mean_sd"]) > 0.5
    assert abs(s["bias_mean_sd"]) < 0.02
    assert abs(s["r2_resid_var"]) < 0.01
    assert s["frac_outside_2sd"] < plain["frac_outside_2sd"]


def test_position_term_finds_planted_effects_and_its_null_does_not():
    t = _assay(alpha_sd=1.0, reps=30, seed=1)
    real = crossfit_assay(t, lambdas=(1.0,))
    null = crossfit_assay(t, lambdas=(1.0,),
                          permute_rng=np.random.default_rng(7))
    r = t.residual.to_numpy()
    assert r2_resid_var(r, real[("position", 1.0)]) > 0.3
    assert r2_resid_var(r, null[("position", 1.0)]) < 0.02


def test_pair_term_finds_planted_interaction_that_position_cannot():
    u = np.array([1.5, -1.5, 1.0, -1.0, 0.5, -0.5, 1.2, -1.2])  # mean zero
    t = _assay(pair_u=u, reps=30, seed=2)
    real = crossfit_assay(t, lambdas=(1.0,))
    null = crossfit_assay(t, lambdas=(1.0,),
                          permute_rng=np.random.default_rng(8))
    r = t.residual.to_numpy()
    pos = r2_resid_var(r, real[("position", 1.0)])
    rank1 = r2_resid_var(r, real[("pair_rank1", 1.0)])
    assert rank1 > pos + 0.2, (pos, rank1)
    assert r2_resid_var(r, null[("pair_rank1", 1.0)]) < 0.02


def test_no_structure_means_no_gain_out_of_fold():
    """On pure noise every fitted model must sit at or below zero out of fold.

    The bound is tighter than what an IN-SAMPLE fit would score here (cell
    means on 28 cells x 20 pairs explain ~27/559 = 0.048 of pure noise), so this
    fails if the inner split stops separating fit from score.
    """
    t = _assay(reps=20, seed=3)
    pr = crossfit_assay(t, lambdas=(1.0,))
    r = t.residual.to_numpy()
    for key, v in pr.items():
        assert r2_resid_var(r, v) < 0.01, key


def test_a_pair_never_predicts_itself():
    """Changing one residual must not change that pair's own prediction."""
    t = _assay(alpha_sd=0.5, seed=4)
    before = crossfit_assay(t, lambdas=(10.0,))
    t2 = t.copy()
    t2.loc[17, "residual"] += 50.0
    after = crossfit_assay(t2, lambdas=(10.0,))
    for key in before:
        assert after[key][17] == pytest.approx(before[key][17], abs=1e-9), key
        # ...while it does move predictions for pairs it trained
        assert np.abs(after[key] - before[key]).max() > 1e-3, key


def test_lowrank_term_is_symmetric_and_rank_k():
    rng = np.random.default_rng(5)
    n_pos = 9
    m = rng.normal(size=(n_pos, n_pos))
    s = m + m.T
    np.fill_diagonal(s, 0.0)
    c = np.ones((n_pos, n_pos)) * 4.0
    np.fill_diagonal(c, 0.0)
    for k in (1, 2, 3):
        low = lowrank_pair_term(s * 4.0, c, 1.0, k)
        assert np.allclose(low, low.T)
        assert np.linalg.matrix_rank(low, tol=1e-8) <= k


def test_outer_selection_uses_only_training_assays():
    """The lambda for a test assay is chosen on the OTHER folds' assays only."""
    mod = _load_script()
    rng = np.random.default_rng(6)
    lams = mod.LAMBDAS
    p1 = []
    for i in range(12):
        r2 = {mod._key(m, lam): float(rng.normal())
              for m in mod.FITTED for lam in lams}
        p1.append({"assay": f"A{i:02d}", "bias": float(rng.normal()),
                   "null_bias": [0.0], "r2_real": r2, "r2_null": [r2]})
    plan, folds = mod.outer_selection(p1, n_perm=1)
    by = {d["assay"]: d for d in p1}
    tested = [a for f in folds for a in f["test_assays"]]
    assert sorted(tested) == sorted(by)            # every assay tested once
    for f in folds:
        train = [a for a in by if a not in f["test_assays"]]
        for m in mod.FITTED:
            med = {lam: float(np.median([by[a]["r2_real"][mod._key(m, lam)]
                                         for a in train])) for lam in lams}
            assert f["lambda"][m]["train_median_r2"] == med
            assert f["lambda"][m]["chosen"] == max(lams,
                                                   key=lambda x: (med[x], x))
            for a in f["test_assays"]:
                assert plan[a]["real"][m] == f["lambda"][m]["chosen"]
        assert f["pooled_offset"] == float(np.median([by[a]["bias"]
                                                      for a in train]))


# ------------------------------------------------------------------ artefact
@pytest.fixture(scope="module")
def art():
    return json.loads(ART.read_text())


def test_artefact_names_its_reproducing_command(art):
    prov = art["provenance"]
    assert prov["command"].startswith("python scripts/alt_baseline.py")
    assert re.fullmatch(r"[0-9a-f]{40}", prov["git_head"])
    assert prov["code_uncommitted_at_run"] == "", \
        "artefact was produced by uncommitted code"


def test_artefact_was_produced_by_this_code(art):
    """A stale artefact (code edited, not re-run) must fail, not drift."""
    import hashlib
    for rel, digest in art["provenance"]["code_sha256"].items():
        if rel == "src/epsceil/pairs.py":
            continue        # owned elsewhere; recorded, not pinned
        got = hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
        assert got == digest, f"{rel} changed since results were produced"


def test_plain_additivity_matches_committed_summary(art):
    """Like-for-like: the recomputed baseline IS the README's baseline."""
    s = pd.read_csv(SUMMARY)
    s = s[s.n_pairs >= 500]
    assert art["n_assays"] == len(s) == 48
    assert art["n_pairs"] == int(s.n_pairs.sum())
    plain = next(m for m in art["models"] if m["model"] == "additive")
    for k in ("auc_direction", "frac_outside_2sd", "bias_mean_sd"):
        assert plain["median"][k] == pytest.approx(float(s[k].median()),
                                                   abs=1e-9), k
    per = {d["assay"]: d["models"]["additive"] for d in art["per_assay"]}
    for row in s.itertuples():
        assert per[row.assay]["frac_outside_2sd"] == pytest.approx(
            row.frac_outside_2sd, abs=1e-9)


def test_every_fitted_model_has_a_null_at_zero_spread_gain(art):
    """The brief's admissibility rule: no fitted result without its null, and
    the null must sit at zero improvement in explained residual variance."""
    for m in art["models"]:
        if m["model"] == "additive":
            assert "null_median" not in m
            continue
        assert "null_median" in m, m["model"]
        assert m["null_median"]["r2_resid_var"] < 0.01, m["model"]
        assert m["null_median_delta_vs_additive"]["auc_direction"] < 0.005
