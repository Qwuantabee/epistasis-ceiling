"""Tests for scripts/esm_scale.py - the corpus-scale ESM-2 probe.

What has to be true for its verdicts to mean anything:

  * the streamed scaler + PCA equals sklearn's StandardScaler + PCA (the shipped
    estimator) - the 5,121-wide matrix is never built, so this is the only check;
  * the controls are what they claim: shuffled rows stay inside their assay,
    noise is one fixed draw per pair whichever rows a pass selects;
  * head (b)'s closed-form gradient-descent path equals iterated descent, and
    the validation MSE it stops on is the real held-out MSE;
  * the pipeline FINDS a planted pair-level signal (so a null from it is
    informative) and does not call one where there is none;
  * the verdict rule judges against the LARGER control, and the committed
    artefact's verdicts, fraction and incremental rho recompute from its own
    fold values under the rule the script carries.
"""
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "results" / "esm_scale.json"

_spec = importlib.util.spec_from_file_location("esm_scale", ROOT / "scripts" / "esm_scale.py")
es = importlib.util.module_from_spec(_spec)
sys.modules["esm_scale"] = es
_spec.loader.exec_module(es)


def test_window_is_the_shipped_window():
    seq = "ACDEFGHIKLMNPQRSTVWY"
    assert es.window(seq, 1, 3) == "ACDE"
    assert es.window(seq, 10, 3) == "HIKLMNP"          # centred on L, residue 10
    assert es.window(seq, 20, 3) == "TVWY"
    assert len(es.window(seq, 10, 8)) == 17


def test_pairs_sharing_positions_share_window_rows():
    df = pd.DataFrame({"assay": ["X", "X", "X", "Y"], "pos_a": [2, 2, 5, 2],
                       "pos_b": [5, 7, 7, 5]})
    seqs = {"X": "MKTAYIAKQR", "Y": "MKTAYIAKQR"}      # same parent: same windows
    windows, ia, ib = es.window_ids(df, seqs, 2)
    assert len(windows) == 3                            # positions 2, 5, 7
    assert ia[0] == ia[1] == ia[3] and ib[0] == ib[3] and ib[1] == ib[2]
    for i in range(4):
        assert windows[ia[i]] == es.window(seqs[df.assay[i]], int(df.pos_a[i]), 2)
        assert windows[ib[i]] == es.window(seqs[df.assay[i]], int(df.pos_b[i]), 2)


def test_streamed_scaler_and_pca_match_sklearn():
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(1)
    W = (rng.normal(size=(40, 5)) * [1, 3, 0.5, 2, 1]
         + [10, -4, 0, 2, 50]).astype(np.float32)
    n = 600
    ia, ib = rng.integers(0, 40, n), rng.integers(0, 40, n)
    lsep = np.log1p(rng.integers(1, 60, n)).astype(np.float32)
    be = es.NumpyBackend()
    src = es.PairFeatures(be, be.put(W), ia, ib, lsep, "full", "esm", chunk=97)
    X = np.hstack([W[ia], W[ib], np.abs(W[ia] - W[ib]), W[ia] * W[ib],
                   lsep[:, None]]).astype(np.float64)
    assert src.dim == X.shape[1] == 21
    sel = rng.random(n) < 0.7
    n_tr, mu = es.stream_mean(src, sel)
    _, S, _, _ = es.stream_scatter(src, sel, mu)
    sigma, V, _ = es.pca_from_scatter(n_tr, S, 6)
    Z = es.stream_project(src, np.ones(n, bool), mu, V / sigma[:, None])
    sc = StandardScaler().fit(X[sel])
    Zs = PCA(n_components=6, svd_solver="full").fit(sc.transform(X[sel])).transform(
        sc.transform(X))
    assert n_tr == sel.sum()
    assert np.allclose(mu, sc.mean_, rtol=1e-5, atol=1e-3)
    assert np.allclose(sigma, sc.scale_, rtol=1e-4)
    # same subspace (rotation-free check) and the leading components agree up to sign
    coef, *_ = np.linalg.lstsq(Z.astype(np.float64), Zs, rcond=None)
    assert np.abs(Z @ coef - Zs).max() < 1e-3 * Zs.std()
    for j in range(3):
        sgn = np.sign(Z[:, j] @ Zs[:, j])
        assert np.allclose(sgn * Z[:, j], Zs[:, j], atol=1e-3 * Zs[:, j].std())


def test_shuffle_stays_within_assay_and_moves_rows():
    groups = np.array(["a"] * 50 + ["b"] * 30 + ["c"])
    perm = es.within_group_permutation(groups, seed=0)
    assert sorted(perm.tolist()) == list(range(81))
    assert (groups[perm] == groups).all()
    assert (perm[:50] != np.arange(50)).mean() > 0.5


def test_noise_is_one_fixed_draw_per_pair_at_the_heads_width():
    rng = np.random.default_rng(3)
    W = rng.normal(size=(10, 4)).astype(np.float32)
    n = 100
    ia, ib = rng.integers(0, 10, n), rng.integers(0, 10, n)
    be = es.NumpyBackend()
    for layout, width in (("full", 17), ("pooled", 8)):
        noise = es.PairFeatures(be, W, ia, ib, np.zeros(n), layout, "noise", seed=5,
                                chunk=16)
        real = es.PairFeatures(be, W, ia, ib, np.zeros(n), layout, "esm", chunk=16)
        assert noise.dim == real.dim == width
        everything = np.ones(n, bool)
        some = rng.random(n) < 0.4
        a = np.vstack([noise.get(c, everything[lo:hi]) for c, lo, hi, _ in
                       es._chunks(noise, everything)])
        b = np.vstack([noise.get(c, some[lo:hi]) for c, lo, hi, _ in
                       es._chunks(noise, some)])
        assert np.array_equal(a[some], b)


def test_gd_closed_form_equals_iterated_descent_and_scores_real_val_mse():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(300, 6)) @ rng.normal(size=(6, 6)) + 3.0
    y = X @ rng.normal(size=6) + rng.normal(size=300)
    tr, va = np.arange(200), np.arange(200, 300)
    mu, ybar = X[tr].mean(0), y[tr].mean()
    Xc, Vc = X[tr] - mu, X[va] - mu
    S_t, sxy_t = Xc.T @ Xc, Xc.T @ (y[tr] - ybar)
    S_v, sxy_v = Vc.T @ Vc, Vc.T @ (y[va] - ybar)
    syy_v = float(((y[va] - ybar) ** 2).sum())
    sigma = es.standard_scale(200, S_t)
    G, h = S_t / 200 / np.outer(sigma, sigma), sxy_t / 200 / sigma
    lam, U = np.linalg.eigh(G)
    eta = 1.0 / lam.max()
    w = np.zeros(6)
    for t in range(1, 51):
        w = w - eta * (G @ w - h)
        if t in (1, 7, 50):
            closed = U @ (es.gd_filter(lam, eta, t) * (U.T @ h))
            assert np.allclose(w, closed, rtol=1e-8, atol=1e-10)
    w_raw, info = es.early_stopped_linear(200, S_t, sxy_t, 100, S_v, sxy_v, syy_v,
                                          steps=np.array([1, 7, 50]))
    pred = ybar + (X[va] - mu) @ w_raw
    assert math.isclose(((pred - y[va]) ** 2).mean(), info["val_mse"], rel_tol=1e-8)
    assert info["steps"] in (1, 7, 50)


def test_verdict_judges_against_the_larger_control():
    noise = [0.00, 0.01, -0.01, 0.00, 0.005]
    shuf = [0.05, 0.06, 0.04, 0.05, 0.055]
    v = es.verdict([0.15, 0.16, 0.14, 0.15, 0.155], {"noise": noise, "shuffled": shuf})
    assert v["verdict"] == "SUPPORTED" and v["control"] == "shuffled"
    # far above noise, level with the shuffle: a null, not a finding
    v = es.verdict(shuf, {"noise": noise, "shuffled": shuf})
    assert v["verdict"] == "NOT_SUPPORTED" and v["delta"] == 0.0
    # a tight null rules out a 0.02 gain
    v = es.verdict([0.001, 0.0, 0.002, -0.001, 0.0], {"noise": noise, "shuffled": [0.0] * 5})
    assert v["verdict"] == "NOT_SUPPORTED" and v["delta_plus_2se"] < es.MIN_GAIN
    # a noisy null cannot rule it out
    v = es.verdict([0.10, -0.10, 0.08, -0.06, 0.0], {"noise": [0.0] * 5, "shuffled": [0.0] * 5})
    assert v["verdict"] == "INCONCLUSIVE"
    v = es.verdict([0.1, float("nan"), 0.1, 0.1, 0.1], {"noise": [0.0] * 5,
                                                       "shuffled": [0.0] * 5})
    assert v["verdict"] == "INCONCLUSIVE"


def _synthetic(signal, seed=0, n_assays=10, n_pos=14, per_assay=200, emb=6):
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(n_assays * n_pos, emb)).astype(np.float32)
    u = rng.normal(size=emb)
    ia, ib, groups, sep = [], [], [], []
    for g in range(n_assays):
        for _ in range(per_assay):
            p, q = sorted(rng.choice(n_pos, 2, replace=False))
            ia.append(g * n_pos + p)
            ib.append(g * n_pos + q)
            groups.append(f"A{g}")
            sep.append(q - p)
    ia, ib, groups = np.array(ia), np.array(ib), np.array(groups)
    s = (W[ia] + W[ib]) @ u
    s = (s - s.mean()) / s.std()
    eps = rng.normal(size=len(s))
    y = np.exp(0.8 * s + 0.3 * eps) if signal else np.exp(0.5 * eps)
    prior_abs = rng.exponential(size=len(s))
    return y, prior_abs, groups, W, ia, ib, np.log1p(np.array(sep, float))


def _run(signal):
    from sklearn.model_selection import GroupKFold
    y, prior_abs, groups, W, ia, ib, lsep = _synthetic(signal)
    folds = list(GroupKFold(n_splits=5).split(np.zeros(len(y)), groups=groups))
    rec = es.run_heads(y, prior_abs, groups, W, ia, ib, lsep, es.NumpyBackend(),
                       folds, seed=0, chunk=256, log=lambda s: None)
    return es.summarise(rec)


def test_pipeline_finds_a_planted_pair_signal_in_every_head():
    out = _run(signal=True)
    for h in ("a", "b", "c"):
        assert out[h]["verdict"]["verdict"] == "SUPPORTED", (h, out[h]["verdict"])
        assert out[h]["rho_mean"]["esm"] > 0.5
    assert out["c"]["incremental_rho_over_prior"]["esm"]["mean"] > 0.5


def test_pipeline_calls_no_signal_when_there_is_none():
    out = _run(signal=False)
    for h in ("a", "b", "c"):
        assert out[h]["verdict"]["verdict"] != "SUPPORTED", (h, out[h]["verdict"])


def test_post_hoc_arm_is_reported_beside_c_not_instead_of_it():
    from sklearn.model_selection import GroupKFold
    y, prior_abs, groups, W, ia, ib, lsep = _synthetic(True)
    folds = list(GroupKFold(n_splits=5).split(np.zeros(len(y)), groups=groups))
    rec = es.run_heads(y, prior_abs, groups, W, ia, ib, lsep, es.NumpyBackend(),
                       folds, seed=0, chunk=256, log=lambda s: None,
                       prior_alt_abs=prior_abs * 3.0)
    out = es.summarise(rec)
    assert set(out) == {"a", "b", "c", "c_posthoc"}
    assert out["c_posthoc"]["head"].startswith("POST HOC")
    # a constant rescale of the prior is the same prior: the arms must agree
    assert np.allclose(out["c"]["rho"]["esm"], out["c_posthoc"]["rho"]["esm"], atol=0.02)


def test_raw_unit_prior_ranks_backwards_when_assay_units_differ():
    """Why c_posthoc exists. Inside every assay |residual| rises with |predicted|,
    but the assays with large raw units have small sd-scaled residuals, so one
    pooled slope on the raw prior is negative and ranks each assay backwards;
    dividing by the assay's own sd of predicted (label-free) restores it."""
    from sklearn.model_selection import GroupKFold
    rng = np.random.default_rng(4)
    groups, raw, y = [], [], []
    for g in range(10):
        level = g % 5
        z = np.abs(rng.normal(size=300))
        raw.append(z * 10.0 ** level)
        y.append(np.exp(0.8 * z - 0.6 * level + 0.3 * rng.normal(size=300)))
        groups += [f"A{g}"] * 300
    raw, y, groups = np.concatenate(raw), np.concatenate(y), np.array(groups)
    sd = pd.Series(raw).groupby(groups).transform("std").to_numpy()
    within = {}
    for name, prior in (("raw", np.log1p(raw)), ("rescaled", np.log1p(raw / sd))):
        vals = []
        for tr, te in GroupKFold(n_splits=5).split(raw, groups=groups):
            pred, _ = es.ridge_predict(prior[tr, None], y[tr], prior[te, None])
            vals.append(es.fold_metrics(pred, y[te], y[tr], groups[te])
                        ["rho_within_assay_mean"])
        within[name] = float(np.mean(vals))
    assert within["raw"] < 0 < within["rescaled"]


def test_selection_covers_every_eligible_assay_and_states_the_fraction(tmp_path):
    pairs = tmp_path / "pairs"
    pairs.mkdir()
    rng = np.random.default_rng(0)

    def table(name, n):
        pd.DataFrame({"mutant": ["A1V:C2V"] * n, "a": ["A1V"] * n, "b": ["C2V"] * n,
                      "observed": 0.0, "predicted": rng.normal(size=n),
                      "residual": rng.normal(size=n), "pos_a": 1, "pos_b": 2,
                      "sep": 1}).to_csv(pairs / f"{name}.csv", index=False)

    for name, n in (("BIG", 1000), ("MID", 400), ("SMALL", 250), ("NOSEQ", 900)):
        table(name, n)
    seqs = {"BIG": "AC" * 10, "MID": "AC" * 10, "SMALL": "AC" * 10}
    df, sel = es.load_pairs(pairs, seqs, max_per_assay=500, seed=0)
    assert set(df.assay) == {"BIG", "MID"}
    assert sel["pairs_eligible"] == 1400 and sel["pairs_used"] == 900
    assert sel["fraction_of_eligible_pairs"] == 900 / 1400
    assert sel["capped_assays_original_sizes"] == {"BIG": 1000}
    assert sel["pairs_in_all_pair_tables"] == 2550
    _, sel = es.load_pairs(pairs, seqs, max_per_assay=0, seed=0)
    assert sel["pairs_used"] == 1400 and sel["fraction_of_eligible_pairs"] == 1.0


def test_embedding_cache_is_ignored_and_untracked():
    assert "results/emb_cache/" in (ROOT / ".gitignore").read_text().splitlines()
    tracked = subprocess.run(["git", "ls-files", "results/emb_cache"], cwd=ROOT,
                             capture_output=True, text=True).stdout.strip()
    assert tracked == ""


def test_artefact_carries_the_scripts_rule_and_recomputes():
    doc = json.loads(ART.read_text())
    assert doc["verdict_rule"] == json.loads(json.dumps(es.VERDICT_RULE))
    assert doc["status"] in ("preregistered", "complete")
    if doc["status"] == "preregistered":
        return
    assert doc["preregistration"]["rule_identical_to_preregistered"] is True
    p = doc["primary"]
    sel = p["selection"]
    assert sel["max_assays_smoke_limit"] is None
    assert sel["assays_used"] == sel["assays_eligible"]
    assert sum(sel["per_assay_pairs_used"].values()) == sel["pairs_used"]
    assert math.isclose(sel["fraction_of_eligible_pairs"],
                        sel["pairs_used"] / sel["pairs_eligible"])
    assert doc["headline"]["pairs_used"] == sel["pairs_used"]
    for h in ("a", "b", "c", "c_posthoc"):
        H = p["heads"][h]
        assert all(len(H["rho"][m]) == 5 for m in es.MODES)
        v = es.verdict(H["rho"]["esm"], {"noise": H["rho"]["noise"],
                                         "shuffled": H["rho"]["shuffled"]})
        assert v["verdict"] == H["verdict"]["verdict"]
        if h != "c_posthoc":                    # post hoc never enters the headline
            assert v["verdict"] == doc["headline"]["verdicts"][h]
        assert math.isclose(v["delta"], H["verdict"]["delta"], abs_tol=1e-12)
    assert set(doc["headline"]["verdicts"]) == {"a", "b", "c"}
    c = p["heads"]["c"]
    inc = np.array(c["rho"]["esm"]) - np.array(c["rho_prior_only"]["folds"])
    assert math.isclose(inc.mean(), doc["headline"]["incremental_rho_of_esm_over_prior"],
                        abs_tol=1e-12)
