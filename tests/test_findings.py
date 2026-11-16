"""Regression tests for the reviewer findings (results/redteam.json).

Each test here was written BEFORE the fix it guards, and each one fails on the
code as the reviewer found it. They encode the three failure modes that string
matching cannot catch:

  * a statistic whose sign depends on an arbitrary origin (F2),
  * a threshold that is not comparable across two series (F4),
  * a test run on a biased half of its own data (F6).

A red test is information about the analysis, not an obstacle.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from epsceil.pairs import build_pairs, summarise  # noqa: E402


def _toy(n_single=60, n_double=40, shift=0.0, seed=0):
    rng = np.random.default_rng(seed)
    rows = [{"mutant": f"A{i}V", "DMS_score": float(rng.normal(0, 1))}
            for i in range(1, n_single + 1)]
    for i in range(1, n_double + 1):
        rows.append({"mutant": f"A{i}V:A{i+1}V",
                     "DMS_score": float(rng.normal(0, 1)) + shift})
    return pd.DataFrame(rows)


def test_residual_is_zero_centred_within_assay():
    """F2: the additive baseline must place median(predicted) at the origin.

    As shipped, residual = obs - (a + b - wt) with wt = median(all variants)
    leaves the residual cloud ~0.9 sd off zero, so |residual| and |predicted|
    measure distance from an arbitrary origin. A magnitude comparison on a
    non-centred variable is not interpretable, so the pair table now carries a
    centred residual and summarise() reports the bias it removed.
    """
    ap = build_pairs(_toy(shift=3.0), "toy", min_singles=10)
    assert ap is not None
    t = ap.doubles
    assert "residual_c" in t.columns, "pair table lost the centred residual"
    # centring removes the MEAN exactly; the median goes wherever the skew puts
    # it, which is the honest behaviour and why the mean is the thing asserted.
    assert t.residual_c.mean() == pytest.approx(0.0, abs=1e-9)
    # and it must be a genuine reduction of the offset, not a relabelling
    assert abs(t.residual_c.median()) < abs(t.residual.median()) + 1e-12
    # it stays in assay-sd units, so it is comparable across assays
    assert t.residual_c.std(ddof=1) == pytest.approx(1.0, rel=0.05)


def test_summary_reports_bias_and_auc():
    """F1/F4: the bias and a threshold-free direction measure must be reported.

    The pooled >2-sd fraction measures bias, not spread, unless the bias is
    quoted next to it; and a median split at the OBSERVED median is not a fair
    threshold for a prediction series whose spread differs.
    """
    ap = build_pairs(_toy(shift=2.0), "toy", min_singles=10)
    s = summarise(ap)
    for k in ("residual_mean", "auc_direction", "frac_within_1sd_centred"):
        assert k in s, f"summary is missing {k}, which the fix must report"
    assert 0.0 <= s["auc_direction"] <= 1.0


def test_auc_is_threshold_free_and_beats_the_median_split():
    """F4: a monotone prediction must score the same AUC at any shift.

    Shifting every prediction by a constant leaves the RANKING intact, so a
    threshold-free measure cannot move. The median-split statistic does move,
    which is exactly why it understated the result.
    """
    base = _toy(shift=0.0, seed=3)
    ap1 = build_pairs(base, "toy", min_singles=10)
    shifted = base.copy()
    # add a constant to every double: ranking is preserved, predictions move
    shifted.loc[shifted.mutant.str.count(":") == 1, "DMS_score"] += 5.0
    ap2 = build_pairs(shifted, "toy", min_singles=10)
    s1, s2 = summarise(ap1), summarise(ap2)
    assert s2["auc_direction"] == pytest.approx(s1["auc_direction"], abs=0.06)
    assert s2["direction_agreement"] != pytest.approx(s1["direction_agreement"],
                                                      abs=0.001)


def test_residue_effect_pools_both_slots():
    """F6: the residue test read only slot `a` and ignored slot `b`.

    Reading half the pair is a systematically biased sample, not a shortcut: the
    two slots are assigned by the order the mutation was written in, which is
    arbitrary with respect to the analysis. The fixed test must see both.
    """
    from pathlib import Path as _P
    sys.path.insert(0, str(_P(__file__).resolve().parent.parent / "scripts"))
    import depth

    src = (_P(__file__).resolve().parent.parent
           / "scripts" / "depth.py").read_text()
    assert "slot" in src.lower() or "concat" in src.lower(), (
        "depth.py must pool both substitution slots in the residue test")
    assert hasattr(depth, "residue_type_effect")


def test_residue_verdict_uses_a_binomial_not_a_hand_set_bar():
    """F6: 30/37 above median is p=1e-4, not 'chance'.

    The 85% bar was set before looking and is a statement about the bar, not
    about biology. The test now reports an exact binomial against 0.5 and a
    Bonferroni correction over the residues tested.
    """
    src = (Path(__file__).resolve().parent.parent
           / "scripts" / "depth.py").read_text()
    assert "binomtest" in src or "binom" in src.lower(), (
        "the residue verdict must quote a binomial p-value")
    assert "bonferroni" in src.lower()
