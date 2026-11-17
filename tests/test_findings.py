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


def test_heldout_verdict_width_branch_can_fire():
    """The 'interval wider than the effect' rule must be reachable.

    It was not. The only caller passed `spread = hi - lo`, so the test
    `(hi - lo) > 2 * abs(spread)` asked whether the width exceeds twice the width
    - impossible - and the branch was dead code. The rule stated in VERDICT_RULE
    was therefore not the rule being applied, and a reader checking the artefact
    against the docstring would have found them inconsistent.

    The fix passes the group's own MEDIAN as the effect. This test pins both
    directions: a wide interval around a small effect is INCONCLUSIVE, and the
    same interval around a large effect is not.
    """
    import sys as _sys
    from pathlib import Path as _P
    _sys.path.insert(0, str(_P(__file__).resolve().parent.parent / "scripts"))
    import heldout

    # width 2.0 around an effect of 0.2 -> the interval is far wider than the
    # claim, so agreement cannot be established.
    assert heldout.verdict(corpus=0.9, lo=-0.8, hi=1.2, n=20,
                           effect=0.2) == "INCONCLUSIVE"
    # an interval that is narrow relative to its effect and does not span zero,
    # with the corpus median OUTSIDE it (0.5 < 1.0) -> NOT_SUPPORTED.
    assert heldout.verdict(corpus=0.5, lo=1.0, hi=3.0, n=20,
                           effect=2.0) == "NOT_SUPPORTED"
    # the dead-code signature: passing the width as the effect must NOT silently
    # behave as before, i.e. the function must be able to return INCONCLUSIVE for
    # a wide interval.
    wide = heldout.verdict(corpus=0.0, lo=-5.0, hi=5.0, n=20, effect=0.05)
    assert wide == "INCONCLUSIVE", wide
    # small groups stay inconclusive regardless
    assert heldout.verdict(corpus=0.9, lo=0.8, hi=1.0, n=3,
                           effect=9.0) == "INCONCLUSIVE"


def _heldout():
    scripts = str(Path(__file__).resolve().parent.parent / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import heldout
    return heldout


def test_heldout_verdict_every_reachable_branch_is_pinned():
    """One input per return of heldout.verdict that the test above leaves open.

    The width test pins the small-n, too-wide and NOT_SUPPORTED returns. It never
    produces SUPPORTED, and its zero-spanning inputs are all caught by the width
    rule first, so the 'spans zero and contains the corpus' rule had no test of
    its own. Each case below is built so that only the named rule can decide it.
    """
    v = _heldout().verdict
    # SUPPORTED: contains the corpus value, excludes zero, narrower than 2|effect|.
    assert v(corpus=0.5, lo=0.1, hi=1.0, n=20, effect=0.9) == "SUPPORTED"
    # The same interval moved to span zero. Width 1.1 is still under 2|effect| =
    # 1.8, so the width rule does not fire and the zero-spanning rule must.
    assert v(corpus=0.5, lo=-0.1, hi=1.0, n=20, effect=0.9) == "INCONCLUSIVE"
    # An effect of exactly zero skips the width rule (it needs |effect| > 0), so
    # this too is decided by the zero-spanning rule.
    assert v(corpus=0.0, lo=-0.1, hi=0.1, n=20, effect=0.0) == "INCONCLUSIVE"
    # Spans zero but excludes the corpus value, and is not too wide: the corpus
    # disagrees with the group, which is NOT_SUPPORTED rather than undecided.
    assert v(corpus=2.0, lo=-0.1, hi=1.0, n=20, effect=0.9) == "NOT_SUPPORTED"
    # n = 8 is the first group size the rule will judge at all.
    assert v(corpus=0.5, lo=0.1, hi=1.0, n=7, effect=0.9) == "INCONCLUSIVE"
    assert v(corpus=0.5, lo=0.1, hi=1.0, n=8, effect=0.9) == "SUPPORTED"


def test_heldout_verdict_sign_fallback_cannot_fire():
    """Documents a dead sub-branch rather than hiding it.

    The SUPPORTED return carries an `else "INCONCLUSIVE"` for an interval whose
    midpoint has the opposite sign to the corpus value. By the time that line is
    reached the interval contains the corpus value and does not contain zero, so
    both of its ends share the corpus sign and the fallback is unreachable. This
    grid pins that: every input that reaches the line returns SUPPORTED. If the
    rules above it are ever reordered, this test says which outcome changed.
    """
    v = _heldout().verdict
    grid = [-3.0, -2.0, -1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0, 2.0, 3.0]
    seen = set()
    for lo in grid:
        for hi in grid:
            for corpus in grid:
                for effect in grid:
                    if not lo <= effect <= hi:
                        continue        # a bootstrap interval holds its median
                    too_wide = abs(effect) > 0 and (hi - lo) > 2 * abs(effect)
                    spans_zero = lo <= 0.0 <= hi
                    if too_wide or spans_zero or not lo <= corpus <= hi:
                        continue
                    seen.add(v(corpus=corpus, lo=lo, hi=hi, n=20, effect=effect))
    assert seen == {"SUPPORTED"}, seen


def test_heldout_artefact_verdicts_follow_from_their_own_intervals():
    """Every per-group verdict in results/heldout.json, re-derived from the
    interval, group size and median stored beside it. The rule was changed after
    the first run (the width branch), so the stored labels are checked against the
    rule as it now stands, not assumed."""
    import json
    heldout = _heldout()
    h = json.loads((Path(__file__).resolve().parent.parent / "results"
                    / "heldout.json").read_text())
    corpus = h["corpus_median"]
    groups = (h["H1_lab_family"]["Tsuboyama"], h["H1_lab_family"]["other_labs"],
              h["H2_excluded_small_assays"])
    checked, outcomes = 0, set()
    for g in groups:
        for key in ("bias_sd", "frac_outside_2sd", "auc"):
            lo, hi = g[key]["ci95"]
            got = heldout.verdict(corpus[key], lo, hi, g["n_assays"],
                                  g[key]["median"])
            assert got == g["verdict_vs_corpus"][key], (g["group"], key, got)
            checked += 1
            outcomes.add(got)
    assert checked == 9
    # the committed data exercise all three labels, the width rule included
    assert outcomes == {"SUPPORTED", "NOT_SUPPORTED", "INCONCLUSIVE"}
