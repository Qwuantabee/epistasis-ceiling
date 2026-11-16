"""Invariants for the additivity measurement.

Each test is written against a specific way the analysis could produce a
plausible-looking but wrong number, not against an obvious crash.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from epsceil.pairs import (  # noqa: E402
    build_pairs, parse_mutant, parse_mutation, summarise, wt_reference)


def test_single_mutation_parses_position_and_residues():
    m = parse_mutation("T159M")
    assert (m.wt, m.position, m.mut) == ("T", 159, "M")
    assert m.key == "T159M"


def test_stop_codon_token_parses():
    # Nonsense mutations appear in several assays; rejecting them would silently
    # drop a whole class of variant rather than erroring.
    m = parse_mutation("Q40*")
    assert m is not None and m.mut == "*"


def test_double_mutant_splits_on_colon():
    ms = parse_mutant("T159M:D166V")
    assert [m.key for m in ms] == ["T159M", "D166V"]


@pytest.mark.parametrize("spelling", ["T159M,D166V", "T159M-D166V"])
def test_alternative_separators_are_accepted(spelling):
    # Returning None here would drop real pairs from some assays; the separator
    # set is explicit rather than guessed per-row.
    ms = parse_mutant(spelling)
    assert ms is not None and len(ms) == 2


def test_garbage_mutant_returns_none_not_a_wrong_pair():
    assert parse_mutant("not_a_mutation") is None
    assert parse_mutant("") is None
    assert parse_mutant(None) is None


def test_additivity_residual_sign_is_observed_minus_predicted():
    """If this were reversed, every downstream correlation flips sign."""
    df = pd.DataFrame({
        "mutant": ["A1V", "A1V:C2D", "C2D"],
        "DMS_score": [1.0, 5.0, 2.0],
    })
    ap = build_pairs(df, "toy", min_singles=1)
    assert ap is not None
    row = ap.doubles.iloc[0]
    # centre = median([1,5,2]) = 2 ; predicted = 1 + 2 - 2 = 1 ; residual = 5 - 1 = 4
    assert row.observed == 5.0
    assert row.predicted == pytest.approx(1.0)
    assert row.residual_raw == pytest.approx(4.0)


def test_residual_is_normalised_by_its_own_assay_sd():
    """Raw and normalised columns must both be present and differ by the sd."""
    rng = np.random.default_rng(0)
    singles = [f"A{i}V" for i in range(1, 61)]
    rows = [{"mutant": s, "DMS_score": float(rng.normal(0, 1))} for s in singles]
    # ten doubles, because build_pairs requires min_singles doubles as well as
    # singles; the first version of this test supplied one and asserted against a
    # threshold the fixture could never meet.
    rows.append({"mutant": "A1V:A2V", "DMS_score": 3.0})
    for i in range(3, 13):
        rows.append({"mutant": f"A{i}V:A{i+1}V", "DMS_score": float(rng.normal(0, 1))})
    ap = build_pairs(pd.DataFrame(rows), "toy", min_singles=10)
    assert ap is not None
    d = ap.doubles.iloc[0]
    assert d.residual_scale_sd > 0
    assert d.residual == pytest.approx(d.residual_raw / d.residual_scale_sd)


def test_wt_reference_is_the_median_and_so_shift_invariant():
    """An additive shift of the whole assay must not change any residual."""
    a = pd.Series([1.0, 2.0, 3.0])
    assert wt_reference(a) == 2.0
    shifted = a + 100.0
    assert wt_reference(shifted) == 102.0


def test_shift_invariance_of_residuals_end_to_end():
    base = pd.DataFrame({
        "mutant": ["A1V", "C2D", "A1V:C2D"],
        "DMS_score": [0.5, 0.7, 1.4],
    })
    shifted = base.assign(DMS_score=base.DMS_score + 50.0)
    r1 = build_pairs(base, "toy", min_singles=1).doubles.residual_raw.iloc[0]
    r2 = build_pairs(shifted, "toy", min_singles=1).doubles.residual_raw.iloc[0]
    assert r1 == pytest.approx(r2)


def test_doubles_without_both_singles_are_excluded_not_assumed():
    """A missing single must drop the pair, never be read as zero effect."""
    df = pd.DataFrame({
        "mutant": ["A1V", "A1V:C2D"],
        "DMS_score": [1.0, 2.0],
    })
    ap = build_pairs(df, "toy", min_singles=1)
    assert ap is None or len(ap.doubles) == 0


def test_summary_reports_the_quantities_the_readme_quotes():
    rng = np.random.default_rng(1)
    singles = [f"A{i}V" for i in range(1, 41)]
    rows = [{"mutant": s, "DMS_score": float(rng.normal(0, 1))} for s in singles]
    for i in range(1, 30):
        rows.append({"mutant": f"A{i}V:A{i+1}V",
                     "DMS_score": float(rng.normal(0, 1))})
    ap = build_pairs(pd.DataFrame(rows), "toy", min_singles=10)
    s = summarise(ap)
    for k in ("n_pairs", "coverage", "spearman_additive", "direction_agreement",
              "frac_within_1sd", "residual_sd", "residual_raw_sd"):
        assert k in s, f"summary lost {k}, which the README quotes"
    assert 0.0 <= s["frac_within_1sd"] <= 1.0
    assert 0.0 <= s["direction_agreement"] <= 1.0


def test_sequence_separation_is_absolute():
    """A pair must not have negative separation, or distance analyses invert."""
    df = pd.DataFrame({
        "mutant": ["A10V", "A2D", "A10V:A2D"],
        "DMS_score": [1.0, 1.0, 1.0],
    })
    ap = build_pairs(df, "toy", min_singles=1)
    assert ap.doubles.sep.iloc[0] == 8
