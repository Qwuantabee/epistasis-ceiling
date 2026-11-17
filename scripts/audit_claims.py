#!/usr/bin/env python3
"""Check every number the README quotes against a committed artefact - in place.

The previous project published a headline that was 23x wrong because the error
column was inflated by a unit slip, and 44 green tests could not see it: the
accuracy was right and the error was wrong. Prose is where numbers rot. This
script is the mechanical guard.

Review round 2 showed the first version was a substring search: a claim passed if
any accepted spelling appeared ANYWHERE in the README, so 34 of 40 wrong READMEs
passed, including the headline AUC changed from 0.884 to 0.784, two table rows
with their values swapped, and flipped signs. It also failed OPEN: deleting an
artefact silently removed its claims and still printed a clean pass.

So now:

* every claim carries one or more ANCHORS - a regex for the place the number
  belongs (a phrase in a sentence, or a table row and column). At EVERY match of
  an anchor, the first number after it (within a short window; for a table cell,
  before the next "|") must be one of the claim's spellings, compared as a whole
  signed token: "-0.50" does not satisfy "0.50", "0.884" does not satisfy "0.88",
  and a value moved to another row fails in the row it left. An anchor that
  matches nowhere is a failure too, so deleting the sentence is not a way to pass;
* a claim with no anchor is still checked the old way but is printed as
  UNANCHORED, and --strict (what CI runs) fails if any claim is unanchored;
* every artefact the claims need is REQUIRED: a missing file, or a missing key
  inside one, is a failure, never a smaller checklist.

Usage: python scripts/audit_claims.py [--readme README.md] [--results DIR] [--strict]
Exit 0 when every claim matches in place; 1 with a list of problems otherwise.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import NamedTuple

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"

# Every artefact a README number is checked against. Missing => audit FAILS.
REQUIRED_ARTEFACTS = (
    "assay_summary.csv", "invariant_pooled.json", "depth.json",
    "esm_context.json", "heldout.json", "figure_stats.json", "alt_baseline.json",
    "esm_scale.json", "esm_tcrit.json", "prior_units.json", "reference.json",
    # Added with the round-2 holdout work: the README quotes numbers from both, so
    # losing either must fail the audit rather than silently shrink the checklist.
    "position_holdout.json", "monotone_fit.json",
    # The 200-seed re-run of the other-labs AUC verdict (DESIGN.md rule).
    "seed_sweep.json",
)

WINDOW = 90          # chars after an anchor in which its number must appear

# a signed number token: -0.94, +0.313, 0.00e+00, 813,014, 12.6%, 47/48
NUM = re.compile(r"(?<![\w.])([+\-\u2212]?\d[\d,]*(?:\.\d+)?(?:e[+\-]\d+)?%?"
                 r"(?:/\d+)?)")
VERDICT_WORD = re.compile(r"\b(NOT_SUPPORTED|SUPPORTED|INCONCLUSIVE|WITHDRAWN)\b")


class Claim(NamedTuple):
    label: str
    spellings: list
    source: str
    anchors: tuple = ()          # regexes (re.MULTILINE); empty = UNANCHORED
    kind: str = "number"         # "number" or "verdict"


class MissingArtefact(Exception):
    pass


def _norm(tok: str) -> str:
    return tok.replace("\u2212", "-").rstrip(",")


def _pct(x: float) -> list:
    """Prose may round a percentage to one decimal or to a whole number."""
    return [f"{x:.1f}%", f"{round(x)}%"]


def _signed(v: float, fmt: str) -> list:
    """A positive value may be written with or without '+'; a negative one only
    with its sign. Either way a flipped sign is a different token. A value that
    rounds to zero may carry either sign."""
    s = fmt.format(v)
    bare = s.lstrip("+-")
    if float(s) == 0:
        return [bare, "+" + bare, "-" + bare]
    if s.startswith("-"):
        return [s]
    return [bare, "+" + bare]


def cell(row_label: str, col: int, extra: str = "") -> str:
    """Anchor for the col-th value cell (1-based) of the table row whose first
    cell matches row_label. 'extra' lets an anchor continue inside the cell
    (e.g. up to the '(' of a parenthesised null)."""
    return (r"^\|\s*" + row_label + r"\s*\|" + r"[^|\n]*\|" * (col - 1)
            + r"\s*" + extra)


def first_token(text: str, start: int, cell_mode: bool, kind: str):
    region = text[start:start + WINDOW]
    if cell_mode:
        region = region.split("|", 1)[0]
    pat = VERDICT_WORD if kind == "verdict" else NUM
    m = pat.search(region)
    return _norm(m.group(1)) if m else None


def check_claim(readme: str, c: Claim) -> list:
    """Problems for one claim; an empty list means it holds in place."""
    spell = {_norm(s) for s in c.spellings}
    if not c.anchors:
        return [] if any(s in readme for s in c.spellings) else \
            [f"{c.label}: none of {c.spellings} appears anywhere (from {c.source})"]
    probs = []
    for a in c.anchors:
        hits = 0
        for m in re.finditer(a, readme, re.MULTILINE):
            hits += 1
            line = readme.count("\n", 0, m.start()) + 1
            cell_mode = m.group(0).rstrip().endswith("|") or a.startswith(r"^\|")
            tok = first_token(readme, m.end(), cell_mode, c.kind)
            if tok is None:
                probs.append(f"{c.label}: README line {line} has the anchor but no "
                             f"{'verdict' if c.kind == 'verdict' else 'number'} "
                             f"after it (expected {c.spellings[0]})")
            elif tok not in spell:
                probs.append(f"{c.label}: README line {line} says {tok}, the "
                             f"artefact says {c.spellings[0]} (from {c.source})")
        # EVERY anchor must still match: a listed place that vanished is a
        # sentence or row that was deleted or reworded, so its number is no
        # longer checked there - that must be visible, not a quiet pass.
        if hits == 0:
            probs.append(f"{c.label}: anchor /{a[:60]}/ matched nowhere - the "
                         f"sentence or row that carried {c.spellings[0]} is gone or "
                         f"reworded (from {c.source})")
    return probs


def audit(readme: str, claims: list, strict: bool = False) -> dict:
    """In-process audit, used by main() and by the mutation tests."""
    problems = []
    for c in claims:
        problems.extend(check_claim(readme, c))
    unanchored = [c.label for c in claims if not c.anchors]
    est = estimator_problems(readme)
    ok = not problems and not est and not (strict and unanchored)
    return {"ok": ok, "problems": problems, "estimator": est,
            "unanchored": unanchored}


def _load(res: Path, name: str):
    p = res / name
    if not p.exists():
        raise MissingArtefact(name)
    return pd.read_csv(p) if name.endswith(".csv") else json.loads(p.read_text())


def need(d, *keys, src: str = ""):
    """d[k1][k2]..., or MissingArtefact naming the key path: a claim whose number
    vanished from its artefact must fail, not quietly leave the checklist."""
    cur = d
    for k in keys:
        try:
            cur = cur[k]
        except (KeyError, IndexError, TypeError):
            raise MissingArtefact(f"{src} lacks {'.'.join(map(str, keys))}")
        if cur is None:
            raise MissingArtefact(f"{src} has null {'.'.join(map(str, keys))}")
    return cur


def build_claims(res: Path = RES) -> tuple:
    """(claims, missing). 'missing' names every required artefact or key that
    was absent; main() fails on it. Nothing here is optional any more: the old
    'if <artefact>.exists()' guards made the audit shrink instead of fail."""
    claims: list = []
    missing: list = []

    def add(label, spellings, source, *anchors, kind="number"):
        claims.append(Claim(label, list(spellings), source, tuple(anchors), kind))

    def block(name, fn):
        try:
            fn(_load(res, name))
        except MissingArtefact as e:
            missing.append(str(e))

    # --- assay_summary.csv --------------------------------------------------------
    def summary(s):
        big = s[s.n_pairs >= 500]
        src = "assay_summary.csv"
        add("assays", [str(len(s))], src + " rows",
            cell(r"assays with both singles and doubles", 1),
            r"total use all ", r"48 of the ")
        add("pairs", [f"{int(s.n_pairs.sum()):,}"], src + " n_pairs sum",
            cell(r"pairs with both singles measured", 1), r"use all 65 assays\*\* \(")
        # coverage = n_pairs / doubles measured, so the quotient is the integer
        # number of doubles each assay measured, with or without their singles.
        add("doubles measured", [f"{round(float((s.n_pairs / s.coverage).sum())):,}"],
            src + " sum of n_pairs / coverage",
            r"both singles and doubles, ")
        add("min coverage", [f"{s.coverage.min():.1%}"], src + " coverage min",
            cell(r"coverage of doubles by their singles", 1), r"covers only ")
        add("median coverage", [f"{s.coverage.median():.1%}"], src + " coverage median",
            cell(r"coverage of doubles by their singles", 1, r"[^|\n]*median "),
            r"median assay covers ")
        add("assays in every median", [str(len(big))], src + " rows >=500 pairs",
            r"\*\*On the ", r"fraction \u2014 uses the ", r"Every median above is over ")
        add("pairs in every median", [f"{int(big.n_pairs.sum()):,}"],
            src + " n_pairs sum >=500", r"at least 500 measured double mutants \(",
            r"at least 500\s+pairs\*\* \(")
        small = s[s.n_pairs < 500]
        add("assays outside the 48", [str(len(small))], src + " rows <500 pairs",
            r"\(817,120 eligible pairs\)\. The ", r"\*\*15 of the ", r"\*\*11 of the ")
        add("pairs outside the 48", [f"{int(small.n_pairs.sum()):,}"],
            src + " n_pairs sum <500", r"assays outside the 48 contribute ",
            r"dropped for having under 500 pairs \(")
        add("median spearman", [f"{big.spearman_additive.median():.3f}"],
            src + " spearman median, >=500 pairs (the stated denominator)",
            cell(r"median assay Spearman \(additive vs observed\)", 1))
        auc = f"{need(big, 'auc_direction', src=src).median():.3f}"
        add("median AUC", [auc], src + " auc_direction median >=500",
            cell(r"median AUC \(threshold-free direction\)", 1),
            r"ranks the double correctly \u2014 median\s+AUC ",
            r"a good \*ranker\* \(AUC ", r"the threshold-free AUC is ",
            r"good at is ranking:\s+\*\*AUC ", r"\(their interval, 0\.926 to\s+0\.956, excludes ",
            r"dropping Tsuboyama alone moves it from\s+", r"around the corpus ",
            r"Median AUC goes ", r"What survives is the ranking result, AUC ",
            r"the size of the gap \(AUC ")
        add("median agreement, own medians",
            [f"{big.direction_agreement_own_medians.median():.3f}"],
            src + " own-median agreement", r"at their own medians gives ")
        add("median agreement, observed-median split",
            [f"{big.direction_agreement.median():.3f}"], src + " direction_agreement",
            r"in the right direction\"\. That\s+")
        b2 = big.frac_outside_2sd.median()
        add("per-assay median >2sd, median centring", [f"{b2:.1%}"],
            src + " frac_outside_2sd median",
            cell(r"pairs beyond 2 sd from additive, per-assay \*\*median\*\*, median "
                 r"centring", 1),
            cell(r"per-assay \*\*median\*\* share beyond 2 sd, median centring", 1),
            r"figure is the centred one, \*\*4\.6%\*\*; ",
            r"the beyond-2-sd share from ")
        b2c = big.frac_outside_2sd_centred.median()
        add("per-assay median >2sd, centred", [f"{b2c:.1%}"],
            src + " frac_outside_2sd_centred median",
            cell(r"the same, each assay's own offset removed \(convention-free\)", 1),
            cell(r"per-assay median after centring each assay", 1),
            r"The convention-free figure is the centred one, ",
            r"the beyond-2-sd share from 12\.6% to\s+", r"the convention-free share is ",
            r"the size of the gap \(AUC 0\.884 ranking; ")
        add("median within 1sd fraction", [f"{big.frac_within_1sd.median():.3f}"],
            src + " frac_within_1sd median >=500",
            cell(r"median assay, fraction of pairs within 1 sd, median centring", 1))
        # The uncentred share above moves with the centring constant; the centred
        # one is the convention-free figure, so the README quotes both.
        add("median within 1sd fraction, centred",
            [f"{need(big, 'frac_within_1sd_centred', src=src).median():.3f}"],
            src + " frac_within_1sd_centred median >=500",
            cell(r"within 1 sd, each assay's own offset removed "
                 r"\(convention-free\)", 1))
        add("median residual bias (median centring)",
            _signed(big.bias_mean_sd.median(), "{:.2f}"), src + " bias_mean_sd median",
            r"the residual\s+cloud sits a median ",
            r"under that convention the median bias over the 48\s+assays is ",
            r"changes sign\" \(\+0\.50 against ")
        add("withdrawn headline bias, magnitude",
            [f"{abs(big.bias_mean_sd.median()):.2f}"], src + " |bias_mean_sd median|",
            r"biased by a\s+median ", r"\"biased by\s+", r"Anyone quoting \"",
            r"quote the AUC and not the\s+")
        n2 = int(big.n_outside_2sd.sum())
        pooled = n2 / big.n_pairs.sum()
        add("pooled >2sd fraction", [f"{pooled:.1%}"], src + " n_outside_2sd / n_pairs",
            cell(r"pooled share beyond 2 sd \u2014 \*\*what this README used to call "
                 r"40\.8%\*\*", 1))
        add("pooled >2sd count", [f"{n2:,}"], src + " n_outside_2sd sum",
            cell(r"pooled share beyond 2 sd \u2014 \*\*what this README used to call "
                 r"40\.8%\*\*", 1, r"[^|(\n]*\("))
        # the denominator of the same cell: a mutation sweep showed 813,014 could be
        # changed here and the audit still passed (it was anchored only elsewhere)
        add("pooled >2sd denominator", [f"{int(big.n_pairs.sum()):,}"],
            src + " n_pairs sum, >=500 pairs",
            cell(r"pooled share beyond 2 sd — \*\*what this README used to call "
                 r"40\.8%\*\*", 1, r"[^|(\n]*\([\d,]+ / "))
        pooled_c = big.n_outside_2sd_centred.sum() / big.n_pairs.sum()
        add("pooled >2sd fraction, centred", [f"{pooled_c:.1%}"],
            src + " n_outside_2sd_centred / n_pairs",
            cell(r"pooled share beyond 2 sd, centred", 1))
        add("largest assay, pairs", [f"{int(s.n_pairs.max()):,}"], src + " n_pairs max",
            r"SPG1_STRSG_Olson_2014, holds ")
    block("assay_summary.csv", summary)

    # --- invariant_pooled.json -------------------------------------------------------
    def invariant(i):
        src = "invariant_pooled.json"
        mp = need(i, "magnitude_prior_verdict", src=src)
        # The old audit also 'checked' mp['null_shuffle_observed_rebuilt'] - a key
        # this artefact never had, so the claim was silently skipped on every run.
        # The README's null (0.502, 144 permutations) is depth.json's and is checked
        # there; this file's 3-permutation null (0.500) is not quoted.
        add("magnitude prior observed (centred)",
            _signed(need(mp, "observed_centred_mean", src=src), "{:+.3f}"),
            src + " observed_centred_mean",
            cell(r"observed rho\(\\\|residual\\\|, \\\|prediction\\\|\), both centred", 1))
        add("magnitude prior exceeds null", [need(mp, "assays_observed_exceeds_null",
            src=src)], src + " assays_observed_exceeds_null",
            cell(r"assays where observed exceeds the null", 1))
        add("AUC (invariant_pooled)", [f"{need(i, 'auc_median', src=src):.3f}"],
            src + " auc_median", cell(r"median AUC \(threshold-free direction\)", 1))
        # Quoted in the text as the robustness check on the centring convention.
        add("magnitude prior median-centring rho",
            _signed(need(i, "rho_centred_by_median_median", src=src), "{:+.3f}"),
            src + " rho_centred_by_median_median",
            r"Median-centring instead of mean-centring gives ")
        add("beyond 2sd centred (invariant_pooled)",
            [f"{100 * need(i, 'frac_beyond_2sd_centred_median_of_assays', src=src):.1f}%"],
            src + " frac_beyond_2sd_centred_median_of_assays",
            cell(r"per-assay median after centring each assay", 1))
    block("invariant_pooled.json", invariant)

    # --- depth.json --------------------------------------------------------------------
    def depth(dj):
        src = "depth.json"
        h2 = need(dj, "H2_position", src=src)
        add("position r2", [f"{need(h2, 'observed_r2_median', src=src):.3f}"],
            src + " H2 observed_r2_median",
            r"which sequence positions are involved is ", r"rather than the raw ")
        add("position null", [f"{need(h2, 'permutation_null_median', src=src):.3f}"],
            src + " H2 permutation_null_median", r"against a permutation null of ")
        add("position positions per assay",
            [f"{need(h2, 'positions_per_assay_median', src=src):.0f}"],
            src + " H2 positions_per_assay_median", r"The median assay has only ")
        # F11 (review round 2): the README attributed the GROUPING count to the
        # assay. Both numbers are now published and both are audited, so neither can
        # masquerade as the other and the union figure cannot go stale silently.
        add("position union median",
            [f"{need(h2, 'positions_union_median', src=src):.0f}"],
            src + " H2 positions_union_median", r"the median is ")
        add("position union min",
            [f"{need(h2, 'positions_union_min', src=src):.0f}"],
            src + " H2 positions_union_min", r"\(min ")
        add("position union max",
            [f"{need(h2, 'positions_union_max', src=src):.0f}"],
            src + " H2 positions_union_max", r"max ")
        # "no assay has only 2" is IMPLIED by the audited min of 3, so it needs no
        # separate claim: a minimum of 3 cannot coexist with a count of 2. Claiming
        # the word "no" directly was wrong - the audit checks numbers, and a
        # numeric claim whose anchor is followed by a word fails - which is how this
        # entry was caught.
        add("position hot-decile coverage",
            [f"{need(h2, 'hot_decile_position_pair_coverage_median', src=src):.0%}"],
            src + " H2 hot decile coverage", r"touches a median of ")
        add("single-mutation r2", [f"{need(h2, 'single_mutation_r2_median_slot_a', src=src):.3f}"],
            src + " H2 single_mutation_r2_median_slot_a",
            r"instead of by position gives R\u00b2\s+")
        add("position r2 demeaned", [f"{need(h2, 'observed_r2_demeaned_median', src=src):.3f}"],
            src + " H2 observed_r2_demeaned_median", r"It shrinks to ")
        # The R2-by-group-count figures were quoted from a table that had no claim
        # behind it, so a stale bin could ship while the audit stayed green.
        add("position r2 groups 3-5",
            [f"{need(h2, 'r2_by_position_count', '(2,5]', 'r2_median', src=src):.3f}"],
            src + " H2 r2_by_position_count (2,5]",
            r"median\s+R\u00b2 is ")
        add("position r2 groups 21-60",
            [f"{need(h2, 'r2_by_position_count', '(20,60]', 'r2_median', src=src):.3f}"],
            src + " H2 r2_by_position_count (20,60]", r"groups and ")
        h1 = need(dj, "H1_residue", src=src)
        top = need(h1, "top", src=src)
        k0 = sorted(top)[0]
        add("residue top fraction", [f"{need(top, k0, 'frac_above_assay_median', src=src):.0%}"],
            src + " H1 top frac", r"median residual in\s+\*\*")
        add("residue rank reliability",
            _signed(need(h1, "split_half_rank_reliability", src=src), "{:.2f}"),
            src + " H1 split-half", r"independent assay halves at only\s+\*\*\u03c1 ",
            r"rank correlation across assay halves ")
        add("residue A after confound removal",
            [f"{need(h1, 'A_frac_above_after_magnitude_removal', src=src):.2f}"],
            src + " H1 confound check", r"alanine's fraction drops to ")
        add("residue verdict", [need(h1, "verdict", src=src)], src + " H1 verdict",
            r"Both were wrong\. Verdict ", r"\*\*Residue identity: ", kind="verdict")
        h3 = need(dj, "H3_magnitude_prior_centred", src=src)
        add("magnitude prior null (depth)",
            [f"{need(h3, 'permutation_null_centred_median', src=src):.3f}"],
            src + " H3 permutation null",
            cell(r"null: within-assay permutation of \x60observed\x60, residual rebuilt", 1))
        add("magnitude prior centred rho (depth)",
            _signed(need(h3, "rho_centred_median", src=src), "{:+.3f}"),
            src + " H3 centred rho",
            cell(r"observed rho\(\\\|residual\\\|, \\\|prediction\\\|\), both centred", 1))
        # Numbers the page quoted with no claim behind them until a mutation
        # sweep of every numeric token showed the audit never looked at them.
        h3x = need(dj, "H3_magnitude_prior_centred", src=src)
        add("magnitude prior observed minus null",
            _signed(need(h3x, "observed_centred_minus_null", src=src), "{:.3f}"),
            src + " H3 observed_centred_minus_null", cell(r"observed minus null", 1))
        bias_bins = need(h3x, "rho_by_bias_bin", src=src)
        add("raw rho, assays offset 0.75-1.5 sd",
            _signed(need(bias_bins, "(0.75, 1.5]", "rho_repo_median", src=src), "{:.3f}"),
            src + " H3 rho_by_bias_bin (0.75, 1.5]", r"zero was placed — it is\s+")
        add("raw rho, assays offset under 0.25 sd",
            _signed(need(bias_bins, "(-1.0, 0.25]", "rho_repo_median", src=src), "{:+.3f}"),
            src + " H3 rho_by_bias_bin (-1.0, 0.25]", r"1\.5 sd and ")
        h2x = need(dj, "H2_position", src=src)
        add("position contrast, assays with 2 groups",
            [str(need(h2x, "assays_with_only_2_positions", src=src))],
            src + " H2 assays_with_only_2_positions", r"groups in this contrast \(")
        cov = need(h2x, "hot_decile_position_pair_coverage_range", src=src)
        add("hot-decile coverage, lowest assay", [f"{cov[0]:.0%}"],
            src + " H2 hot decile coverage range[0]", r"of pairs \(range\s+")
        add("hot-decile coverage, highest assay", [f"{cov[1]:.0%}"],
            src + " H2 hot decile coverage range[1]", r"of pairs \(range\s+\d+%–")
        add("alanine Bonferroni p",
            [f"{need(dj, 'H1_residue', 'top', 'A', 'p_bonferroni', src=src):.3f}"],
            src + " H1 top.A.p_bonferroni", r"Bonferroni p = ")
    block("depth.json", depth)

    # --- esm_context.json (the 4.4% probe) -----------------------------------------------
    def esm_context(e):
        src = "esm_context.json"
        add("esm rho", _signed(need(e, "spearman_mean", src=src), "{:+.3f}"),
            src + " spearman_mean", r"sequence separation: \u03c1 = ")
        add("esm spearman sd", [f"{need(e, 'spearman_sd', src=src):.3f}"],
            src + " spearman_sd", r"\u03c1 = \*\*-?[\d.]+ \u00b1 ")
        ctl = need(e, "controls", src=src)
        add("esm noise floor", _signed(need(ctl, "noise", "spearman_mean", src=src),
            "{:+.3f}"), src + " controls.noise", r"pure-noise control of ")
        add("esm prior-only rho", _signed(need(ctl, "prior_only", "spearman_mean",
            src=src), "{:+.3f}"), src + " controls.prior_only",
            r"worse than the prior alone\s+\(")
        add("esm incremental value", [f"{abs(need(e, 'incremental_value_of_esm_over_prior', src=src)):.3f}"],
            src + " incremental value", r"\(\+0\.231\)\*\* by ")
        sel = need(e, "selection", src=src)
        add("esm pairs used", [f"{int(need(sel, 'pairs_used', src=src)):,}"],
            src + " selection.pairs_used", r"The probe above used ")
        add("esm fraction", [f"{100 * need(sel, 'fraction_of_available', src=src):.1f}%"],
            src + " selection.fraction_of_available", r"The probe above used 36,000 pairs \(")
        add("esm noise floor sd",
            [f"{need(e, 'controls', 'noise', 'spearman_sd', src=src):.3f}"],
            src + " controls.noise.spearman_sd",
            r"pure-noise control of \+\d\.\d+ ± ")
        add("esm plus prior rho",
            _signed(need(e, "controls", "esm_plus_prior", "spearman_mean", src=src),
                    "{:.3f}"),
            src + " controls.esm_plus_prior.spearman_mean",
            r"the free prior\*\*: ρ = ")
    block("esm_context.json", esm_context)

    # --- heldout.json ------------------------------------------------------------------
    def heldout(h):
        src = "heldout.json"
        h1 = need(h, "H1_lab_family", src=src)
        ts, ot = need(h1, "Tsuboyama", src=src), need(h1, "other_labs", src=src)
        add("heldout n Tsuboyama in corpus", [str(need(ts, "n_assays", src=src))],
            src + " H1 Tsuboyama n_assays", r"that split is ", r"the same lab as ")
        add("heldout n other labs in corpus", [str(need(ot, "n_assays", src=src))],
            src + " H1 other_labs n_assays", r"that split is 33 against ",
            r"for the 33 Tsuboyama assays\s+against \*\*0\.941\*\* for the ")
        add("heldout bias Tsuboyama (median centring)",
            _signed(need(ts, "bias_sd", "median", src=src), "{:.2f}"),
            src + " H1 Tsuboyama bias_sd median", r"bias medians also\s+differ, ")
        add("heldout bias other labs (median centring)",
            _signed(need(ot, "bias_sd", "median", src=src), "{:.2f}"),
            src + " H1 other_labs bias_sd median", r"differ, \*\*-0\.79\*\* against ")
        add("heldout auc Tsuboyama", [f"{need(ts, 'auc', 'median', src=src):.3f}"],
            src + " H1 Tsuboyama auc", r"but not at one level: AUC ")
        add("heldout auc other labs", [f"{need(ot, 'auc', 'median', src=src):.3f}"],
            src + " H1 other_labs auc", r"for the 33 Tsuboyama assays\s+against ")
        oci = need(ot, "auc", "ci95", src=src)
        add("heldout auc other labs CI low", [f"{oci[0]:.3f}"], src + " other_labs auc ci95",
            r"\(their interval, ")
        add("heldout auc other labs CI high", [f"{oci[1]:.3f}"], src + " other_labs auc ci95",
            r"\(their interval, 0\.926 to\s+")
        add("heldout other labs AUC verdict",
            [need(ot, "verdict_vs_corpus", "auc", src=src)], src + " other_labs verdict",
            r"labs' AUC against the corpus median is ", kind="verdict")
        d = need(h1, "difference_Tsuboyama_minus_other", "bias_sd", src=src)
        add("heldout bias difference p", [f"{need(d, 'p_two_sided', src=src):.4f}"],
            src + " H1 bootstrap p", r"\(bootstrap p ")
        h2 = need(h, "H2_excluded_small_assays", src=src)
        add("heldout excluded n same lab", [str(need(h2, "n_same_lab_as_corpus", src=src))],
            src + " H2 n_same_lab_as_corpus", r"the three 48-assay medians\. But \*\*")
        add("heldout excluded n in ESM probe",
            [str(need(h2, "in_esm_scale_probe", "n_assays", src=src))],
            src + " H2 in_esm_scale_probe", r"of the corpus assays, and \*\*")
        add("heldout excluded bias (withdrawn, quoted)",
            _signed(need(h2, "bias_sd", "median", src=src), "{:+.2f}"),
            src + " H2 bias_sd median", r"and even changes sign\" \(")
        add("heldout excluded auc", [f"{need(h2, 'auc', 'median', src=src):.3f}"],
            src + " H2 auc median", r"Their ranking agrees with\s+the corpus \(\*\*AUC ")
        eci = need(h2, "auc", "ci95", src=src)
        add("heldout excluded auc CI low", [f"{eci[0]:.3f}"], src + " H2 auc ci95",
            r"0\.848\*\*, interval ")
        add("heldout excluded auc CI high", [f"{eci[1]:.3f}"], src + " H2 auc ci95",
            r"0\.848\*\*, interval 0\.755 to ")
        add("heldout excluded auc verdict",
            [need(h2, "verdict_vs_corpus", "auc", src=src)], src + " H2 verdict",
            r"around the corpus 0\.884;\s+verdict ", kind="verdict")
        h3 = need(h, "H3_leave_one_lab_out", src=src)
        rng = need(h3, "auc_range", src=src)
        add("loo auc range low", [f"{rng[0]:.3f}"], src + " H3 auc_range",
            r"moves the median AUC between ")
        add("loo auc range high", [f"{rng[1]:.3f}"], src + " H3 auc_range",
            r"moves the median AUC between 0\.880 and ",
            r"dropping Tsuboyama alone moves it from\s+0\.884 to ")
        add("loo verdict", [need(h3, "verdict", src=src)], src + " H3 verdict",
            r"artefact's verdict is ", kind="verdict")
    block("heldout.json", heldout)

    # --- figure_stats.json --------------------------------------------------------------
    def figstats(f):
        src = "figure_stats.json"
        b = need(f, "bias_vs_tail_fraction", src=src)
        add("tail vs |bias| r", _signed(need(b, "pearson_abs_bias", src=src), "{:+.2f}"),
            src + " pearson_abs_bias", r"the share past 2 sd correlates ")
        # The "tail vs spread r" claim (pearson_residual_sd) is withdrawn: its x
        # variable is 1 for every assay by construction. The figure and the
        # artefact no longer carry it, so there is nothing left to check.
    block("figure_stats.json", figstats)

    # --- alt_baseline.json: the table, cell by cell ---------------------------------------
    def alt(a):
        src = "alt_baseline.json"
        rows = {"additive": r"plain additivity \(recomputed\)",
                "additive+pooled_offset": r"\+ one offset learned on the other assays",
                "additive+assay_offset": r"\+ the assay's own offset",
                "additive+position": r"\+ position",
                "additive+position+pair_rank3": r"\+ position \+ position-pair rank 3"}
        models = {str(m.get("model")): m for m in need(a, "models", src=src)}
        for name, row in rows.items():
            m = need(models, name, src=src)
            med = need(m, "median", src=src)
            r2 = need(med, "r2_resid_var", src=src)
            r2s = ["0", "0.000", "-0.000"] if round(r2, 3) == 0 else [f"{r2:.3f}"]
            add(f"alt {name} r2", r2s, f"{src} {name} median.r2_resid_var", cell(row, 1))
            add(f"alt {name} auc", [f"{need(med, 'auc_direction', src=src):.3f}"],
                f"{src} {name} median.auc_direction", cell(row, 2))
            add(f"alt {name} beyond2", [f"{100 * need(med, 'frac_outside_2sd', src=src):.1f}%"],
                f"{src} {name} median.frac_outside_2sd", cell(row, 3))
            bias = need(med, "bias_mean_sd", src=src)
            add(f"alt {name} bias", ["0.00", "-0.00", "+0.00"] if round(bias, 2) == 0
                else _signed(bias, "{:.2f}"), f"{src} {name} median.bias_mean_sd",
                cell(row, 4))
            if name in ("additive+position", "additive+position+pair_rank3"):
                nul = need(m, "null_median", src=src)
                add(f"alt {name} r2 null", [f"{need(nul, 'r2_resid_var', src=src):.3f}"],
                    f"{src} {name} null r2", cell(row, 1, r"[^|(\n]*\("))
                add(f"alt {name} auc null", [f"{need(nul, 'auc_direction', src=src):.3f}"],
                    f"{src} {name} null auc", cell(row, 2, r"[^|(\n]*\("))
                add(f"alt {name} beyond2 null",
                    [f"{100 * need(nul, 'frac_outside_2sd', src=src):.1f}%"],
                    f"{src} {name} null beyond2", cell(row, 3, r"[^|(\n]*\("))
        pos, best = models["additive+position"], models["additive+position+pair_rank3"]
        add("alt best r2 (prose)", [f"{best['median']['r2_resid_var']:.3f}"],
            src + " pair_rank3 r2", r"The best model removes a\s+median ",
            r"They remove a median ")
        add("alt position r2 (prose)", [f"{pos['median']['r2_resid_var']:.3f}"],
            src + " position r2", r"\(position alone ")
        add("alt pair beats null", [need(best, "assays_real_beats_own_null", "r2_resid_var",
            src=src)], src + " pair_rank3 beats null", r"model beats its own null in ")
        add("alt position beats null", [need(pos, "assays_real_beats_own_null",
            "r2_resid_var", src=src)], src + " position beats null",
            r"\(the position\s+term alone in ")
        add("alt pair auc (prose)", [f"{best['median']['auc_direction']:.3f}"],
            src + " pair_rank3 auc", r"Median AUC goes 0\.884 \u2192 ")
        add("alt pair auc wins", [need(best, "assays_auc_above_additive", src=src)],
            src + " pair_rank3 auc wins", r"the pair model wins in only ")
        rk = str(need(a, "verdict", "gap_survives", "ranking", src=src))
        gain = re.search(r"median gain ([+\-][0-9.]+)", rk)
        if gain is None:
            raise MissingArtefact(src + " verdict.gap_survives.ranking lacks the gain")
        add("alt pair median auc gain", [gain.group(1)], src + " gap_survives.ranking",
            r"a median gain of\s+")
        own = models["additive+assay_offset"]
        add("alt own offset beyond2 (prose)",
            [f"{100 * own['median']['frac_outside_2sd']:.1f}%"],
            src + " assay_offset beyond2", r"the beyond-2-sd share from 12\.6% to\s+")
    block("alt_baseline.json", alt)

    # --- esm_scale.json -------------------------------------------------------------
    def esm_scale(e):
        src = "esm_scale.json"
        sel = need(e, "primary", "selection", src=src)
        add("esm_scale pairs", [f"{int(need(sel, 'pairs_used', src=src)):,}"],
            src + " primary.selection.pairs_used", r"covers \*\*59 assays and ")
        frac = f"{100 * need(sel, 'fraction_of_eligible_pairs', src=src):.1f}%"
        add("esm_scale corpus fraction", [frac], src + " fraction_of_eligible_pairs",
            r"assays and 319,014 pairs \u2014 ", r"esm_scale\.py\x60 covers ",
            r"^\| head \| ", r"^\| head \|[^|\n]*\|\s*")
        add("esm_scale assays", [str(need(sel, "assays_used", src=src))],
            src + " assays_used", r"\(\x60scripts/esm_scale\.py\x60\) covers \*\*",
            r"corpus-scale ESM probe uses ", r"are among the ")
        rule = str(need(sel, "rule", src=src))
        mp = re.search(r">= (\d+) pairs", rule)
        if mp is None:
            raise MissingArtefact(src + " selection.rule lacks the pair minimum")
        add("esm_scale pair minimum", [mp.group(1)], src + " selection.rule",
            r"those\s+with at least ")
        add("esm_scale eligible pairs", [f"{int(need(sel, 'pairs_eligible', src=src)):,}"],
            src + " selection.pairs_eligible", r"parent sequence \(",
            r"a \*\*100%\*\* run \(")
        heads = need(e, "primary", "heads", src=src)
        anchors = {"a": (r"Head \(a\) scores ", r"against a shuffled control of "),
                   "b": (r"; \(b\) ", r"\(b\) -?[\d.]+ against\s+"),
                   "c": (r"; \(c\) ", r"\(c\) [+\-]?[\d.]+ against\s+")}
        for hid, (ah, ac) in anchors.items():
            vd = need(heads, hid, "verdict", src=src)
            add(f"esm_scale head {hid} rho", _signed(need(vd, "head_mean", src=src),
                "{:+.3f}"), f"{src} heads.{hid}.head_mean", ah)
            add(f"esm_scale head {hid} control", _signed(need(vd, "control_mean",
                src=src), "{:+.3f}"), f"{src} heads.{hid}.control_mean", ac)
        add("esm_scale incremental over prior",
            _signed(need(e, "headline", "incremental_rho_of_esm_over_prior", src=src),
                    "{:+.3f}"), src + " headline incremental rho",
            r"gain of ESM over the free prior is\s+\*\*")
        inc = need(e, "primary", "heads", "c", "incremental_rho_over_prior", src=src)
        add("esm_scale incremental over prior, SE",
            [f"{need(inc, 'esm', 'se', src=src):.3f}"],
            src + " heads.c.incremental_rho_over_prior.esm.se",
            r"free prior is\s+\*\*\+?\d\.\d+ ± ")
        add("esm_scale shuffled incremental",
            _signed(need(inc, "shuffled", "mean", src=src), "{:+.3f}"),
            src + " heads.c.incremental_rho_over_prior.shuffled.mean",
            r"within an assay gives\s+\*\*")
        add("esm_scale shuffled incremental, SE",
            [f"{need(inc, 'shuffled', 'se', src=src):.3f}"],
            src + " heads.c.incremental_rho_over_prior.shuffled.se",
            r"within an assay gives\s+\*\*\+?\d\.\d+ ± ")
    block("esm_scale.json", esm_scale)

    # --- esm_tcrit.json -------------------------------------------------------------
    def tcrit(t):
        src = "esm_tcrit.json"
        add("esm 2 SE interval coverage",
            [f"{100 * need(t, 'coverage_of_2se_interval', src=src):.1f}%"],
            src + " coverage_of_2se_interval", r"over five folds is an \*\*")
        add("esm t critical value", [f"{need(t, 't_crit_0975', src=src):.3f}"],
            src + " t_crit_0975", r"t\(4\) critical value of \*\*")
    block("esm_tcrit.json", tcrit)

    # --- prior_units.json -----------------------------------------------------------
    def prior_units(u):
        src = "prior_units.json"
        add("prior units within raw", _signed(need(u, "within_assay_raw_units",
            "median_rho", src=src), "{:+.3f}"), src + " within raw",
            r"rank correlation is\s+\*\*")
        add("prior units within sd-scaled", _signed(need(u, "within_assay_sd_scaled",
            "median_rho", src=src), "{:+.3f}"), src + " within sd-scaled",
            r"in raw units and \*\*(?=[^*\n]*\*\* after dividing)")
        add("prior units pooled raw", _signed(need(u, "pooled_across_assays_raw_units",
            src=src), "{:+.3f}"), src + " pooled raw", r"the same quantity reads \*\*")
        add("prior units pooled sd-scaled", _signed(need(u,
            "pooled_across_assays_sd_scaled", src=src), "{:+.3f}"),
            src + " pooled sd-scaled",
            r"in raw units and \*\*(?=[^*\n]*\*\* once each assay)")
        add("prior units max deviation",
            [f"{need(u, 'max_abs_deviation_within_assay', src=src):.2e}"],
            src + " max_abs_deviation_within_assay", r"maximum difference of \*\*")
    block("prior_units.json", prior_units)

    # --- reference.json: both centring conventions ------------------------------------
    def reference(r):
        src = "reference.json"
        n_anch = need(r, "n_assays_anchored_at_zero", src=src)
        add("reference anchored assays", [str(n_anch)], src + " n_assays_anchored_at_zero",
            r"For the ", r"read \*\*-?[\d.]+\*\* sd under the median centring \(positive "
            r"in \*\*\d+/")
        mc = need(r, "median_bias_under_median_centring", src=src)
        w0 = need(r, "median_bias_under_wt0_for_anchored_assays", src=src)
        add("reference anchored bias, median centring",
            _signed(need(mc, "anchored_at_zero_assays", src=src), "{:.2f}"),
            src + " anchored, median centring", r"assays scored as a \u0394\u0394G from "
            r"wild type\s+it reads ", r"the same 33 assays\s+read \*\*")
        add("reference anchored bias, wild type = 0",
            _signed(need(w0, "anchored_at_zero_assays", src=src), "{:+.2f}"),
            src + " anchored, wt0", r"centring this repo used and ",
            r"\(positive in \*\*\d+/\d+\*\*\) against\s+\*\*",
            r"the Tsuboyama median\s+becomes ")
        add("reference all-48 bias, median centring",
            _signed(need(mc, "all_assays", src=src), "{:.2f}"), src + " all, median",
            r"under that convention the median bias over the 48\s+assays is ")
        add("reference anchored positive, median centring",
            [f"{need(r, 'anchored_positive_under_median_centring', src=src)}/{n_anch}"],
            src + " anchored_positive_under_median_centring",
            r"under the median centring \(positive in ")
        add("reference anchored positive, wild type = 0",
            [f"{need(r, 'anchored_positive_under_wt0', src=src)}/{n_anch}"],
            src + " anchored_positive_under_wt0", r"wild type = 0, positive in ",
            r"wild type = 0 \(positive in ")
        add("reference >2sd, zero as wild type",
            [f"{100 * need(r, 'median_frac_outside_2sd_wt0', src=src):.1f}%"],
            src + " median_frac_outside_2sd_wt0",
            cell(r"per-assay median share beyond 2 sd, zero taken as wild type in every "
                 r"assay \(exact for the 33 \u0394\u0394G assays\)", 1),
            r"12\.6% and ")
        add("reference two-number recalibration",
            [f"{need(r, 'linear_recalibration_two_numbers_median_r2', src=src):.3f}"],
            src + " linear_recalibration_two_numbers_median_r2",
            r"scored on the same metric \u2014 removes a median ",
            r"with no\s+positions removes ")
    block("reference.json", reference)

    # --- position_holdout.json: whole-position-pair holdout ------------------------
    def position_holdout(h):
        src = "position_holdout.json"
        pa = need(h, "position_pairs_per_assay", src=src)
        add("holdout median position pairs", [f"{need(pa, 'median', src=src):.1f}"],
            src + " position_pairs_per_assay.median",
            r"assays have a median of\s+\*\*")
        bins = need(pa, "assays_per_bin", src=src)
        add("holdout assays in bin 2", [str(need(bins, "2", src=src))],
            src + " assays_per_bin[2]",
            r"position pairs, and the split is bimodal \u2014 \*\*")
        add("holdout assays in bin 3-9", [str(need(bins, "3-9", src=src))],
            src + " assays_per_bin[3-9]",
            r"\*\* assays hold only 2\s+position pairs, \*\*")
        add("holdout assays in bin >=10", [str(need(bins, ">=10", src=src))],
            src + " assays_per_bin[>=10]",
            r"\*\* hold 3\u20139, and \*\*")
        sp = need(h, "summary", "position", src=src)
        allb = need(sp, "all", src=src)
        add("holdout position r2, random split",
            [f"{need(allb, 'random_pair', 'median_r2', src=src):.3f}"],
            src + " position.all.random_pair.median_r2",
            r"assays position R\u00b2 falls from \*\*")
        add("holdout position r2, whole-pair split",
            _signed(need(allb, "whole_position_pair", "median_r2", src=src),
                    "{:.3f}"),
            src + " position.all.whole_position_pair.median_r2",
            r"R\u00b2 falls from \*\*[\d.]+\*\* to \*\*")
        add("holdout assays above null, all",
            [str(need(allb, "whole_position_pair", "assays_above_null", src=src))],
            src + " position.all.whole_position_pair.assays_above_null",
            r"to \*\*-?[\d.]+\*\* \(")
        ten = need(sp, ">=10", src=src)
        add("holdout position keep share >=10",
            [f"{round(100 * need(h, 'verdict', 'position', '>=10', 'ratio', src=src)):.0f}%"],
            src + " verdict.position.>=10.ratio",
            r"training: there position keeps \*\*")
        add("holdout position r2 >=10 random",
            [f"{need(ten, 'random_pair', 'median_r2', src=src):.3f}"],
            src + " position.>=10.random_pair.median_r2",
            r"keeps \*\*\d+%\*\* of its R\u00b2, scoring \*\*")
        add("holdout position r2 >=10 whole",
            [f"{need(ten, 'whole_position_pair', 'median_r2', src=src):.3f}"],
            src + " position.>=10.whole_position_pair.median_r2",
            r"on the\s+random split and \*\*")
        r3 = need(h, "summary", "pair_rank3", ">=10", src=src)
        add("holdout rank3 keep share >=10",
            [f"{round(100 * need(h, 'verdict', 'pair_rank3', '>=10', 'ratio', src=src)):.0f}%"],
            src + " verdict.pair_rank3.>=10.ratio",
            r"split; pair-rank-3 keeps \*\*")
        add("holdout rank3 r2 >=10 random",
            [f"{need(r3, 'random_pair', 'median_r2', src=src):.3f}"],
            src + " pair_rank3.>=10.random_pair.median_r2",
            r"pair-rank-3 keeps \*\*\d+%\*\*,\s+scoring \*\*")
        add("holdout rank3 r2 >=10 whole",
            [f"{need(r3, 'whole_position_pair', 'median_r2', src=src):.3f}"],
            src + " pair_rank3.>=10.whole_position_pair.median_r2",
            r"scoring \*\*\d+\.\d+\*\* and \*\*")
        r3all = need(h, "summary", "pair_rank3", "all", src=src)
        add("holdout rank3 r2, random split, all",
            [f"{need(r3all, 'random_pair', 'median_r2', src=src):.3f}"],
            src + " pair_rank3.all.random_pair.median_r2", r"falls just as far, ")
        add("holdout rank3 r2, whole-pair split, all",
            _signed(need(r3all, "whole_position_pair", "median_r2", src=src), "{:.3f}"),
            src + " pair_rank3.all.whole_position_pair.median_r2",
            r"falls just as far, \d\.\d+ to ")
        # one sentence covers both models, so both must carry the same count
        for mdl in ("position", "pair_rank3"):
            add(f"holdout {mdl} above null, >=10",
                [str(need(h, "summary", mdl, ">=10", "whole_position_pair",
                          "assays_above_null", src=src))],
                f"{src} {mdl}.>=10.whole_position_pair.assays_above_null",
                r"Both beat their null in \*\*")
        pdx = need(h, "summary", "position", ">=10", "paired_whole_minus_random", src=src)
        pci = need(pdx, "ci95_bootstrap", src=src)
        add("holdout paired loss >=10", [f"{abs(need(pdx, 'median', src=src)):.3f}"],
            src + " position.>=10.paired_whole_minus_random.median",
            r"position R²\s+drops by ")
        add("holdout paired loss >=10, CI near bound", [f"{abs(pci[1]):.3f}"],
            src + " position.>=10.paired_whole_minus_random.ci95[1]",
            r"drops by \d\.\d+ \[")
        add("holdout paired loss >=10, CI far bound", [f"{abs(pci[0]):.3f}"],
            src + " position.>=10.paired_whole_minus_random.ci95[0]",
            r"drops by \d\.\d+ \[\d\.\d+, ")
        add("holdout paired loss >=10, assays lower",
            [str(need(pdx, "assays_whole_lower", src=src))],
            src + " position.>=10.paired_whole_minus_random.assays_whole_lower",
            r"\], lower in ")
        seen = [p["seen"]["both"] for p in need(h, "per_assay", src=src)
                if p["n_cells"] >= 10]
        add("holdout >=10, assays with every test pair seen",
            [str(sum(v == 1.0 for v in seen))],
            src + " per_assay seen.both == 1, n_cells >= 10", r"every test pair in ")
        add("holdout >=10, assays with >=99.9% of test pairs seen",
            [str(sum(v >= 0.999 for v in seen))],
            src + " per_assay seen.both >= 0.999, n_cells >= 10", r"test pairs in ")
        add("holdout >=10, lowest share of test pairs seen", [f"{min(seen):.0%}"],
            src + " per_assay min seen.both, n_cells >= 10",
            r"test pairs in \d+, and ")
    block("position_holdout.json", position_holdout)

    # --- monotone_fit.json: is "position" a calibration curve? ---------------------
    def monotone_fit(m):
        src = "monotone_fit.json"
        models = need(m, "models", src=src)
        by_name = {r["model"]: r for r in models}
        add("monotone isotonic r2",
            [f"{need(by_name['isotonic'], 'median_r2', src=src):.3f}"],
            src + " models[isotonic].median_r2",
            r"transform of the additive prediction alone\s+removes \*\*")
        add("position r2 (monotone artefact)",
            [f"{need(by_name['position'], 'median_r2', src=src):.3f}"],
            src + " models[position].median_r2",
            r"residual variance against position's \*\*")
        add("position+isotonic r2",
            [f"{need(by_name['position+isotonic'], 'median_r2', src=src):.3f}"],
            src + " models[position+isotonic].median_r2",
            r"`position \+ isotonic` reaches \*\*")
        paired = need(m, "paired", src=src)
        d32 = need(paired, "position_minus_isotonic (3-2)", src=src)
        add("position minus isotonic",
            _signed(need(d32, "median", src=src), "{:.3f}"),
            src + " paired 3-2 median",
            r"In the paired comparison position sits\s+\*\*")
        add("position minus isotonic assays positive",
            [str(need(d32, "assays_positive", src=src))],
            src + " paired 3-2 assays_positive",
            r"below the monotone curve and is positive in only ")
        d42 = need(paired, "position+isotonic_minus_isotonic (4-2)", src=src)
        add("position adds over isotonic",
            _signed(need(d42, "median", src=src), "{:.3f}"),
            src + " paired 4-2 median",
            r"term's own contribution once the curve is present is \*\*")
        add("position adds over isotonic, beats null",
            [f"{round(need(d42, 'beats_null_share', src=src) * need(m, 'n_assays', src=src)):.0f}/{need(m, 'n_assays', src=src)}"],
            src + " paired 4-2 beats_null_share x n_assays",
            r"above its permutation null in \*\*")
        surv = need(m, "position_share_surviving", src=src)
        add("position share surviving",
            [f"{round(100 * need(surv, 'median_D42_over_median_position_r2', src=src)):.0f}%"],
            src + " median_D42_over_median_position_r2",
            r"assays\. That is \*\*")
        ci32 = need(d32, "ci95_bootstrap", src=src)
        add("position minus isotonic, CI low", _signed(ci32[0], "{:.3f}"),
            src + " paired 3-2 ci95[0]", r"position sits\s+\*\*-?\d\.\d+\*\* \[")
        add("position minus isotonic, CI high", _signed(ci32[1], "{:.3f}"),
            src + " paired 3-2 ci95[1]",
            r"position sits\s+\*\*-?\d\.\d+\*\* \[-?\d\.\d+, ")
        ci42 = need(d42, "ci95_bootstrap", src=src)
        add("position adds over isotonic, CI low", _signed(ci42[0], "{:.3f}"),
            src + " paired 4-2 ci95[0]",
            r"curve is present is \*\*\d\.\d+\*\* \[")
        add("position adds over isotonic, CI high", _signed(ci42[1], "{:.3f}"),
            src + " paired 4-2 ci95[1]",
            r"curve is present is \*\*\d\.\d+\*\* \[\d\.\d+, ")
        add("monotone rule eps", [f"{need(m, 'rule', 'eps', src=src):.2f}"],
            src + " rule.eps", r"lower bound clears ", r"judged by the\s+same ")
    block("monotone_fit.json", monotone_fit)

    for name in REQUIRED_ARTEFACTS:
        if not (res / name).exists() and name not in missing:
            missing.append(name)
    # --- seed_sweep.json: the other-labs AUC verdict over 200 seeds ----------------
    def seed_sweep(d):
        src = "seed_sweep.json"
        ag = need(d, "aggregate", src=src)
        cnt = need(ag, "verdict_counts", src=src)
        add("seed sweep n seeds", [str(need(ag, "n_seeds", src=src))],
            src + " aggregate.n_seeds", r"Re-run over ")
        add("seed sweep seeds SUPPORTED", [str(cnt.get("SUPPORTED", 0))],
            src + " verdict_counts.SUPPORTED", r"gives SUPPORTED on \*\*")
        add("seed sweep seeds NOT_SUPPORTED", [str(cnt.get("NOT_SUPPORTED", 0))],
            src + " verdict_counts.NOT_SUPPORTED", r"and NOT_SUPPORTED on \*\*")
        cl = need(ag, "clearance", src=src)
        add("seed sweep clearance median",
            _signed(need(cl, "median", src=src), "{:+.3f}"),
            src + " clearance.median", r"0\.884\) has median \*\*")
        add("seed sweep clearance p2.5",
            _signed(need(cl, "p2_5", src=src), "{:+.3f}"),
            src + " clearance.p2_5", r"and runs from \*\*")
        add("seed sweep clearance p97.5",
            _signed(need(cl, "p97_5", src=src), "{:+.3f}"),
            src + " clearance.p97_5", r"and runs from \*\*[^*]+\*\* to\s+\*\*")
        add("seed sweep aggregated verdict",
            [need(ag, "verdict", src=src).split()[0]], src + " aggregate.verdict",
            r"so the seed-aggregated verdict is ", kind="verdict")

    block("seed_sweep.json", seed_sweep)
    return claims, sorted(set(missing))


def estimator_problems(readme: str) -> list[str]:
    """Check that each quoted number carries the estimator its artefact used.

    This is the check the first version could not do, and its absence certified
    three wrong claims as clean (reviewer F9): '40.8%' matched the pooled artefact
    while sitting in a table of per-assay medians; '0.693' matched a median split
    while the sentence implied a threshold-free accuracy; '81%' matched the residue
    artefact while the sentence called it chance. A number and its estimator are
    two claims, and only one of them was being checked.

    Each entry is (pattern that must appear, pattern that must NOT appear, why).
    """
    problems: list[str] = []

    # A withdrawn phrase is allowed ONLY inside an explicit correction. The
    # README must be able to say "an earlier version claimed X", because that is
    # the honest way to retire a claim - otherwise the only way to pass this
    # check is to hide the mistake. So each occurrence is examined in a window and
    # passes when the surrounding text marks it as withdrawn.
    RETRACTION = re.compile(
        r"withdrawn|earlier version|earlier draft|used to (say|call|report)|"
        r"is dead|no longer|an earlier|correction|must not|do not quote|"
        r"said \u201d|previously",
        re.IGNORECASE)

    rules = [
        # a pooled fraction must never be labelled as a per-assay median
        (r"per-assay\s+\*\*median\*\*",
         r"pairs deviating >2 assay-sd from additive",
         "the 2-sd row must be labelled per-assay median; the pooled fraction "
         "must not be presented as one"),
        (None, r"41% of (individual |the )?pairs|misses by more than 2 assay-sd on\s+\n?\s*41%",
         "the pooled 41% was withdrawn as a headline; it must not be quoted as "
         "the miss rate"),
        (None, r"resolves only 69%|resolves [0-9]+% of them in the right direction",
         "the observed-median split is not a threshold-free accuracy; AUC is"),
        (None, r"fails hardest where the singles look mild|"
               r"largest where the singles look mild",
         "the mild-singles prior is withdrawn; it failed its own permutation null"),
        # the null ESM result must be stated as null somewhere
        (r"no usable signal|does not tell you|indistinguishable from",
         None,
         "a null ESM result must be stated as null, not softened"),
    ]
    for must, must_not, why in rules:
        if must_not:
            for m in re.finditer(must_not, readme, re.IGNORECASE):
                window = readme[max(0, m.start() - 400):m.end() + 400]
                if not RETRACTION.search(window):
                    problems.append(f"withdrawn phrase present: {why} "
                                    f"(at char {m.start()})")
        if must and not re.search(must, readme, re.IGNORECASE):
            problems.append(f"missing required labelling: {why}")

    # the >2sd number must never appear without an adjacent estimator label
    for m in re.finditer(r"40\.?9?%", readme):
        window = readme[max(0, m.start() - 260):m.end() + 260]
        if re.search(r"pooled", window, re.IGNORECASE) is None and \
                re.search(r"used to call", window, re.IGNORECASE) is None:
            problems.append(
                "a pooled >2-sd percentage appears without the word 'pooled' "
                "nearby; the pooled and per-assay figures differ 3-fold")
    return problems


def published_text(root: Path = ROOT, readme: str | None = None) -> str:
    """README.md, then every docs/*.md in name order: all the pages a reader is
    sent to from the README. Claims are checked on this text, so a section moved
    out of the README keeps its guard. `readme` replaces README.md only (the
    docs pages are still appended). Line numbers in problems count from the top
    of this concatenation."""
    parts = [Path(readme).read_text() if readme else (root / "README.md").read_text()]
    for p in sorted((root / "docs").glob("*.md")):
        parts.append(f"<!-- {p.relative_to(root).as_posix()} -->\n" + p.read_text())
    return "\n".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--readme", default=None,
                    help="README to check in place of README.md; docs/*.md are "
                         "always appended")
    ap.add_argument("--results", default=str(RES),
                    help="artefact directory (tests point this at a damaged copy)")
    ap.add_argument("--strict", action="store_true",
                    help="also fail if any claim has no anchor (what CI runs)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    readme = published_text(ROOT, args.readme)
    claims, missing = build_claims(Path(args.results))
    out = audit(readme, claims, strict=args.strict)
    n_anch = len(claims) - len(out["unanchored"])
    bad = {p.split(":", 1)[0] for p in out["problems"]}
    print(f"{len(claims) - len(bad)}/{len(claims)} claims hold in place; "
          f"{n_anch}/{len(claims)} anchored to a sentence or table cell")
    if missing:
        print(f"\nMISSING ARTEFACT ({len(missing)}) - the claims that depend on it "
              "cannot be checked, which is a failure, not a smaller checklist:")
        for m in missing:
            print(f"  {m}")
    if out["unanchored"]:
        print(f"\nUNANCHORED ({len(out['unanchored'])}) - checked only as 'appears "
              "somewhere in the README':")
        for u in out["unanchored"]:
            print(f"  {u}")
    if out["estimator"]:
        print(f"\nESTIMATOR LABELS WRONG ({len(out['estimator'])}):")
        for x in out["estimator"]:
            print(f"  {x}")
    if out["problems"]:
        print("\nWRONG, MOVED OR MISSING:")
        for p in out["problems"]:
            print(f"  {p}")
    if args.verbose:
        for c in claims:
            print(f"  OK? {c.label:44s} {c.spellings[0]:>12s} anchors={len(c.anchors)}"
                  f"  <- {c.source}")
    if missing or not out["ok"]:
        if args.strict and out["unanchored"] and not out["problems"]:
            print("\n--strict: every claim must be bound to the place it belongs.")
        print("\nEither the README is stale or an artefact changed or is missing. "
              "Neither may ship.")
        return 1
    print("every published number matches its committed artefact, in place")
    return 0


if __name__ == "__main__":
    sys.exit(main())
