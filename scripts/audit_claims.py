#!/usr/bin/env python3
"""Check every number the README quotes against a committed artefact.

The previous project published a headline that was 23x wrong because the error
column was inflated by a unit slip, and 44 green tests could not see it: the
accuracy was right and the error was wrong. Prose is where numbers rot. This
script is the mechanical guard.

Usage: python scripts/audit_claims.py [--readme README.md]
Exit 0 when every claim matches; 1 with a list of mismatches otherwise.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"


def _pct(x: float) -> list[str]:
    """Accept 40.8% and 41% for the same value - rounding in prose is allowed,
    misquoting is not."""
    return [f"{x:.1f}%", f"{round(x)}%", f"{x:.2f}%"]


def build_claims() -> tuple[list[tuple[str, list[str], str]], list[str]]:
    """(claims, unverifiable).

    Rewritten after review round 1: the artefacts gained shift-invariant
    statistics, and several numbers this audit used to check were withdrawn
    because they were artefacts (a pooled fraction beside per-assay medians, and
    a correlation that depended on an arbitrary zero point). Checking a withdrawn
    number would have kept the audit green on a false claim, so the checklist
    follows the corrected artefacts.
    """
    claims: list[tuple[str, list[str], str]] = []
    unverifiable: list[str] = []

    summ_p = RES / "assay_summary.csv"
    inv_p = RES / "invariant_pooled.json"
    prior_p = RES / "prior_pooled.json"
    esm_p = RES / "esm_context.json"
    depth_p = RES / "depth.json"

    if summ_p.exists():
        s = pd.read_csv(summ_p)
        claims.append(("assays", [str(len(s))], "assay_summary.csv rows"))
        claims.append(("pairs", [f"{int(s.n_pairs.sum()):,}"],
                       "assay_summary.csv n_pairs sum"))
        # every median is over the 48 assays with >=500 pairs, as the README says
        claims.append(("median spearman",
                       [f"{s[s.n_pairs >= 500].spearman_additive.median():.3f}"],
                       "assay_summary.csv spearman median, >=500 pairs"))
        claims.append(("min coverage", [f"{s.coverage.min():.1%}"],
                       "assay_summary.csv coverage min"))
        # F8: both denominators are published, so both are audited. The old README
        # used 65 assays for coverage and 48 for every correlation, silently.
        big = s[s.n_pairs >= 500]
        claims.append(("assays in correlation claims", [str(len(big))],
                       "assay_summary.csv rows with >=500 pairs"))
        claims.append(("pairs in correlation claims",
                       [f"{int(big.n_pairs.sum()):,}"],
                       "assay_summary.csv n_pairs sum, >=500 pairs"))
        # F4: AUC replaces the median split, but the split is still published and
        # must be labelled as the split it is.
        if "auc_direction" in big.columns:
            claims.append(("median AUC", [f"{big.auc_direction.median():.3f}"],
                           "assay_summary.csv auc_direction median"))
            claims.append(("median agreement, own medians",
                           [f"{big.direction_agreement_own_medians.median():.3f}"],
                           "assay_summary.csv own-median agreement"))
            claims.append(("median agreement, observed-median split",
                           [f"{big.direction_agreement.median():.3f}"],
                           "assay_summary.csv direction_agreement"))
        # F1: the per-assay median beyond 2 sd, plus the bias that makes the pooled
        # version uninterpretable.
        if "frac_outside_2sd" in big.columns:
            claims.append(("per-assay median >2sd fraction",
                           [f"{big.frac_outside_2sd.median():.1%}",
                            f"{big.frac_outside_2sd.median():.3f}"],
                           "assay_summary.csv frac_outside_2sd median"))
        if "frac_within_1sd" in big.columns:
            # Audited on the SAME 48-assay denominator the paragraph promises. The
            # README quoted 0.437, which is the all-65 median: the row sat in a
            # table whose text says every fraction uses the 48-assay set, so the
            # number and its stated denominator disagreed by construction.
            claims.append(("median within 1sd fraction",
                           [f"{big.frac_within_1sd.median():.3f}"],
                           "assay_summary.csv frac_within_1sd median, >=500 pairs"))
        if "bias_mean_sd" in big.columns:
            claims.append(("median residual bias",
                           [f"{big.bias_mean_sd.median():.2f}",
                            f"{abs(big.bias_mean_sd.median()):.2f}"],
                           "assay_summary.csv bias_mean_sd median"))
        # THE POOLED FRACTIONS, from integer counts in a COMMITTED file. These used
        # to be checkable only from results/pairs/ (101 MB, gitignored), so a fresh
        # clone could not verify them and the audit had to announce reduced
        # coverage. Counting rows makes the same numbers reproducible with nothing
        # but assay_summary.csv, which is why the counts are published beside the
        # fractions rather than instead of them.
        if {"n_outside_2sd", "n_outside_2sd_centred", "n_within_1sd"} <= \
                set(big.columns):
            pooled = float(big.n_outside_2sd.sum() / big.n_pairs.sum())
            claims.append(("pooled >2sd fraction",
                           [f"{pooled:.1%}", f"{pooled:.3f}"],
                           "assay_summary.csv n_outside_2sd / n_pairs, >=500 "
                           "pairs (integer counts)"))
            pooled_c = float(big.n_outside_2sd_centred.sum() / big.n_pairs.sum())
            claims.append(("pooled >2sd fraction, centred",
                           [f"{pooled_c:.1%}", f"{pooled_c:.3f}"],
                           "assay_summary.csv n_outside_2sd_centred / n_pairs, "
                           ">=500 pairs"))

    if inv_p.exists():
        i = json.loads(inv_p.read_text())
        # Only what the README actually claims now. The uncentred rho and the
        # centred terciles were WITHDRAWN as artefacts, so checking them would
        # keep the audit green on numbers the repo no longer stands behind.
        for key, label, fmts in (
            ("rho_centred_median", "centred rho (mean)", ("{:+.3f}", "{:.3f}")),
            ("auc_median", "AUC", ("{:.3f}",)),
            ("frac_beyond_2sd_raw_median_of_assays", "beyond 2sd median", None),
            ("frac_beyond_2sd_raw_pooled", "beyond 2sd pooled", None),
            ("frac_beyond_2sd_centred_median_of_assays", "beyond 2sd centred", None),
        ):
            v = i.get(key)
            if v is None:
                continue
            if fmts is None:
                claims.append((label, _pct(100 * v), f"invariant_pooled.json {key}"))
            else:
                claims.append((label, [f.format(v) for f in fmts],
                               f"invariant_pooled.json {key}"))
        v = i.get("residual_bias_sd_median")
        if v is not None:
            claims.append(("residual bias sd", [f"{abs(v):.2f}"],
                           "invariant_pooled.json"))
        mp = i.get("magnitude_prior_verdict", {})
        for key, label in (("null_shuffle_observed_rebuilt", "correct null"),
                           ("observed_centred_mean", "observed centred")):
            if key in mp:
                claims.append((label, [f"{mp[key]:+.3f}", f"{mp[key]:.3f}"],
                               "invariant_pooled.json magnitude_prior_verdict"))
        if "assays_observed_exceeds_null" in mp:
            claims.append(("exceeds null", [mp["assays_observed_exceeds_null"]],
                           "invariant_pooled.json"))

    if depth_p.exists():
        dj = json.loads(depth_p.read_text())
        h2 = dj.get("H2_position", {})
        for key, label in (("observed_r2_median", "position r2"),
                           ("permutation_null_median", "position null")):
            if key in h2:
                claims.append((label, [f"{h2[key]:.3f}"], "depth.json H2"))
        h1 = dj.get("H1_residue", {})
        top = h1.get("top", {})
        if top:
            k0 = sorted(top)[0]
            v = top[k0].get("frac_above_assay_median")
            if v is not None:
                claims.append(("residue top fraction", [f"{v:.0%}"],
                               "depth.json H1"))
        # F6: the reliability and confound checks are what make the residue
        # verdict INCONCLUSIVE, so they must be on the page and audited.
        if h1.get("split_half_rank_reliability") is not None:
            claims.append(("residue rank reliability",
                           [f"{h1['split_half_rank_reliability']:.2f}"],
                           "depth.json H1 split-half"))
        if h1.get("A_frac_above_after_magnitude_removal") is not None:
            claims.append(("residue A after confound removal",
                           [f"{h1['A_frac_above_after_magnitude_removal']:.2f}"],
                           "depth.json H1 confound check"))
        # F5: position-count context, so the R2 cannot be read as a hot-residue map
        for key, fmt in (("positions_per_assay_median", "{:.0f}"),
                         ("hot_decile_position_pair_coverage_median", "{:.0%}"),
                         ("single_mutation_r2_median_slot_a", "{:.3f}"),
                         ("observed_r2_demeaned_median", "{:.3f}")):
            if h2.get(key) is not None:
                claims.append((f"position {key}", [fmt.format(h2[key])],
                               "depth.json H2"))
        # F2/F3: the withdrawn magnitude prior must be reported WITH its null,
        # and the null must be the construction-preserving one.
        h3 = dj.get("H3_magnitude_prior_centred", {})
        if h3.get("permutation_null_centred_median") is not None:
            claims.append(("magnitude prior null",
                           [f"{h3['permutation_null_centred_median']:.3f}"],
                           "depth.json H3 permutation null"))
        if h3.get("rho_centred_median") is not None:
            claims.append(("magnitude prior centred rho",
                           [f"{h3['rho_centred_median']:+.3f}",
                            f"{h3['rho_centred_median']:.3f}"],
                           "depth.json H3 centred rho"))

    if esm_p.exists():
        e = json.loads(esm_p.read_text())
        rho = e.get("spearman_mean")
        if rho is not None:
            claims.append(("esm rho", [f"{rho:+.3f}", f"{rho:.3f}"],
                           "esm_context.json"))
        if e.get("n_pairs"):
            claims.append(("esm pairs", [f"{int(e['n_pairs']):,}"],
                           "esm_context.json"))
        # F7: the noise floor and the free-prior comparison must be on the page.
        # A null ESM result without its noise floor is not evidence of anything.
        ctl = e.get("controls") or {}
        nz = (ctl.get("noise") or {}).get("spearman_mean")
        if nz is not None:
            claims.append(("esm noise floor", [f"{nz:+.3f}"],
                           "esm_context.json controls.noise"))
        pri = (ctl.get("prior_only") or {}).get("spearman_mean")
        if pri is not None:
            claims.append(("esm prior-only rho", [f"{pri:+.3f}"],
                           "esm_context.json controls.prior_only"))
        sd = (ctl.get("esm_only") or {}).get("spearman_sd")
        if sd is not None:
            claims.append(("esm spearman sd", [f"{sd:.3f}"],
                           "esm_context.json controls.esm_only"))
        inc = e.get("incremental_value_of_esm_over_prior")
        if inc is not None:
            claims.append(("esm incremental value", [f"{abs(inc):.3f}"],
                           "esm_context.json incremental value"))
        sel = e.get("selection") or {}
        if sel.get("pairs_used"):
            claims.append(("esm pairs used", [f"{int(sel['pairs_used']):,}"],
                           "esm_context.json selection"))

    # --- held-out tests (scripts/heldout.py) ---------------------------------
    # Every number in the "Does any of it hold out of sample?" section is in the
    # artefact, including the intervals, because an interval is the claim there:
    # the point of the section is that the bias does NOT replicate, and a bare
    # median without its interval cannot show that.
    hold_p = RES / "heldout.json"
    if hold_p.exists():
        h = json.loads(hold_p.read_text())
        ts = (h.get("H1_lab_family") or {}).get("Tsuboyama") or {}
        ot = (h.get("H1_lab_family") or {}).get("other_labs") or {}
        for grp, tag in ((ts, "Tsuboyama"), (ot, "other labs")):
            b = (grp.get("bias_sd") or {})
            if b.get("median") is not None:
                claims.append((f"heldout bias {tag}",
                               [f"{b['median']:+.2f}", f"{b['median']:.2f}",
                                f"{abs(b['median']):.2f}"],
                               f"heldout.json H1 {tag} bias_sd median"))
            a = (grp.get("auc") or {})
            if a.get("median") is not None:
                claims.append((f"heldout auc {tag}",
                               [f"{a['median']:.3f}"],
                               f"heldout.json H1 {tag} auc median"))
        d = ((h.get("H1_lab_family") or {})
             .get("difference_Tsuboyama_minus_other") or {}).get("bias_sd") or {}
        if d.get("diff_median") is not None:
            claims.append(("heldout bias difference",
                           [f"{d['diff_median']:.2f}",
                            f"{abs(d['diff_median']):.2f}"],
                           "heldout.json H1 bias difference"))
        if d.get("p_two_sided") is not None:
            claims.append(("heldout bias difference p",
                           [f"{d['p_two_sided']:.4f}"],
                           "heldout.json H1 bootstrap p"))
        sh = h.get("H2_excluded_small_assays") or {}
        sb = sh.get("bias_sd") or {}
        if sb.get("median") is not None:
            claims.append(("heldout excluded bias",
                           [f"{sb['median']:+.2f}", f"{sb['median']:.2f}"],
                           "heldout.json H2 bias_sd median"))
        sa = (sh.get("auc") or {}).get("median")
        if sa is not None:
            claims.append(("heldout excluded auc", [f"{sa:.3f}"],
                           "heldout.json H2 auc median"))
        loo = h.get("H3_leave_one_lab_out") or {}
        for key, label in (("auc_range", "loo auc range"),
                           ("bias_range", "loo bias range")):
            rng = loo.get(key) or []
            if len(rng) == 2:
                fmt = (lambda v: f"{v:.3f}") if key == "auc_range" \
                    else (lambda v: f"{v:+.2f}")
                claims.append((label, [fmt(rng[0]), fmt(rng[1])],
                               f"heldout.json H3 {key}"))

    # --- figure statistics (scripts/make_figures.py) -------------------------
    # Quoted in the README next to the scatter, so they are checked like any other
    # number rather than living only in a function's stdout.
    fig_p = RES / "figure_stats.json"
    if fig_p.exists():
        f = (json.loads(fig_p.read_text()).get("bias_vs_tail_fraction") or {})
        for key, label in (("pearson_abs_bias", "tail vs |bias| r"),
                           ("pearson_residual_sd", "tail vs spread r")):
            v = f.get(key)
            if v is not None:
                claims.append((label, [f"{v:+.2f}", f"{abs(v):.2f}"],
                               f"figure_stats.json {key}"))

    # --- stronger baselines (scripts/alt_baseline.py) ------------------------
    # The section is a table of model comparisons, so every cell a reader can see
    # is checked against the artefact - including the nulls, because the null is
    # the claim there: a model that matches its own permutation null is not
    # structure, and the README says exactly that about the offsets.
    alt_p = RES / "alt_baseline.json"
    if alt_p.exists():
        a = json.loads(alt_p.read_text())
        # ONLY the models the README table shows. Claiming on all seven would
        # demand that every fitted variant appear on the page, which is backwards:
        # the audit follows the prose, it does not dictate the table. Rank 1 and 2
        # are in the artefact for completeness and are not argued from here.
        shown = ("additive", "additive+pooled_offset", "additive+assay_offset",
                 "additive+position", "additive+position+pair_rank3")
        for m in a.get("models", []):
            name = str(m.get("model"))
            if name not in shown:
                continue
            med = m.get("median") or {}
            nul = m.get("null_median") or {}
            for key, short, fmts in (
                    ("r2_resid_var", "r2", ("{:.3f}", "{:.0f}")),
                    ("auc_direction", "auc", ("{:.3f}",)),
                    ("frac_outside_2sd", "beyond2", None),
                    ("bias_mean_sd", "bias", ("{:.2f}",))):
                v = med.get(key)
                if v is None:
                    continue
                if fmts is None:
                    spell = _pct(100 * v)
                else:
                    spell = [f.format(v) for f in fmts]
                claims.append((f"alt_baseline {name} {short}", spell,
                               f"alt_baseline.json {name} median.{key}"))
            nv = nul.get("r2_resid_var")
            if nv is not None:
                claims.append((f"alt_baseline {name} r2 null",
                               [f"{nv:.3f}"],
                               f"alt_baseline.json {name} null r2"))
        # the per-assay win counts the prose quotes
        for m in a.get("models", []):
            if m.get("model") in ("additive+position+pair_rank3",
                                  "additive+position"):
                bl = m.get("assays_real_beats_own_null") or {}
                if isinstance(bl, dict) and bl.get("r2_resid_var"):
                    claims.append((f"alt_baseline {m['model']} beats null",
                                   [str(bl["r2_resid_var"])],
                                   f"alt_baseline.json {m['model']} "
                                   "assays_real_beats_own_null"))
        # ONLY the pair model's AUC win count is argued in the prose ("wins in
        # only 34/48 assays"). The position-only 28/48 is in the artefact and the
        # table's other columns, but no sentence claims it, so claiming it here
        # would be the audit inventing a fact rather than checking one.
        for m in a.get("models", []):
            if m.get("model") == "additive+position+pair_rank3" \
                    and m.get("assays_auc_above_additive"):
                claims.append(("alt_baseline pair_rank3 auc wins",
                               [str(m["assays_auc_above_additive"])],
                               "alt_baseline.json pair_rank3 auc wins"))

    # --- corpus-scale ESM probe (scripts/esm_scale.py) -----------------------
    esm_scale_p = RES / "esm_scale.json"
    if esm_scale_p.exists():
        e = json.loads(esm_scale_p.read_text())
        sel = ((e.get("primary") or {}).get("selection")) or {}
        for key, label, fmt in (
                ("pairs_used", "esm_scale pairs", "{:,}"),
                ("fraction_of_eligible_pairs", "esm_scale corpus fraction", None),
                ("assays_used", "esm_scale assays", "{}")):
            v = sel.get(key)
            if v is None:
                continue
            spell = ([f"{100 * v:.1f}%"] if fmt is None
                     else [fmt.format(v) if fmt != "{}" else str(v)])
            claims.append((label, spell, f"esm_scale.json selection.{key}"))
        heads = (e.get("primary") or {}).get("heads") or {}
        for hid in ("a", "b", "c"):
            vd = (heads.get(hid) or {}).get("verdict") or {}
            # head_mean and control_mean are both quoted in the prose ("(a) scores
            # +0.026 against a shuffled control of +0.055"). delta is not: the
            # README reports the two arms and lets the reader subtract, so
            # claiming delta would demand a number the page never states.
            for key, label in (("head_mean", f"esm_scale head {hid} rho"),
                               ("control_mean", f"esm_scale head {hid} control")):
                v = vd.get(key)
                if v is not None:
                    claims.append((label, [f"{v:+.3f}", f"{v:.3f}"],
                                   f"esm_scale.json primary.heads.{hid}.{key}"))
        inc = (e.get("headline") or {}).get("incremental_rho_of_esm_over_prior")
        if inc is not None:
            claims.append(("esm_scale incremental over prior",
                           [f"{inc:+.3f}"],
                           "esm_scale.json headline incremental rho"))
        # the 100% sensitivity run must be checkable too. Its column in the README
        # states VERDICTS, not rho values, so the audit checks the one number the
        # page does carry from that run (the pair count) and the verdict
        # correspondence is checked structurally by
        # tests/test_claims.py::test_esm_verdict_table_matches_the_artefact.
        # Claiming a rho here would demand a number the README never prints, which
        # is the audit inventing a requirement rather than checking a claim.
        sens = e.get("sensitivity") or []
        if sens:
            ssh = (sens[0].get("selection") or {})
            if ssh.get("pairs_used"):
                claims.append(("esm_scale full-corpus pairs",
                               [f"{int(ssh['pairs_used']):,}"],
                               "esm_scale.json sensitivity selection.pairs_used"))

    # --- prior units (scripts/prior_units.py) --------------------------------
    pu_p = RES / "prior_units.json"
    if pu_p.exists():
        u = json.loads(pu_p.read_text())
        for grp, key, label in (
                ("within_assay_raw_units", "median_rho",
                 "prior units within raw"),
                ("within_assay_sd_scaled", "median_rho",
                 "prior units within sd-scaled")):
            v = (u.get(grp) or {}).get(key)
            if v is not None:
                claims.append((label, [f"{v:+.3f}", f"{v:.3f}"],
                               f"prior_units.json {grp}.{key}"))
        for key, label in (("pooled_across_assays_raw_units",
                            "prior units pooled raw"),
                           ("pooled_across_assays_sd_scaled",
                            "prior units pooled sd-scaled")):
            v = u.get(key)
            if v is not None:
                claims.append((label, [f"{v:+.3f}", f"{v:.3f}"],
                               f"prior_units.json {key}"))
        dev = u.get("max_abs_deviation_within_assay")
        if dev is not None:
            claims.append(("prior units max deviation", [f"{dev:.2e}",
                                                         "0.00e+00"],
                           "prior_units.json max_abs_deviation_within_assay"))

    return claims, unverifiable


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--readme", default=str(ROOT / "README.md"))
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    readme = Path(args.readme).read_text()
    claims, unverifiable = build_claims()
    if not claims:
        print("no artefacts found - run build_pairs.py and prior.py first")
        return 1

    missing = []
    for label, spellings, source in claims:
        if not any(sp in readme for sp in spellings):
            missing.append((label, spellings, source))

    print(f"{len(claims) - len(missing)}/{len(claims)} claims found in the README")
    if unverifiable:
        print(f"\nUNVERIFIED ({len(unverifiable)}) - coverage is REDUCED, not clean:")
        for u in unverifiable:
            print(f"  {u}")
    est = estimator_problems(readme)
    if est:
        print(f"\nESTIMATOR LABELS WRONG ({len(est)}):")
        for e in est:
            print(f"  {e}")
    if missing:
        print("\nMISSING OR MISQUOTED:")
        for label, spellings, source in missing:
            print(f"  {label:32s} expected one of {spellings}  (from {source})")
        print("\nEither the README is stale or the artefact changed. Neither may "
              "ship.")
        return 1
    if est:
        print("\nEvery number matches its artefact, but a number printed under the "
              "wrong estimator is still a wrong claim.")
        return 1
    if args.verbose:
        for label, spellings, source in claims:
            print(f"  OK {label:32s} {spellings[0]:>12s}  <- {source}")
    if unverifiable:
        print("every checkable number matches; the ones above could not be checked")
    else:
        print("every published number matches a committed artefact")
    return 0


if __name__ == "__main__":
    sys.exit(main())
