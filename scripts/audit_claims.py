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
        claims.append(("median spearman", [f"{s.spearman_additive.median():.3f}"],
                       "assay_summary.csv spearman median"))
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
        # F4: AUC replaces the median split as the direction measure, and the
        # split is still published so the two can be compared.
        if "auc_direction" in big.columns:
            claims.append(("median AUC", [f"{big.auc_direction.median():.3f}"],
                           "assay_summary.csv auc_direction median"))
            claims.append(("median agreement, own medians",
                           [f"{big.direction_agreement_own_medians.median():.3f}"],
                           "assay_summary.csv own-median agreement"))
        # F1: the per-assay median beyond 2 sd, plus the bias that makes the pooled
        # version uninterpretable.
        if "frac_outside_2sd" in big.columns:
            claims.append(("per-assay median >2sd fraction",
                           [f"{big.frac_outside_2sd.median():.1%}",
                            f"{big.frac_outside_2sd.median():.3f}"],
                           "assay_summary.csv frac_outside_2sd median"))
        if "bias_mean_sd" in big.columns:
            claims.append(("median residual bias",
                           [f"{big.bias_mean_sd.median():.2f}",
                            f"{abs(big.bias_mean_sd.median()):.2f}"],
                           "assay_summary.csv bias_mean_sd median"))
        # F8: both denominators are published, so both are audited. The old README
        # used 65 assays for coverage and 48 for correlations without saying so.
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
        # F1: the per-assay median beyond 2 sd, and the bias that makes the pooled
        # version uninterpretable.
        if "frac_outside_2sd" in big.columns:
            claims.append(("per-assay median >2sd fraction",
                           [f"{big.frac_outside_2sd.median():.1%}",
                            f"{big.frac_outside_2sd.median():.3f}"],
                           "assay_summary.csv frac_outside_2sd median"))
        if "bias_mean_sd" in big.columns:
            claims.append(("median residual bias",
                           [f"{big.bias_mean_sd.median():.2f}",
                            f"{abs(big.bias_mean_sd.median()):.2f}"],
                           "assay_summary.csv bias_mean_sd median"))

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
