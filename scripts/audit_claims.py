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

    if esm_p.exists():
        e = json.loads(esm_p.read_text())
        rho = e.get("spearman_mean")
        if rho is not None:
            claims.append(("esm rho", [f"{rho:+.3f}", f"{rho:.3f}"],
                           "esm_context.json"))
        if e.get("n_pairs"):
            claims.append(("esm pairs", [f"{int(e['n_pairs']):,}"],
                           "esm_context.json"))

    return claims, unverifiable


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
    if missing:
        print("\nMISSING OR MISQUOTED:")
        for label, spellings, source in missing:
            print(f"  {label:32s} expected one of {spellings}  (from {source})")
        print("\nEither the README is stale or the artefact changed. Neither may "
              "ship.")
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
