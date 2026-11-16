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


def build_claims() -> list[tuple[str, list[str], str]]:
    """(label, acceptable spellings, source-of-truth description)."""
    claims: list[tuple[str, list[str], str]] = []

    summ_p = RES / "assay_summary.csv"
    prior_p = RES / "prior_pooled.json"
    esm_p = RES / "esm_context.json"
    probe_p = RES / "esm_probe.json"

    if summ_p.exists():
        s = pd.read_csv(summ_p)
        n_assays = len(s)
        n_pairs = int(s.n_pairs.sum())
        claims.append(("assays", [str(n_assays)], "assay_summary.csv rows"))
        claims.append(("pairs", [f"{n_pairs:,}", str(n_pairs)],
                       "assay_summary.csv n_pairs sum"))
        claims.append(("median spearman", [f"{s.spearman_additive.median():.3f}"],
                       "assay_summary.csv spearman median"))
        claims.append(("median direction agreement",
                       [f"{s.direction_agreement.median():.3f}"],
                       "assay_summary.csv direction median"))
        claims.append(("min coverage", [f"{s.coverage.min():.1%}"],
                       "assay_summary.csv coverage min"))

    # pooled strong-epistasis fraction, recomputed from the pair files
    pair_files = sorted((RES / "pairs").glob("*.csv"))
    if pair_files:
        import numpy as np
        tot = 0
        strong = 0
        for p in pair_files:
            d = pd.read_csv(p, usecols=["residual"])
            if len(d) < 500:
                continue
            tot += len(d)
            strong += int((d.residual.abs() > 2).sum())
        if tot:
            frac = strong / tot
            claims.append(("strong epistasis fraction", _pct(100 * frac),
                           "pair files, |residual| > 2 sd"))

    if prior_p.exists():
        pj = json.loads(prior_p.read_text())
        claims.append(("rho vs abs prediction",
                       [f"{pj['rho_abspred_absres_median']:+.3f}",
                        f"{pj['rho_abspred_absres_median']:.3f}"],
                       "prior_pooled.json"))
        claims.append(("rho vs signed prediction",
                       [f"{pj['rho_pred_absres_median']:+.3f}"],
                       "prior_pooled.json"))
        claims.append(("rho vs separation",
                       [f"{pj['rho_sep_absres_median']:+.3f}"],
                       "prior_pooled.json"))
        claims.append(("simulation null",
                       [f"{abs(pj['simulation_null']['mean']):.3f}"],
                       "prior_pooled.json simulation_null"))
        terc = pj.get("median_abs_residual_by_additive_tercile", {})
        if "small" in terc:
            claims.append(("tercile small", [f"{terc['small']:.3f}"],
                           "prior_pooled.json terciles"))
        if "large" in terc:
            claims.append(("tercile large", [f"{terc['large']:.3f}"],
                           "prior_pooled.json terciles"))

    if esm_p.exists():
        e = json.loads(esm_p.read_text())
        rho = e.get("spearman_mean")
        if rho is not None:
            claims.append(("esm spearman", [f"{rho:+.3f}", f"{rho:.3f}"],
                           "esm_context.json"))
        imp = e.get("mae_improvement_mean")
        if imp is not None:
            claims.append(("esm mae change", [f"{abs(imp):.2f}"],
                           "esm_context.json"))
        if e.get("n_pairs"):
            claims.append(("esm pairs", [f"{int(e['n_pairs']):,}"],
                           "esm_context.json"))

    depth_p = RES / "depth.json"
    if depth_p.exists():
        dj = json.loads(depth_p.read_text())
        h2 = dj.get("H2_position", {})
        if "observed_r2_median" in h2:
            claims.append(("position r2", [f"{h2['observed_r2_median']:.3f}"],
                           "depth.json H2_position"))
        if "permutation_null_median" in h2:
            claims.append(("position null",
                           [f"{h2['permutation_null_median']:.3f}"],
                           "depth.json H2_position"))
        if "observed_exceeds_null_in" in h2 and "n_assays" in h2:
            claims.append(("position assays",
                           [f"{h2['observed_exceeds_null_in']}/{h2['n_assays']}"],
                           "depth.json H2_position"))
        h1 = dj.get("H1_residue", {})
        if h1.get("top"):
            top = sorted(h1["top"].items())[0][1]
            if "frac_above_assay_median" in top:
                claims.append(("residue top fraction",
                               [f"{top['frac_above_assay_median']:.0%}"],
                               "depth.json H1_residue"))

    if probe_p.exists():
        q = json.loads(probe_p.read_text())
        r = q.get("spearman_distance")
        if r is not None:
            claims.append(("distance spearman", [f"{r:.2f}"],
                           "esm_probe.json"))

    return claims


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--readme", default=str(ROOT / "README.md"))
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    readme = Path(args.readme).read_text()
    claims = build_claims()
    if not claims:
        print("no artefacts found - run build_pairs.py and prior.py first")
        return 1

    missing = []
    for label, spellings, source in claims:
        if not any(sp in readme for sp in spellings):
            missing.append((label, spellings, source))

    print(f"{len(claims) - len(missing)}/{len(claims)} claims found in the README")
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
    print("every published number matches a committed artefact")
    return 0


if __name__ == "__main__":
    sys.exit(main())
