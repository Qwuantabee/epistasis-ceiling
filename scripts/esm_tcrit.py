#!/usr/bin/env python3
"""The ESM verdicts under a t critical value, beside the preregistered 2 SE rule.

WHY THIS EXISTS. scripts/esm_scale.py judges each head by delta +/- 2 SE over
K = 5 folds. With 4 degrees of freedom that is not a 95% interval: P(|T4| < 2) is
about 0.88. Review round 2 (science F6) recomputed the cells with t(0.975, 4) and
found that two of the three NOT_SUPPORTED cells become INCONCLUSIVE.

The preregistered rule is NOT changed retroactively: esm_scale.json keeps its
verdicts, and the README keeps them in their own column, labelled as the rule
committed in b323e50 before any head was fitted. This script only adds a second
column computed from the same stored per-fold arrays, so a reader can see which
labels depend on the critical value.

Reads results/esm_scale.json, writes results/esm_tcrit.json. CPU only, seconds.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"


def judge(lo: float, hi: float, min_gain: float) -> str:
    """The preregistered decision rule, with the interval passed in."""
    if lo > 0:
        return "SUPPORTED"
    if hi < min_gain:
        return "NOT_SUPPORTED"
    return "INCONCLUSIVE"


def cell(head: dict, crit: float) -> dict:
    """Recompute one (head, sample size) cell from its per-fold rho arrays."""
    v = head["verdict"]
    rho = head["rho"]
    cstar = v["control"]
    d = np.asarray(rho["esm"], float) - np.asarray(rho[cstar], float)
    k = len(d)
    delta = float(d.mean())
    se = float(d.std(ddof=1) / math.sqrt(k))
    # the stored numbers must be what the per-fold arrays give, or this column
    # would be judging a different run than the preregistered one
    if not (math.isclose(delta, v["delta"], abs_tol=1e-12)
            and math.isclose(se, v["se"], abs_tol=1e-12)):
        raise SystemExit("stored delta/se do not match the per-fold arrays")
    lo2, hi2 = delta - 2 * se, delta + 2 * se
    lot, hit = delta - crit * se, delta + crit * se
    pre = judge(lo2, hi2, v["min_gain"])
    if pre != v["verdict"]:
        raise SystemExit("recomputed 2 SE verdict differs from the stored one")
    return {"control": cstar, "k_folds": k, "delta": delta, "se": se,
            "interval_2se": [lo2, hi2], "verdict_2se_preregistered": pre,
            "interval_t": [lot, hit], "verdict_t": judge(lot, hit, v["min_gain"]),
            "min_gain": v["min_gain"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--esm", default=str(RES / "esm_scale.json"))
    ap.add_argument("--out", default=str(RES / "esm_tcrit.json"))
    args = ap.parse_args()
    p = Path(args.esm)
    if not p.exists():
        raise SystemExit(f"{p} not found; run scripts/esm_scale.py first")
    e = json.loads(p.read_text())

    runs = {"primary": e["primary"], "full": (e.get("sensitivity") or [None])[0]}
    k = len(e["primary"]["heads"]["a"]["rho"]["esm"])
    crit = float(stats.t.ppf(0.975, k - 1))
    coverage_2se = float(stats.t.cdf(2, k - 1) - stats.t.cdf(-2, k - 1))
    out = {"question": "Which ESM verdicts depend on using 2 SE rather than a t "
                       "critical value over K folds?",
           "k_folds": k, "df": k - 1, "t_crit_0975": crit,
           "coverage_of_2se_interval": coverage_2se,
           "preregistered_rule": "delta +/- 2 SE (esm_scale.py VERDICT_RULE, "
                                 "committed in b323e50); unchanged",
           "t_rule": "the same decision rule with delta +/- t(0.975, K-1) SE",
           "cells": {}}
    for run_name, run in runs.items():
        if run is None:
            continue
        frac = (run.get("selection") or {}).get("fraction_of_eligible_pairs")
        for hid in ("a", "b", "c"):
            h = (run.get("heads") or {}).get(hid)
            if h is None:
                continue
            out["cells"][f"{run_name}:{hid}"] = {
                "run": run_name, "head": hid,
                "fraction_of_eligible_pairs": frac, **cell(h, crit)}
    changed = [c for c, v in out["cells"].items()
               if v["verdict_t"] != v["verdict_2se_preregistered"]]
    out["cells_whose_label_changes"] = changed
    out["n_cells_supported_under_either_rule"] = sum(
        1 for v in out["cells"].values()
        if "SUPPORTED" in (v["verdict_t"], v["verdict_2se_preregistered"]))
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(f"K={k}  t(0.975,{k - 1})={crit:.3f}  coverage of 2 SE={coverage_2se:.3f}")
    for c, v in out["cells"].items():
        print(f"  {c:10s} 2SE {v['verdict_2se_preregistered']:14s} "
              f"t {v['verdict_t']:14s} t-interval [{v['interval_t'][0]:+.3f}, "
              f"{v['interval_t'][1]:+.3f}]")
    print(f"labels that change: {changed}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
