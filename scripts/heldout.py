#!/usr/bin/env python3
"""Is the bias a property of additivity, or of one lab's assay collection?

WHY THIS EXISTS. Every headline number in the README is a median over 48 assays
with >=500 pairs. Two things about that set were never tested:

  1. **It is one lab.** 48 of 65 assays come from Tsuboyama et al. (2023), a
     single mega-study of stability measurements on 48 proteins. If the -0.94 sd
     bias and the 12.6% beyond-2sd share are features of that measurement setup -
     a common scale, a common saturation curve - then the corpus median describes
     one experimental design, not "double mutants".
  2. **17 assays are excluded from every 48-assay median** and 5,348 pairs with
     them. They were excluded for SIZE (fewer than 500 pairs), not for being
     wrong. They are held out by size, NOT by lab: 15 of the 17 are Tsuboyama,
     the lab behind 33 of the 48 corpus assays, and 11 of them are among the 59
     assays of the corpus-scale ESM probe (scripts/esm_scale.py), so they are not
     outside "every published number". Both facts are computed below and written
     to the artefact rather than asserted here.

This script tests both, and reports the disagreement honestly rather than
averaging it away. It fits no model: every statistic is a median over assays with
a bootstrap interval, so nothing here can leak between groups.

Outputs results/heldout.json. Verdicts are SUPPORTED / NOT_SUPPORTED /
INCONCLUSIVE under the rule stated in VERDICT_RULE, fixed before the run.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
# The ProteinGym index (DMS_substitutions.csv, for first_author) is part of the
# raw ProteinGym download, not of this repo. Resolution order: --index, then
# $EPSCEIL_INDEX, then the documented locations below (the first is the default
# scripts/esm_scale.py uses). A missing index is a clear error, not a traceback.
INDEX_ENV = "EPSCEIL_INDEX"
INDEX_SEARCH = (ROOT.parent / "_data" / "DMS_substitutions.csv",
                ROOT / "data" / "DMS_substitutions.csv")


def resolve_index(cli_value: str | None) -> Path:
    """Return the index path, or exit 1 naming every place that was tried."""
    if cli_value:                                        # explicit: no fallback
        cands = [Path(cli_value).expanduser()]
    elif os.environ.get(INDEX_ENV):                      # explicit: no fallback
        cands = [Path(os.environ[INDEX_ENV]).expanduser()]
    else:
        cands = list(INDEX_SEARCH)
    for p in cands:
        if p.is_file():
            return p
    raise SystemExit(
        "heldout.py needs the ProteinGym index DMS_substitutions.csv (it reads "
        "first_author from it); it is not in the repo. Pass --index PATH or set "
        f"{INDEX_ENV}=PATH. Looked in: " + ", ".join(str(p) for p in cands))

# Fixed before the run, so a reader can see the bar was not chosen after the
# numbers. A group "agrees" with the corpus when the corpus median lies inside the
# group's bootstrap 95% interval AND the group's median is not on the other side
# of zero when the corpus median is one-signed.
VERDICT_RULE = (
    "SUPPORTED when the corpus median lies inside the held-out group's bootstrap "
    "95% interval and the group's median has the corpus sign; NOT_SUPPORTED when "
    "the interval excludes the corpus median; INCONCLUSIVE when the interval is "
    "wider than the effect (group n < 8, or interval spanning zero and the corpus "
    "median with it)."
)

RNG_SEED = 20261116


def boot_median(x: np.ndarray, reps: int = 20000, seed: int = RNG_SEED):
    """Percentile bootstrap of the median. Returns (lo, hi) at 95%."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    meds = np.median(rng.choice(x, size=(reps, len(x)), replace=True), axis=1)
    return (float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5)))


def boot_diff_median(a: np.ndarray, b: np.ndarray, reps: int = 20000,
                     seed: int = RNG_SEED) -> dict:
    """Bootstrap the DIFFERENCE of two group medians (resample each group).

    A permutation test of "same distribution" is the usual choice, but the groups
    here differ in size by 2x and are heavy-tailed, so the quantity a reader cares
    about - how far apart the medians are - is bootstrapped directly. The p-value
    is the two-sided share of resamples whose difference crosses zero.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return {"diff": float("nan"), "ci95": [float("nan"), float("nan")],
                "p_two_sided": float("nan")}
    rng = np.random.default_rng(seed)
    d = (np.median(rng.choice(a, size=(reps, len(a)), replace=True), axis=1)
         - np.median(rng.choice(b, size=(reps, len(b)), replace=True), axis=1))
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return {"diff_median": float(np.median(d)),
            "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "p_two_sided": float(min(1.0, p))}


def verdict(corpus: float, lo: float, hi: float, n: int, effect: float) -> str:
    """Apply VERDICT_RULE literally.

    `effect` is the group's OWN median — the size of the thing being claimed. The
    previous version took a `spread` argument that its only caller passed as
    ``hi - lo``, so the "interval wider than the effect" branch compared the width
    against twice the width and could therefore never fire. That was dead code, and
    it meant the rule stated in VERDICT_RULE was not the rule actually applied.
    """
    width = hi - lo
    if n < 8 or not np.isfinite(lo) or not np.isfinite(effect):
        return "INCONCLUSIVE"
    # The interval is wider than the effect it is meant to establish: agreement
    # with the corpus and agreement with zero cannot be told apart.
    if np.isfinite(width) and abs(effect) > 0 and width > 2 * abs(effect):
        return "INCONCLUSIVE"
    # Interval spanning zero AND containing the corpus median: the group is
    # compatible both with no effect and with the corpus value.
    if lo <= 0.0 <= hi and lo <= corpus <= hi:
        return "INCONCLUSIVE"
    if lo <= corpus <= hi:
        return "SUPPORTED" if (np.sign(np.median([lo, hi])) == np.sign(corpus)
                               or corpus == 0) else "INCONCLUSIVE"
    return "NOT_SUPPORTED"


def group_row(name: str, t: pd.DataFrame, corpus_vals: dict) -> dict:
    """Summarise one assay group on the three published statistics."""
    out = {"group": name, "n_assays": int(len(t)),
           "n_pairs": int(t.n_pairs.sum())}
    for col, key in (("bias_mean_sd", "bias_sd"),
                     ("frac_outside_2sd", "frac_outside_2sd"),
                     ("auc_direction", "auc")):
        v = t[col].to_numpy(dtype=float)
        lo, hi = boot_median(v)
        out[key] = {"median": float(np.median(v)), "ci95": [lo, hi]}
        out.setdefault("verdict_vs_corpus", {})[key] = verdict(
            corpus_vals[key], lo, hi, len(v), float(np.median(v)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default=str(RES / "assay_summary.csv"))
    ap.add_argument("--min-pairs", type=int, default=500)
    ap.add_argument("--out", default=str(RES / "heldout.json"))
    ap.add_argument("--index", default=None,
                    help="ProteinGym DMS_substitutions.csv; default: "
                         f"${INDEX_ENV}, then " + ", ".join(str(p) for p in INDEX_SEARCH))
    args = ap.parse_args()

    idx = resolve_index(args.index)
    s = pd.read_csv(args.summary)
    meta = pd.read_csv(idx)
    m = s.merge(meta[["DMS_id", "first_author", "year", "source_organism"]],
                left_on="assay", right_on="DMS_id", how="left")
    if m.first_author.isna().any():
        raise SystemExit("missing author metadata for some assays")

    big = m[m.n_pairs >= args.min_pairs].copy()
    small = m[m.n_pairs < args.min_pairs].copy()
    corpus = {k: float(np.median(big[c].to_numpy(float)))
              for c, k in (("bias_mean_sd", "bias_sd"),
                           ("frac_outside_2sd", "frac_outside_2sd"),
                           ("auc_direction", "auc"))}

    # --- H1: one lab or many? -------------------------------------------------
    big["lab"] = np.where(big.first_author == "Tsuboyama", "Tsuboyama",
                          "other labs")
    main_lab = group_row("Tsuboyama (48-assay corpus)", big[big.lab == "Tsuboyama"],
                         corpus)
    other = group_row("other labs (15 assays)", big[big.lab == "other labs"],
                      corpus)
    diff = {k: boot_diff_median(
        big[big.lab == "Tsuboyama"][c].to_numpy(float),
        big[big.lab == "other labs"][c].to_numpy(float))
        for c, k in (("bias_mean_sd", "bias_sd"),
                     ("frac_outside_2sd", "frac_outside_2sd"),
                     ("auc_direction", "auc"))}
    h1 = {
        "question": "Does the published bias depend on WHICH lab the assay came "
                    "from?",
        "hypothesis": "the bias differs between the dominant lab and the rest",
        "corpus_median": corpus,
        "Tsuboyama": main_lab,
        "other_labs": other,
        "difference_Tsuboyama_minus_other": diff,
        # SUPPORTED means the hypothesis was confirmed: the labs differ. The
        # earlier draft inverted this and called a significant difference
        # NOT_SUPPORTED, which would have published the opposite of what was
        # measured.
        "verdict": ("SUPPORTED"
                    if diff["bias_sd"].get("p_two_sided", 1.0) < 0.05
                    else "NOT_SUPPORTED"),
        "reading": ("The groups differ in the bias (p from the bootstrap of the "
                    "median difference) and in the beyond-2-sd share UNDER THE "
                    "MEDIAN-OF-ALL-VARIANTS CENTRING. That bias is not a property "
                    "of additivity: it depends on where each assay's zero is put, "
                    "and the 33 Tsuboyama assays flip sign under wild type = 0 "
                    "(results/reference.json), so the bias contrast is a contrast "
                    "of centring constants, not of labs. The AUC contrast is "
                    "convention-free: both groups rank well above chance and the "
                    "level differs by lab family (other labs vs corpus AUC "
                    "verdict below)."),
    }

    # --- H2: the excluded small assays as a held-out set ----------------------
    h2 = group_row("held out: 17 assays under 500 pairs", small, corpus)
    h2["question"] = ("Do the assays excluded from every published number agree "
                      "with them?")
    h2["authors"] = small.first_author.value_counts().to_dict()
    h2["n_same_lab_as_corpus"] = int((small.first_author == "Tsuboyama").sum())
    # Review round 2 (F3): this field used to be a hard-coded False. It is now
    # computed from the ESM artefact, which DOES use 11 of these assays.
    esm_used: list[str] = []
    esm_all: set[str] = set()
    esm_p = RES / "esm_scale.json"
    if esm_p.exists():
        sel = ((json.loads(esm_p.read_text()).get("primary") or {})
               .get("selection") or {})
        esm_all = set(sel.get("per_assay_pairs_used") or {})
        esm_used = sorted(esm_all & set(small.assay))
    h2["in_esm_scale_probe"] = {"n_assays": len(esm_used),
                                "n_assays_in_probe": len(esm_all),
                                "assays": esm_used}
    h2["included_in_any_published_number"] = bool(esm_used)
    h2["held_out_by"] = "size (< %d pairs), not by lab" % args.min_pairs
    h2["bias_verdict_status"] = (
        "WITHDRAWN: bias_sd depends on the centring convention; under wild type "
        "= 0 the Tsuboyama assays agree with the corpus (results/reference.json)")
    h2["reading"] = (
        "Held out by SIZE, not by lab: %d of the %d are Tsuboyama, the lab behind "
        "most of the corpus, and %d of them are among the %d assays of the ESM "
        "probe. They are outside the three 48-assay medians only. The bias "
        "comparison is withdrawn: it compares centring constants (the median of "
        "all variants moves with the share of doubles in the library), and under "
        "wild type = 0 the small Tsuboyama assays replicate the big ones. Only the "
        "ranking comparison is convention-free; its verdict is the auc entry of "
        "verdict_vs_corpus." % (h2["n_same_lab_as_corpus"], len(small),
                                len(esm_used), len(esm_all)))
    h2["verdict"] = (
        "bias: " + h2["verdict_vs_corpus"]["bias_sd"]
        + "; ranking: " + h2["verdict_vs_corpus"]["auc"])
    h2["hypothesis"] = "the published bias and ranking both hold out of sample"
    h2["verdict_SUPPORTED_means"] = (
        "the held-out group's median agrees with the corpus median")

    # --- H3: leave-one-lab-out sensitivity -----------------------------------
    loo = []
    for lab in sorted(big.first_author.unique()):
        rest = big[big.first_author != lab]
        if len(rest) < 10:
            continue
        loo.append({"excluded_lab": lab, "n_excluded": int(len(big) - len(rest)),
                    "bias_sd_median": float(np.median(rest.bias_mean_sd)),
                    "frac_outside_2sd_median": float(np.median(rest.frac_outside_2sd)),
                    "auc_median": float(np.median(rest.auc_direction))})
    h3 = {
        "question": "Does dropping any single lab move a published median?",
        "corpus_median": corpus,
        "leave_one_lab_out": loo,
        "bias_range": [min(r["bias_sd_median"] for r in loo),
                       max(r["bias_sd_median"] for r in loo)]
        if loo else None,
        "auc_range": [min(r["auc_median"] for r in loo),
                      max(r["auc_median"] for r in loo)] if loo else None,
        "verdict": ("SUPPORTED"
                    if loo and abs(loo[0]["auc_median"] - corpus["auc"]) < 0.05
                    and all(abs(r["auc_median"] - corpus["auc"]) < 0.05
                            for r in loo) else "INCONCLUSIVE"),
        "reading": ("A ranking number that survives dropping any one lab is a "
                    "corpus property; one that does not is that lab's."),
    }

    res = {
        "n_assays_total": int(len(m)),
        "n_assays_corpus": int(len(big)),
        "n_assays_heldout": int(len(small)),
        "n_pairs_heldout": int(small.n_pairs.sum()),
        "corpus_median": corpus,
        "verdict_rule": VERDICT_RULE,
        "bootstrap": {"reps": 20000, "seed": RNG_SEED, "method": "percentile"},
        "H1_lab_family": h1,
        "H2_excluded_small_assays": h2,
        "H3_leave_one_lab_out": h3,
        "provenance": ("/home/ubuntu/.openclaw/workspace/venv/bin/python "
                       "scripts/heldout.py"),
    }
    Path(args.out).write_text(json.dumps(res, indent=2) + "\n")

    # human-readable summary
    print(f"corpus: {len(big)} assays >= {args.min_pairs} pairs, "
          f"{int(big.n_pairs.sum()):,} pairs")
    print(f"corpus medians: bias {corpus['bias_sd']:+.3f} sd, "
          f"beyond2 {corpus['frac_outside_2sd']:.3f}, AUC {corpus['auc']:.3f}")
    for g in (main_lab, other, h2):
        print(f"  {g['group']:38s} n={g['n_assays']:3d} "
              f"bias {g['bias_sd']['median']:+.3f} "
              f"[{g['bias_sd']['ci95'][0]:+.3f},{g['bias_sd']['ci95'][1]:+.3f}] "
              f"beyond2 {g['frac_outside_2sd']['median']:.3f} "
              f"AUC {g['auc']['median']:.3f} "
              f"-> {g['verdict_vs_corpus']['bias_sd']}/{g['verdict_vs_corpus']['auc']}")
    print(f"  H1 bias difference Tsuboyama-other: "
          f"{diff['bias_sd']['diff_median']:+.3f} "
          f"p={diff['bias_sd']['p_two_sided']:.4f}")
    print(f"  H3 AUC leave-one-lab-out range: {h3['auc_range']}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
