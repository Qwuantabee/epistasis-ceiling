# Design notes

Rules written down before the run they govern. Each section names the script that
applies it and the artefact it writes.

## Seed-aggregated verdict: other labs' AUC against the corpus median

**Why.** `results/heldout.json` records one percentile-bootstrap interval for the
median AUC of the 15 non-Tsuboyama assays (seed 20261116, 20,000 reps) and the
verdict `NOT_SUPPORTED` (the interval excludes the corpus median). With 15 assays
the bootstrap median can land on only a few of the 15 values, so the lower bound
jumps between two of them and the verdict depends on the seed. A one-seed verdict
is not a property of the data. This rule replaces it with a statement over a fixed
set of seeds.

**Disclosure.** This rule is not blind. A review sweep over seeds 0-199 (outside
the repo) already reported that the lower bound falls below the corpus median on
most seeds. The rule below was chosen so that a verdict is published only if it
holds on nearly every seed, whichever way that comes out.

**Inputs, unchanged.** `results/assay_summary.csv`, assays with `n_pairs >= 500`;
the 15 whose assay id does not contain `Tsuboyama` are "other labs"; the corpus
median is the median `auc_direction` over all of those assays. The per-seed
interval and verdict are computed by the unmodified `heldout.boot_median`
(20,000 reps) and `heldout.verdict`. `results/heldout.json` is not rewritten.

**Seeds.** The 200 integers 0, 1, ..., 199, committed as a list in
`results/seed_sweep_seeds.txt` before the run. No seed is added or dropped after.

**Per seed s.** Interval `[lo_s, hi_s]`; verdict `v_s`; clearance
`c_s = lo_s - corpus` when the group median is above the corpus median, else
`corpus - hi_s`. A positive clearance means the interval excludes the corpus
median on the side of the group.

**Reported.** The share of seeds for each verdict; the median of `c_s` across
seeds and its 2.5 and 97.5 percentiles across seeds (`numpy.percentile`,
linear interpolation); the same three numbers for `lo_s` and `hi_s`.

**Verdict rule.**
1. If one verdict holds on at least 190 of the 200 seeds (95%), that is the
   seed-aggregated verdict.
2. Otherwise the seed-aggregated verdict is `INCONCLUSIVE (seed-dependent)`, and
   the README states the share of seeds behind each verdict instead of any single
   one.

The 95% bar matches the coverage of the interval itself: a verdict that a fresh
seed overturns more than one time in twenty is not stated as a finding.

**Script and artefact.** `scripts/seed_sweep.py` writes
`results/seed_sweep.json` (per-seed rows, the aggregate, and this rule's text).
