# Reproducing and layout

[Back to the README](../README.md)

## Reproducing

Expected runtime: the audit and the tests finish in under a minute on a fresh
clone. The full rebuild needs raw ProteinGym, and a GPU for step 8; the step
times the artefacts record are under "How long" below.

```bash
pip install -r requirements.txt

# 1. additivity residuals for every assay (writes results/pairs/*.csv)
python scripts/build_pairs.py \
    --data <ProteinGym dir with <assay>.csv> \
    --index <DMS_substitutions.csv>

# 2. the per-assay summary, rebuilt from the pair tables of step 1 (build_pairs
#    writes it too; this is the cheap re-run after a summary change)
python scripts/rebuild_summary.py

# 3. the magnitude prior and the null that withdraws it
python scripts/prior.py
python scripts/invariant_stats.py

# 4. the depth probes: position, residue, centred prior
python scripts/depth.py

# 5. does a stronger baseline close the gap? (per-position and rank-k
#    position-pair terms, out-of-fold, each against its own permutation null)
python scripts/alt_baseline.py --n-perm 10 --workers 8

# 5b. the two harder tests of the position term: whole position pairs held out,
#     then a monotone global curve fitted first
python scripts/position_holdout.py --n-perm 10 --workers 9
python scripts/monotone_fit.py --n-perm 5 --workers 6

# 6. the held-out checks: one lab or many, and the assays excluded by size
#    (--index defaults to $EPSCEIL_INDEX, then ../_data/DMS_substitutions.csv)
python scripts/heldout.py --index <DMS_substitutions.csv>
python scripts/seed_sweep.py     # the other-labs AUC verdict over seeds 0-199 (DESIGN.md)

# 6b. both centring conventions, and the two-number recalibration
#     (defaults: $EPSCEIL_DATA / $EPSCEIL_INDEX, then ../_data/)
python scripts/reference.py --data <ProteinGym dir> --index <DMS_substitutions.csv>

# 7. why a pooled rho across assays is not a within-assay rho
python scripts/prior_units.py

# 8. the ESM-2 probe (GPU; needs a parent-sequence index). esm_context.py is the
#    original 4.4% probe kept for comparison; esm_scale.py is the corpus-scale run
#    with three heads and a 100% sensitivity pass
python scripts/esm_context.py --index <DMS_substitutions.csv>
python scripts/esm_scale.py --max-per-assay 50000 --sensitivity-caps 0
python scripts/esm_tcrit.py      # the same verdicts under t(4), beside 2 SE

# 9. the figures, then the claim audit and the tests (figures write the r values
#    the README quotes, so they run before the audit)
python scripts/make_figures.py
python scripts/audit_claims.py --strict
pytest tests/
```

**How long.** On the committed runs, by the runtimes each artefact records:
step 5 took 327 s on 8 workers, `position_holdout.py` 427 s on 9 and
`monotone_fit.py` 52 s on 6. The audit and the tests finish in well under a
minute. Step 8 needs a GPU.

**What needs what.** Everything that reads the pair tables — steps 2 to 5b, 6b,
7 and the figures — needs the gitignored `results/pairs/` from step 1, and on a
fresh clone `rebuild_summary.py`, `prior_units.py` and `make_figures.py` stop
with exit 1 and a message naming that directory and `build_pairs.py` rather than
writing anything partial. With committed files alone a clone can run the audit,
`esm_tcrit.py` and the tests, which is what CI does.

The audit is also a test, and CI runs it with `--strict`. Every audited number is
bound to the sentence or table cell it belongs in (an *anchor*), so a right number
in the wrong row fails, not just a number missing from the page; a missing
artefact fails rather than shrinking the check; and `tests/test_claims.py` mutates
the README in memory (a wrong value, two swapped rows, a flipped sign) and
requires the audit to reject each.

## Layout

```
src/epsceil/pairs.py        mutation parsing, additivity residual, per-assay scaling
src/epsceil/baseline.py     position and rank-k position-pair baselines
scripts/build_pairs.py      every assay -> results/assay_summary.csv + pairs/
scripts/rebuild_summary.py  per-assay summary from the pair tables alone
scripts/prior.py            the magnitude prior + its permutation null
scripts/invariant_stats.py  shift-invariant correlations, AUC, bias accounting
scripts/depth.py            position, residue and centred-prior probes
scripts/alt_baseline.py     stronger baselines, out-of-fold, with permutation nulls
scripts/position_holdout.py whole-position-pair holdout of the position models
scripts/monotone_fit.py     isotonic / polynomial global curve vs the position term
scripts/heldout.py          one-lab-in-the-corpus and size-excluded assays
scripts/seed_sweep.py       that verdict over 200 fixed seeds (rule in DESIGN.md)
scripts/reference.py        both centring conventions; two-number recalibration
scripts/prior_units.py      within-assay vs pooled: what a rho is measured over
scripts/esm_context.py      the GPU probe, grouped CV by assay (4.4% sample)
scripts/esm_scale.py        corpus-scale ESM probe, three heads, two controls
scripts/esm_tcrit.py        the ESM verdicts under t(4) beside the 2 SE rule
scripts/make_figures.py     figures from committed artefacts
scripts/audit_claims.py     every README number against a committed artefact
tests/                      invariants, estimator labels, the claim audit
tests/test_alt_baseline.py  the stronger-baseline table's nulls and means
tests/test_esm_scale.py     the ESM verdict rule and its controls
tests/test_position_holdout.py  both holdout verdicts re-derived from per-assay rows
```

MIT licence for this repository's own code. ProteinGym data, the ESM-2 weights
and UniProt keep their own terms; see [NOTICE](NOTICE).

## Data notes

- **`results/pairs/` is gitignored** (reproducible from raw ProteinGym with step 1).
  That used to mean a fresh clone could not check the pooled fraction.
  `assay_summary.csv` now carries the integer counts (`n_outside_2sd`,
  `n_outside_2sd_centred`, `n_within_1sd`) beside every fraction, so the pooled
  number is `n_outside_2sd.sum() / n_pairs.sum()` — computable in a clone from a
  committed file, and exact where a fraction times a row count only round-trips
  by luck. A missing artefact now fails the audit instead of shrinking it.
- **Two assays were skipped** (F7YBW8_MESOW_Aakre_2015, GCN4_YEAST_Staller_2018)
  for having too few singles or doubles by the stated threshold; see
  `results/skipped.txt`.
