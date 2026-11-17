# Can a protein language model do better?

[Back to the README](../README.md)

Three probes. The first two were in the earlier draft; the third is the one that
was actually needed, and the earlier ESM numbers were withdrawn.

- **Distance alone**: ρ = -0.03, MAE worse than a constant by 0.086 sd. Nothing.
- **ESM-2 (650M) alone**, window embeddings of both mutated positions plus their
difference, product and sequence separation: ρ = **-0.003 ± 0.049**, against a
**pure-noise control of +0.007 ± 0.014** run through the identical pipeline (both
mean ± sd across the 5 folds). MAE
is 2% worse than predicting the training median. Sequence context carries no
usable signal here.
- **ESM-2 + the free prior**: ρ = -0.007, i.e. **worse than the prior alone
(+0.231)** by 0.238. Adding 5,121 ESM features to the one variable a lab already
has makes the prediction worse.

The earlier draft reported ρ = +0.064 and an MAE "worse than a constant by 1.43
sd". Both are withdrawn. The MAE was estimator failure, not a property of ESM-2:
a 1.43-sd deficit implies mean absolute error of 1.55 on a target whose median is
1.44, so the model was predicting outside the data range — 5,121 near-collinear
features, a StandardScaler, Ridge alpha=10 and no target transform on a
non-negative right-skewed target. Two further defects: there was no noise control
at all, and the assay list was chosen by alphabetical glob order, which silently
dropped the assay that is 65% of the corpus. The re-run fixes all four and reports
the noise floor beside every number.

**So sequence context does not tell you where additivity breaks — and neither, on
these data, does anything else tested here.** The three candidate priors are all
null: the magnitude of the singles (fails a construction-preserving permutation),
residue identity (rank correlation across assay halves +0.50, and the one
Bonferroni-significant residue falls to 0.52 against a chance 0.50 once magnitude
is removed), and ESM-2 context
(indistinguishable from pure noise, and actively harmful on top of the prior).
What survives is the ranking result, AUC 0.884, which does not let you pick the
pairs in advance.

### The same test at corpus scale

The probe above used 36,000 pairs (**4.4%** of the corpus) and one estimator, so it
could only say "this estimator found nothing". The re-run
(`scripts/esm_scale.py`) covers **59 assays and 319,014 pairs — 39.0%** of the
corpus at a 50,000-pair-per-assay cap, with a **100%** run (817,120 pairs) as a
sensitivity check, and three heads instead of one. The verdict rule was committed
in `b323e50` before any head was fitted.

| head | 39.0% of corpus, 2·SE (preregistered) | 39.0% of corpus, t(4) | 100% of corpus, 2·SE (preregistered) | 100% of corpus, t(4) |
|---|---|---|---|---|
| (a) the shipped PCA(100)+RidgeCV estimator | INCONCLUSIVE | INCONCLUSIVE | NOT_SUPPORTED | NOT_SUPPORTED |
| (b) early-stopped linear on window embeddings, no interaction terms | NOT_SUPPORTED | INCONCLUSIVE | INCONCLUSIVE | INCONCLUSIVE |
| (c) the free prior always present, ESM block on top | INCONCLUSIVE | INCONCLUSIVE | NOT_SUPPORTED | INCONCLUSIVE |

The 2·SE columns are the rule committed in `b323e50` before any head was fitted,
and they stay as preregistered. But 2·SE over five folds is an **88.4%** interval,
not a 95% one; with the t(4) critical value of **2.776** (`scripts/esm_tcrit.py`)
two of the three NOT_SUPPORTED cells become INCONCLUSIVE.

**No head beats its larger control at either sample size, under either rule.**
Head (a) scores +0.026 against a shuffled control of +0.055; (b) -0.017 against
+0.002; (c) +0.031 against +0.058. The nominal gain of ESM over the free prior is
**+0.154 ± 0.084** — but shuffling the ESM rows within an assay gives
**+0.181 ± 0.094**, *more* (here ± is the standard error over the 5 folds, not
the sd used above), so that gain comes from knowing which assay a pair
belongs to, not from the pair. Only head (a) at full corpus excludes a 0.02 gain
under both rules; the other five cells cannot exclude one at t(4). The null is a
statement about this pipeline at corpus scale with five folds of power, not a
demonstration that no sequence model could help.

One defect the re-run exposed is worth publishing because it is a trap any pooled
evaluation of these data will hit: `|predicted|` is in raw assay units while the
residual is in per-assay standard deviations, so pooling pairs across assays lets
the scale differences drive the ordering. Measured directly
(`scripts/prior_units.py`), the free prior's **within-assay** rank correlation is
**+0.313** in raw units and **+0.313** after dividing by each assay's own sd — a
maximum difference of **0.00e+00** across all 48 assays, because a positive
per-assay constant cannot reorder anything inside an assay. Pooled across assays
the same quantity reads **-0.025** in raw units and **+0.105** once each assay is
put on its own scale. A pooled rho that goes negative here is measuring the
pooling, not the prior.

## What the probe leaves untested

**The ESM probe is three heads at corpus scale, still one architecture.**
`esm_scale.py` covers 39.0% of the corpus and a 100% sensitivity pass with three
heads, and no head beats its larger control at either size — so the null is no
longer a statement about a 4.4% sample or a single estimator. What it still does
not show is that no model can: one architecture (ESM-2 650M), one window size
(8 residues each side), wild-type sequence only, and a supervised model trained on matched
data remains the obvious untried test. Two further limits are recorded in the
artefact — overlapping windows for pairs closer than 17 residues mean heads (a)
and (c) do not fully separate window content from window overlap, and a
wild-type window is identical for every substitution pair at the same two
positions, so these features can rank position pairs but not the substitutions
within one.
