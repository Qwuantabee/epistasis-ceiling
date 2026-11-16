# How far do single-mutant measurements get you on a double mutant?

**On 818,362 measured double mutants from 65 assays, adding the two single-mutant
effects ranks the double correctly (AUC 0.884) but is biased: the
additive prediction sits a median **0.94 assay-sd
away from the truth**, so a fixed tolerance derived from it is not a reliability
measure. Where it fails worst is where the singles look **most dramatic** — rho =
+0.313 on a shift-invariant statistic, positive in
47/48 assays.**

![Additivity ceiling](figures/ceiling.png)

Protein engineers screen singles because they are cheap and doubles explode
combinatorially: 1,000 singles imply ~500,000 pairs. The bet is that effects add.
This repo measures exactly how good that bet is, on the largest public collection
of measured doubles (ProteinGym, 1.77M multi-mutants), and then asks the only
useful follow-up: **can you tell in advance which pairs will break?**

## The measurement

For every double mutant whose two singles were also measured:

```
residual = observed(double) − ( single_a + single_b − centre )
```

`centre` is the assay's own median, so the baseline is invariant to an additive
shift in the assay scale. Residuals are then divided by each assay's own residual
sd, because ProteinGym assays are **not** on a common scale — the Somermeyer
deep-mutational scans report raw fluorescence in the thousands (residual sd up to
6,098) against a median of 0.69 elsewhere. Pooling those raw would have let three
assays dominate every aggregate number.

| | |
|---|---|
| assays with both singles and doubles | **65** |
| pairs with both singles measured | **818,362** |
| coverage of doubles by their singles | **30.2%–100%**, median **98.5%** |
| median assay Spearman (additive vs observed) | **0.760** |
| median direction agreement (median split) | **0.693** |
| pairs beyond 2 sd from additive, per-assay **median** | **12.6%** |
| median assay, fraction of pairs within 1 sd | **0.437** |

So additivity is a good *ranker* (AUC 0.884) and a **biased**
predictor: its median error is 0.94 assay-sd,
and 12.6% of pairs in a typical assay sit
more than 2 sd from it. An earlier version of this README said 41% — that was a
pooled figure printed beside per-assay medians, and most of it was the model's
systematic offset rather than its spread (see the correction below).

## A withdrawal, and what does survive

**The magnitude prior is withdrawn.** An earlier version of this README claimed
"additivity fails where the singles look mild". That claim is dead, and the reason
matters more than the claim did.

`residual = observed - predicted`, so `|residual|` contains `|predicted|` **by
construction**. Any correlation between the two is therefore partly structural, and
the two obvious controls disagree:

| statistic | median across 48 assays |
|---|---|
| observed rho(\|residual\|, \|prediction\|), both centred | +0.313 |
| null: shuffle the residual alone | +0.003 |
| **null: shuffle `observed` and REBUILD the residual** (the correct one) | **+0.505** |

The correct null preserves the construction and destroys only the pairing — and it
is **larger** than the observed value. Observed exceeds it in only
11/48 assays. So the statistic carries no usable
signal about epistasis in either direction, and reporting it with a flipped sign
would have been just as wrong as the original.

Two further reasons it was never trustworthy, both found by review: the additive
baseline is identified only up to one constant per assay, and sweeping that
constant over [-2, +2] kcal/mol swings the correlation from **-0.597 to +0.597** on
a single assay; and the simulation null that appeared to validate it drew
mean-zero noise, so it could not in principle detect the offset that produced the
sign.

**What replaces it:** the two priors that have valid nulls and survive them, in the
next section. Neither is as quotable as the withdrawn one, which is the point.

### The additive model is biased, which changes what "a fixed tolerance" means

The residual mean is **0.94 assay-sd** per assay,
not zero. A "fraction beyond a fixed number of sds" is then measuring that offset,
not the model's spread — and a pooled fraction is not comparable to a per-assay
median:

| how it is counted | value |
|---|---|
| pooled share beyond 2 sd — **what this README used to call 40.8%** | 40.9% |
| per-assay **median** share beyond 2 sd | 12.6% |
| per-assay median after centring each assay | 4.6% |

The honest figure is the middle row, **12.6%**;
the 41% was that number blended with the model's systematic bias. What additivity
*is* good at is ranking: **AUC 0.884** against the observed effect,
which is why screening by additive prediction still beats not screening at all.

## Where the failures concentrate

The magnitude prior above is one signal. Two further probes, each with a null:

**Positions matter, and this is the strongest prior in the repo.** The share of
residual variance explained by sequence position alone is **0.140**
against a permutation null of 0.002 (analytic
0.002), and it exceeds that null in
**47/48 assays**. Roughly one seventh of the
epistasis in these data is attributable to *which residues are involved*, not how
mild their singles look — so a lab that has already run a double-mutant
experiment can flag its hot positions for the next round at no extra cost.

**Residue identity does matter, though modestly.** With both substitution slots
pooled (the earlier version read only slot `a`) and a binomial test instead of a
threshold picked by eye, alanine sits above its own assay's median residual in
**73% of 48 assays**
(Bonferroni p = 0.042), and the residue ranking replicates at
rho +0.87 across a split-half. Verdict **INCONCLUSIVE**. An earlier draft called
this NOT_SUPPORTED on an arbitrary 85% bar — that bar was not a test.

Neither probe explains away the earlier finding; they are different axes of the
same residual.

### Two ways this could have been manufactured, and why it wasn't

Both were tested before the reviewer round, because each would have produced the
findings above out of nothing:

- **Per-assay normalisation.** Dividing each assay's residual by its own sd is a
  scalar per assay, so it cannot change a within-assay rank correlation.
  Recomputing on the raw residuals gives **-0.260 raw against -0.260 normalised**,
  with a maximum difference of exactly 0.0 across all 48 assays. The normalisation
  makes assays comparable in spread; it does not create the effect.
- **Position clustering as a count artefact.** If a high r² simply meant "few
  pairs per position", it would track position count. It does not: ρ = +0.027
  against pairs-per-position and -0.055 against number of positions, and 44 of 48
  assays still exceed five times the analytic null. The clustering is real.

## Can a protein language model do better?

Two probes, both with GroupKFold cross-validation **by protein** (pairs from one
protein share a sequence and a scale, so a random split would leak):

- **Distance alone**: rho = -0.03, MAE worse than a constant by 0.086 sd. Nothing.
- **ESM-2 (650M) context**, window embeddings of both mutated positions plus
  their difference, product and sequence separation — 5,121 features, 35,110
  pairs: rho = **+0.064 ± 0.127**. **The MAE figure from that run is withdrawn**:
  a pure-noise prediction scores -0.11 sd on the same metric, so the reported
  -1.43 sd was dominated by a scale mismatch in the estimator, not by the model.
  The rho is the usable number and it is indistinguishable from zero.

**Sequence context does not tell you where additivity breaks.** That is a
negative result and it is reported as one: a 650M-parameter protein language model
does not beat "predict the training median" at ranking which double mutants will
violate additivity. The plain magnitude prior above is the better tool, and it is
free.

## Reproducing

```bash
pip install -r requirements.txt

# 1. additivity residuals for every assay (writes results/pairs/*.csv)
python scripts/build_pairs.py \
    --data <ProteinGym dir with <assay>.csv> \
    --index <DMS_substitutions.csv>

# 2. the ceiling and the per-assay table
#    (build_pairs writes results/assay_summary.csv)

# 3. the prior, with its simulation null
python scripts/prior.py

# 4. the ESM-2 probe (GPU; needs a parent-sequence index)
python scripts/esm_context.py --index <DMS_substitutions.csv>

pytest tests/
```

## Limits, stated plainly

- **Additivity is not the only null model.** A model that fits single-mutant
  effects with a per-position term would beat plain additivity; this repo measures
  the crudest baseline first, deliberately, because that is what a lab actually
  uses when it "just adds the singles".
- **ProteinGym is not one experiment.** Assays differ in scale, saturation and
  what "fitness" means (fluorescence, growth, binding). The normalisation makes
  them comparable in *spread*, not in *meaning*; the per-assay table is the real
  result and the pooled numbers are summaries.
- **Coverage is per assay, not global.** A double whose singles were never
  measured is excluded, which biases toward whatever the DMS library covered. The
  median assay covers 98.5% of its doubles, but the range is wide and one assay
  (F7YBW8_MESOW_Ding_2023) covers only **30.2%** (26 of 86). An earlier draft of
  this README said "≥93.6% per assay", which the coverage column contradicts —
  caught by `scripts/audit_claims.py`.
- **The ESM probe is one architecture, one feature set and one window size.** It
  shows *this* model does not rank epistasis from context; it does not prove no
  model can. A trained supervised model on matched data is the obvious next test,
  and it was not run.
- **Two assays were skipped** (F7YBW8_MESOW_Aakre_2015, GCN4_YEAST_Staller_2018)
  for having too few singles or doubles by the stated threshold; see
  `results/skipped.txt`.

## Layout

```
src/epsceil/pairs.py        mutation parsing, additivity residual, per-assay scaling
scripts/build_pairs.py      every assay -> results/assay_summary.csv + pairs/
scripts/prior.py            the magnitude prior + its simulation null
scripts/esm_context.py      the GPU probe, grouped CV by protein
scripts/make_figures.py     figures from committed artefacts
tests/                      invariants: parsing, sign, normalisation, null
```

MIT licence for this repository's own code. ProteinGym data keeps its own terms.
