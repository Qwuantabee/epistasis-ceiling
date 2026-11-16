# How far do single-mutant measurements get you on a double mutant?

**On 818,362 measured double mutants from 65 assays, the additive shortcut —
predict the double by adding the two singles — misses by more than 2 assay-sd on
41% of pairs, and it resolves only 69% of them in the right direction. The pairs
where it fails are the ones that look least suspicious.**

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
| pairs deviating >2 assay-sd from additive | **40.8%** |
| median assay, fraction of pairs within 1 sd | **0.437** |

So additivity is a decent *ranker* and a poor *predictor*. Spearman 0.76 sounds
respectable until you notice that 41% of individual predictions are more than two
standard deviations out, which is the number that decides whether you must
measure the pair.

## The part that changes what you'd do

Sequence separation — the obvious "are they close in the structure" prior — is
worth nothing here: ρ = −0.010 median across assays.

What does carry information is counter-intuitive, and it took two passes to get
the sign right:

| correlation of \|residual\| with | median across 48 assays |
|---|---|
| **\|additive prediction\|** (how dramatic the singles look) | **-0.260** |
| signed additive prediction | +0.379 |
| observed score | +0.041 |
| sequence separation | -0.010 |
| *simulation null* (observed independent of predicted) | −0.001 ± 0.002 |

**Additivity fails hardest where the singles look mild.** Splitting pairs by
|additive prediction| into terciles, median |residual| is **1.516 sd for the
mildest third and 0.876 for the most dramatic third**. You cannot find the risky
pairs by looking for dramatic single mutations — the quiet ones are where the
shortcut breaks.

The signed +0.379 is a separate asymmetry between negative and positive
predictions. An earlier draft of this repo read it as "big effects are harder",
which the tercile table disproves; both numbers are reported so the misreading
cannot recur. The simulation null rules out the obvious artefact (the residual
contains `predicted` by construction — with the two independent, ρ is 0.001).

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

**Residue identity does not matter.** Testing whether particular substituting
residues carry larger residuals, the strongest candidate (alanine, in
37 assays) sits above its own assay median in only 81% of them — below
the 85% threshold set before looking, and against a 50% null. Verdict **NOT_SUPPORTED**, reported as a
negative rather than dropped.

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
  pairs: ρ = **+0.064 ± 0.127**, MAE **worse** than a constant by 1.43 sd.

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
