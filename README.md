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
| coverage of doubles by their singles | **≥93.6%** per assay |
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
| **\|additive prediction\|** (how dramatic the singles look) | **−0.260** |
| signed additive prediction | +0.379 |
| observed score | +0.041 |
| sequence separation | −0.010 |
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

## Can a protein language model do better?

Two probes, both with GroupKFold cross-validation **by protein** (pairs from one
protein share a sequence and a scale, so a random split would leak):

- **Distance alone**: ρ = −0.03, MAE worse than a constant by 0.086 sd. Nothing.
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
  measured is excluded, which biases toward the parts of sequence space the DMS
  library happened to cover. 93.6%–100% coverage per assay makes this small but
  not zero.
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
