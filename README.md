# How far do single-mutant measurements get you on a double mutant?

**On the 48 ProteinGym assays with at least 500 measured double mutants (813,014
pairs), adding the two single-mutant effects ranks the double correctly — median
AUC 0.884.** An earlier version also called the additive prediction biased by a
median 0.94 assay-sd. That is withdrawn: the offset is set by where each assay's
zero is put, not by additivity. For the 33 assays scored as a ΔΔG from wild type
it reads -0.79 sd under the median centring this repo used and +1.15 sd under
wild type = 0, positive in 33/33. Two candidate priors were tested and both
**fail their own null**. The honest answer is that neither how dramatic the singles
look nor which residues they involve tells you in advance where additivity breaks.

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

`centre` is the median of every measured variant in the assay (not a measured
wild type; see the section on the offset below), so the baseline is invariant to
an additive shift in the assay scale. Residuals are then divided by each assay's own residual
sd, because ProteinGym assays are **not** on a common scale — the Somermeyer
deep-mutational scans report raw fluorescence in the thousands (residual sd up to
6,098) against a median of 0.69 elsewhere. Pooling those raw would have let three
assays dominate every aggregate number.

| | |
|---|---|
| assays with both singles and doubles | **65** |
| pairs with both singles measured | **818,362** |
| coverage of doubles by their singles | **30.2%–100%**, median **98.5%** |
| median assay Spearman (additive vs observed) | **0.763** |
| median AUC (threshold-free direction) | **0.884** |
| pairs beyond 2 sd from additive, per-assay **median**, median centring | **12.6%** |
| the same, each assay's own offset removed (convention-free) | **4.6%** |
| median assay, fraction of pairs within 1 sd | **0.431** |

Three denominators meet on this page, stated once here so no row is ambiguous:
**coverage and the pair total use all 65 assays** (818,362 pairs); **every median
— each correlation, AUC and fraction — uses the 48 assays with at least 500
pairs** (813,014 pairs); and the **corpus-scale ESM probe uses 59 assays**, those
with at least 300 pairs and a parent sequence (817,120 eligible pairs). The 17
assays outside the 48 contribute 5,348 pairs between them and cannot move a
median; the earlier version of this README silently mixed the first two.

So additivity is a good *ranker* (AUC 0.884). Whether it is also an offset
predictor cannot be read off these data, because the offset depends on the
centring convention (see the section on the offset below). An earlier
version said "41% of pairs miss by more than 2 sd" — that was a pooled figure
printed beside per-assay medians, and most of it was the model's systematic offset
rather than its spread.

It also said additivity "resolves only 69% of them in the right direction". That
0.693 is a median split at the OBSERVED median while the prediction sits above
that median 76% of the time, so it measures the offset too. Splitting both series
at their own medians gives 0.802, and the threshold-free AUC is **0.884** — which
is the number to quote.

## A withdrawal, and what does survive

**The magnitude prior is withdrawn.** An earlier version of this README claimed
"additivity fails where the singles look mild". That claim is dead, and the reason
matters more than the claim did.

`residual = observed - predicted`, so `|residual|` contains `|predicted|` **by
construction**. Any correlation between the two is therefore partly structural.
Centring both series within assay removes the arbitrary zero point, and then the
honest control — permute `observed` within assay and **rebuild** the residual —
can be applied:

| statistic | median across 48 assays |
|---|---|
| observed rho(\|residual\|, \|prediction\|), both centred | +0.313 |
| null: within-assay permutation of `observed`, residual rebuilt | **0.502** |
| observed minus null | **−0.189** |
| assays where observed exceeds the null | **2/48** |

The null preserves the construction and destroys only the pairing — and it is
**larger** than the observed value. That is expected rather than mysterious:
large `|predicted|` inflates `|residual|` mechanically even when the double is
independent of its singles, so this statistic cannot be positive evidence of
anything. It carries no usable signal about epistasis in either direction, and
reporting it with a flipped sign would have been just as wrong as the original.
Median-centring instead of mean-centring gives +0.373, and the same verdict.

Two further reasons it was never trustworthy: the additive baseline is identified
only up to one constant per assay, and under the median centring the residual
cloud sits a median **-0.94** assay-sd from zero, so the raw statistic tracks where
zero was placed — it is
-0.469 in assays offset 0.75–1.5 sd and +0.058 in assays offset under 0.25. And
the simulation null that appeared to validate it drew mean-zero Gaussian noise,
so it could not in principle detect the offset that produced the sign.

**What replaces it:** the position and residue probes below. Neither closes the
gap, and the section says how far each one gets.

### The additive model's offset is a convention, which changes what "a fixed tolerance" means

The residual mean is not zero, but its size and its sign are set by where each
assay's zero is put, not by additivity. This repo centres each assay on the median
of all its measured variants; under that convention the median bias over the 48
assays is **-0.94** sd. For the 33 assays whose score is a ΔΔG relative to wild
type, the data carry their own reference, wild type = 0, and the same 33 assays
read **-0.79** sd under the median centring (positive in **5/33**) against
**+1.15** sd under wild type = 0 (positive in **33/33**). A number that flips sign
with the choice of origin is not a finding about the model, so the "biased by
0.94 sd" headline is withdrawn (`scripts/reference.py`, `results/reference.json`).
The median of all variants moves with the share of doubles in each library, which
is why it is a poor stand-in for wild type.

What does not depend on the convention: AUC, and every statistic computed after
removing each assay's own offset. A "fraction beyond a fixed number of sds" does
depend on it, and a pooled fraction is not comparable to a per-assay median:

| how it is counted | value |
|---|---|
| pooled share beyond 2 sd — **what this README used to call 40.8%** | 40.9% (332,751 / 813,014) |
| per-assay **median** share beyond 2 sd, median centring | 12.6% |
| per-assay median share beyond 2 sd, zero taken as wild type in every assay (exact for the 33 ΔΔG assays) | 21.0% |
| per-assay median after centring each assay | 4.6% |
| pooled share beyond 2 sd, centred | 5.8% |

The convention-free figure is the centred one, **4.6%**; 12.6% and 21.0% count the
same pairs from two different zeros. What additivity *is* good at is ranking:
**AUC 0.884** against the observed effect, which no per-assay constant can move,
and which is why screening by additive prediction still beats not screening at all.

The same argument, measured per assay rather than asserted pool-wide: under the
median centring, across the 48 assays the share past 2 sd correlates **+0.93**
with how far that assay's additive prediction sits from zero, and only **+0.12**
with its own residual spread.

![The tail tracks the offset, not the spread](figures/bias_not_spread.png)

So "beyond 2 sd" is mostly a restatement of the offset. A tolerance built on it
would flag the assays where the model is shifted, not the pairs where it is wrong.

## Does any of it hold out of sample?

Every median above is over 48 assays. Two things about that set were never tested,
and both matter for anyone who wants to use these results.

**The corpus is mostly one lab.** 48 of the 65 assays are Tsuboyama et al. (2023),
a single study of stability measurements; the other 17 assays come from 13 papers.
Within the 48-assay subset that split is 33 against 15. Both groups rank well
above chance, but not at one level: AUC **0.839** for the 33 Tsuboyama assays
against **0.941** for the 15 others, and the artefact's own verdict for the other
labs' AUC against the corpus median is **NOT_SUPPORTED** (their interval, 0.926 to
0.956, excludes 0.884). Under the median centring the two groups' bias medians also
differ, **-0.79** against **-1.50** sd (bootstrap p 0.0004), but that is a
difference between centring constants rather than labs: the Tsuboyama median
becomes +1.15 under wild type = 0, and the other labs' wild-type score is not in
the data. Leave-one-lab-out moves the median AUC between 0.880 and 0.941, and the
artefact's verdict is **INCONCLUSIVE**: dropping Tsuboyama alone moves it from
0.884 to 0.941, outside the script's tolerance. What holds is the weaker
statement: every lab family ranks doubles well above chance, and the level
differs by lab family.

**The 17 assays excluded for size are held out by size, not by lab.** They were
dropped for having under 500 pairs (5,348 pairs between them), so they are outside
the three 48-assay medians. But **15 of the 17** are Tsuboyama, the same lab as 33
of the corpus assays, and **11 of the 17** are among the 59 assays of the ESM
probe, so they are not outside every published number. Their ranking agrees with
the corpus (**AUC 0.848**, interval 0.755 to 0.920 around the corpus 0.884;
verdict SUPPORTED). An earlier version also reported that their bias "does not
replicate and even changes sign" (+0.50 against -0.94). That is withdrawn: it
compares centring constants — the median of all variants moves with the share of
doubles in a library, and the small assays have fewer — and under wild type = 0
the small Tsuboyama assays agree with the big ones. Anyone quoting "0.94 sd" as a
property of the method is quoting a centring convention.

## Where the failures concentrate

Two probes with valid nulls — one holds, one does not, and the second one is a
correction of a correction.

**Positions matter, but less than the raw number suggests.** The share of residual
variance explained by which sequence positions are involved is **0.140**
against a permutation null of 0.002, exceeding it in **47/48 assays**. Three
caveats, all measured, because the first version of this section overstated it:

- The median assay has only **4** groups in this contrast (11 assays have just 2),
  because the grouping uses the lower of the two positions. Counting **distinct
  positions in the assay** — the union of both slots — the median is **7** (min 3,
  max 234) and **no** assay has only 2. So this is usually a four-group contrast,
  not a map of hot residues; an earlier version of this line attributed the grouping
  count to the assay. It is weakest exactly where the promise would matter: median
  R² is 0.189 for 3–5 groups and 0.054 for 21–60.
- It shrinks to **0.092** once the assay's own bias is removed, and the verdict is
  judged on that centred figure rather than the raw 0.140.
- Grouping by **which single mutation** is involved instead of by position gives R²
  **0.343** — a strictly finer grouping, and therefore an upper bound on how much
  is specifically about position. Pairs sharing a mutation are similar largely
  because single-mutant effects are shared.

Flagging the hottest 10% of positions touches a median of **23%** of pairs (range
1%–67%), so the operational promise is real but modest — not "at no extra cost".

**Residue identity: INCONCLUSIVE, and the earlier negative was wrong too.** With
both substitution slots pooled (the earlier version read only slot `a`, an
arbitrary and non-random half of each pair) and an exact binomial instead of a
threshold picked by eye, alanine sits above its own assay's median residual in
**73% of 48 assays**, Bonferroni p = 0.042. But two independent checks say that is
not residue-specific epistasis:

- the residue ranking reproduces across independent assay halves at only
  **ρ +0.50** — chance;
- alanine's fraction drops to **0.52**, exactly chance, once the magnitude
  relationship is regressed out within assay.

So the marginal p-value is the magnitude confound leaking through. An earlier
draft called this NOT_SUPPORTED on an arbitrary 85% bar, which was a statement
about the bar and not a test; a later one nearly called it SUPPORTED on a p-value
alone. Both were wrong. Verdict **INCONCLUSIVE**: no residue shows a reproducible,
magnitude-independent effect at this sample size.

### Two ways this could have been manufactured, and why it wasn't

Both were tested before the reviewer round, because each would have produced a
finding out of nothing:

- **Per-assay normalisation.** Dividing each assay's residual by its own sd is a
  scalar per assay, so it cannot change a within-assay rank correlation.
  Recomputing on the raw residuals gives **-0.260 raw against -0.260 normalised**,
  with a maximum difference of exactly 0.0 across all 48 assays. The normalisation
  makes assays comparable in spread; it does not create an effect.
- **Position clustering as a count artefact.** If a high r² simply meant "few
  pairs per position", it would track position count. It does not: ρ = +0.027
  against pairs-per-position and -0.055 against number of positions, and 44 of 48
  assays still exceed five times the analytic null. The clustering is not a count
  artefact — the caveats above are about what the number *means*, not whether it
  is real.

## Does a stronger baseline close the gap?

Everything above judges additivity — the crudest model a lab could use. The obvious
objection is that the gap is just a weak baseline, so it was measured against
stronger ones: a per-position term, and a low-rank position-pair interaction. Every
per-pair prediction is made by a fit that never saw that pair (out-of-fold, inner
KFold(5) within the assay), the threshold is the same 2 sd for every model, and each
fitted model is compared to a permutation null of itself.

| model | R² of residual var. (null) | AUC (null) | beyond 2 sd (null) | bias, sd (median centring) |
|---|---|---|---|---|
| plain additivity (recomputed) | 0 | 0.884 | 12.6% | **-0.94** |
| + one offset learned on the other assays | 0 | 0.884 | 6.7% | -0.07 |
| + the assay's own offset | -0.001 | 0.884 | 4.6% | 0.00 |
| + position | **0.333** (-0.001) | 0.903 (0.884) | 2.5% (4.6%) | 0.00 |
| + position + position-pair rank 3 | **0.381** (-0.006) | **0.905** (0.884) | 2.5% (4.5%) | 0.00 |

Three things follow, and none of them is position-specific epistasis.

**The offset rows are bookkeeping, not findings.** Removing one offset per assay —
even the offset's own permutation null, which contains no structure at all — takes
the median bias to 0.00 by construction and the beyond-2-sd share from 12.6% to
**4.6%**, the centred figure above. Under the median centring 8 of the 12.6 points
were the additive prediction sitting off zero; the convention-free share is 4.6%.

**Position is mostly absorbing a global calibration.** The best model removes a
median **0.381** of the residual variance (position alone **0.333**), and the pair
model beats its own null in **48/48 assays** on variance removed (the position
term alone in 47/48), so this is not a fitting artefact. But a two-number linear
recalibration of the additive prediction — no positions, no pairs, fitted
out-of-fold and scored on the same metric — removes a median **0.500**, more than
either (`scripts/reference.py`, `results/reference.json`). The additive prediction
is over-dispersed, and most of what the position term buys is that global
correction, not positional structure. The gap is not closed: about half of the
residual variance is left after the recalibration, and these data cannot split
that half into measurement noise and epistasis.

**Ranking barely moves.** Median AUC goes 0.884 → **0.905**, a median gain of
+0.0081, and the pair model wins in only **34/48 assays**. A model that needs 4/5 of
the assay's own measured doubles to gain 8 points of AUC is not what a lab with
singles-only can use.

That last caveat is the important one, and it is in `results/alt_baseline.json`
rather than here: every per-assay row is fitted on 4/5 of **that protein's own
doubles**, so these numbers bound what a stronger baseline can achieve, not what a
singles-only design can. Held-out whole position pairs were not run. What the
table and the recalibration establish together is narrower than the earlier
reading ("the gap is not an artefact of a weak baseline"): a global recalibration
does more than position, and about half of the residual variance survives it — a
gap, but not evidence that the gap is positional.

## Can a protein language model do better?

Three probes. The first two were in the earlier draft; the third is the one that
was actually needed, and the earlier ESM numbers were withdrawn.

- **Distance alone**: ρ = -0.03, MAE worse than a constant by 0.086 sd. Nothing.
- **ESM-2 (650M) alone**, window embeddings of both mutated positions plus their
difference, product and sequence separation: ρ = **-0.003 ± 0.049**, against a
**pure-noise control of +0.007 ± 0.014** run through the identical pipeline. MAE
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
residue identity (rank correlation across assay halves +0.50), and ESM-2 context
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
**+0.181 ± 0.094**, *more*, so that gain comes from knowing which assay a pair
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

## Reproducing

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

# 6. the held-out checks: one lab or many, and the assays excluded by size
#    (--index defaults to $EPSCEIL_INDEX, then ../_data/DMS_substitutions.csv)
python scripts/heldout.py --index <DMS_substitutions.csv>

# 6b. both centring conventions, and the two-number recalibration
python scripts/reference.py

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

**What needs what.** Everything that reads the pair tables — steps 2 to 5, 6b,
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

## Limits, stated plainly

- **Additivity is not the only null model — and now that is measured, not promised.**
  The per-position and rank-k position-pair rows above are those stronger baselines.
  They remove a median 0.381 of the residual variance and need 4/5 of the assay's
  own measured doubles to do it; a two-number linear recalibration with no
  positions removes 0.500, so most of what they capture is a global calibration,
  and the position R² is not evidence of position-*specific* epistasis. A monotone
  global-epistasis fit beyond that linear map (saturation, a floor) was not run,
  and whole position pairs were never held out.
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
- **The ESM probe is three heads at corpus scale, still one architecture.**
  `esm_scale.py` covers 39.0% of the corpus and a 100% sensitivity pass with three
  heads, and no head beats its larger control at either size — so the null is no
  longer a statement about a 4.4% sample or a single estimator. What it still does
  not show is that no model can: one architecture (ESM-2 650M), one window size
  (8 residues), wild-type sequence only, and a supervised model trained on matched
  data remains the obvious untried test. Two further limits are recorded in the
  artefact — overlapping windows for pairs closer than 17 residues mean heads (a)
  and (c) do not fully separate window content from window overlap, and a
  wild-type window is identical for every substitution pair at the same two
  positions, so these features can rank position pairs but not the substitutions
  within one.
- **Every prior tested here is null, and that is the result.** The magnitude prior
  fails a construction-preserving permutation, residue identity does not replicate
  across assay halves, and ESM-2 context is indistinguishable from noise. A reader
  who came for "which pairs will break" should leave without an answer; what the
  repo establishes is the size of the gap (AUC 0.884 ranking; 4.6% of pairs beyond
  2 sd once each assay's own offset is removed; an offset whose sign is a centring
  convention) and that three obvious ways to close it do not work. The held-out
  section narrows even that: every lab family ranks well above chance, at levels
  that differ by lab, and the offset is a convention — so quote the AUC and not the
  0.94 sd.
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

## Layout

```
src/epsceil/pairs.py        mutation parsing, additivity residual, per-assay scaling
src/epsceil/baseline.py     position and rank-k position-pair baselines
scripts/build_pairs.py      every assay -> results/assay_summary.csv + pairs/
scripts/rebuild_summary.py  per-assay summary from the pair tables alone
scripts/prior.py            the magnitude prior + its permutation null
scripts/invariant_stats.py  shift-invariant correlations, AUC, bias accounting
scripts/depth.py            position, residue and centred-prior probes
scripts/alt_baseline.py    stronger baselines, out-of-fold, with permutation nulls
scripts/heldout.py          one-lab-in-the-corpus and size-excluded assays
scripts/reference.py        both centring conventions; two-number recalibration
scripts/prior_units.py      within-assay vs pooled: what a rho is measured over
esm_context.py              the GPU probe, grouped CV by assay (4.4% sample)
scripts/esm_scale.py        corpus-scale ESM probe, three heads, two controls
scripts/esm_tcrit.py        the ESM verdicts under t(4) beside the 2 SE rule
scripts/make_figures.py     figures from committed artefacts
scripts/audit_claims.py     every README number against a committed artefact
tests/                      invariants, estimator labels, the claim audit
tests/test_alt_baseline.py  the stronger-baseline table's nulls and means
tests/test_esm_scale.py     the ESM verdict rule and its controls
```

MIT licence for this repository's own code. ProteinGym data keeps its own terms.
