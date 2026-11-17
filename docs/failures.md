# Where the failures concentrate

[Back to the README](../README.md)

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

**Two harder tests, both of which cut the position result down.** The position
R² of 0.333 in the README's [stronger-baseline table](../README.md#does-a-stronger-baseline-close-the-gap)
comes from a random pair split, so a held-out pair usually shares both of its
positions with training pairs — the most favourable split available. Holding out
**whole position pairs** (no position pair seen in training appears in test) and
allowing a **monotone global nonlinearity** both shrink it:

- **Whole-position-pair holdout** (`results/position_holdout.json`). Across all 48
  assays position R² falls from **0.333** to **-0.134** (30/48 above their null;
  verdict COLLAPSES) and the pair-rank-3 model falls just as far, 0.381 to -0.134.
  Most of that collapse is arithmetic, not refutation: assays have a median of
  **5.5** position pairs, and the split is bimodal — **9** assays hold only 2
  position pairs, **23** hold 3–9, and **16** hold 10 or more. In the small bins the
  held-out cells have positions the model never saw and it can
  only predict the other cells' mean. **The informative stratum is the 16 assays
  with 10 or more position pairs**, where held-out positions are nearly always
  seen in training (every test pair in 9 of the 16 assays, at least 99.9% of
  test pairs in 14, and 63% in the lowest). With held-out positions seen in
  training: there position keeps **60%** of its R², scoring **0.236** on the
  random split and **0.142** on the whole-pair split; pair-rank-3 keeps **84%**,
  scoring **0.241** and **0.201**. Both beat their null in **16/16** assays.
  Those shares are ratios of stratum medians, the artefact's own rule (HOLDS at
  0.75 or more, COLLAPSES below 0.25, otherwise PARTIAL; position is PARTIAL and
  pair-rank-3 HOLDS here). Paired within assay the loss is smaller: the median assay's position R²
  drops by 0.030 [0.019, 0.058], lower in 16/16.

- **Monotone global fit** (`results/monotone_fit.json`). If single-effect size
  clusters by position, a global saturation or floor would produce
  position-clustered residuals with no position-specific epistasis. Fitted
  out-of-fold, a monotone (isotonic) transform of the additive prediction alone
  removes **0.576** of the residual variance against position's **0.333** — so the
  curve alone beats position head-to-head. In the paired comparison position sits
  **-0.249** [-0.275, -0.196] below the monotone curve and is positive in only 6/48
  assays. Allowing both, `position + isotonic` reaches **0.707**, and the position
  term's own contribution once the curve is present is **0.047** [0.038, 0.069],
  above its permutation null in **48/48** assays. That is **14%** of the 0.333.

So the honest statement is narrower than "positions matter": **most of the
position signal was a calibration curve, and a small position-specific part —
about 0.05 of variance, 14% of the original figure — survives both tests.** The
verdict in the artefact is SUPPORTED for that residual contribution and
NOT_SUPPORTED for position beating the monotone curve; both readings are stored.
The rule, written into `scripts/monotone_fit.py` before the fits were run, decides
on the nested comparison (what position adds once the curve is present): SUPPORTED
when the interval's lower bound clears 0.02 of variance and at least 75% of assays
beat their label-permutation null. The head-to-head comparison is judged by the
same 0.02 bar on its interval and is the reading that fails.

**Residue identity: INCONCLUSIVE, and the earlier negative was wrong too.** With
both substitution slots pooled (the earlier version read only slot `a`, an
arbitrary and non-random half of each pair) and an exact binomial instead of a
threshold picked by eye, alanine sits above its own assay's median residual in
**73% of 48 assays**, Bonferroni p = 0.042. But two checks stop that short of
residue-specific epistasis:

- the residue ranking reproduces across independent assay halves at only
  **ρ +0.50**. That is moderate agreement, not chance: chance for a rank
  correlation is 0. An earlier version of this line called +0.50 chance, which
  confuses a correlation with a coin-flip rate. Something about the ranking does
  reproduce, so this check alone does not dismiss it;
- alanine's fraction drops to **0.52**, against a chance rate of 0.50, once the
  magnitude relationship is regressed out within assay.

The second check is the one that decides it: what reproduces is consistent with
the magnitude confound leaking through the marginal p-value. An earlier
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
