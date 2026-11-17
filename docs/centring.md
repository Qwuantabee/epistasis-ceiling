# The additive model's offset is a convention

[Back to the README](../README.md)

An earlier version also called the additive prediction biased by a
median 0.94 assay-sd. That is withdrawn: the offset is set by where each assay's
zero is put, not by additivity. For the 33 assays scored as a ΔΔG from wild type
it reads -0.79 sd under the median centring this repo used and +1.15 sd under
wild type = 0, positive in 33/33.

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

The median centring was avoidable for most of the corpus. Those 33 assays are 33
of the 48 behind every median, their score is already a difference from wild
type, and the additive reference is therefore exactly 0 with no constant to
choose. The remaining 15 have no wild-type score in the data, so no
convention-free signed offset exists for them, and none is quoted.

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

## A withdrawn figure panel

**Withdrawn:** an earlier version of this line and figure added "and only +0.12
with its own residual spread", with a second panel. That was not a measurement:
every residual is divided by its assay's own sd, so the spread is 1 for all 48
assays by construction and the correlation was with floating-point rounding. The
figure is now one panel and `results/figure_stats.json` no longer holds the value.
