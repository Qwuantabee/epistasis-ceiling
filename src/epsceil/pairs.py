"""What a single-mutant measurement does and does not tell you about a double.

The unit of study is the ADDITIVITY RESIDUAL:

    residual = observed(double) - (single_a + single_b - wt)

where `wt` centres the assay. If doubles were additive, every residual would be
zero and one round of single-mutant screening would answer every pair question.

Everything downstream compares against that baseline, so the baseline is computed
first, once, and never refitted. Nothing here trains a model.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

# A mutant token looks like "T159M" or "T159*" (stop). Parse position and the
# wild-type/mutant residues so we can reason about distance and identity.
_MUT = re.compile(r"^([A-Za-z])(\d+)([A-Za-z*])$")


@dataclass(frozen=True)
class Mutation:
    wt: str
    position: int
    mut: str

    @property
    def key(self) -> str:
        return f"{self.wt}{self.position}{self.mut}"


def parse_mutation(token: str) -> Mutation | None:
    m = _MUT.match(token.strip())
    if not m:
        return None
    return Mutation(m.group(1), int(m.group(2)), m.group(3))


def parse_mutant(mutant: str) -> list[Mutation] | None:
    """Split "T159M:D166V" into Mutations. None if any token is unparseable.

    Also handles the "T159M,D166V" and "T159M-D166V" spellings that appear in a
    few assays; returning None rather than guessing keeps a malformed row out of
    the statistics instead of silently becoming a wrong pair.
    """
    if not isinstance(mutant, str) or not mutant.strip():
        return None
    # Separator precedence matters: a lone token like "T159M" contains no
    # separator, and splitting on "-" would also break a mutation spelled with a
    # dash inside it. Try colon, then comma, then a dash BETWEEN two mutation
    # tokens (verified by parsing both halves), so "T159M-D166V" works while a
    # nonsense dash string still returns None.
    for sep in (":", ","):
        if sep in mutant:
            tokens = mutant.split(sep)
            break
    else:
        tokens = [mutant]
        if "-" in mutant:
            parts = mutant.split("-")
            if len(parts) == 2 and all(parse_mutation(p) for p in parts):
                tokens = parts
    out = []
    for t in tokens:
        p = parse_mutation(t)
        if p is None:
            return None
        out.append(p)
    return out


def wt_reference(scores: pd.Series) -> float:
    """The centring constant for an assay.

    DMS scores are already relative to wild type in most ProteinGym assays, but
    the scale differs per assay. The MEDIAN of all measured variants is used as
    the reference rather than an assumed zero: it is robust, and it makes the
    additivity baseline invariant to an additive shift in the assay.

    (Using 0 instead would still give the same RESIDUALS - subtraction cancels the
    constant - but it would shift `predicted` and make the plotted agreement
    between predicted and observed depend on assay convention.)
    """
    return float(np.median(scores.to_numpy()))


@dataclass
class AssayPairs:
    assay: str
    doubles: pd.DataFrame      # mutant, a, b, observed, predicted, residual
    singles: dict[str, float]
    wt: float
    n_singles_measured: int
    n_doubles_total: int

    @property
    def coverage(self) -> float:
        return len(self.doubles) / max(self.n_doubles_total, 1)


def build_pairs(df: pd.DataFrame, assay: str,
                min_singles: int = 50) -> AssayPairs | None:
    """Compute the additivity residual for every double whose singles exist."""
    df = df.dropna(subset=["mutant", "DMS_score"]).copy()
    df["parsed"] = df.mutant.map(parse_mutant)
    df = df[df.parsed.notna()]

    singles = df[df.parsed.map(len) == 1]
    doubles = df[df.parsed.map(len) == 2]
    if len(singles) < min_singles or len(doubles) < min_singles:
        return None

    s_map = {row.parsed[0].key: float(row.DMS_score)
             for row in singles.itertuples() if row.parsed}
    wt = wt_reference(df.DMS_score)

    rows = []
    for row in doubles.itertuples():
        a, b = row.parsed[0].key, row.parsed[1].key
        if a not in s_map or b not in s_map:
            continue
        predicted = s_map[a] + s_map[b] - wt
        rows.append({
            "mutant": row.mutant, "a": a, "b": b,
            "observed": float(row.DMS_score), "predicted": float(predicted),
            "residual": float(row.DMS_score) - predicted,
            "pos_a": row.parsed[0].position, "pos_b": row.parsed[1].position,
        })
    if not rows:
        return None
    t = pd.DataFrame(rows)
    t["sep"] = (t.pos_b - t.pos_a).abs()

    # NORMALISE WITHIN ASSAY. ProteinGym assays are NOT on a common scale: most
    # are relative fitness around 0-1, but some (e.g. the Somermeyer deep
    # mutational scans) report raw fluorescence in the thousands. Pooling them
    # raw would let three assays dominate every aggregate statistic - their
    # residual sd was 2340 / 1377 / 6098 against a median of 0.69 across the
    # other 62. Each assay is therefore divided by its own residual sd, so the
    # residual is in "assay standard deviations" and is comparable everywhere.
    # This is a rescaling of the SAME quantity, not a new measurement, and
    # residual_raw is kept so nothing is hidden.
    t["residual_raw"] = t["residual"]
    sd = float(t["residual_raw"].std(ddof=1))
    t["residual"] = t["residual_raw"] / sd if sd > 0 else t["residual_raw"]
    t["residual_scale_sd"] = sd

    return AssayPairs(assay=assay, doubles=t, singles=s_map, wt=wt,
                      n_singles_measured=len(s_map), n_doubles_total=len(doubles))


def summarise(ap: AssayPairs) -> dict:
    r = ap.doubles.residual.to_numpy()
    from scipy.stats import spearmanr
    rho = float(spearmanr(ap.doubles.predicted, ap.doubles.observed).statistic)
    med = float(np.median(ap.doubles.observed))
    # direction agreement on a median split: does additivity at least rank the
    # pair above/below the cohort median correctly?
    agree = float(np.mean(
        np.sign(ap.doubles.predicted - med) == np.sign(ap.doubles.observed - med)))
    return {
        "assay": ap.assay,
        "n_pairs": int(len(ap.doubles)),
        "coverage": ap.coverage,
        "n_singles_measured": ap.n_singles_measured,
        "residual_mean": float(r.mean()),
        "residual_sd": float(r.std(ddof=1)) if len(r) > 1 else float("nan"),
        "residual_median": float(np.median(r)),
        "residual_p05": float(np.percentile(r, 5)),
        "residual_p95": float(np.percentile(r, 95)),
        "spearman_additive": rho,
        "direction_agreement": agree,
        # the headline: what fraction of pairs does additivity get right by more
        # than a stated tolerance, in assay-sd units
        # `r` is already in assay-sd units, so these are fractions of pairs whose
        # additivity residual is within 0.5 / 1.0 sd of the assay's own spread.
        "frac_within_0.5sd": float(np.mean(np.abs(r) <= 0.5)),
        "frac_within_1sd": float(np.mean(np.abs(r) <= 1.0)),
        "residual_scale_sd": float(ap.doubles.residual_scale_sd.iloc[0]),
        "residual_raw_sd": float(ap.doubles.residual_raw.std(ddof=1)),
    }
