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

    (`wt` cancels out of the RESIDUAL only up to the fact that it sets where zero
    sits: residual = observed - (a + b - wt), so d(residual)/d(wt) = +1. An
    earlier version of this docstring claimed the constant cancels. It does not
    cancel from the residual - every residual in the assay moves by exactly wt -
    which is harmless for a correlation between two signed quantities but NOT
    harmless for anything involving |residual|. See residual_c below.)
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


def add_centred_columns(t: pd.DataFrame) -> pd.DataFrame:
    """Derive residual_c / predicted_c on a pair table loaded from disk.

    results/pairs/ is gitignored and predates the centring fix, so every consumer
    that reads a pair CSV calls this rather than trusting the file to have the
    columns. One function, so the definition cannot drift between scripts.

    THE CONVENTION IS MEAN-CENTRING FOR BOTH SERIES, named here once and used
    everywhere (depth.py, prior.py, esm_context.py, invariant_stats.py). Mixing
    mean-centring one series with median-centring the other produces a number that
    is neither convention; that mistake sat in this file for one revision and moved
    the statistic from 0.313 to 0.328 without any of the data changing. Depth also
    reports the median-centred value (+0.373) as a robustness check, and the two
    agree on the conclusion, which is the point of reporting both.
    """
    t = t.copy()
    if "residual_c" not in t.columns:
        t["residual_c"] = (t.residual
                           - t.groupby("assay").residual.transform("mean")
                           if "assay" in t.columns
                           else t.residual - float(t.residual.mean()))
    if "predicted_c" not in t.columns:
        t["predicted_c"] = (t.predicted
                            - t.groupby("assay").predicted.transform("mean")
                            if "assay" in t.columns
                            else t.predicted - float(t.predicted.mean()))
    return t


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

    # CENTRE THE RESIDUAL. The additive baseline is identified only up to one
    # constant per assay: wt is a convention, not a measurement, so the residual
    # cloud can sit anywhere on the number line. That is invisible to a
    # correlation between two signed variables and decisive for |residual|: the
    # real data's cloud sits a median -0.94 assay-sd off zero, so a comparison of
    # |residual| with |predicted| was comparing two distances from two different
    # arbitrary origins. residual_c removes that one constant per assay, so a
    # magnitude comparison is a comparison of spreads. residual is kept unchanged
    # because the bias is itself a finding (doubles are systematically worse than
    # additive) and nothing here hides it.
    t["residual_c"] = t["residual"] - float(t["residual"].mean())
    t["predicted_c"] = t["predicted"] - float(t["predicted"].mean())

    return AssayPairs(assay=assay, doubles=t, singles=s_map, wt=wt,
                      n_singles_measured=len(s_map), n_doubles_total=len(doubles))


def direction_auc(pred: np.ndarray, obs: np.ndarray) -> float:
    """Threshold-free direction accuracy: P(pred ranks a random high pair above a
    random low pair), splitting observed at its median.

    Why not a median split on both series: `predicted` is wider than
    `observed` (sd ratio 1.39 median) and sits above the observed median 76% of
    the time, so splitting both at the OBSERVED median puts most predictions on
    one side and reports 0.693 where the rank information is worth 0.884. The
    split statistic is kept for continuity but AUC is the honest number.
    """
    from scipy.stats import rankdata
    y = (obs > np.median(obs)).astype(int)
    n1, n0 = int(y.sum()), int((1 - y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    rk = rankdata(pred)
    return float((rk[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def summarise_table(t: pd.DataFrame, assay: str,
                    coverage: float = float("nan"),
                    n_singles: int = 0) -> dict:
    """Per-assay summary from a pair table alone.

    This exists so the summary can be rebuilt from results/pairs/*.csv without the
    raw ProteinGym CSVs, and so the definition lives in ONE place: summarise()
    and scripts/rebuild_summary.py both call it, and cannot drift apart.
    """
    from scipy.stats import spearmanr
    r = t.residual.to_numpy()
    rc = t.residual_c.to_numpy() if "residual_c" in t.columns else \
        t.residual - float(t.residual.mean())
    rho = float(spearmanr(t.predicted, t.observed).statistic)
    med = float(np.median(t.observed))
    pm = float(np.median(t.predicted.to_numpy()))
    agree = float(np.mean(
        np.sign(t.predicted - med) == np.sign(t.observed - med)))
    agree_own = float(np.mean(
        np.sign(t.predicted - pm) == np.sign(t.observed - med)))
    return {
        "assay": assay,
        "n_pairs": int(len(t)),
        "coverage": coverage,
        "n_singles_measured": n_singles,
        "residual_mean": float(r.mean()),
        "residual_sd": float(r.std(ddof=1)) if len(r) > 1 else float("nan"),
        "residual_median": float(np.median(r)),
        "residual_p05": float(np.percentile(r, 5)),
        "residual_p95": float(np.percentile(r, 95)),
        "spearman_additive": rho,
        "direction_agreement": agree,
        "direction_agreement_own_medians": agree_own,
        "auc_direction": direction_auc(t.predicted.to_numpy(),
                                       t.observed.to_numpy()),
        "bias_mean_sd": float(r.mean()),
        "bias_median_sd": float(np.median(r)),
        "frac_within_0.5sd": float(np.mean(np.abs(r) <= 0.5)),
        "frac_within_1sd": float(np.mean(np.abs(r) <= 1.0)),
        "frac_within_1sd_centred": float(np.mean(np.abs(rc) <= 1.0)),
        "frac_outside_2sd": float(np.mean(np.abs(r) > 2.0)),
        "frac_outside_2sd_centred": float(np.mean(np.abs(rc) > 2.0)),
        "residual_scale_sd": float(t.residual_scale_sd.iloc[0])
        if "residual_scale_sd" in t.columns else float("nan"),
        "residual_raw_sd": float(t.residual_raw.std(ddof=1))
        if "residual_raw" in t.columns else float("nan"),
    }


def summarise(ap: AssayPairs) -> dict:
    """Per-assay summary, delegating to summarise_table so the two cannot drift."""
    return summarise_table(ap.doubles, ap.assay, ap.coverage,
                           ap.n_singles_measured)
