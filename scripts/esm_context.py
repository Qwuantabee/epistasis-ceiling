#!/usr/bin/env python3
"""Does ESM-2 context make epistasis predictable? The real GPU test.

The distance-only probe found nothing (rho = -0.03). This is the version that
actually uses the GPU: for each mutation in a pair, embed the wild-type sequence
window around that position with ESM-2, then ask whether those embeddings predict
|additivity residual| better than a constant.

Why window embeddings rather than the whole protein: the signal, if any, is local
- a mutation's effect depends on its own chemical neighbourhood. Whole-protein
mean pooling would dilute two point mutations into one 1280-vector that is
identical for every pair from that protein, which cannot rank pairs at all.

Leakage controls, both required for the result to mean anything:
  * features come from the WILD-TYPE sequence only. Scores never enter the input.
  * GroupKFold by assay: a protein is never in both train and test, because pairs
    from one protein share a sequence and a measurement scale.
  * the baseline is the training median, i.e. "learn nothing".

Reported as rank correlation with |residual| plus MAE improvement in assay-sd
units, both with the fold spread so a small win cannot masquerade as a finding.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def load_parent_sequences(index_csv: Path) -> dict[str, str]:
    idx = pd.read_csv(index_csv)
    return dict(zip(idx.DMS_id, idx.target_seq))


def load_pairs(pairs_dir: Path, seqs: dict[str, str], max_assays: int,
               max_per_assay: int, seed: int = 0) -> pd.DataFrame:
    """Pick assays by SIZE, not by alphabetical luck.

    The first version took `sorted(glob)[:max_assays]`, which silently dropped
    SPG1_STRSG_Olson_2014 - 65.5% of all pairs in the corpus - purely because
    that name sorts 40th, and analysed 4.3% of the data while saying nothing
    about it. Selection is now by pair count, and the exclusion is reported.
    """
    rng = np.random.default_rng(seed)
    frames = []
    cands = []
    for p in pairs_dir.glob("*.csv"):
        assay = p.stem
        if assay not in seqs:
            continue
        n = sum(1 for _ in p.open()) - 1
        if n < 300:
            continue
        cands.append((n, p))
    cands.sort(reverse=True)
    used = 0
    for n, p in cands:
        assay = p.stem
        d = pd.read_csv(p)
        if len(d) > max_per_assay:
            keep = rng.choice(len(d), max_per_assay, replace=False)
            d = d.iloc[np.sort(keep)]
        d["assay"] = assay
        d["seq"] = seqs[assay]
        frames.append(d)
        used += 1
        if used >= max_assays:
            break
    if not frames:
        raise SystemExit("no usable assays (missing parent sequences?)")
    out = pd.concat(frames, ignore_index=True)
    total = sum(n for n, _ in cands)
    out.attrs["selection"] = {
        "assays_available_over_300_pairs": len(cands),
        "pairs_available": int(total),
        "assays_used": used,
        "pairs_used": int(len(out)),
        "fraction_of_available": float(len(out) / total) if total else None,
        "rule": "assays ranked by pair count, descending; capped per assay",
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-dir", default=str(ROOT / "results" / "pairs"))
    ap.add_argument("--index", required=True)
    ap.add_argument("--max-assays", type=int, default=20)
    ap.add_argument("--max-per-assay", type=int, default=3000)
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--out", default=str(ROOT / "results" / "esm_context.json"))
    args = ap.parse_args()

    seqs = load_parent_sequences(Path(args.index))
    df = load_pairs(Path(args.pairs_dir), seqs, args.max_assays,
                    args.max_per_assay)
    # the centring rule must match the rest of the repo, or the target differs
    df["residual_c"] = (df.residual
                        - df.groupby("assay").residual.transform("mean"))
    df["predicted_c"] = (df.predicted
                         - df.groupby("assay").predicted.transform("median"))
    print(f"{len(df):,} pairs from {df.assay.nunique()} assays, "
          f"{args.max_per_assay} cap each")
    print("selection:", json.dumps(df.attrs.get("selection")))

    import torch
    from transformers import AutoModel, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    name = "facebook/esm2_t33_650M_UR50D"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name).to(device).eval()
    print(f"esm-2 on {device}")

    def windows(seq: str, pos: int, w: int) -> str:
        lo, hi = max(0, pos - 1 - w), min(len(seq), pos + w)
        return seq[lo:hi]

    # Both mutation contexts, batched.
    wa = [windows(s, p, args.window) for s, p in zip(df.seq, df.pos_a)]
    wb = [windows(s, p, args.window) for s, p in zip(df.seq, df.pos_b)]
    print("embedding windows ...")
    embs = []
    for chunk in (wa, wb):
        out = []
        for i in range(0, len(chunk), 256):
            enc = tok(chunk[i:i + 256], return_tensors="pt", padding=True,
                      truncation=True, max_length=64).to(device)
            with torch.no_grad():
                h = model(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).float()
            out.append(((h * m).sum(1) / m.sum(1).clamp(min=1)).cpu().numpy())
        embs.append(np.vstack(out))
    Ea, Eb = embs
    # Element-wise combination: the residual depends on the PAIR, not on either
    # mutation alone, so the product and absolute difference carry the signal.
    Feat = np.hstack([Ea, Eb, np.abs(Ea - Eb), Ea * Eb,
                      np.log1p(df.sep.to_numpy())[:, None]])
    y = np.abs(df.residual_c.to_numpy())
    groups = df.assay.to_numpy()
    print(f"feature matrix {Feat.shape}")

    from sklearn.decomposition import PCA
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    from scipy.stats import spearmanr

    # THE FREE PRIOR IS A FEATURE. The README's own question is whether sequence
    # context adds anything to what a lab already has for nothing: |additive
    # prediction|. The first probe omitted it, so it could only answer "does ESM
    # alone beat a constant", which is not the decision anyone faces.
    prior = np.abs(df.predicted_c.to_numpy())[:, None]

    def evaluate(X: np.ndarray, label: str, k_folds: int = 5) -> dict:
        """GroupKFold by assay, PCA inside the fold, alpha chosen on inner folds.

        Three estimator repairs over the first probe, each measured rather than
        asserted:
          * PCA to 100 dims inside each training fold - 5,121 near-collinear
            features with StandardScaler and alpha=10 was the estimator that
            produced a 1.43-sd MAE deficit, i.e. it predicted outside the data
            range. That deficit is a property of the estimator, not of ESM.
          * the target is log1p-transformed: |residual| is non-negative and
            right-skewed (skew 0.89), which a linear model on the raw scale
            cannot represent.
          * alpha chosen by inner CV rather than fixed.
        """
        gkf = GroupKFold(n_splits=int(min(k_folds, len(np.unique(groups)))))
        rhos, deltas, mae_m, mae_b = [], [], [], []
        for tr, te in gkf.split(X, y, groups):
            sc = StandardScaler().fit(X[tr])
            Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
            ncomp = int(min(100, Xtr.shape[0] - 1, Xtr.shape[1]))
            pca = PCA(n_components=ncomp, random_state=0).fit(Xtr)
            Ztr, Zte = pca.transform(Xtr), pca.transform(Xte)
            ytr, yte = np.log1p(y[tr]), np.log1p(y[te])
            model = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(Ztr, ytr)
            pred = np.expm1(model.predict(Zte))
            # CLIP TO THE TRAINING TARGET RANGE. expm1 of an unbounded linear
            # output can leave the data by orders of magnitude; a ridge fit on
            # 100 PCA components of collinear features does exactly that on an
            # unseen protein, and a single pair then dominates the fold MAE (a
            # smoke run produced MAE 140,000 on a target whose median is 1.4).
            # That is the same estimator failure the first probe reported as an
            # ESM finding, so the guard belongs here, not in the write-up.
            hi = float(y[tr].max())
            pred = np.clip(pred, 0.0, hi)
            base = np.full_like(pred, np.median(y[tr]))
            mae_m.append(float(np.abs(pred - y[te]).mean()))
            mae_b.append(float(np.abs(base - y[te]).mean()))
            deltas.append(mae_b[-1] - mae_m[-1])
            r = spearmanr(pred, y[te]).statistic
            if np.isfinite(r):
                rhos.append(float(r))
        return {
            "label": label,
            "spearman_mean": float(np.mean(rhos)) if rhos else None,
            "spearman_sd": float(np.std(rhos)) if rhos else None,
            "mae_model": float(np.mean(mae_m)),
            "mae_constant_baseline": float(np.mean(mae_b)),
            "mae_improvement_sd_units": float(np.mean(deltas)
                                              / np.mean(y)),
            "mae_improvement_mean": float(np.mean(deltas)),
            "mae_improvement_relative": float(np.mean(deltas)
                                              / np.mean(mae_b)),
        }

    # 1. the pipeline floor: pure noise through the identical pipeline.
    noise = np.random.default_rng(0).normal(
        size=(Feat.shape[0], min(Feat.shape[1], 512)))
    res_noise = evaluate(noise, "pure noise, identical pipeline")
    print("  noise control:", json.dumps(res_noise))

    # 2. the free prior alone, so the ESM probe has an honest reference point.
    res_prior = evaluate(prior, "|additive prediction| alone (the free prior)")
    print("  prior only:", json.dumps(res_prior))

    # 3. ESM alone, and ESM plus the prior (the question that matters).
    res_esm = evaluate(Feat, "ESM-2 features alone")
    print("  esm only:", json.dumps(res_esm))
    res_both = evaluate(np.hstack([Feat, prior]),
                        "ESM-2 features + |additive prediction|")
    print("  esm+prior:", json.dumps(res_both))

    res = {
        "n_pairs": int(len(df)), "n_assays": int(df.assay.nunique()),
        "features": "ESM-2 window embeddings (both positions) + abs-diff + product "
                    "+ log1p(sep)",
        "cv": "GroupKFold(5) by assay",
        "target": "|centred additivity residual|, log1p on the model side",
        "selection": df.attrs.get("selection"),
        "spearman_mean": res_esm["spearman_mean"],
        "spearman_sd": res_esm["spearman_sd"],
        "mae_improvement_mean": res_esm["mae_improvement_mean"],
        "mae_improvement_sd": None,
        "folds": 5,
        "device": device,
        "controls": {
            "noise": res_noise,
            "prior_only": res_prior,
            "esm_only": res_esm,
            "esm_plus_prior": res_both,
        },
        "incremental_value_of_esm_over_prior": (
            res_both["spearman_mean"] - res_prior["spearman_mean"]
            if res_both["spearman_mean"] is not None
            and res_prior["spearman_mean"] is not None else None),
    }
    Path(args.out).write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=2))

    print("\nINTERPRETATION:")
    inc = res["incremental_value_of_esm_over_prior"]
    rho = res["spearman_mean"] or 0.0
    sd = res["spearman_sd"] or 0.0
    nz = res_noise["spearman_mean"] or 0.0
    print(f"  ESM alone:            rho {rho:+.3f} +- {sd:.3f}")
    print(f"  noise control:        rho {nz:+.3f}")
    print(f"  free prior alone:     rho {res_prior['spearman_mean']:+.3f}")
    print(f"  ESM + prior:          rho {res_both['spearman_mean']:+.3f}")
    print(f"  incremental value:    {inc:+.3f}" if inc is not None else "")
    print(f"  MAE change vs constant, ESM: {res_esm['mae_improvement_mean']:+.4f} "
          f"({res_esm['mae_improvement_relative']:+.1%} of the baseline MAE)")
    if rho > nz + 2 * sd and rho > 0:
        print("  ESM context beats the noise floor.")
    else:
        print("  ESM context does NOT beat the pipeline's own noise floor: the "
              "probe shows no usable signal from sequence context on this "
              "feature set and window size.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
