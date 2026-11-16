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
    rng = np.random.default_rng(seed)
    frames = []
    files = sorted(pairs_dir.glob("*.csv"))
    used = 0
    for p in files:
        assay = p.stem
        if assay not in seqs:
            continue
        d = pd.read_csv(p)
        if len(d) < 300:
            continue
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
    return pd.concat(frames, ignore_index=True)


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
    print(f"{len(df):,} pairs from {df.assay.nunique()} assays, "
          f"{args.max_per_assay} cap each")

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
    y = np.abs(df.residual.to_numpy())
    groups = df.assay.to_numpy()
    print(f"feature matrix {Feat.shape}")

    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    from scipy.stats import spearmanr

    gkf = GroupKFold(n_splits=5)
    rhos, deltas = [], []
    for tr, te in gkf.split(Feat, y, groups):
        sc = StandardScaler().fit(Feat[tr])
        m = Ridge(alpha=10.0).fit(sc.transform(Feat[tr]), y[tr])
        pred = m.predict(sc.transform(Feat[te]))
        base = np.full_like(pred, np.median(y[tr]))
        deltas.append(float(np.abs(base - y[te]).mean()
                            - np.abs(pred - y[te]).mean()))
        r = spearmanr(pred, y[te]).statistic
        if np.isfinite(r):
            rhos.append(float(r))

    res = {
        "n_pairs": int(len(df)), "n_assays": int(df.assay.nunique()),
        "features": "ESM-2 window embeddings (both positions) + abs-diff + product "
                    "+ log1p(sep)",
        "cv": "GroupKFold(5) by assay",
        "spearman_mean": float(np.mean(rhos)) if rhos else None,
        "spearman_sd": float(np.std(rhos)) if rhos else None,
        "mae_improvement_mean": float(np.mean(deltas)),
        "mae_improvement_sd": float(np.std(deltas)),
        "folds": len(deltas),
        "device": device,
    }
    Path(args.out).write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=2))

    print("\nINTERPRETATION:")
    rho = res["spearman_mean"] or 0.0
    imp = res["mae_improvement_mean"]
    sd = res["spearman_sd"] or 0.0
    if imp > 0 and rho - 2 * sd > 0:
        print(f"  ESM context predicts where additivity fails: rho {rho:+.3f} "
              f"+- {sd:.3f}, MAE improves {imp:+.4f} sd over the constant. Signal.")
    else:
        print(f"  No usable signal: rho {rho:+.3f} +- {sd:.3f}, MAE change "
              f"{imp:+.4f} sd. Sequence context does not tell you where the "
              f"additivity assumption breaks.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
