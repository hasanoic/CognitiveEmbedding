"""Standalone BnPC (real gold Bangla paraphrase corpus) evaluation --
added after the user supplied the raw BnPC CSVs directly (Kaggle-hosted,
previously NOT ACCESSIBLE, see data/registry.py). Kept separate from
evaluate.py's ~65-minute full run so this gold-data check doesn't require
re-running everything: it loads whatever checkpoint currently exists in
results/core_model.pt and reports the real gold Bangla number alongside the
existing synthetic BanglaParaphrase proxy and the LaBSE baseline, so the two
Bangla metrics can be sanity-checked against each other.

`datasets` must be imported before `torch` (see data/loaders.py docstring).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import datasets  # noqa: F401 -- import-order fix, must precede torch

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cogembed.data.loaders import load_bnpc_pairs
from cogembed.models.backbone import BackboneConfig, apply_whitening, fit_whitening
from cogembed.models.cognitive_embedding import CognitiveEmbeddingModel

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def cosine_sim_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_n = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)
    b_n = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-9)
    return (a_n * b_n).sum(axis=-1)


def auroc_for(model, pairs) -> dict:
    def embed_all(sentences):
        with torch.no_grad():
            return np.stack([model.embed(s).numpy() for s in sentences])

    e1, e2 = embed_all([p["s1"] for p in pairs]), embed_all([p["s2"] for p in pairs])
    labels = np.array([p["label"] for p in pairs])
    raw_auroc = roc_auc_score(labels, cosine_sim_np(e1, e2))

    fit = np.concatenate([e1, e2], axis=0)
    mu, w = fit_whitening(fit)
    e1w, e2w = apply_whitening(e1, mu, w), apply_whitening(e2, mu, w)
    white_auroc = roc_auc_score(labels, cosine_sim_np(e1w, e2w))
    return {"raw_auroc": float(raw_auroc), "whitened_auroc": float(white_auroc), "n_pairs": len(pairs)}


def labse_auroc_for(pairs) -> dict:
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("sentence-transformers/LaBSE")
    mdl = AutoModel.from_pretrained("sentence-transformers/LaBSE")
    mdl.eval()

    @torch.no_grad()
    def labse_embed(sentence: str) -> np.ndarray:
        enc = tok(sentence, return_tensors="pt", truncation=True, max_length=64)
        out = mdl(**enc)
        mf = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (out.last_hidden_state * mf).sum(1) / mf.sum(1).clamp(min=1e-9)
        return torch.nn.functional.normalize(pooled, dim=-1)[0].numpy()

    def embed_all(sentences):
        return np.stack([labse_embed(s) for s in sentences])

    e1, e2 = embed_all([p["s1"] for p in pairs]), embed_all([p["s2"] for p in pairs])
    labels = np.array([p["label"] for p in pairs])
    return {"raw_auroc": float(roc_auc_score(labels, cosine_sim_np(e1, e2))), "n_pairs": len(pairs)}


def main() -> None:
    print("Loading frozen backbone + trained core checkpoint...")
    model = CognitiveEmbeddingModel(BackboneConfig())
    core_ckpt = RESULTS_DIR / "core_model.pt"
    if core_ckpt.exists():
        model.core.load_state_dict(torch.load(core_ckpt))
        print(f"Loaded trained core from {core_ckpt}")
    else:
        print("No trained core checkpoint found -- evaluating UNTRAINED core.")

    bnpc_test = load_bnpc_pairs("test")
    print(f"\n=== BnPC (real gold Bangla paraphrase, n={len(bnpc_test)}) ===")
    ours_bnpc = auroc_for(model, bnpc_test)
    print(f"  ours raw_auroc={ours_bnpc['raw_auroc']:.4f} whitened_auroc={ours_bnpc['whitened_auroc']:.4f}")

    print("\n=== LaBSE on BnPC (same gold test set) ===")
    labse_bnpc = labse_auroc_for(bnpc_test)
    print(f"  labse raw_auroc={labse_bnpc['raw_auroc']:.4f}")

    results = {
        "bnpc_ours": ours_bnpc,
        "bnpc_labse": labse_bnpc,
    }
    out_path = RESULTS_DIR / "tables" / "bnpc_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
