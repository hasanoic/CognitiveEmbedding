"""Supplementary check -- evaluate_multilingual.py only reported WHITENED
'ours' scores. Given whitening has already reversed direction (hurt rather
than helped) on STS12, STS15, and BnPC in earlier runs of this project, the
apparent weak/negative training effect on Bangla and Telugu needs to be
checked against the RAW (unwhitened) number before concluding anything about
the training itself -- this isolates whitening-methodology artifacts from
genuine architecture/training effects. Re-embeds with the current
core_model.pt only (no baseline/LaBSE recompute -- those are unaffected by
this question and already logged).
"""

from __future__ import annotations

import sys
from pathlib import Path

import datasets  # noqa: F401

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cogembed.data.loaders import load_bnpc_pairs, load_semrel_arabic, load_semrel_hindi, load_semrel_telugu, load_stsb
from cogembed.models.backbone import BackboneConfig, apply_whitening, fit_whitening
from cogembed.models.cognitive_embedding import CognitiveEmbeddingModel

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

LANGUAGES = {
    "english": ("sts", lambda: load_stsb("test")),
    "bangla": ("auroc", lambda: load_bnpc_pairs("test")),
    "telugu": ("sts", lambda: load_semrel_telugu("test")),
    "hindi": ("sts", lambda: load_semrel_hindi("test")),
    "arabic": ("sts", lambda: load_semrel_arabic("test")),
}


def cosine_sim_np(a, b):
    a_n = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)
    b_n = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-9)
    return (a_n * b_n).sum(axis=-1)


def score(kind, sims, pairs):
    if kind == "sts":
        gold = np.array([p["score"] for p in pairs])
        rho, _ = spearmanr(sims, gold)
        return float(rho)
    labels = np.array([p["label"] for p in pairs])
    return float(roc_auc_score(labels, sims))


def main():
    print("Loading trained core checkpoint...")
    model = CognitiveEmbeddingModel(BackboneConfig())
    model.core.load_state_dict(torch.load(RESULTS_DIR / "core_model.pt"))

    def embed_all(sentences):
        with torch.no_grad():
            return np.stack([model.embed(s).numpy() for s in sentences])

    for lang, (kind, loader) in LANGUAGES.items():
        pairs = loader()
        e1, e2 = embed_all([p["s1"] for p in pairs]), embed_all([p["s2"] for p in pairs])
        raw_score = score(kind, cosine_sim_np(e1, e2), pairs)

        fit = np.concatenate([e1, e2], axis=0)
        mu, w = fit_whitening(fit)
        e1w, e2w = apply_whitening(e1, mu, w), apply_whitening(e2, mu, w)
        white_score = score(kind, cosine_sim_np(e1w, e2w), pairs)

        print(f"{lang}: raw={raw_score:.4f} whitened={white_score:.4f} (n={len(pairs)})")


if __name__ == "__main__":
    main()
