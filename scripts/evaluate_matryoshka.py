"""Accuracy-vs-size Pareto evaluation for the Matryoshka-trained checkpoint
(core_model_matryoshka.pt, see train_matryoshka.py). For each of the 5
languages and each nested dimension tier (768/256/128/64), truncates the
SAME trained embedding to the first D dims and scores it -- no retraining
per tier, that's the whole point of Matryoshka training.

Includes a NAIVE-truncation contrast: LaBSE was never Matryoshka-trained, so
chopping its 768-dim output down to the first D dims is not expected to
degrade gracefully (LaBSE's information isn't prioritized into early
dimensions the way MRL training targets). Reporting this side-by-side is
the actual evidence for "the training, not just the truncation, matters" --
without it, a skeptical reviewer could reasonably ask whether ANY embedding
truncates fine and Matryoshka training added nothing.

`datasets` must be imported before `torch` (see data/loaders.py docstring).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import datasets  # noqa: F401

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cogembed.data.loaders import load_bnpc_pairs, load_semrel_arabic, load_semrel_hindi, load_semrel_telugu, load_stsb
from cogembed.losses import MATRYOSHKA_DIMS
from cogembed.models.backbone import BackboneConfig, FrozenMultilingualBackbone, apply_whitening, fit_whitening
from cogembed.models.cognitive_embedding import CognitiveEmbeddingCore

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


def score_at_dims(e1_full: np.ndarray, e2_full: np.ndarray, pairs: list, kind: str, dims) -> dict:
    out = {}
    for d in dims:
        d = min(d, e1_full.shape[-1])
        e1, e2 = e1_full[:, :d], e2_full[:, :d]
        raw = score(kind, cosine_sim_np(e1, e2), pairs)
        fit = np.concatenate([e1, e2], axis=0)
        mu, w = fit_whitening(fit)
        white = score(kind, cosine_sim_np(apply_whitening(e1, mu, w), apply_whitening(e2, mu, w)), pairs)
        out[d] = {"raw": raw, "whitened": white, "best": max(raw, white)}
    return out


def main():
    print("Loading frozen backbone + Matryoshka-trained core...")
    backbone = FrozenMultilingualBackbone(BackboneConfig())
    core = CognitiveEmbeddingCore(backbone.hidden_dim)
    ckpt = RESULTS_DIR / "core_model_matryoshka.pt"
    core.load_state_dict(torch.load(ckpt))
    core.eval()
    print(f"Loaded {ckpt}")

    print("Loading LaBSE (for naive-truncation contrast)...")
    from transformers import AutoModel, AutoTokenizer

    labse_tok = AutoTokenizer.from_pretrained("sentence-transformers/LaBSE")
    labse_mdl = AutoModel.from_pretrained("sentence-transformers/LaBSE")
    labse_mdl.eval()

    @torch.no_grad()
    def labse_embed(sentence):
        enc = labse_tok(sentence, return_tensors="pt", truncation=True, max_length=64)
        out = labse_mdl(**enc)
        mf = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (out.last_hidden_state * mf).sum(1) / mf.sum(1).clamp(min=1e-9)
        return torch.nn.functional.normalize(pooled, dim=-1)[0].numpy()

    def embed_ours(sentences):
        with torch.no_grad():
            out = []
            for s in sentences:
                h, m = backbone.encode_tokens(s)
                out.append(core(h, m).numpy())
            return np.stack(out)

    results = {}
    for lang, (kind, loader) in LANGUAGES.items():
        pairs = loader()
        print(f"\n=== {lang} (n={len(pairs)}) ===")

        e1, e2 = embed_ours([p["s1"] for p in pairs]), embed_ours([p["s2"] for p in pairs])
        ours_by_dim = score_at_dims(e1, e2, pairs, kind, MATRYOSHKA_DIMS)

        l1 = np.stack([labse_embed(p["s1"]) for p in pairs])
        l2 = np.stack([labse_embed(p["s2"]) for p in pairs])
        labse_by_dim = score_at_dims(l1, l2, pairs, kind, MATRYOSHKA_DIMS)

        results[lang] = {"ours": ours_by_dim, "labse_naive_truncation": labse_by_dim}
        for d in MATRYOSHKA_DIMS:
            d_eff = min(d, e1.shape[-1])
            print(f"  dim={d_eff:4d}  ours(best)={ours_by_dim[d_eff]['best']:.4f}   "
                  f"labse_naive_trunc(best)={labse_by_dim[d_eff]['best']:.4f}")

    out_path = RESULTS_DIR / "tables" / "matryoshka_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
