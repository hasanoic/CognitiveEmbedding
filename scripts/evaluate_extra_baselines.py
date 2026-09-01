"""Additional modern baselines, added per external review feedback that
LaBSE alone (2020) is a dated sole comparison point (see conversation log).
Deliberately narrowed from a much longer suggested list (BGE-M3, mE5, MuRIL,
AraBERT, MiniLM) down to two, chosen for a FAIR, feasible comparison on this
project's CPU-only compute:
  - mE5-base: a modern (2023) multilingual retrieval-focused encoder, sized
    comparably to our own backbone (~278M vs XLM-R-base's ~270M) -- unlike
    BGE-M3 (much larger, and its headline Matryoshka-sizing feature is
    better compared against once we've built our own compression story, not
    before -- see conversation log), this is a same-weight-class comparison.
  - all-MiniLM-L6-v2: English-only, tiny (384-dim, ~22M params) -- cheap to
    run, and directly supports the efficiency-tier comparison for English.
  MuRIL and AraBERT were excluded: both are raw encoders (like our own
  frozen backbone), not off-the-shelf sentence embedders -- a fair
  comparison would require training a NEW pooling head for each, which is
  disproportionate effort for a secondary specialist baseline here.

E5 models require a "query: " prefix on BOTH sides for symmetric similarity
tasks (this is the E5 authors' own evaluation protocol, also how MTEB scores
E5 on STS-style tasks) -- omitting it is a known way to under-report E5's
real performance, so it's applied here deliberately.

Reuses the same 5-language test sets as evaluate_multilingual.py so results
are directly comparable to the existing baseline/ours/LaBSE table.

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


def make_mean_pool_embedder(tok, mdl, prefix: str = ""):
    @torch.no_grad()
    def embed(sentence: str) -> np.ndarray:
        text = f"{prefix}{sentence}" if prefix else sentence
        enc = tok(text, return_tensors="pt", truncation=True, max_length=128)
        out = mdl(**enc)
        mf = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (out.last_hidden_state * mf).sum(1) / mf.sum(1).clamp(min=1e-9)
        return torch.nn.functional.normalize(pooled, dim=-1)[0].numpy()
    return embed


def main():
    from transformers import AutoModel, AutoTokenizer

    print("Loading mE5-base (intfloat/multilingual-e5-base)...")
    e5_tok = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-base")
    e5_mdl = AutoModel.from_pretrained("intfloat/multilingual-e5-base")
    e5_mdl.eval()
    e5_embed = make_mean_pool_embedder(e5_tok, e5_mdl, prefix="query: ")

    print("Loading all-MiniLM-L6-v2 (English only)...")
    mini_tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    mini_mdl = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    mini_mdl.eval()
    mini_embed = make_mean_pool_embedder(mini_tok, mini_mdl)

    results = {}
    for lang, (kind, loader) in LANGUAGES.items():
        pairs = loader()
        print(f"\n=== {lang} (n={len(pairs)}) ===")

        e1 = np.stack([e5_embed(p["s1"]) for p in pairs])
        e2 = np.stack([e5_embed(p["s2"]) for p in pairs])
        e5_score = score(kind, cosine_sim_np(e1, e2), pairs)
        print(f"  mE5-base: {e5_score:.4f}")

        results[lang] = {"n_pairs": len(pairs), "me5_base": e5_score}

        if lang == "english":
            m1 = np.stack([mini_embed(p["s1"]) for p in pairs])
            m2 = np.stack([mini_embed(p["s2"]) for p in pairs])
            mini_score = score(kind, cosine_sim_np(m1, m2), pairs)
            results[lang]["minilm_l6_v2"] = mini_score
            print(f"  MiniLM-L6-v2: {mini_score:.4f}")

    out_path = RESULTS_DIR / "tables" / "extra_baselines_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
