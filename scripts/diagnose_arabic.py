"""Cheap, no-training diagnostics requested by peer review specifically for
Arabic: (1) tokenized sequence length and truncation rate at max_length=64
per language, to check whether Arabic text is disproportionately truncated;
(2) an anisotropy proxy (mean pairwise cosine similarity of raw, unwhitened
mean-pooled embeddings) per language, to check whether Arabic's embedding
geometry is unusually collapsed relative to the other four languages. Both
are measurements only -- no training -- so they are cheap enough to run
directly rather than merely disclose as untested.

`datasets` must be imported before `torch` (see data/loaders.py docstring).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import datasets  # noqa: F401

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cogembed.data.loaders import (
    load_bnpc_pairs, load_semrel_arabic, load_semrel_hindi, load_semrel_telugu, load_stsb,
)
from cogembed.models.backbone import mean_pool

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
MAX_LENGTH = 64
N_SAMPLE = 300  # sentences per language for the anisotropy proxy, kept small since this is a measurement, not training

BACKBONES = {
    "english": "roberta-base",
    "bangla": "csebuetnlp/banglabert",
    "telugu": "l3cube-pune/telugu-bert",
    "hindi": "l3cube-pune/hindi-bert-v2",
    "arabic": "aubmindlab/bert-base-arabertv2",
}


def get_sentences(lang: str) -> list[str]:
    if lang == "english":
        pairs = load_stsb("test")
    elif lang == "bangla":
        pairs = load_bnpc_pairs("test")
    elif lang == "telugu":
        pairs = load_semrel_telugu("test")
    elif lang == "hindi":
        pairs = load_semrel_hindi("test")
    elif lang == "arabic":
        pairs = load_semrel_arabic("test")
    sentences = []
    for p in pairs:
        sentences.append(p["s1"])
        sentences.append(p["s2"])
    return sentences


def run_language(lang: str, repo: str) -> dict:
    print(f"\n=== {lang} ({repo}) ===")
    tokenizer = AutoTokenizer.from_pretrained(repo)
    model = AutoModel.from_pretrained(repo)
    model.eval()

    sentences = get_sentences(lang)
    sample = sentences[:N_SAMPLE]

    # 1. tokenized length / truncation rate (no truncation applied here, to see the true length)
    lengths = [len(tokenizer(s, truncation=False)["input_ids"]) for s in sample]
    lengths = np.array(lengths)
    truncated_frac = float((lengths > MAX_LENGTH).mean())

    # 2. anisotropy proxy: mean pairwise cosine similarity of raw mean-pooled embeddings
    embs = []
    with torch.no_grad():
        for s in sample:
            enc = tokenizer(s, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
            out = model(**enc, output_hidden_states=True)
            layers = torch.stack(out.hidden_states, dim=0)
            avg = layers.mean(dim=0)[0]
            mask = enc["attention_mask"][0]
            embs.append(mean_pool(avg, mask).numpy())
    embs = np.stack(embs)
    norm = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9)
    sims = norm @ norm.T
    iu = np.triu_indices(len(sims), k=1)
    mean_pairwise_cos = float(sims[iu].mean())

    result = {
        "n_sentences": len(sample),
        "mean_token_length": float(lengths.mean()),
        "median_token_length": float(np.median(lengths)),
        "max_token_length": int(lengths.max()),
        "fraction_truncated_at_64": truncated_frac,
        "mean_pairwise_cosine_raw_embeddings": mean_pairwise_cos,
    }
    print(f"  mean_token_length={result['mean_token_length']:.1f} median={result['median_token_length']:.1f} "
          f"max={result['max_token_length']} truncated_frac={truncated_frac:.3f}")
    print(f"  anisotropy proxy (mean pairwise cosine, raw embeddings): {mean_pairwise_cos:.4f}")
    return result


def main():
    results = {}
    for lang, repo in BACKBONES.items():
        results[lang] = run_language(lang, repo)

    print("\n=== Summary ===")
    for lang, r in results.items():
        print(f"  {lang}: mean_len={r['mean_token_length']:.1f} truncated={r['fraction_truncated_at_64']:.3f} "
              f"anisotropy={r['mean_pairwise_cosine_raw_embeddings']:.4f}")

    out_path = RESULTS_DIR / "tables" / "arabic_diagnostics.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
