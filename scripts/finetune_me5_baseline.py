"""Reduced-epoch, reduced-data full fine-tune of multilingual-E5-base on
English STS-B -- closes the last remaining "missing baseline" gap named by
the peer review: every other comparison against E5/LaBSE in this paper uses
them zero-shot, with no training on our data at all.

This is explicitly NOT the same protocol as the main experiments. Fully
fine-tuning a 278M-parameter backbone (no frozen-feature caching -- every
step is a real forward+backward through the whole model) was measured at
42.3s/step (batch=8) on this project's CPU-only hardware; matching the main
experiments' full protocol (2,200 STS-B training pairs, many epochs) would
take multiple hours for a single language. Instead, this runs a genuinely
reduced protocol -- a small training subset, few epochs -- exactly the
"even a reduced-epoch protocol would be informative" fallback the review
itself suggested, and reports it as such rather than presenting it as
equivalent to the main experiments' scale.

Evaluation is on the FULL STS-B test set (1,379 pairs), same metric
(Spearman correlation) and same mean-pooling convention as every other
comparison in this paper, so the result is directly comparable to
Table tab:baseline-head and the existing zero-shot mE5 number in
results/tables/extra_baselines_results.json.

`datasets` must be imported before `torch` (see data/loaders.py docstring).
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import datasets  # noqa: F401

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cogembed.data.loaders import RANDOM_SEED, load_stsb

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
REPO = "intfloat/multilingual-e5-base"
TRAIN_N = 300  # reduced from the main experiments' 2,200 -- see module docstring
EPOCHS = 2
BATCH_SIZE = 8
LR = 2e-5
TEMPERATURE = 0.05
MAX_LENGTH = 64


def mean_pool(last_hidden_state, attention_mask):
    mf = attention_mask.unsqueeze(-1).float()
    return (last_hidden_state * mf).sum(1) / mf.sum(1).clamp(min=1e-9)


def embed_batch(tokenizer, model, sentences, prefix="query: "):
    enc = tokenizer([prefix + s for s in sentences], return_tensors="pt", padding=True,
                     truncation=True, max_length=MAX_LENGTH)
    out = model(**enc)
    return mean_pool(out.last_hidden_state, enc["attention_mask"])


def info_nce_loss(e1, e2, temperature):
    e1n, e2n = F.normalize(e1, dim=-1), F.normalize(e2, dim=-1)
    logits = e1n @ e2n.T / temperature
    labels = torch.arange(e1.shape[0])
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


def main():
    t0 = time.time()
    torch.manual_seed(RANDOM_SEED)
    rng = random.Random(RANDOM_SEED)

    print(f"Loading {REPO} (full fine-tune, no frozen backbone)...")
    tokenizer = AutoTokenizer.from_pretrained(REPO)
    model = AutoModel.from_pretrained(REPO)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    train_all = load_stsb("train")
    train_pairs = [p for p in train_all if p["score"] >= 3.0]
    rng.shuffle(train_pairs)
    train_pairs = train_pairs[:TRAIN_N]
    test_pairs = load_stsb("test")
    print(f"Reduced training set: {len(train_pairs)} pairs (main experiments use ~2,200). "
          f"Test set (unchanged, full): {len(test_pairs)} pairs.")

    print(f"\nFine-tuning ({EPOCHS} epochs, batch={BATCH_SIZE}, lr={LR})...")
    for epoch in range(EPOCHS):
        perm = np.random.RandomState(RANDOM_SEED + epoch).permutation(len(train_pairs))
        total_loss, n_steps = 0.0, 0
        for start in range(0, len(train_pairs), BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            if len(idx) < 2:
                continue
            batch = [train_pairs[i] for i in idx]
            optimizer.zero_grad()
            e1 = embed_batch(tokenizer, model, [p["s1"] for p in batch])
            e2 = embed_batch(tokenizer, model, [p["s2"] for p in batch])
            loss = info_nce_loss(e1, e2, TEMPERATURE)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_steps += 1
            print(f"  epoch {epoch} step {n_steps}/{(len(train_pairs) + BATCH_SIZE - 1) // BATCH_SIZE} "
                  f"loss={loss.item():.4f}  ({time.time() - t0:.0f}s elapsed)")
        print(f"  epoch {epoch} mean_loss={total_loss / max(n_steps, 1):.4f}")

    print("\nEvaluating on full STS-B test set...")
    model.eval()

    def embed_all(sentences):
        embs = []
        with torch.no_grad():
            for i in range(0, len(sentences), 16):
                batch = sentences[i:i + 16]
                embs.append(embed_batch(tokenizer, model, batch).numpy())
        return np.concatenate(embs, axis=0)

    e1 = embed_all([p["s1"] for p in test_pairs])
    e2 = embed_all([p["s2"] for p in test_pairs])
    e1n = e1 / (np.linalg.norm(e1, axis=-1, keepdims=True) + 1e-9)
    e2n = e2 / (np.linalg.norm(e2, axis=-1, keepdims=True) + 1e-9)
    sims = (e1n * e2n).sum(-1)
    gold = np.array([p["score"] for p in test_pairs])
    rho, _ = spearmanr(sims, gold)

    result = {
        "model": REPO, "train_n": len(train_pairs), "epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR,
        "test_n": len(test_pairs), "finetuned_spearman": float(rho),
        "seconds": time.time() - t0,
        "note": "reduced-epoch, reduced-data protocol -- see module docstring; not the same scale as the main experiments",
    }
    print(f"\n=== Reduced fine-tune of {REPO} on English STS-B ===")
    print(f"  trained on {len(train_pairs)} pairs, {EPOCHS} epochs")
    print(f"  test Spearman (fine-tuned): {rho:.4f}")
    print("  (compare against results/tables/extra_baselines_results.json's zero-shot me5_base "
          "and paper Table tab:baseline-head)")

    out_path = RESULTS_DIR / "tables" / "me5_finetuned_reduced.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nResults written to {out_path}")
    print(f"Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
