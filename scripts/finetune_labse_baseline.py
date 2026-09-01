"""Reduced-epoch, reduced-data full fine-tune of LaBSE on English STS-B --
mirrors finetune_me5_baseline.py exactly, closing the other half of the same
peer-review gap: "the proposed model should be compared with fine-tuned
models." The mE5 reduced fine-tune covered one of the two baselines; every
comparison against LaBSE in this paper was still zero-shot only until this
script.

This is explicitly NOT the same protocol as the main experiments, for the
same reason given for mE5: fully fine-tuning a 470.9M-parameter backbone (no
frozen-feature caching -- every step is a real forward+backward through the
whole model) is far more expensive than this project's main experiments,
which only ever backpropagate through a small trainable head on cached,
frozen features. Runs the same genuinely-reduced protocol used for mE5 (small
training subset, few epochs) so the two results are directly comparable to
each other, not just to the zero-shot numbers.

All 470,926,848 parameters are updated by Adam, including the embedding
table -- the same "fully fine-tuned, no frozen features" protocol used for
mE5, so the two results are directly symmetric. (Development note, not a
property of this protocol: an early run of this script crashed with a CPU
out-of-memory error, traced to Adam's momentum-buffer allocation for LaBSE's
unusually large 501,153-token, 100+-language embedding table -- but the
crash occurred while a second, concurrent process was competing for the same
memory; run alone, as intended, this script completes the full update
without needing to freeze anything. Noted here only so a future run that
hits the same error checks for a concurrent process before concluding the
hardware can't support it.)

Evaluation is on the FULL STS-B test set (1,379 pairs), same metric
(Spearman correlation) and same mean-pooling convention as every other
comparison in this paper (LaBSE needs no query/passage prefix, unlike mE5),
so the result is directly comparable to Table tab:baseline-head and the
existing zero-shot LaBSE number implied by Table tab:specialist's "vs. LaBSE"
column.

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
REPO = "sentence-transformers/LaBSE"
TRAIN_N = 300  # matches finetune_me5_baseline.py -- see module docstring
EPOCHS = 2
BATCH_SIZE = 8
LR = 2e-5
TEMPERATURE = 0.05
MAX_LENGTH = 64


def mean_pool(last_hidden_state, attention_mask):
    mf = attention_mask.unsqueeze(-1).float()
    return (last_hidden_state * mf).sum(1) / mf.sum(1).clamp(min=1e-9)


def embed_batch(tokenizer, model, sentences):
    # No query/passage prefix for LaBSE, unlike mE5 -- matches the zero-shot
    # convention already used for LaBSE elsewhere in this project (see
    # multi_seed_eval.py's labse_embed).
    enc = tokenizer(sentences, return_tensors="pt", padding=True,
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

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  total params: {total_params:,} (all trainable)")

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
        "total_params": total_params, "trainable_params": total_params,
        "seconds": time.time() - t0,
        "note": "reduced-epoch, reduced-data protocol -- see module docstring; not the same scale as the main experiments. All parameters trainable (symmetric with the mE5 fine-tune protocol).",
    }
    print(f"\n=== Reduced fine-tune of {REPO} on English STS-B ===")
    print(f"  trained on {len(train_pairs)} pairs, {EPOCHS} epochs")
    print(f"  test Spearman (fine-tuned): {rho:.4f}")
    print("  (compare against the zero-shot LaBSE number implied by Table tab:specialist's "
          "'vs. LaBSE' column and paper Table tab:baseline-head)")

    out_path = RESULTS_DIR / "tables" / "labse_finetuned_reduced.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nResults written to {out_path}")
    print(f"Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
