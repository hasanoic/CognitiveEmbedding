"""Experiment: does fusing the Predictive Head's signal into the core
embedding help, or does it repeat Memory-Aware Retrieval's negative result
(poc_memory_coala.py: fusing memory content into the embedding via a gated
residual made things WORSE than keeping memory as a separate scorer)? This
is the one fusion combination not yet tested (see conversation log).

Design: the trained predictive_head.pt's `pool_proj` (encode()) is used as
a FROZEN third feature view -- it was already trained via next-sentence
InfoNCE on cnn_dailymail, so its output encodes "predictively useful"
sentence features distinct from what attention/composition learn from a
similarity objective. Freezing it (rather than retraining it jointly on
STS) keeps the question clean: does PREDICTIVELY-trained signal specifically
help similarity judgments, not just "does adding more trainable capacity
help" (a different, less interesting question).

Scope: ENGLISH ONLY (STS-B). The predictive head was only ever trained on
English discourse -- testing this fusion on Bangla/Telugu would confound
"does fusion help" with "does an English-only predictive head transfer
zero-shot," which is a separate, unresolved question. Two arms, same data,
same recipe as train.py's original English-only Stage 1 (pre-multilingual):
  - baseline: CognitiveEmbeddingCore (attention + composition only --
    matches every other reported result in this project)
  - fused:    attention + composition + frozen predictive view, concat +
    project

`datasets` must be imported before `torch` (see data/loaders.py docstring).
"""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path

import datasets  # noqa: F401

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cogembed.data.loaders import RANDOM_SEED, load_stsb
from cogembed.losses import info_nce_loss
from cogembed.models.backbone import BackboneConfig, FrozenMultilingualBackbone, apply_whitening, fit_whitening
from cogembed.models.cognitive_embedding import CognitiveEmbeddingCore
from cogembed.models.compositional_aggregator import CompositionalAggregator
from cogembed.models.predictive_head import PredictiveHead
from cogembed.models.semantic_attention import SemanticAttentionPooling

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
SEED = 42
TRAIN_SIZE, VAL_SIZE, EPOCHS, BATCH_SIZE, LR, TEMPERATURE = 2200, 500, 15, 32, 1e-3, 0.05


class PredictiveFusedCore(nn.Module):
    """attention + composition + FROZEN predictive-head view, concat + project."""

    def __init__(self, hidden_dim: int, frozen_predictive_head: PredictiveHead):
        super().__init__()
        self.attention = SemanticAttentionPooling(hidden_dim, n_heads=4)
        self.composition = CompositionalAggregator(hidden_dim, gru_hidden=128)
        self.frozen_predictive_head = frozen_predictive_head
        for p in self.frozen_predictive_head.parameters():
            p.requires_grad = False
        self.combine = nn.Linear(hidden_dim * 3, hidden_dim)

    def forward(self, token_features, mask):
        attn = self.attention(token_features, mask)
        comp = self.composition(token_features, mask)
        with torch.no_grad():
            pred = self.frozen_predictive_head.encode(token_features, mask)
        return self.combine(torch.cat([attn, comp, pred], dim=-1))


def cosine_sim_np(a, b):
    a_n = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)
    b_n = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-9)
    return (a_n * b_n).sum(axis=-1)


def cache_pairs(backbone, pairs):
    return [backbone.encode_tokens(p["s1"]) for p in pairs], [backbone.encode_tokens(p["s2"]) for p in pairs]


def pool_batch(module, cache_batch):
    return torch.stack([module(h, m) for h, m in cache_batch])


def train_and_eval(name, module, train_cache, val_cache, val_gold, test_cache, test_pairs, test_gold):
    optimizer = torch.optim.Adam([p for p in module.parameters() if p.requires_grad], lr=LR)
    c1, c2 = train_cache
    v1, v2 = val_cache
    best_val, best_state = -1.0, None
    for epoch in range(EPOCHS):
        module.train()
        perm = np.random.RandomState(SEED + epoch).permutation(len(c1))
        for start in range(0, len(c1), BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            if len(idx) < 2:
                continue
            optimizer.zero_grad()
            e1, e2 = pool_batch(module, [c1[i] for i in idx]), pool_batch(module, [c2[i] for i in idx])
            loss = info_nce_loss(e1, e2, TEMPERATURE)
            loss.backward()
            optimizer.step()
        module.eval()
        with torch.no_grad():
            vv1, vv2 = pool_batch(module, v1).numpy(), pool_batch(module, v2).numpy()
        val_rho, _ = spearmanr(cosine_sim_np(vv1, vv2), val_gold)
        if val_rho > best_val:
            best_val, best_state = val_rho, {k: v.clone() for k, v in module.state_dict().items()}
        print(f"  [{name}] epoch {epoch:2d} val_spearman={val_rho:.4f} (best={best_val:.4f})")
    module.load_state_dict(best_state)

    module.eval()
    t1, t2 = test_cache
    with torch.no_grad():
        te1, te2 = pool_batch(module, t1).numpy(), pool_batch(module, t2).numpy()
    raw, _ = spearmanr(cosine_sim_np(te1, te2), test_gold)
    fit = np.concatenate([te1, te2], axis=0)
    mu, w = fit_whitening(fit)
    white, _ = spearmanr(cosine_sim_np(apply_whitening(te1, mu, w), apply_whitening(te2, mu, w)), test_gold)
    return {"best_val": best_val, "test_raw": float(raw), "test_whitened": float(white), "test_best": float(max(raw, white))}


def main():
    t_start = time.time()
    print("Loading frozen backbone + trained predictive head...")
    backbone = FrozenMultilingualBackbone(BackboneConfig())
    hidden_dim = backbone.hidden_dim
    predictive_head = PredictiveHead(hidden_dim)
    predictive_head.load_state_dict(torch.load(RESULTS_DIR / "predictive_head.pt"))
    predictive_head.eval()

    rng = random.Random(RANDOM_SEED)
    train_all = rng.sample(load_stsb("train"), TRAIN_SIZE)
    train_pairs = [r for r in train_all if r["score"] >= 3.0]
    val_pairs = rng.sample(load_stsb("validation"), VAL_SIZE)
    test_pairs = load_stsb("test")
    print(f"Train/val/test: {len(train_pairs)}/{len(val_pairs)}/{len(test_pairs)}")

    print("Caching token features (shared across both arms)...")
    train_cache = cache_pairs(backbone, train_pairs)
    val_cache = cache_pairs(backbone, val_pairs)
    test_cache = cache_pairs(backbone, test_pairs)
    val_gold = np.array([r["score"] for r in val_pairs])
    test_gold = np.array([r["score"] for r in test_pairs])
    print(f"Caching done at {time.time() - t_start:.1f}s\n")

    torch.manual_seed(SEED)
    baseline_core = CognitiveEmbeddingCore(hidden_dim, n_heads=4, gru_hidden=128)
    print("=== Arm 1: baseline (attention + composition) ===")
    baseline_result = train_and_eval("baseline", baseline_core, train_cache, val_cache, val_gold, test_cache, test_pairs, test_gold)

    torch.manual_seed(SEED)
    fused_core = PredictiveFusedCore(hidden_dim, predictive_head)
    print("\n=== Arm 2: fused (attention + composition + frozen predictive view) ===")
    fused_result = train_and_eval("fused", fused_core, train_cache, val_cache, val_gold, test_cache, test_pairs, test_gold)

    print("\n=== Summary ===")
    print(f"  baseline: test_raw={baseline_result['test_raw']:.4f} test_whitened={baseline_result['test_whitened']:.4f} best={baseline_result['test_best']:.4f}")
    print(f"  fused:    test_raw={fused_result['test_raw']:.4f} test_whitened={fused_result['test_whitened']:.4f} best={fused_result['test_best']:.4f}")
    print(f"  delta (fused - baseline, best-of): {fused_result['test_best'] - baseline_result['test_best']:+.4f}")

    import json
    out_path = RESULTS_DIR / "tables" / "ablation_predictive_fusion.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"baseline": baseline_result, "fused": fused_result}, indent=2))
    print(f"\nResults written to {out_path}")
    print(f"Total time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
