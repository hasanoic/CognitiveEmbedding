"""Ablation: does Semantic Attention and Compositional Aggregator BOTH
contribute to the core embedding, or does one dominate/the other become
redundant once combined? Both were validated SEPARATELY in early POCs
(see semantic_attention.py / compositional_aggregator.py docstrings) before
being fused into CognitiveEmbeddingCore -- but no leave-one-out ablation was
ever run WITHIN the final pipeline to confirm both still pull weight once
combined. This closes that gap directly (see conversation log: "did you do
the experiment").

Three arms, same architecture interface (forward(token_features, mask) ->
[hidden_dim] vector), same data, same caching (via the multi_seed_eval.py
caching-reuse pattern -- one backbone pass, three cheap head-training runs):
  - attention_only:   SemanticAttentionPooling alone
  - composition_only: CompositionalAggregator alone
  - combined:         CognitiveEmbeddingCore (both, concatenated + projected)
                       -- this IS the architecture used everywhere else in
                       this project (multi-seed, mE5 comparison, bitext,
                       specialist backbones).

Trained on EN(STS-B)+BN(BnPC)+TE(SemRel) jointly, same recipe as Stage 1 of
train.py, WITHOUT the NLI pretrain stage -- deliberately, to isolate the
pooling-mechanism question cleanly rather than convolve it with the NLI
pretrain's own effect (already separately established). Single seed (42).

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
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cogembed.data.loaders import (
    RANDOM_SEED,
    load_bnpc_pairs,
    load_semrel_arabic,
    load_semrel_hindi,
    load_semrel_telugu,
    load_stsb,
)
from cogembed.losses import info_nce_loss
from cogembed.models.backbone import BackboneConfig, FrozenMultilingualBackbone, apply_whitening, fit_whitening
from cogembed.models.cognitive_embedding import CognitiveEmbeddingCore
from cogembed.models.compositional_aggregator import CompositionalAggregator
from cogembed.models.semantic_attention import SemanticAttentionPooling

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
SEED = 42
CORE_TRAIN_SIZE, CORE_VAL_SIZE, BN_TRAIN_SIZE = 2200, 500, 2200
MULTI_EPOCHS, BATCH_SIZE, LR, TEMPERATURE = 15, 32, 1e-3, 0.05

TEST_LANGUAGES = {
    "english": ("sts", lambda: load_stsb("test")),
    "bangla": ("auroc", lambda: load_bnpc_pairs("test")),
    "telugu": ("sts", lambda: load_semrel_telugu("test")),
    "hindi": ("sts", lambda: load_semrel_hindi("test")),
    "arabic": ("sts", lambda: load_semrel_arabic("test")),
}

ARMS = {
    "attention_only": lambda hidden_dim: SemanticAttentionPooling(hidden_dim, n_heads=4),
    "composition_only": lambda hidden_dim: CompositionalAggregator(hidden_dim, gru_hidden=128),
    "combined": lambda hidden_dim: CognitiveEmbeddingCore(hidden_dim, n_heads=4, gru_hidden=128),
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


def cache_pairs(backbone, pairs):
    return [backbone.encode_tokens(p["s1"]) for p in pairs], [backbone.encode_tokens(p["s2"]) for p in pairs]


def pool_batch(module, cache_batch):
    return torch.stack([module(h, m) for h, m in cache_batch])


def train_arm(arm_name, hidden_dim, train_cache, val_cache, val_gold_data):
    torch.manual_seed(SEED)
    module = ARMS[arm_name](hidden_dim)
    optimizer = torch.optim.Adam(module.parameters(), lr=LR)
    languages = list(train_cache.keys())

    best_macro, best_state = -1.0, None
    for epoch in range(MULTI_EPOCHS):
        module.train()
        epoch_rng = random.Random(SEED + epoch)
        all_batches = []
        for lang in languages:
            c1, c2 = train_cache[lang]
            idx = list(range(len(c1)))
            epoch_rng.shuffle(idx)
            for start in range(0, len(idx), BATCH_SIZE):
                chunk = idx[start:start + BATCH_SIZE]
                if len(chunk) >= 2:
                    all_batches.append((c1, c2, chunk))
        epoch_rng.shuffle(all_batches)
        for c1, c2, chunk in all_batches:
            optimizer.zero_grad()
            e1 = pool_batch(module, [c1[i] for i in chunk])
            e2 = pool_batch(module, [c2[i] for i in chunk])
            loss = info_nce_loss(e1, e2, TEMPERATURE)
            loss.backward()
            optimizer.step()

        module.eval()
        with torch.no_grad():
            scores = []
            for lang in languages:
                v1, v2 = val_cache[lang]
                vv1, vv2 = pool_batch(module, v1).numpy(), pool_batch(module, v2).numpy()
                kind, gold_or_labels = val_gold_data[lang]
                sims = cosine_sim_np(vv1, vv2)
                if kind == "sts":
                    rho, _ = spearmanr(sims, gold_or_labels)
                    scores.append(rho)
                else:
                    scores.append(roc_auc_score(gold_or_labels, sims))
        macro = float(np.mean(scores))
        if macro > best_macro:
            best_macro, best_state = macro, {k: v.clone() for k, v in module.state_dict().items()}
    module.load_state_dict(best_state)
    return module, best_macro


def main():
    t_start = time.time()
    print("Loading frozen backbone (xlm-roberta-base)...")
    backbone = FrozenMultilingualBackbone(BackboneConfig())
    hidden_dim = backbone.hidden_dim
    rng = random.Random(RANDOM_SEED)

    print("Caching EN/BN/TE train+val (shared across all 3 arms)...")
    en_train_all = rng.sample(load_stsb("train"), CORE_TRAIN_SIZE)
    en_train = [r for r in en_train_all if r["score"] >= 3.0]
    en_val = rng.sample(load_stsb("validation"), CORE_VAL_SIZE)

    bn_train_all = load_bnpc_pairs("train")
    bn_train = [r for r in bn_train_all if r["label"] == 1]
    if len(bn_train) > BN_TRAIN_SIZE:
        bn_train = rng.sample(bn_train, BN_TRAIN_SIZE)
    bn_val = load_bnpc_pairs("validation")
    if len(bn_val) > CORE_VAL_SIZE:
        bn_val = rng.sample(bn_val, CORE_VAL_SIZE)

    te_train_all = load_semrel_telugu("train")
    te_train = [r for r in te_train_all if r["score"] >= 0.5]
    te_val = load_semrel_telugu("dev")

    train_cache = {"en": cache_pairs(backbone, en_train), "bn": cache_pairs(backbone, bn_train), "te": cache_pairs(backbone, te_train)}
    val_cache = {"en": cache_pairs(backbone, en_val), "bn": cache_pairs(backbone, bn_val), "te": cache_pairs(backbone, te_val)}
    val_gold_data = {
        "en": ("sts", np.array([r["score"] for r in en_val])),
        "bn": ("auroc", np.array([r["label"] for r in bn_val])),
        "te": ("sts", np.array([r["score"] for r in te_val])),
    }
    print(f"  EN train/val: {len(en_train)}/{len(en_val)}  BN: {len(bn_train)}/{len(bn_val)}  TE: {len(te_train)}/{len(te_val)}")

    print("Caching 5-language test sets (shared across all 3 arms)...")
    test_cache = {}
    for lang, (kind, loader) in TEST_LANGUAGES.items():
        pairs = loader()
        c1, c2 = cache_pairs(backbone, pairs)
        test_cache[lang] = {"pairs": pairs, "kind": kind, "c1": c1, "c2": c2}
        print(f"  {lang}: {len(pairs)} pairs")
    print(f"Caching done at {time.time() - t_start:.1f}s\n")

    results = {}
    for arm_name in ARMS:
        print(f"=== Training arm: {arm_name} ===")
        t0 = time.time()
        module, best_macro = train_arm(arm_name, hidden_dim, train_cache, val_cache, val_gold_data)
        print(f"  best macro val (EN/BN/TE): {best_macro:.4f}  ({time.time() - t0:.1f}s)")

        module.eval()
        arm_results = {}
        with torch.no_grad():
            for lang, d in test_cache.items():
                pairs, kind = d["pairs"], d["kind"]
                e1, e2 = pool_batch(module, d["c1"]).numpy(), pool_batch(module, d["c2"]).numpy()
                raw = score(kind, cosine_sim_np(e1, e2), pairs)
                fit = np.concatenate([e1, e2], axis=0)
                mu, w = fit_whitening(fit)
                white = score(kind, cosine_sim_np(apply_whitening(e1, mu, w), apply_whitening(e2, mu, w)), pairs)
                arm_results[lang] = {"raw": raw, "whitened": white, "best": max(raw, white)}
                print(f"    {lang}: raw={raw:.4f} whitened={white:.4f} best={max(raw, white):.4f}")
        results[arm_name] = {"macro_val": best_macro, "test": arm_results}
        print()

    print("=== Ablation summary (best-of raw/whitened, per language) ===")
    header = "arm".ljust(18) + "".join(lang.ljust(10) for lang in TEST_LANGUAGES)
    print(header)
    for arm_name, r in results.items():
        row = arm_name.ljust(18) + "".join(f"{r['test'][lang]['best']:.4f}".ljust(10) for lang in TEST_LANGUAGES)
        print(row)

    import json
    out_path = RESULTS_DIR / "tables" / "ablation_core_components.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults written to {out_path}")
    print(f"Total time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
