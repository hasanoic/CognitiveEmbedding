"""Matryoshka-trained core -- same architecture, data, and two-stage recipe
(NLI pretrain -> multilingual EN/BN/TE fine-tune) as train.py, but using
matryoshka_info_nce_loss(_hard_negatives) instead of the plain InfoNCE
losses (see losses.py docstring for the technique -- Kusupati et al. 2022).
The core module still outputs one 768-dim vector; what changes is that the
first 256/128/64 dims of that SAME vector are also independently trained to
be usable embeddings, so truncating at inference time degrades gracefully
instead of catastrophically. This directly answers "what vector size does
this aim for" (see conversation log) -- not one size, a nested set served by
one model, evaluated across the whole tier by evaluate_matryoshka.py.

Single seed (42, matching the existing reference checkpoint) -- this is
about the SIZE-elasticity property, not a second robustness study; multi-
seed variance is already covered by multi_seed_eval.py for the base model.

Saves to results/core_model_matryoshka.pt -- does NOT touch core_model.pt
(the existing single-seed reference checkpoint already used for the main
reported numbers).

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

from cogembed.data.loaders import RANDOM_SEED, load_bnpc_pairs, load_nli_triplets, load_semrel_telugu, load_stsb
from cogembed.losses import matryoshka_info_nce_loss, matryoshka_info_nce_loss_hard_negatives
from cogembed.models.backbone import BackboneConfig, FrozenMultilingualBackbone
from cogembed.models.cognitive_embedding import CognitiveEmbeddingCore

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
SEED = 42

CORE_TRAIN_SIZE, CORE_VAL_SIZE, BN_TRAIN_SIZE = 2200, 500, 2200
NLI_TRIPLETS, NLI_EPOCHS, MULTI_EPOCHS = 6000, 6, 15
BATCH_SIZE, LR, TEMPERATURE = 32, 1e-3, 0.05


def cosine_sim_np(a, b):
    a_n = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)
    b_n = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-9)
    return (a_n * b_n).sum(axis=-1)


def cache_pairs(backbone, pairs):
    c1 = [backbone.encode_tokens(r["s1"]) for r in pairs]
    c2 = [backbone.encode_tokens(r["s2"]) for r in pairs]
    return c1, c2


def cache_sentences(backbone, sentences):
    return [backbone.encode_tokens(s) for s in sentences]


def pool_batch(core, cache_batch):
    return torch.stack([core(h, m) for h, m in cache_batch])


def main():
    t_start = time.time()
    print("Loading frozen backbone (xlm-roberta-base)...")
    backbone = FrozenMultilingualBackbone(BackboneConfig())
    hidden_dim = backbone.hidden_dim

    rng = random.Random(RANDOM_SEED)

    print("Caching NLI triplets...")
    nli_triplets = load_nli_triplets(max_premises=NLI_TRIPLETS)
    rng.shuffle(nli_triplets)
    n_val = max(1, int(len(nli_triplets) * 0.05))
    nli_val, nli_train = nli_triplets[:n_val], nli_triplets[n_val:]
    nli_train_a = cache_sentences(backbone, [t["anchor"] for t in nli_train])
    nli_train_p = cache_sentences(backbone, [t["positive"] for t in nli_train])
    nli_train_n = cache_sentences(backbone, [t["hard_negative"] for t in nli_train])
    nli_val_a = cache_sentences(backbone, [t["anchor"] for t in nli_val])
    nli_val_p = cache_sentences(backbone, [t["positive"] for t in nli_val])
    nli_val_n = cache_sentences(backbone, [t["hard_negative"] for t in nli_val])
    print(f"  NLI train/val: {len(nli_train)}/{len(nli_val)}")

    print("Caching EN/BN/TE train+val...")
    en_train_all = rng.sample(load_stsb("train"), min(CORE_TRAIN_SIZE, len(load_stsb("train"))))
    en_train = [r for r in en_train_all if r["score"] >= 3.0]
    en_val = rng.sample(load_stsb("validation"), min(CORE_VAL_SIZE, len(load_stsb("validation"))))

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

    en_c1, en_c2 = cache_pairs(backbone, en_train)
    bn_c1, bn_c2 = cache_pairs(backbone, bn_train)
    te_c1, te_c2 = cache_pairs(backbone, te_train)
    en_v1, en_v2 = cache_pairs(backbone, en_val)
    bn_v1, bn_v2 = cache_pairs(backbone, bn_val)
    te_v1, te_v2 = cache_pairs(backbone, te_val)
    en_val_gold = np.array([r["score"] for r in en_val])
    bn_val_labels = np.array([r["label"] for r in bn_val])
    te_val_gold = np.array([r["score"] for r in te_val])
    print(f"  EN train/val: {len(en_train)}/{len(en_val)}  BN: {len(bn_train)}/{len(bn_val)}  TE: {len(te_train)}/{len(te_val)}")
    print(f"Caching done at {time.time() - t_start:.1f}s\n")

    torch.manual_seed(SEED)
    core = CognitiveEmbeddingCore(hidden_dim)

    # ---- Stage 0: NLI pretrain (Matryoshka loss) ----
    print("=== Stage 0: NLI pretrain (Matryoshka) ===")
    optimizer = torch.optim.Adam(core.parameters(), lr=LR)
    n_train = len(nli_train_a)
    best_val, best_state = float("inf"), None
    for epoch in range(NLI_EPOCHS):
        core.train()
        perm = np.random.RandomState(SEED + epoch).permutation(n_train)
        losses = []
        for start in range(0, n_train, BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            if len(idx) < 2:
                continue
            optimizer.zero_grad()
            a = pool_batch(core, [nli_train_a[i] for i in idx])
            p = pool_batch(core, [nli_train_p[i] for i in idx])
            n = pool_batch(core, [nli_train_n[i] for i in idx])
            loss = matryoshka_info_nce_loss_hard_negatives(a, p, n, TEMPERATURE)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        core.eval()
        with torch.no_grad():
            va, vp, vn = pool_batch(core, nli_val_a), pool_batch(core, nli_val_p), pool_batch(core, nli_val_n)
            val_loss = matryoshka_info_nce_loss_hard_negatives(va, vp, vn, TEMPERATURE).item()
        if val_loss < best_val:
            best_val, best_state = val_loss, {k: v.clone() for k, v in core.state_dict().items()}
        print(f"[nli] epoch {epoch:2d} train_loss={np.mean(losses):.4f} val_loss={val_loss:.4f} (best={best_val:.4f})")
    core.load_state_dict(best_state)

    # ---- Stage 1: multilingual joint fine-tune (Matryoshka loss) ----
    print("\n=== Stage 1: multilingual joint fine-tune (Matryoshka) ===")
    optimizer = torch.optim.Adam(core.parameters(), lr=LR)
    languages = [("en", en_c1, en_c2), ("bn", bn_c1, bn_c2), ("te", te_c1, te_c2)]
    best_macro, best_state = -1.0, None
    for epoch in range(MULTI_EPOCHS):
        core.train()
        epoch_rng = random.Random(SEED + epoch)
        all_batches = []
        for lang, c1, c2 in languages:
            idx = list(range(len(c1)))
            epoch_rng.shuffle(idx)
            for start in range(0, len(idx), BATCH_SIZE):
                chunk = idx[start:start + BATCH_SIZE]
                if len(chunk) >= 2:
                    all_batches.append((c1, c2, chunk))
        epoch_rng.shuffle(all_batches)
        for c1, c2, chunk in all_batches:
            optimizer.zero_grad()
            e1 = pool_batch(core, [c1[i] for i in chunk])
            e2 = pool_batch(core, [c2[i] for i in chunk])
            loss = matryoshka_info_nce_loss(e1, e2, TEMPERATURE)
            loss.backward()
            optimizer.step()
        core.eval()
        with torch.no_grad():
            ev1, ev2 = pool_batch(core, en_v1).numpy(), pool_batch(core, en_v2).numpy()
            bv1, bv2 = pool_batch(core, bn_v1).numpy(), pool_batch(core, bn_v2).numpy()
            tv1, tv2 = pool_batch(core, te_v1).numpy(), pool_batch(core, te_v2).numpy()
        # Model selection on the FULL 768-dim macro val score (matches the base
        # model's selection criterion -- full-size fidelity is still the primary
        # objective; smaller tiers are a bonus property, not what we optimize
        # checkpoint choice for).
        en_rho, _ = spearmanr(cosine_sim_np(ev1, ev2), en_val_gold)
        bn_auroc = roc_auc_score(bn_val_labels, cosine_sim_np(bv1, bv2))
        te_rho, _ = spearmanr(cosine_sim_np(tv1, tv2), te_val_gold)
        macro = (en_rho + bn_auroc + te_rho) / 3
        if macro > best_macro:
            best_macro, best_state = macro, {k: v.clone() for k, v in core.state_dict().items()}
        print(f"[multi] epoch {epoch:2d} val en={en_rho:.4f} bn={bn_auroc:.4f} te={te_rho:.4f} macro={macro:.4f} (best={best_macro:.4f})")
    core.load_state_dict(best_state)

    torch.save(best_state, RESULTS_DIR / "core_model_matryoshka.pt")
    print(f"\nTotal time: {time.time() - t_start:.1f}s")
    print(f"Best macro val (768-dim): {best_macro:.4f}")
    print(f"Saved to {RESULTS_DIR / 'core_model_matryoshka.pt'}")


if __name__ == "__main__":
    main()
