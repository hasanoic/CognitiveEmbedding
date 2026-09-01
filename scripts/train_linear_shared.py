"""Linear-head counterpart to train.py's Stages 0-1, on the SHARED XLM-R-base
backbone -- closes a gap a reviewer identified: the fair linear-head
baseline (scripts/baseline_linear_head.py, Table tab:baseline-head) was only
ever run on the specialist per-language backbones for the similarity/
paraphrase task. The downstream classification-transfer evaluation
(evaluate_classification.py, Table tab:classification) has never had a
linear-head comparison at all, only "Ours" (cognitive embedding) vs LaBSE --
and it MUST use the shared cross-lingual backbone, not a specialist one,
because cross-lingual classifier transfer requires all languages to live in
the same embedding space (see evaluate_classification.py's docstring).

This script trains a LinearProjectionHead (the identical class used in
baseline_linear_head.py: mean-pool + one nn.Linear(768,768), no attention,
no composition) through the SAME two stages as train.py's Stage 0
(NLI pretrain) and Stage 1 (EN/BN/TE multilingual joint fine-tune), on the
same shared XLM-R-base backbone, same data, same hyperparameters -- the only
difference from train.py is the head architecture. Stage 2 (Predictive Head)
is skipped: it is irrelevant to the classification-transfer question this
experiment exists to answer.

Saves to results/core_model_linear.pt. Does not touch or overwrite
core_model.pt (the cognitive-embedding checkpoint already used for every
published number). Feeds into train_linear_crosslingual.py next.

`datasets` must be imported before `torch` (see data/loaders.py docstring).
"""

from __future__ import annotations

import random
import time
from pathlib import Path

import datasets  # noqa: F401 -- import-order fix, must precede torch

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cogembed.data.loaders import (
    RANDOM_SEED,
    load_bnpc_pairs,
    load_nli_triplets,
    load_semrel_telugu,
    load_stsb,
)
from cogembed.losses import info_nce_loss, info_nce_loss_hard_negatives
from cogembed.models.backbone import BackboneConfig, mean_pool
from cogembed.models.cognitive_embedding import CognitiveEmbeddingModel

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Same hyperparameters as train.py's defaults -- kept identical so the only
# variable between this run and the cognitive-embedding one is the head.
CORE_TRAIN_SIZE, CORE_VAL_SIZE, BN_TRAIN_SIZE = 2200, 500, 2200
MULTILINGUAL_EPOCHS = 15
NLI_TRIPLETS, NLI_EPOCHS = 8000, 6
BATCH_SIZE, LR, TEMPERATURE = 32, 1e-3, 0.05


class LinearProjectionHead(nn.Module):
    """Identical to baseline_linear_head.py's class -- mean-pool (the
    project's canonical mean_pool from backbone.py) + one linear layer.
    Duplicated here (not imported) because baseline_linear_head.py is a
    standalone script, not a module other scripts import from."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, token_features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.proj(mean_pool(token_features, mask))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def cosine_sim_np(a, b):
    a_n = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)
    b_n = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-9)
    return (a_n * b_n).sum(axis=-1)


def train_nli_pretrain(model: CognitiveEmbeddingModel) -> dict:
    triplets = load_nli_triplets(max_premises=NLI_TRIPLETS)
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(triplets)
    n_val = max(1, int(len(triplets) * 0.05))
    val_triplets, train_triplets = triplets[:n_val], triplets[n_val:]
    print(f"[nli] Train triplets: {len(train_triplets)}, Val triplets: {len(val_triplets)}")

    print("[nli] Caching token features (frozen backbone forward pass, done once)...")

    def cache(sents):
        return [model.backbone.encode_tokens(s) for s in sents]

    train_anchor = cache([t["anchor"] for t in train_triplets])
    train_pos = cache([t["positive"] for t in train_triplets])
    train_neg = cache([t["hard_negative"] for t in train_triplets])
    val_anchor = cache([t["anchor"] for t in val_triplets])
    val_pos = cache([t["positive"] for t in val_triplets])
    val_neg = cache([t["hard_negative"] for t in val_triplets])

    optimizer = torch.optim.Adam(model.core.parameters(), lr=LR)

    def pool_batch(cache_batch):
        return torch.stack([model.embed_tokens(h, m) for h, m in cache_batch])

    n_train = len(train_triplets)
    best_val, best_state = float("inf"), None
    for epoch in range(NLI_EPOCHS):
        model.core.train()
        perm = np.random.RandomState(RANDOM_SEED + epoch).permutation(n_train)
        losses = []
        for start in range(0, n_train, BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            if len(idx) < 2:
                continue
            optimizer.zero_grad()
            a = pool_batch([train_anchor[i] for i in idx])
            p = pool_batch([train_pos[i] for i in idx])
            n = pool_batch([train_neg[i] for i in idx])
            loss = info_nce_loss_hard_negatives(a, p, n, TEMPERATURE)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        model.core.eval()
        with torch.no_grad():
            va, vp, vn = pool_batch(val_anchor), pool_batch(val_pos), pool_batch(val_neg)
            val_loss = info_nce_loss_hard_negatives(va, vp, vn, TEMPERATURE).item()
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.core.state_dict().items()}
        print(f"[nli] epoch {epoch:2d} train_loss={np.mean(losses) if losses else float('nan'):.4f} val_loss={val_loss:.4f} (best={best_val:.4f})")

    model.core.load_state_dict(best_state)
    return {"best_val_nli_loss": best_val, "n_train_triplets": n_train}


def train_multilingual(model: CognitiveEmbeddingModel) -> dict:
    from scipy.stats import spearmanr

    rng = random.Random(RANDOM_SEED)

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

    print(f"[multi] EN train/val: {len(en_train)}/{len(en_val)}  BN train/val: {len(bn_train)}/{len(bn_val)}  TE train/val: {len(te_train)}/{len(te_val)}")
    print("[multi] Caching token features for EN+BN+TE (frozen backbone forward pass, done once)...")

    def cache_pairs(pairs):
        c1 = [model.backbone.encode_tokens(r["s1"]) for r in pairs]
        c2 = [model.backbone.encode_tokens(r["s2"]) for r in pairs]
        return c1, c2

    en_c1, en_c2 = cache_pairs(en_train)
    bn_c1, bn_c2 = cache_pairs(bn_train)
    te_c1, te_c2 = cache_pairs(te_train)
    en_v1, en_v2 = cache_pairs(en_val)
    bn_v1, bn_v2 = cache_pairs(bn_val)
    te_v1, te_v2 = cache_pairs(te_val)

    en_val_gold = np.array([r["score"] for r in en_val])
    bn_val_labels = np.array([r["label"] for r in bn_val])
    te_val_gold = np.array([r["score"] for r in te_val])

    optimizer = torch.optim.Adam(model.core.parameters(), lr=LR)

    def pool_batch(cache_batch):
        return torch.stack([model.embed_tokens(h, m) for h, m in cache_batch])

    languages = [("en", en_c1, en_c2), ("bn", bn_c1, bn_c2), ("te", te_c1, te_c2)]

    best_macro, best_state = -1.0, None
    for epoch in range(MULTILINGUAL_EPOCHS):
        model.core.train()
        epoch_rng = random.Random(RANDOM_SEED + epoch)
        all_batches = []
        for lang, c1, c2 in languages:
            idx = list(range(len(c1)))
            epoch_rng.shuffle(idx)
            for start in range(0, len(idx), BATCH_SIZE):
                chunk = idx[start:start + BATCH_SIZE]
                if len(chunk) >= 2:
                    all_batches.append((lang, c1, c2, chunk))
        epoch_rng.shuffle(all_batches)

        losses = {"en": [], "bn": [], "te": []}
        for lang, c1, c2, chunk in all_batches:
            optimizer.zero_grad()
            e1 = pool_batch([c1[i] for i in chunk])
            e2 = pool_batch([c2[i] for i in chunk])
            loss = info_nce_loss(e1, e2, TEMPERATURE)
            loss.backward()
            optimizer.step()
            losses[lang].append(loss.item())

        model.core.eval()
        with torch.no_grad():
            ev1, ev2 = pool_batch(en_v1).numpy(), pool_batch(en_v2).numpy()
            bv1, bv2 = pool_batch(bn_v1).numpy(), pool_batch(bn_v2).numpy()
            tv1, tv2 = pool_batch(te_v1).numpy(), pool_batch(te_v2).numpy()
        en_rho, _ = spearmanr(cosine_sim_np(ev1, ev2), en_val_gold)
        bn_auroc = roc_auc_score(bn_val_labels, cosine_sim_np(bv1, bv2))
        te_rho, _ = spearmanr(cosine_sim_np(tv1, tv2), te_val_gold)
        macro = (en_rho + bn_auroc + te_rho) / 3
        if macro > best_macro:
            best_macro = macro
            best_state = {k: v.clone() for k, v in model.core.state_dict().items()}
        print(
            f"[multi] epoch {epoch:2d} losses(en/bn/te)="
            f"{np.mean(losses['en']) if losses['en'] else float('nan'):.4f}/"
            f"{np.mean(losses['bn']) if losses['bn'] else float('nan'):.4f}/"
            f"{np.mean(losses['te']) if losses['te'] else float('nan'):.4f} | "
            f"val en_spearman={en_rho:.4f} bn_auroc={bn_auroc:.4f} te_spearman={te_rho:.4f} "
            f"macro={macro:.4f} (best={best_macro:.4f})"
        )

    model.core.load_state_dict(best_state)
    torch.save(best_state, RESULTS_DIR / "core_model_linear.pt")
    return {"best_macro_val": best_macro}


def main() -> None:
    set_seed(RANDOM_SEED)
    start = time.time()

    print("Loading frozen backbone (xlm-roberta-base), swapping core for LinearProjectionHead...")
    model = CognitiveEmbeddingModel(BackboneConfig())
    model.core = LinearProjectionHead(model.backbone.hidden_dim)

    print("\n=== Stage 0: NLI pretrain (SNLI+MultiNLI, SimCSE-supervised) ===")
    nli_result = train_nli_pretrain(model)

    print("\n=== Stage 1: multilingual joint fine-tune (EN=STS-B, BN=BnPC, TE=SemRel-Telugu) ===")
    core_result = train_multilingual(model)

    print(f"\nTotal training time: {time.time() - start:.1f}s")
    print(f"NLI pretrain best val loss: {nli_result['best_val_nli_loss']:.4f} ({nli_result['n_train_triplets']} triplets)")
    print(f"Core best macro val (mean of en_spearman/bn_auroc/te_spearman): {core_result['best_macro_val']:.4f}")
    print(f"Checkpoint saved to {RESULTS_DIR / 'core_model_linear.pt'}")


if __name__ == "__main__":
    main()
