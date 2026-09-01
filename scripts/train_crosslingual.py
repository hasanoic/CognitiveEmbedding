"""Cross-lingual alignment fine-tune -- Stage 2 on top of the existing
core_model.pt (NLI pretrain + EN/BN/TE monolingual joint fine-tune, see
train.py). Added because evaluate_bitext_retrieval.py revealed a large gap
(R@1 ~0.26-0.46 vs LaBSE's ~1.0) that the per-language STS/AUROC numbers
never surfaced: monolingual similarity training ("these two English
sentences mean the same thing") gives the model no reason to learn
CROSS-LINGUAL alignment ("this English sentence and this Bangla sentence
mean the same thing") -- different skills, and only the first was ever
trained.

SCOPE CHANGE, stated explicitly (not silent): Hindi and Arabic were
deliberately kept as PURE zero-shot languages in train.py (no training
signal in any form) to test transfer via the shared backbone alone. This
script adds FLORES parallel pairs for all four non-English languages,
including Hindi and Arabic -- because the bitext gap affected all four
roughly equally, not just the untrained-monolingual ones. After this stage,
Hindi/Arabic are no longer "pure zero-shot": they remain zero-shot on
MONOLINGUAL similarity (never trained on Hindi-Hindi or Arabic-Arabic
pairs) but DO receive cross-lingual alignment signal (English-Hindi,
English-Arabic parallel pairs). This is a different, still-meaningful
transfer question, not the same claim as before -- report accordingly.

Uses FLORES `dev` split (997 pairs/language, see loaders.py) for training,
split into train/val here; `devtest` (used by evaluate_bitext_retrieval.py)
stays untouched for the final evaluation, so there's no leakage.

Also re-checks the 5-language monolingual test sets after this stage, to
catch catastrophic forgetting -- adding a new training objective on top of
an already-converged checkpoint risks eroding what Stage 0/1 already
learned; this needs to be measured, not assumed safe.

Saves to results/core_model_crosslingual.pt -- does not overwrite
core_model.pt.

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
    load_flores_parallel_pairs,
    load_semrel_arabic,
    load_semrel_hindi,
    load_semrel_telugu,
    load_stsb,
)
from cogembed.losses import info_nce_loss
from cogembed.models.backbone import BackboneConfig, apply_whitening, fit_whitening
from cogembed.models.cognitive_embedding import CognitiveEmbeddingModel

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
CROSSLINGUAL_LANGS = ["bangla", "telugu", "hindi", "arabic"]
EPOCHS, BATCH_SIZE, LR, TEMPERATURE = 12, 32, 5e-4, 0.05
_SMOKE_TRUNCATE_N = None  # None for the real run -- caps pairs/language for a fast smoke test

TEST_LANGUAGES = {
    "english": ("sts", lambda: load_stsb("test")),
    "bangla": ("auroc", lambda: load_bnpc_pairs("test")),
    "telugu": ("sts", lambda: load_semrel_telugu("test")),
    "hindi": ("sts", lambda: load_semrel_hindi("test")),
    "arabic": ("sts", lambda: load_semrel_arabic("test")),
}


def cosine_sim_np(a, b):
    """Element-wise (row i of a vs row i of b) -- for STS/AUROC pair scoring."""
    a_n = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)
    b_n = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-9)
    return (a_n * b_n).sum(axis=-1)


def cosine_sim_matrix(a, b):
    """Full pairwise (row i of a vs EVERY row of b) -- for retrieval/Recall@1."""
    a_n = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)
    b_n = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-9)
    return a_n @ b_n.T


def score(kind, sims, pairs):
    if kind == "sts":
        gold = np.array([p["score"] for p in pairs])
        rho, _ = spearmanr(sims, gold)
        return float(rho)
    labels = np.array([p["label"] for p in pairs])
    return float(roc_auc_score(labels, sims))


def recall_at_1(sims: np.ndarray) -> float:
    n = sims.shape[0]
    preds = sims.argmax(axis=1)
    return float((preds == np.arange(n)).mean())


def main():
    t_start = time.time()
    print("Loading frozen backbone + existing core_model.pt...")
    model = CognitiveEmbeddingModel(BackboneConfig())
    model.core.load_state_dict(torch.load(RESULTS_DIR / "core_model.pt"))

    rng = random.Random(RANDOM_SEED)

    print("Caching FLORES dev parallel pairs (EN-BN, EN-TE, EN-HI, EN-AR)...")
    train_cache, val_cache = {}, {}
    for lang in CROSSLINGUAL_LANGS:
        pairs = load_flores_parallel_pairs(lang, split="dev")
        rng.shuffle(pairs)
        if _SMOKE_TRUNCATE_N is not None:
            pairs = pairs[:_SMOKE_TRUNCATE_N]
        n_val = max(1, int(len(pairs) * 0.12))
        val_pairs, train_pairs = pairs[:n_val], pairs[n_val:]
        train_cache[lang] = (
            [model.backbone.encode_tokens(p["s1"]) for p in train_pairs],
            [model.backbone.encode_tokens(p["s2"]) for p in train_pairs],
        )
        val_cache[lang] = (
            [model.backbone.encode_tokens(p["s1"]) for p in val_pairs],
            [model.backbone.encode_tokens(p["s2"]) for p in val_pairs],
        )
        print(f"  {lang}: train={len(train_pairs)} val={len(val_pairs)}")

    print(f"Caching done at {time.time() - t_start:.1f}s\n")

    def pool_batch(cache_batch):
        return torch.stack([model.embed_tokens(h, m) for h, m in cache_batch])

    optimizer = torch.optim.Adam(model.core.parameters(), lr=LR)
    best_val_r1, best_state = -1.0, None
    for epoch in range(EPOCHS):
        model.core.train()
        epoch_rng = random.Random(RANDOM_SEED + epoch)
        all_batches = []
        for lang in CROSSLINGUAL_LANGS:
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
            e1 = pool_batch([c1[i] for i in chunk])
            e2 = pool_batch([c2[i] for i in chunk])
            loss = info_nce_loss(e1, e2, TEMPERATURE)
            loss.backward()
            optimizer.step()

        model.core.eval()
        with torch.no_grad():
            r1s = []
            for lang in CROSSLINGUAL_LANGS:
                vc1, vc2 = val_cache[lang]
                v1, v2 = pool_batch(vc1).numpy(), pool_batch(vc2).numpy()
                r1s.append(recall_at_1(cosine_sim_matrix(v1, v2)))
        val_r1 = float(np.mean(r1s))
        if val_r1 > best_val_r1:
            best_val_r1, best_state = val_r1, {k: v.clone() for k, v in model.core.state_dict().items()}
        print(f"[xling] epoch {epoch:2d} val_bitext_recall@1(mean over 4 langs)={val_r1:.4f} "
              f"per-lang={['%.3f' % r for r in r1s]} (best={best_val_r1:.4f})")

    model.core.load_state_dict(best_state)
    torch.save(best_state, RESULTS_DIR / "core_model_crosslingual.pt")
    print(f"\nCross-lingual fine-tune done at {time.time() - t_start:.1f}s, best val bitext R@1={best_val_r1:.4f}")

    # ---- Regression check: did this hurt monolingual STS/AUROC? ----
    print("\n=== Regression check: monolingual test sets (before was multi-seed mean, see conversation log) ===")

    def embed_all(sentences):
        with torch.no_grad():
            return np.stack([model.embed(s).numpy() for s in sentences])

    for lang, (kind, loader) in TEST_LANGUAGES.items():
        pairs = loader()
        if _SMOKE_TRUNCATE_N is not None:
            pairs = pairs[:_SMOKE_TRUNCATE_N]
        e1, e2 = embed_all([p["s1"] for p in pairs]), embed_all([p["s2"] for p in pairs])
        raw = score(kind, cosine_sim_np(e1, e2), pairs)
        fit = np.concatenate([e1, e2], axis=0)
        mu, w = fit_whitening(fit)
        white = score(kind, cosine_sim_np(apply_whitening(e1, mu, w), apply_whitening(e2, mu, w)), pairs)
        print(f"  {lang}: raw={raw:.4f} whitened={white:.4f} best={max(raw, white):.4f}")

    print(f"\nTotal time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
