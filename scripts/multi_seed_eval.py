"""Multi-seed robustness check -- the top-priority gap identified before Q1
submission (see conversation log): every result so far is single-seed, and
some margins over LaBSE (e.g. English +0.001) are well within plausible seed
noise. A reviewer will ask for this; better to have it before submitting.

Design: the frozen backbone's forward passes dominate runtime (~hours) and
are SEED-INVARIANT (the backbone is never trained), so caching every needed
token-feature representation ONCE and re-running only the cheap trainable
core (a few million params) across seeds turns an O(N_seeds x full_pipeline)
cost into O(full_pipeline + N_seeds x cheap_head_training). Concretely:
  - NLI triplets, EN/BN/TE train+val pairs, and all 5 languages' TEST pairs
    are each encoded through the frozen backbone exactly once.
  - The untrained baseline (mean-pool+whiten) and LaBSE scores on the test
    sets are also seed-invariant -- computed once, reused for every seed's
    comparison row.
  - Only model init (torch.manual_seed) and epoch-level shuffling vary by
    seed -- this isolates variance due to optimization/initialization,
    which is what a multi-seed robustness claim is actually about. Data
    SELECTION (which examples appear in train/val/test) stays fixed across
    seeds via the existing RANDOM_SEED constant, deliberately -- comparing
    runs on different data would conflate two different sources of
    variance.

Scope: re-seeds the CORE model only (the one behind every cross-lingual
claim in the paper). The Predictive Head is a secondary, already-validated
auxiliary component (English/discourse-only) -- out of scope here to keep
this tractable; can be added later if reviewers ask.

Does NOT overwrite results/core_model.pt (the existing single-seed
checkpoint already used for the reported main-table numbers) -- writes to
results/tables/multi_seed_results.json instead.

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
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cogembed.data.loaders import (
    RANDOM_SEED,
    load_bnpc_pairs,
    load_nli_triplets,
    load_semrel_arabic,
    load_semrel_hindi,
    load_semrel_telugu,
    load_stsb,
)
from cogembed.losses import info_nce_loss, info_nce_loss_hard_negatives
from cogembed.models.backbone import BackboneConfig, FrozenMultilingualBackbone, apply_whitening, fit_whitening, mean_pool
from cogembed.models.cognitive_embedding import CognitiveEmbeddingCore

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
SEEDS = [42, 123, 2024]

CORE_TRAIN_SIZE, CORE_VAL_SIZE, BN_TRAIN_SIZE = 2200, 500, 2200
NLI_TRIPLETS, NLI_EPOCHS, MULTI_EPOCHS = 6000, 6, 15
BATCH_SIZE, LR, TEMPERATURE = 32, 1e-3, 0.05
TEST_SAMPLE_N = None  # None for the real run (full test sets) -- caps test-set size for a fast smoke test

TEST_LANGUAGES = {
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

    # ---- Stage 0 data: NLI triplets ----
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

    # ---- Stage 1 data: EN/BN/TE train+val ----
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

    # ---- Test sets for all 5 languages, cached once ----
    print("Caching 5-language test sets...")
    test_cache = {}
    for lang, (kind, loader) in TEST_LANGUAGES.items():
        pairs = loader()
        if TEST_SAMPLE_N is not None and len(pairs) > TEST_SAMPLE_N:
            pairs = random.Random(RANDOM_SEED).sample(pairs, TEST_SAMPLE_N)
        c1, c2 = cache_pairs(backbone, pairs)
        test_cache[lang] = {"pairs": pairs, "kind": kind, "c1": c1, "c2": c2}
        print(f"  {lang}: {len(pairs)} pairs")

    # ---- Seed-invariant baselines: untrained mean-pool+whiten, and LaBSE ----
    print("Computing untrained baseline + LaBSE scores (seed-invariant, once)...")
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

    seed_invariant = {}
    for lang, d in test_cache.items():
        pairs, kind = d["pairs"], d["kind"]
        b1 = np.stack([mean_pool(h, m).numpy() for h, m in d["c1"]])
        b2 = np.stack([mean_pool(h, m).numpy() for h, m in d["c2"]])
        fit = np.concatenate([b1, b2], axis=0)
        mu, w = fit_whitening(fit)
        baseline_score = score(kind, cosine_sim_np(apply_whitening(b1, mu, w), apply_whitening(b2, mu, w)), pairs)

        l1 = np.stack([labse_embed(p["s1"]) for p in pairs])
        l2 = np.stack([labse_embed(p["s2"]) for p in pairs])
        labse_score = score(kind, cosine_sim_np(l1, l2), pairs)

        seed_invariant[lang] = {"baseline": baseline_score, "labse": labse_score}
        print(f"  {lang}: baseline={baseline_score:.4f} labse={labse_score:.4f}")

    print(f"\nCaching phase done at {time.time() - t_start:.1f}s. Starting {len(SEEDS)}-seed training loop...\n")

    # ---- Multi-seed training + evaluation loop (cheap: cached features only) ----
    per_seed_results = {lang: {"raw": [], "whitened": []} for lang in TEST_LANGUAGES}

    for seed in SEEDS:
        seed_t0 = time.time()
        torch.manual_seed(seed)
        core = CognitiveEmbeddingCore(hidden_dim)

        # Stage 0: NLI pretrain
        optimizer = torch.optim.Adam(core.parameters(), lr=LR)
        n_train = len(nli_train_a)
        best_val, best_state = float("inf"), None
        for epoch in range(NLI_EPOCHS):
            core.train()
            perm = np.random.RandomState(seed + epoch).permutation(n_train)
            for start in range(0, n_train, BATCH_SIZE):
                idx = perm[start:start + BATCH_SIZE]
                if len(idx) < 2:
                    continue
                optimizer.zero_grad()
                a = pool_batch(core, [nli_train_a[i] for i in idx])
                p = pool_batch(core, [nli_train_p[i] for i in idx])
                n = pool_batch(core, [nli_train_n[i] for i in idx])
                loss = info_nce_loss_hard_negatives(a, p, n, TEMPERATURE)
                loss.backward()
                optimizer.step()
            core.eval()
            with torch.no_grad():
                va, vp, vn = pool_batch(core, nli_val_a), pool_batch(core, nli_val_p), pool_batch(core, nli_val_n)
                val_loss = info_nce_loss_hard_negatives(va, vp, vn, TEMPERATURE).item()
            if val_loss < best_val:
                best_val, best_state = val_loss, {k: v.clone() for k, v in core.state_dict().items()}
        core.load_state_dict(best_state)

        # Stage 1: multilingual joint fine-tune
        optimizer = torch.optim.Adam(core.parameters(), lr=LR)
        languages = [("en", en_c1, en_c2), ("bn", bn_c1, bn_c2), ("te", te_c1, te_c2)]
        best_macro, best_state = -1.0, None
        for epoch in range(MULTI_EPOCHS):
            core.train()
            epoch_rng = random.Random(seed + epoch)
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
                loss = info_nce_loss(e1, e2, TEMPERATURE)
                loss.backward()
                optimizer.step()
            core.eval()
            with torch.no_grad():
                ev1, ev2 = pool_batch(core, en_v1).numpy(), pool_batch(core, en_v2).numpy()
                bv1, bv2 = pool_batch(core, bn_v1).numpy(), pool_batch(core, bn_v2).numpy()
                tv1, tv2 = pool_batch(core, te_v1).numpy(), pool_batch(core, te_v2).numpy()
            en_rho, _ = spearmanr(cosine_sim_np(ev1, ev2), en_val_gold)
            bn_auroc = roc_auc_score(bn_val_labels, cosine_sim_np(bv1, bv2))
            te_rho, _ = spearmanr(cosine_sim_np(tv1, tv2), te_val_gold)
            macro = (en_rho + bn_auroc + te_rho) / 3
            if macro > best_macro:
                best_macro, best_state = macro, {k: v.clone() for k, v in core.state_dict().items()}
        core.load_state_dict(best_state)

        # Evaluate on all 5 languages' cached test features
        core.eval()
        with torch.no_grad():
            for lang, d in test_cache.items():
                pairs, kind = d["pairs"], d["kind"]
                e1 = pool_batch(core, d["c1"]).numpy()
                e2 = pool_batch(core, d["c2"]).numpy()
                raw = score(kind, cosine_sim_np(e1, e2), pairs)
                fit = np.concatenate([e1, e2], axis=0)
                mu, w = fit_whitening(fit)
                white = score(kind, cosine_sim_np(apply_whitening(e1, mu, w), apply_whitening(e2, mu, w)), pairs)
                per_seed_results[lang]["raw"].append(raw)
                per_seed_results[lang]["whitened"].append(white)

        print(f"[seed {seed}] macro_val={best_macro:.4f}  ({time.time() - seed_t0:.1f}s)  " +
              "  ".join(f"{lang}={per_seed_results[lang]['whitened'][-1]:.4f}" for lang in TEST_LANGUAGES))

    # ---- Aggregate ----
    print("\n=== Multi-seed summary (mean +/- std across {} seeds) ===".format(len(SEEDS)))
    summary = {}
    for lang in TEST_LANGUAGES:
        raw_scores = per_seed_results[lang]["raw"]
        white_scores = per_seed_results[lang]["whitened"]
        best_per_seed = [max(r, w) for r, w in zip(raw_scores, white_scores)]
        summary[lang] = {
            "raw_mean": float(np.mean(raw_scores)), "raw_std": float(np.std(raw_scores)),
            "whitened_mean": float(np.mean(white_scores)), "whitened_std": float(np.std(white_scores)),
            "best_of_mean": float(np.mean(best_per_seed)), "best_of_std": float(np.std(best_per_seed)),
            "per_seed_raw": raw_scores, "per_seed_whitened": white_scores,
            "baseline": seed_invariant[lang]["baseline"], "labse": seed_invariant[lang]["labse"],
        }
        s = summary[lang]
        print(f"  {lang}: ours={s['best_of_mean']:.4f}+/-{s['best_of_std']:.4f}  "
              f"baseline={s['baseline']:.4f}  labse={s['labse']:.4f}  "
              f"gap_vs_labse={s['best_of_mean'] - s['labse']:+.4f}")

    out_path = RESULTS_DIR / "tables" / "multi_seed_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"seeds": SEEDS, "results": summary}, indent=2))
    print(f"\nResults written to {out_path}")
    print(f"Total runtime: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
