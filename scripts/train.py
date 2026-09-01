"""Full-pipeline training script -- trains the validated CognitiveEmbedding
components on the datasets confirmed accessible in this project (see
src/cogembed/data/registry.py for what's excluded and why).

Three training stages, run sequentially (see cognitive_embedding.py's
docstring for why Predictive Head is NOT forced into one joint forward pass
-- it was only validated at the discourse level):

  Stage 0: CognitiveEmbeddingCore NLI pretrain -- SimCSE-supervised
           contrastive loss (Gao et al. 2021) on SNLI+MultiNLI premise/
           entailment/contradiction triplets, model-selected on NLI val
           loss. Added because the LaBSE comparison showed our STS-B-only
           training (~1,200 pairs) losing to a model pretrained on
           billions of parallel sentences -- more contrastive signal, not
           a new equation, is the fix (see data/loaders.py docstring).
           Skippable via --skip-nli-pretrain to reproduce the earlier,
           STS-B-only result.
  Stage 1: CognitiveEmbeddingCore (Attention + Composition combined) via
           MULTILINGUAL joint InfoNCE fine-tune -- interleaved, per-batch-
           monolingual batches from English (STS-B), Bangla (BnPC), and
           Telugu (SemRel2024), three typologically unrelated language
           families (Germanic, Indo-Aryan, Dravidian), all updating the
           SAME shared core weights. Model-selected on the MACRO-AVERAGE of
           the three languages' own validation metrics, not English alone
           -- selecting on English-only would silently re-bias the
           checkpoint toward English even with multilingual data present.
           This directly tests whether the architecture is language-general
           (comparable gains across unrelated families) rather than an
           English-specific mechanism that happens to transfer via the
           frozen multilingual backbone. See evaluate_multilingual.py for
           the zero-shot-only languages (Hindi, Arabic) that receive NO
           training signal in any form -- the real transfer test.
           Continues from Stage 0's weights, so this is a fine-tune, not a
           from-scratch phase. The previous English-only fine-tuned
           checkpoint is preserved as core_model_en_only.pt for comparison.
  Stage 2: PredictiveHead via InfoNCE next-sentence prediction on
           cnn_dailymail, model-selected on a held-out document split
           (Recall@1). English/discourse only -- multilingual discourse
           corpora were not available, this stage's scope is unchanged.

Memory-Aware Retrieval is NOT trained here -- the validated, recommended
configuration is the zero-parameter content_only scorer (see
memory_retrieval.py), evaluated directly in scripts/evaluate.py.

`datasets` must be imported before `torch` (see data/loaders.py docstring)
-- this is why the import order below looks unusual.
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import datasets  # noqa: F401 -- import-order fix, must precede torch

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cogembed.data.loaders import (
    RANDOM_SEED,
    load_bnpc_pairs,
    load_cnn_dailymail_documents,
    load_nli_triplets,
    load_semrel_telugu,
    load_stsb,
)
from cogembed.losses import info_nce_loss, info_nce_loss_hard_negatives
from cogembed.models.backbone import BackboneConfig
from cogembed.models.cognitive_embedding import CognitiveEmbeddingModel

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def cosine_sim_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_n = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)
    b_n = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-9)
    return (a_n * b_n).sum(axis=-1)


def train_core_nli_pretrain(model: CognitiveEmbeddingModel, args) -> dict:
    triplets = load_nli_triplets(max_premises=args.nli_triplets)
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

    optimizer = torch.optim.Adam(model.core.parameters(), lr=args.lr)

    def pool_batch(cache_batch):
        return torch.stack([model.embed_tokens(h, m) for h, m in cache_batch])

    n_train = len(train_triplets)
    best_val, best_state = float("inf"), None
    for epoch in range(args.nli_epochs):
        model.core.train()
        perm = np.random.RandomState(RANDOM_SEED + epoch).permutation(n_train)
        losses = []
        for start in range(0, n_train, args.batch_size):
            idx = perm[start:start + args.batch_size]
            if len(idx) < 2:
                continue
            optimizer.zero_grad()
            a = pool_batch([train_anchor[i] for i in idx])
            p = pool_batch([train_pos[i] for i in idx])
            n = pool_batch([train_neg[i] for i in idx])
            loss = info_nce_loss_hard_negatives(a, p, n, args.temperature)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        model.core.eval()
        with torch.no_grad():
            va, vp, vn = pool_batch(val_anchor), pool_batch(val_pos), pool_batch(val_neg)
            val_loss = info_nce_loss_hard_negatives(va, vp, vn, args.temperature).item()
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.core.state_dict().items()}
        print(f"[nli] epoch {epoch:2d} train_loss={np.mean(losses) if losses else float('nan'):.4f} val_loss={val_loss:.4f} (best={best_val:.4f})")

    model.core.load_state_dict(best_state)
    return {"best_val_nli_loss": best_val, "n_train_triplets": n_train}


def train_core_multilingual(model: CognitiveEmbeddingModel, args) -> dict:
    """Joint multilingual fine-tune: English (STS-B), Bangla (BnPC), Telugu
    (SemRel2024) -- three unrelated language families, same shared core
    weights, per-batch-monolingual batches (avoids the trivial "same
    language = higher similarity" shortcut a mixed-language batch would let
    the model exploit) interleaved in random order each epoch. Model
    selection uses the macro-average of the three languages' own validation
    metrics -- Spearman for EN/TE (continuous relatedness), AUROC for BN
    (binary paraphrase label). These two metrics aren't on identical scales
    (AUROC floor 0.5 vs Spearman floor 0) so the macro-average is a
    documented, transparent approximation, not a calibrated combined score
    -- good enough to prevent English from silently dominating checkpoint
    selection, not a claim of principled cross-metric equivalence."""
    from scipy.stats import spearmanr

    rng = random.Random(RANDOM_SEED)

    en_train_all = rng.sample(load_stsb("train"), min(args.core_train_size, len(load_stsb("train"))))
    en_train = [r for r in en_train_all if r["score"] >= 3.0]
    en_val = rng.sample(load_stsb("validation"), min(args.core_val_size, len(load_stsb("validation"))))

    bn_train_all = load_bnpc_pairs("train")
    bn_train = [r for r in bn_train_all if r["label"] == 1]
    if len(bn_train) > args.bn_train_size:
        bn_train = rng.sample(bn_train, args.bn_train_size)
    bn_val = load_bnpc_pairs("validation")
    if len(bn_val) > args.core_val_size:
        bn_val = rng.sample(bn_val, args.core_val_size)

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

    optimizer = torch.optim.Adam(model.core.parameters(), lr=args.lr)

    def pool_batch(cache_batch):
        return torch.stack([model.embed_tokens(h, m) for h, m in cache_batch])

    languages = [("en", en_c1, en_c2), ("bn", bn_c1, bn_c2), ("te", te_c1, te_c2)]

    best_macro, best_state = -1.0, None
    for epoch in range(args.multilingual_epochs):
        model.core.train()
        epoch_rng = random.Random(RANDOM_SEED + epoch)
        all_batches = []
        for lang, c1, c2 in languages:
            idx = list(range(len(c1)))
            epoch_rng.shuffle(idx)
            for start in range(0, len(idx), args.batch_size):
                chunk = idx[start:start + args.batch_size]
                if len(chunk) >= 2:
                    all_batches.append((lang, c1, c2, chunk))
        epoch_rng.shuffle(all_batches)

        losses = {"en": [], "bn": [], "te": []}
        for lang, c1, c2, chunk in all_batches:
            optimizer.zero_grad()
            e1 = pool_batch([c1[i] for i in chunk])
            e2 = pool_batch([c2[i] for i in chunk])
            loss = info_nce_loss(e1, e2, args.temperature)
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
    torch.save(best_state, RESULTS_DIR / "core_model.pt")
    return {"best_macro_val": best_macro}


def train_predictive_head(model: CognitiveEmbeddingModel, args) -> dict:
    documents = load_cnn_dailymail_documents(args.predictive_documents)
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(documents)
    n_val = max(1, int(len(documents) * 0.15))
    val_docs, train_docs = documents[:n_val], documents[n_val:]

    def pairs_from(docs):
        out = []
        for doc in docs:
            for i in range(len(doc) - 1):
                out.append((doc[i], doc[i + 1]))
        return out

    train_pairs, val_pairs = pairs_from(train_docs), pairs_from(val_docs)
    print(f"[predictive] Train pairs: {len(train_pairs)}, Val pairs: {len(val_pairs)}")

    print("[predictive] Caching token features...")
    train_cur = [model.backbone.encode_tokens(p[0]) for p in train_pairs]
    train_next = [model.backbone.encode_tokens(p[1]) for p in train_pairs]
    val_cur = [model.backbone.encode_tokens(p[0]) for p in val_pairs]
    val_next = [model.backbone.encode_tokens(p[1]) for p in val_pairs]

    optimizer = torch.optim.Adam(model.predictive_head.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    def encode_batch(cache_batch):
        return torch.stack([model.predictive_head.encode(h, m) for h, m in cache_batch])

    def recall_at_1(a, b):
        a = torch.nn.functional.normalize(a, dim=-1)
        b = torch.nn.functional.normalize(b, dim=-1)
        sims = a @ b.T
        preds = sims.argmax(dim=-1)
        labels = torch.arange(sims.shape[0])
        return (preds == labels).float().mean().item()

    n_train = len(train_pairs)
    best_val, best_state = -1.0, None
    for epoch in range(args.predictive_epochs):
        model.predictive_head.train()
        perm = np.random.RandomState(RANDOM_SEED + epoch).permutation(n_train)
        losses = []
        for start in range(0, n_train, args.batch_size):
            idx = perm[start:start + args.batch_size]
            if len(idx) < 2:
                continue
            optimizer.zero_grad()
            cur = encode_batch([train_cur[i] for i in idx])
            nxt = encode_batch([train_next[i] for i in idx])
            pred = model.predictive_head.predict_next(cur)
            loss = info_nce_loss(pred, nxt, args.temperature)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        model.predictive_head.eval()
        with torch.no_grad():
            v_cur, v_next = encode_batch(val_cur), encode_batch(val_next)
            v_pred = model.predictive_head.predict_next(v_cur)
            val_r1 = recall_at_1(v_pred, v_next)
        scheduler.step(val_r1)
        if val_r1 > best_val:
            best_val = val_r1
            best_state = {k: v.clone() for k, v in model.predictive_head.state_dict().items()}
        print(f"[predictive] epoch {epoch:2d} loss={np.mean(losses) if losses else float('nan'):.4f} val_recall@1={val_r1:.4f} (best={best_val:.4f})")

    model.predictive_head.load_state_dict(best_state)
    torch.save(best_state, RESULTS_DIR / "predictive_head.pt")
    return {"best_val_recall@1": best_val}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--core-train-size", type=int, default=2200, help="English (STS-B) train pool size")
    parser.add_argument("--core-val-size", type=int, default=500, help="per-language validation pool size cap")
    parser.add_argument("--bn-train-size", type=int, default=2200, help="Bangla (BnPC) train pool size")
    parser.add_argument("--multilingual-epochs", type=int, default=15)
    parser.add_argument("--nli-triplets", type=int, default=8000)
    parser.add_argument("--nli-epochs", type=int, default=6)
    parser.add_argument("--skip-nli-pretrain", action="store_true")
    parser.add_argument("--predictive-documents", type=int, default=300)
    parser.add_argument("--predictive-epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--skip-predictive", action="store_true")
    args = parser.parse_args()

    set_seed(RANDOM_SEED)
    start = time.time()

    print("Loading frozen backbone (xlm-roberta-base)...")
    model = CognitiveEmbeddingModel(BackboneConfig())

    old_ckpt = RESULTS_DIR / "core_model.pt"
    if old_ckpt.exists():
        backup_path = RESULTS_DIR / "core_model_en_only.pt"
        backup_path.write_bytes(old_ckpt.read_bytes())
        print(f"Backed up previous (English-only fine-tuned) checkpoint to {backup_path}")

    nli_result = {}
    if not args.skip_nli_pretrain:
        print("\n=== Stage 0: NLI pretrain (SNLI+MultiNLI, SimCSE-supervised) ===")
        nli_result = train_core_nli_pretrain(model, args)

    print("\n=== Stage 1: multilingual joint fine-tune (EN=STS-B, BN=BnPC, TE=SemRel-Telugu) ===")
    core_result = train_core_multilingual(model, args)

    predictive_result = {}
    if not args.skip_predictive:
        print("\n=== Stage 2: training PredictiveHead ===")
        predictive_result = train_predictive_head(model, args)

    print(f"\nTotal training time: {time.time() - start:.1f}s")
    if nli_result:
        print(f"NLI pretrain best val loss: {nli_result['best_val_nli_loss']:.4f} ({nli_result['n_train_triplets']} triplets)")
    print(f"Core best macro val (mean of en_spearman/bn_auroc/te_spearman): {core_result['best_macro_val']:.4f}")
    if predictive_result:
        print(f"Predictive head best val Recall@1: {predictive_result['best_val_recall@1']:.4f}")
    print(f"Checkpoints saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
