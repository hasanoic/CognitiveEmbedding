"""Extends Memory-Aware Retrieval and the Predictive Head beyond English-
only (see conversation log: both had exactly one English-only validation
run, unlike the core embedding's now-extensive multi-seed/multilingual
testing). Adds Bangla using the SAME task-appropriate corpus logic already
established for English: encyclopedic text (Wikipedia) is fine for Memory-
Aware Retrieval, but the Predictive Head specifically needs real narrative
coherence (XL-Sum Bengali news articles, not Wikipedia) -- mirroring the
finding that wikitext-2 failed for the Predictive Head while cnn_dailymail
worked (see predictive_head.py docstring).

Single run per language (consolidating existing evidence + one new
language, per explicit scope decision -- not a multi-seed study).

Memory-Aware Retrieval: zero-parameter content_only scorer (no training),
operates on raw mean-pooled frozen-backbone features, exactly as originally
validated -- NOT the trained cognitive core embedding (this module was
validated as a standalone scorer, see memory_retrieval.py).

Predictive Head: trained fresh per language (English reuses the existing
results/predictive_head.pt if present; Bangla trains fresh on XL-Sum).

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
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cogembed.data.loaders import (
    RANDOM_SEED,
    load_bangla_wikipedia_documents,
    load_cnn_dailymail_documents,
    load_wikitext_documents,
    load_xlsum_bengali_documents,
)
from cogembed.losses import info_nce_loss
from cogembed.models.backbone import BackboneConfig, FrozenMultilingualBackbone, mean_pool
from cogembed.models.memory_retrieval import content_only_score
from cogembed.models.predictive_head import PredictiveHead

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
MEMORY_SIZE, N_DISTRACTORS = 5, 5
PREDICTIVE_EPOCHS, BATCH_SIZE, LR, TEMPERATURE = 15, 32, 1e-3, 0.05

MEMORY_CORPORA = {
    "english": lambda: load_wikitext_documents(500),
    "bangla": lambda: load_bangla_wikipedia_documents(500),
}
PREDICTIVE_CORPORA = {
    "english": lambda: load_cnn_dailymail_documents(300),
    "bangla": lambda: load_xlsum_bengali_documents(300),
}


def evaluate_memory(backbone, documents) -> dict:
    rng = random.Random(RANDOM_SEED)
    all_sentences = [s for doc in documents for s in doc]
    items = []
    for doc in documents:
        if len(doc) <= MEMORY_SIZE:
            continue
        i = len(doc) - 1
        query = doc[i]
        true_context = [doc[i - k] for k in range(1, MEMORY_SIZE + 1)]
        distractors = []
        while len(distractors) < N_DISTRACTORS:
            cand = rng.choice(all_sentences)
            if cand not in doc:
                distractors.append(cand)
        items.append({"query": query, "context": true_context, "distractors": distractors})

    def embed(sentence):
        h, m = backbone.encode_tokens(sentence)
        return mean_pool(h, m)

    all_scores, all_labels = [], []
    with torch.no_grad():
        for it in items:
            q = embed(it["query"])
            mem = torch.stack([embed(s) for s in it["context"] + it["distractors"]])
            labels = [1] * len(it["context"]) + [0] * len(it["distractors"])
            scores = content_only_score(q, mem)
            all_scores.extend(scores.tolist())
            all_labels.extend(labels)

    auroc = roc_auc_score(all_labels, all_scores)
    return {"auroc": float(auroc), "n_retrieval_items": len(items)}


def evaluate_predictive(backbone, documents, existing_ckpt: Path | None) -> dict:
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(documents)
    n_val = max(1, int(len(documents) * 0.15))
    val_docs, train_docs = documents[:n_val], documents[n_val:]

    def pairs_from(docs):
        return [(doc[i], doc[i + 1]) for doc in docs for i in range(len(doc) - 1)]

    train_pairs, val_pairs = pairs_from(train_docs), pairs_from(val_docs)

    def cache(pairs, idx):
        return [backbone.encode_tokens(p[idx]) for p in pairs]

    train_cur, train_next = cache(train_pairs, 0), cache(train_pairs, 1)
    val_cur, val_next = cache(val_pairs, 0), cache(val_pairs, 1)

    head = PredictiveHead(backbone.hidden_dim)
    if existing_ckpt is not None and existing_ckpt.exists():
        head.load_state_dict(torch.load(existing_ckpt))

    def encode_batch(batch):
        return torch.stack([head.encode(h, m) for h, m in batch])

    def recall_at_1(a, b):
        a, b = torch.nn.functional.normalize(a, dim=-1), torch.nn.functional.normalize(b, dim=-1)
        preds = (a @ b.T).argmax(dim=-1)
        return (preds == torch.arange(a.shape[0])).float().mean().item()

    # Untrained baseline (before any Bangla-specific training) for comparison
    head.eval()
    with torch.no_grad():
        v_cur0, v_next0 = encode_batch(val_cur), encode_batch(val_next)
        untrained_r1 = recall_at_1(head.predict_next(v_cur0), v_next0)

    optimizer = torch.optim.Adam(head.parameters(), lr=LR)
    best_val, best_state = -1.0, None
    for epoch in range(PREDICTIVE_EPOCHS):
        head.train()
        perm = np.random.RandomState(RANDOM_SEED + epoch).permutation(len(train_pairs))
        for start in range(0, len(train_pairs), BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            if len(idx) < 2:
                continue
            optimizer.zero_grad()
            cur, nxt = encode_batch([train_cur[i] for i in idx]), encode_batch([train_next[i] for i in idx])
            pred = head.predict_next(cur)
            loss = info_nce_loss(pred, nxt, TEMPERATURE)
            loss.backward()
            optimizer.step()
        head.eval()
        with torch.no_grad():
            v_cur, v_next = encode_batch(val_cur), encode_batch(val_next)
            val_r1 = recall_at_1(head.predict_next(v_cur), v_next)
        if val_r1 > best_val:
            best_val, best_state = val_r1, {k: v.clone() for k, v in head.state_dict().items()}
        print(f"    epoch {epoch:2d} val_recall@1={val_r1:.4f} (best={best_val:.4f})")
    head.load_state_dict(best_state)

    chance = 1.0 / len(val_pairs) if val_pairs else 0.0
    return {"untrained_recall@1": untrained_r1, "trained_recall@1": best_val, "chance_recall@1": chance,
            "n_train_pairs": len(train_pairs), "n_val_pairs": len(val_pairs)}


def main():
    t_start = time.time()
    print("Loading frozen backbone (xlm-roberta-base)...")
    backbone = FrozenMultilingualBackbone(BackboneConfig())

    results = {"memory": {}, "predictive": {}}

    print("\n=== Memory-Aware Retrieval (content_only, zero-parameter) ===")
    for lang, loader in MEMORY_CORPORA.items():
        print(f"  {lang}...")
        documents = loader()
        r = evaluate_memory(backbone, documents)
        results["memory"][lang] = r
        print(f"    {lang}: auroc={r['auroc']:.4f} (n_items={r['n_retrieval_items']})")

    print("\n=== Predictive Head (discourse next-sentence prediction) ===")
    existing_ckpt = RESULTS_DIR / "predictive_head.pt"
    for lang, loader in PREDICTIVE_CORPORA.items():
        print(f"  {lang}...")
        documents = loader()
        ckpt = existing_ckpt if lang == "english" else None
        r = evaluate_predictive(backbone, documents, ckpt)
        results["predictive"][lang] = r
        print(f"    {lang}: untrained={r['untrained_recall@1']:.4f} trained={r['trained_recall@1']:.4f} "
              f"chance={r['chance_recall@1']:.4f}")

    print("\n=== Summary ===")
    print("Memory-Aware Retrieval (AUROC, 0.5=chance):")
    for lang, r in results["memory"].items():
        print(f"  {lang}: {r['auroc']:.4f}")
    print("Predictive Head (Recall@1, trained vs untrained vs chance):")
    for lang, r in results["predictive"].items():
        print(f"  {lang}: trained={r['trained_recall@1']:.4f} untrained={r['untrained_recall@1']:.4f} chance={r['chance_recall@1']:.4f}")

    out_path = RESULTS_DIR / "tables" / "memory_predictive_multilingual.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults written to {out_path}")
    print(f"Total time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
