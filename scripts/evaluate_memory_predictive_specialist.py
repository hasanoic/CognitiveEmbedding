"""Extends Memory-Aware Retrieval and the Predictive Head to SPECIALIST
monolingual backbones (RoBERTa for English, BanglaBERT for Bangla) -- the
gap explicitly flagged after the last results table: these two blocks had
only ever been run on the shared XLM-R backbone, unlike the core embedding
(Attention+Composition), which was validated on both shared AND specialist
backbones. Scoped to English + Bangla only, matching the two languages
with discourse-ordered corpora already available in this project (Telugu/
Hindi/Arabic have no discourse corpus loaded yet -- extending those would
need new corpus sourcing, a separate task, not silently added here).

Reuses the SpecialistBackbone wrapper from train_specialist_backbones.py
and the task logic from evaluate_memory_predictive_multilingual.py, with
FRESH Predictive Head training per backbone (a predictive head trained on
XLM-R features cannot be reused on a different backbone's representation
space -- different model, not just a different tokenizer).

Task provenance (neither is a named external benchmark -- both are
evaluation protocols this project built on top of real corpora, stated
explicitly so they aren't mistaken for standard leaderboard datasets):
  - Memory-Aware Retrieval task: query + 5 true-context (the sentences
    immediately preceding it in its own document) + 5 distractors sampled
    uniformly at random from other documents (not distance- or topic-
    matched -- see the paper's Section 5.5 for why we describe it this
    way), built from Wikipedia documents (en: enwiki via wikitext-2, bn:
    Bangla Wikipedia).
  - Predictive Head task: discourse next-sentence discrimination, built
    from narrative news corpora (en: CNN/DailyMail, bn: XL-Sum Bengali).

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
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cogembed.data.loaders import (
    RANDOM_SEED,
    load_bangla_wikipedia_documents,
    load_cnn_dailymail_documents,
    load_wikitext_documents,
    load_xlsum_bengali_documents,
)
from cogembed.losses import info_nce_loss
from cogembed.models.memory_retrieval import content_only_score
from cogembed.models.predictive_head import PredictiveHead

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
MEMORY_SIZE, N_DISTRACTORS = 5, 5
PREDICTIVE_EPOCHS, BATCH_SIZE, LR, TEMPERATURE = 15, 32, 1e-3, 0.05

BACKBONES = {"english": "roberta-base", "bangla": "csebuetnlp/banglabert"}
MEMORY_CORPORA = {"english": lambda: load_wikitext_documents(500), "bangla": lambda: load_bangla_wikipedia_documents(500)}
PREDICTIVE_CORPORA = {"english": lambda: load_cnn_dailymail_documents(300), "bangla": lambda: load_xlsum_bengali_documents(300)}


class SpecialistBackbone:
    """Same interface as FrozenMultilingualBackbone -- copied from
    train_specialist_backbones.py rather than imported, to keep this
    script self-contained (see that file for the original)."""

    def __init__(self, repo: str):
        self.tokenizer = AutoTokenizer.from_pretrained(repo)
        self.model = AutoModel.from_pretrained(repo)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.hidden_dim = self.model.config.hidden_size

    @torch.no_grad()
    def encode_tokens(self, sentence: str):
        enc = self.tokenizer(sentence, return_tensors="pt", truncation=True, max_length=64)
        out = self.model(**enc, output_hidden_states=True)
        layers = torch.stack(out.hidden_states, dim=0)
        avg = layers.mean(dim=0)[0]
        mask = enc["attention_mask"][0]
        return avg, mask


def mean_pool(h, m):
    mf = m.unsqueeze(-1).float()
    return (h * mf).sum(0) / mf.sum(0).clamp(min=1e-9)


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


def evaluate_predictive(backbone, documents) -> dict:
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

    def encode_batch(batch):
        return torch.stack([head.encode(h, m) for h, m in batch])

    def recall_at_1(a, b):
        a, b = torch.nn.functional.normalize(a, dim=-1), torch.nn.functional.normalize(b, dim=-1)
        preds = (a @ b.T).argmax(dim=-1)
        return (preds == torch.arange(a.shape[0])).float().mean().item()

    def recall_at_k(a, b, k):
        """Whole-val-set retrieval: for each of the N=len(val_pairs) query
        sentences, rank all N candidate next-sentences by cosine similarity
        and check whether the true next-sentence is in the top-k. Candidate
        set size is therefore len(val_pairs), not a fixed pool -- disclosed
        explicitly in the returned dict since it differs per language."""
        a, b = torch.nn.functional.normalize(a, dim=-1), torch.nn.functional.normalize(b, dim=-1)
        sims = a @ b.T
        k = min(k, sims.shape[1])
        topk = sims.topk(k, dim=-1).indices
        targets = torch.arange(a.shape[0]).unsqueeze(1)
        return (topk == targets).any(dim=1).float().mean().item()

    head.eval()
    with torch.no_grad():
        v_cur0, v_next0 = encode_batch(val_cur), encode_batch(val_next)
        untrained_r1 = recall_at_1(head.predict_next(v_cur0), v_next0)
        untrained_r3 = recall_at_k(head.predict_next(v_cur0), v_next0, 3)
        untrained_r5 = recall_at_k(head.predict_next(v_cur0), v_next0, 5)

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
    with torch.no_grad():
        v_cur_f, v_next_f = encode_batch(val_cur), encode_batch(val_next)
        pred_f = head.predict_next(v_cur_f)
        trained_r3 = recall_at_k(pred_f, v_next_f, 3)
        trained_r5 = recall_at_k(pred_f, v_next_f, 5)

    n_candidates = len(val_pairs)
    chance_r1 = 1.0 / n_candidates if n_candidates else 0.0
    chance_r3 = min(3, n_candidates) / n_candidates if n_candidates else 0.0
    chance_r5 = min(5, n_candidates) / n_candidates if n_candidates else 0.0
    return {
        "candidate_set_size": n_candidates,
        "untrained_recall@1": untrained_r1, "trained_recall@1": best_val, "chance_recall@1": chance_r1,
        "untrained_recall@3": untrained_r3, "trained_recall@3": trained_r3, "chance_recall@3": chance_r3,
        "untrained_recall@5": untrained_r5, "trained_recall@5": trained_r5, "chance_recall@5": chance_r5,
        "n_train_pairs": len(train_pairs), "n_val_pairs": len(val_pairs),
    }, head


def main():
    t_start = time.time()
    results = {"memory": {}, "predictive": {}}

    for lang, repo in BACKBONES.items():
        print(f"\n{'=' * 15} {lang} ({repo}) {'=' * 15}")
        backbone = SpecialistBackbone(repo)

        print(f"  Memory-Aware Retrieval ({lang})...")
        documents = MEMORY_CORPORA[lang]()
        r = evaluate_memory(backbone, documents)
        results["memory"][lang] = r
        print(f"    auroc={r['auroc']:.4f} (n_items={r['n_retrieval_items']})")

        print(f"  Predictive Head ({lang})...")
        documents = PREDICTIVE_CORPORA[lang]()
        r, head = evaluate_predictive(backbone, documents)
        results["predictive"][lang] = r
        torch.save(head.state_dict(), RESULTS_DIR / f"predictive_head_specialist_{lang}.pt")
        print(f"    candidates={r['candidate_set_size']}  "
              f"R@1: trained={r['trained_recall@1']:.4f} untrained={r['untrained_recall@1']:.4f} chance={r['chance_recall@1']:.4f}  "
              f"R@3: trained={r['trained_recall@3']:.4f} chance={r['chance_recall@3']:.4f}  "
              f"R@5: trained={r['trained_recall@5']:.4f} chance={r['chance_recall@5']:.4f}")

    print("\n=== Summary (specialist monolingual backbones) ===")
    print("Memory-Aware Retrieval (AUROC, 0.5=chance):")
    for lang, r in results["memory"].items():
        print(f"  {lang} ({BACKBONES[lang]}): {r['auroc']:.4f}")
    print("Predictive Head (Recall@k, candidate set size = full val split, disclosed per language):")
    for lang, r in results["predictive"].items():
        print(f"  {lang} ({BACKBONES[lang]}), candidates={r['candidate_set_size']}: "
              f"R@1 trained={r['trained_recall@1']:.4f} chance={r['chance_recall@1']:.4f}  "
              f"R@3 trained={r['trained_recall@3']:.4f} chance={r['chance_recall@3']:.4f}  "
              f"R@5 trained={r['trained_recall@5']:.4f} chance={r['chance_recall@5']:.4f}")

    out_path = RESULTS_DIR / "tables" / "memory_predictive_specialist.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults written to {out_path}")
    print(f"Total time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
