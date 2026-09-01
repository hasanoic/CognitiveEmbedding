"""Memory-scoring variant comparison behind paper Table tab:memory-variants
(Section 5.5) and the H3 memory-fusion result (Section 5.4). This is a
faithful port into the versioned repository of a script that was originally
developed and run outside it (poc_memory_coala.py, building on
poc_memory_retrieval.py's item construction) -- copied here specifically so
the paper's reproducibility claim (all training scripts for every reported
configuration are public) is actually true for this experiment. The
algorithm, hyperparameters, and random seed are unchanged from the original
run; only the module structure is adapted to reuse this repository's
existing backbone/whitening/memory-scoring classes instead of the
standalone helper scripts the original depended on.

Reproducibility note: the zero-parameter variants (content-only, fixed
decay) reproduce the original run's numbers exactly, since they depend only
on RANDOM_SEED-controlled data construction. The trained variants (learned
decay, CoALA-inspired) depend additionally on PyTorch's model-weight
initialization; an initial port of this script omitted `torch.manual_seed`
before model construction and produced different trained-variant numbers
from the original run as a result (a gap first identified in this
project's predictive-head code and, at the time, not yet checked here
too) -- now fixed by seeding immediately at the start of main().

Task ("distance-matched" retrieval): for each of 500 English Wikipedia
documents (wikitext-2) with more than 5 sentences, the last sentence is the
query, its 5 immediately preceding sentences are the true context (distance
1..5), and 5 sentences sampled uniformly at random from OTHER documents are
distractors -- each distractor is ALSO assigned a distance drawn from the
same 1..5 range, not a fixed larger value, specifically so that a
distance-only signal cannot trivially separate true context from
distractors by construction alone. Items are split 181/38/38 into
train/validation/test (seeded, shuffled once); model selection uses best
validation AUROC, and the reported score for every variant is the held-out
test AUROC.

Four scoring variants are compared:
  1. content_only:      score = cos(query, slot), zero trainable parameters
  2. fixed decay:        score = cos(query, slot) - 0.3 * distance, zero parameters
  3. learned decay:      score = cos(query, slot)/tau - lambda*distance, 2 parameters
  4. CoALA-inspired:     learned key/value projections + softmax retrieval +
                         gated residual integration, 984,577 parameters,
                         trained on WHITENED input features with a
                         multi-positive negative-log-likelihood loss over
                         the softmax retrieval weights (the correct loss for
                         a softmax read mechanism with several valid
                         targets -- per-slot binary cross-entropy is a
                         structural mismatch for this mechanism, a bug this
                         project made once already and does not repeat).

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

from cogembed.data.loaders import RANDOM_SEED, load_wikitext_documents
from cogembed.models.backbone import BackboneConfig, FrozenMultilingualBackbone, apply_whitening, fit_whitening, mean_pool
from cogembed.models.memory_retrieval import CoALAInspiredMemory, LearnedDecayMemory, content_only_score

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
N_DOCUMENTS = 500
MEMORY_SIZE, N_DISTRACTORS = 5, 5
N_EPOCHS, LR, KV_DIM = 40, 1e-3, 128
VAL_FRACTION, TEST_FRACTION = 0.15, 0.15


def build_items(documents, rng):
    """Distance-matched construction: distractors get a distance drawn from
    the same 1..MEMORY_SIZE range as true context, not a fixed larger value
    -- an earlier version fixed distractor distance beyond the true range,
    which let a distance-only score solve the task perfectly by construction
    (see project history). Distance-matching the negatives is what makes
    this a fair test of whether content similarity adds anything beyond
    what distance alone would already predict."""
    all_sentences = [s for doc in documents for s in doc]
    items = []
    for doc in documents:
        if len(doc) <= MEMORY_SIZE:
            continue
        i = len(doc) - 1
        query = doc[i]
        true_context = [(doc[i - k], k) for k in range(1, MEMORY_SIZE + 1)]
        distractors = []
        while len(distractors) < N_DISTRACTORS:
            cand = rng.choice(all_sentences)
            if cand not in doc:
                distractors.append((cand, rng.randint(1, MEMORY_SIZE)))
        items.append({"query": query, "true_context": true_context, "distractors": distractors})
    return items


def encode_items(backbone, item_list):
    query_vecs = []
    per_item_slots = []
    for it in item_list:
        h, m = backbone.encode_tokens(it["query"])
        query_vecs.append(mean_pool(h, m).numpy())
        slots = []
        for sent, dist in it["true_context"]:
            h, m = backbone.encode_tokens(sent)
            slots.append((mean_pool(h, m).numpy(), 1, dist))
        for sent, dist in it["distractors"]:
            h, m = backbone.encode_tokens(sent)
            slots.append((mean_pool(h, m).numpy(), 0, dist))
        per_item_slots.append(slots)
    return query_vecs, per_item_slots


def score_zero_param(query_vecs, per_item_slots, lam=0.0):
    scores, labels = [], []
    for q, slots in zip(query_vecs, per_item_slots):
        q_t = torch.tensor(q, dtype=torch.float32)
        for vec, label, dist in slots:
            v_t = torch.tensor(vec, dtype=torch.float32)
            content = float(content_only_score(q_t, v_t.unsqueeze(0)).item())
            scores.append(content - lam * dist)
            labels.append(label)
    return roc_auc_score(labels, scores)


def prep_split(query_vecs, per_item_slots, mu, w):
    wq = apply_whitening(np.array(query_vecs), mu, w)
    prepped = []
    for idx, slots in enumerate(per_item_slots):
        slot_vecs = np.array([v for v, _, _ in slots])
        labels = torch.tensor([l for _, l, _ in slots], dtype=torch.float32)
        wslots = apply_whitening(slot_vecs, mu, w)
        prepped.append((wq[idx], wslots, labels))
    return prepped


LEARNED_DECAY_LR = 0.05  # NOT the shared LR=1e-3 -- a 2-parameter scalar model needs a much
# larger step size to move meaningfully; matches the original poc_memory_retrieval_v2.py exactly.


def precompute_sims_dists(query_vecs, per_item_slots):
    """content similarity is a fixed frozen-embedding quantity -- precompute it once so
    only tau/lambda need gradients each epoch, matching the original script exactly."""
    out = []
    for q, slots in zip(query_vecs, per_item_slots):
        q_t = torch.tensor(q, dtype=torch.float32)
        sims, dists, labels = [], [], []
        for vec, label, dist in slots:
            v_t = torch.tensor(vec, dtype=torch.float32)
            sims.append(float(content_only_score(q_t, v_t.unsqueeze(0)).item()))
            dists.append(dist)
            labels.append(label)
        out.append((torch.tensor(sims), torch.tensor(dists, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)))
    return out


def train_learned_decay(train_q, train_slots, val_q, val_slots, test_q, test_slots):
    model = LearnedDecayMemory()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNED_DECAY_LR)
    train_data = precompute_sims_dists(train_q, train_slots)
    val_data = precompute_sims_dists(val_q, val_slots)
    test_data = precompute_sims_dists(test_q, test_slots)

    def epoch_pass(data, train):
        all_scores, all_labels = [], []
        for sims, dists, labels in data:
            if train:
                optimizer.zero_grad()
            scores = model(sims, dists)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(scores, labels)
            if train:
                loss.backward()
                optimizer.step()
            all_scores.extend(scores.detach().tolist())
            all_labels.extend(labels.tolist())
        return roc_auc_score(all_labels, all_scores)

    best_val, best_state = -1.0, None
    for epoch in range(N_EPOCHS):
        epoch_pass(train_data, train=True)
        with torch.no_grad():
            val_auroc = epoch_pass(val_data, train=False)
        if val_auroc > best_val:
            best_val, best_state = val_auroc, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    with torch.no_grad():
        return epoch_pass(test_data, train=False)


def train_coala(train_prepped, val_prepped, test_prepped, hidden_dim):
    memory = CoALAInspiredMemory(hidden_dim, kv_dim=KV_DIM)
    optimizer = torch.optim.Adam(memory.parameters(), lr=LR)

    def epoch_pass(prepped, train):
        all_alpha, all_labels = [], []
        total_loss = 0.0
        for wq, wslots, labels in prepped:
            if train:
                optimizer.zero_grad()
            h_query = torch.tensor(wq, dtype=torch.float32)
            mem_items = torch.tensor(wslots, dtype=torch.float32)
            k = memory.W_k(mem_items)
            v = memory.W_v(mem_items)
            h_proj = memory.W_k(h_query)
            tau = memory.tau.clamp(min=0.05)
            sims = (k @ h_proj) / (tau * (k.norm(dim=-1) * h_proj.norm() + 1e-9))
            alpha = torch.softmax(sims, dim=0)
            true_mass = (alpha * labels).sum().clamp(min=1e-9)
            loss = -torch.log(true_mass)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            all_alpha.extend(alpha.detach().tolist())
            all_labels.extend(labels.tolist())
        return total_loss / len(prepped), roc_auc_score(all_labels, all_alpha)

    best_val, best_state = -1.0, None
    print(f"\nTraining CoALA-inspired semantic memory ({N_EPOCHS} epochs, multi-positive NLL loss, on WHITENED features)...")
    for epoch in range(N_EPOCHS):
        train_loss, train_auroc = epoch_pass(train_prepped, train=True)
        with torch.no_grad():
            _, val_auroc = epoch_pass(val_prepped, train=False)
        if val_auroc > best_val:
            best_val, best_state = val_auroc, {k: v.clone() for k, v in memory.state_dict().items()}
        if epoch % 5 == 0 or epoch == N_EPOCHS - 1:
            print(f"  epoch {epoch:2d}  train_loss={train_loss:.4f} train_auroc={train_auroc:.4f}  val_auroc={val_auroc:.4f} (best={best_val:.4f})  tau={memory.tau.item():.3f}")
    memory.load_state_dict(best_state)
    with torch.no_grad():
        _, test_auroc = epoch_pass(test_prepped, train=False)
    return test_auroc, sum(p.numel() for p in memory.parameters())


def main():
    start = time.time()
    torch.manual_seed(RANDOM_SEED)
    backbone = FrozenMultilingualBackbone(BackboneConfig())
    print("Loading discourse-ordered corpus (wikitext-2)...")
    documents = load_wikitext_documents(N_DOCUMENTS)
    print(f"Loaded {len(documents)} documents")

    rng = random.Random(RANDOM_SEED)
    items = build_items(documents, rng)
    rng.shuffle(items)
    n_val = max(1, int(len(items) * VAL_FRACTION))
    n_test = max(1, int(len(items) * TEST_FRACTION))
    val_items, test_items = items[:n_val], items[n_val:n_val + n_test]
    train_items = items[n_val + n_test:]
    print(f"Items -- train: {len(train_items)}, val: {len(val_items)}, test: {len(test_items)}")

    print("Caching per-token multi-layer-averaged features for train/val/test (slow part, done once)...")
    train_q, train_slots = encode_items(backbone, train_items)
    val_q, val_slots = encode_items(backbone, val_items)
    test_q, test_slots = encode_items(backbone, test_items)
    print(f"Caching done in {time.time() - start:.1f}s.")

    results = {"n_documents": N_DOCUMENTS, "n_train": len(train_items), "n_val": len(val_items), "n_test": len(test_items)}

    results["content_only"] = {"auroc": score_zero_param(test_q, test_slots, lam=0.0), "trainable_params": 0}
    results["fixed_decay"] = {"auroc": score_zero_param(test_q, test_slots, lam=0.3), "trainable_params": 0}
    print(f"content_only test AUROC: {results['content_only']['auroc']:.4f}")
    print(f"fixed_decay test AUROC: {results['fixed_decay']['auroc']:.4f}")

    ld_auroc = train_learned_decay(train_q, train_slots, val_q, val_slots, test_q, test_slots)
    results["learned_decay"] = {"auroc": ld_auroc, "trainable_params": 2}
    print(f"learned_decay test AUROC: {ld_auroc:.4f}")

    all_train_vecs = np.array(train_q + [v for slots in train_slots for v, _, _ in slots])
    mu, w = fit_whitening(all_train_vecs)
    train_prepped = prep_split(train_q, train_slots, mu, w)
    val_prepped = prep_split(val_q, val_slots, mu, w)
    test_prepped = prep_split(test_q, test_slots, mu, w)
    coala_auroc, coala_params = train_coala(train_prepped, val_prepped, test_prepped, backbone.hidden_dim)
    results["coala_inspired"] = {"auroc": coala_auroc, "trainable_params": coala_params}
    print(f"coala_inspired test AUROC: {coala_auroc:.4f}  (params={coala_params:,})")

    print("\n=== Summary (English, shared XLM-R-base backbone, held-out test split) ===")
    for name in ["content_only", "fixed_decay", "learned_decay", "coala_inspired"]:
        r = results[name]
        print(f"  {name}: test_auroc={r['auroc']:.4f}  trainable_params={r['trainable_params']}")

    out_path = RESULTS_DIR / "tables" / "memory_scoring_variants.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults written to {out_path}")
    print(f"Total time: {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
