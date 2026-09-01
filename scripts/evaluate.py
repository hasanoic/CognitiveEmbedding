"""Full-pipeline evaluation -- loads trained checkpoints (or runs zero-shot
if none exist) and reports every validated metric in one place:
  - Core embedding: English STS-B test Spearman, Bangla paraphrase AUROC
  - Generalization checks (zero-shot, no retraining): SemEval STS12-16
    (broader English), SemRel2024-Hindi (the real cross-lingual test, H5 --
    the core model never saw Hindi during training)
  - Predictive head: discourse next-sentence Recall@1 (held-out documents)
  - Memory-Aware Retrieval: content-only AUROC (zero-parameter, no training
    needed -- see memory_retrieval.py for why this is the recommended
    configuration over the learned-decay/CoALA-inspired alternatives)
  - LaBSE specialist-baseline comparison on the same English/Bangla/Hindi sets

Writes results/tables/full_pipeline_results.json.
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import datasets  # noqa: F401 -- import-order fix, must precede torch

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cogembed.data.loaders import (
    RANDOM_SEED,
    STS_SUITE_REPOS,
    load_bangla_paraphrase_pairs,
    load_semrel_hindi,
    load_sts_suite,
    load_stsb,
    load_wikitext_documents,
)
from cogembed.models.backbone import BackboneConfig, apply_whitening, fit_whitening
from cogembed.models.cognitive_embedding import CognitiveEmbeddingModel
from cogembed.models.memory_retrieval import content_only_score

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def cosine_sim_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_n = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)
    b_n = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-9)
    return (a_n * b_n).sum(axis=-1)


def evaluate_core(model: CognitiveEmbeddingModel) -> dict:
    test_pairs = load_stsb("test")
    bn_pairs = load_bangla_paraphrase_pairs("test", n_positive=200)

    def embed_all(sentences):
        with torch.no_grad():
            return np.stack([model.embed(s).numpy() for s in sentences])

    e1, e2 = embed_all([p["s1"] for p in test_pairs]), embed_all([p["s2"] for p in test_pairs])
    en_gold = np.array([p["score"] for p in test_pairs])
    en_spearman, _ = spearmanr(cosine_sim_np(e1, e2), en_gold)

    b1, b2 = embed_all([p["s1"] for p in bn_pairs]), embed_all([p["s2"] for p in bn_pairs])
    bn_labels = np.array([p["label"] for p in bn_pairs])
    bn_auroc = roc_auc_score(bn_labels, cosine_sim_np(b1, b2))

    # Whitened variants (fit on this same eval pool -- POC-scale simplification,
    # a full run should fit whitening on a separate dev split, see project's
    # data-leakage discipline notes)
    fit_en = np.concatenate([e1, e2], axis=0)
    mu, w = fit_whitening(fit_en)
    e1w, e2w = apply_whitening(e1, mu, w), apply_whitening(e2, mu, w)
    en_spearman_whitened, _ = spearmanr(cosine_sim_np(e1w, e2w), en_gold)

    fit_bn = np.concatenate([b1, b2], axis=0)
    mu_bn, w_bn = fit_whitening(fit_bn)
    b1w, b2w = apply_whitening(b1, mu_bn, w_bn), apply_whitening(b2, mu_bn, w_bn)
    bn_auroc_whitened = roc_auc_score(bn_labels, cosine_sim_np(b1w, b2w))

    return {
        "en_stsb_spearman": float(en_spearman),
        "en_stsb_spearman_whitened": float(en_spearman_whitened),
        "bn_paraphrase_auroc": float(bn_auroc),
        "bn_paraphrase_auroc_whitened": float(bn_auroc_whitened),
        "n_en_test_pairs": len(test_pairs),
        "n_bn_test_pairs": len(bn_pairs),
    }


def evaluate_generalization(model: CognitiveEmbeddingModel) -> dict:
    """Zero-shot generalization: the core model was trained ONLY on English
    STS-B. STS12-16 tests broader English domains; SemRel2024-Hindi is the
    real cross-lingual test (H5) -- Hindi was never seen during training,
    same status as the Bangla eval but with GOLD (not synthetic) labels."""

    def embed_all(sentences):
        with torch.no_grad():
            return np.stack([model.embed(s).numpy() for s in sentences])

    def spearman_for(pairs):
        e1, e2 = embed_all([p["s1"] for p in pairs]), embed_all([p["s2"] for p in pairs])
        gold = np.array([p["score"] for p in pairs])
        raw_rho, _ = spearmanr(cosine_sim_np(e1, e2), gold)
        fit = np.concatenate([e1, e2], axis=0)
        mu, w = fit_whitening(fit)
        e1w, e2w = apply_whitening(e1, mu, w), apply_whitening(e2, mu, w)
        white_rho, _ = spearmanr(cosine_sim_np(e1w, e2w), gold)
        return raw_rho, white_rho, len(pairs)

    results = {}
    for name in STS_SUITE_REPOS:
        pairs = load_sts_suite(name, "test")
        raw_rho, white_rho, n = spearman_for(pairs)
        results[f"{name}_spearman"] = float(raw_rho)
        results[f"{name}_spearman_whitened"] = float(white_rho)
        results[f"{name}_n_pairs"] = n
        print(f"  [gen] {name}: raw={raw_rho:.4f} whitened={white_rho:.4f} (n={n})")

    hindi_pairs = load_semrel_hindi("test")
    hi_raw, hi_white, hi_n = spearman_for(hindi_pairs)
    results["hindi_semrel_str_spearman"] = float(hi_raw)
    results["hindi_semrel_str_spearman_whitened"] = float(hi_white)
    results["hindi_n_pairs"] = hi_n
    print(f"  [gen] hindi (SemRel2024, zero-shot, gold STR): raw={hi_raw:.4f} whitened={hi_white:.4f} (n={hi_n})")
    return results


def evaluate_labse_baseline(en_test_pairs, bn_pairs, hindi_pairs) -> dict:
    """Specialist baseline -- LaBSE, run zero-shot (no training/fine-tuning
    here either, for a fair comparison against the equally zero-shot-on-
    Bangla/Hindi core model). Required per the pipeline doc's baseline list;
    every comparison before this point was only against mean-pool/whitening,
    not against a specialist sentence encoder."""
    from transformers import AutoModel, AutoTokenizer

    print("  Loading LaBSE (sentence-transformers/LaBSE)...")
    tok = AutoTokenizer.from_pretrained("sentence-transformers/LaBSE")
    mdl = AutoModel.from_pretrained("sentence-transformers/LaBSE")
    mdl.eval()

    @torch.no_grad()
    def labse_embed(sentence: str) -> np.ndarray:
        enc = tok(sentence, return_tensors="pt", truncation=True, max_length=64)
        out = mdl(**enc)
        mf = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (out.last_hidden_state * mf).sum(1) / mf.sum(1).clamp(min=1e-9)
        return torch.nn.functional.normalize(pooled, dim=-1)[0].numpy()

    def embed_all(sentences):
        return np.stack([labse_embed(s) for s in sentences])

    e1, e2 = embed_all([p["s1"] for p in en_test_pairs]), embed_all([p["s2"] for p in en_test_pairs])
    en_gold = np.array([p["score"] for p in en_test_pairs])
    en_rho, _ = spearmanr(cosine_sim_np(e1, e2), en_gold)

    b1, b2 = embed_all([p["s1"] for p in bn_pairs]), embed_all([p["s2"] for p in bn_pairs])
    bn_labels = np.array([p["label"] for p in bn_pairs])
    bn_auroc = roc_auc_score(bn_labels, cosine_sim_np(b1, b2))

    h1, h2 = embed_all([p["s1"] for p in hindi_pairs]), embed_all([p["s2"] for p in hindi_pairs])
    hi_gold = np.array([p["score"] for p in hindi_pairs])
    hi_rho, _ = spearmanr(cosine_sim_np(h1, h2), hi_gold)

    return {
        "labse_en_stsb_spearman": float(en_rho),
        "labse_bn_paraphrase_auroc": float(bn_auroc),
        "labse_hindi_semrel_str_spearman": float(hi_rho),
    }


def evaluate_memory_content_only(model: CognitiveEmbeddingModel, n_documents: int = 200) -> dict:
    documents = load_wikitext_documents(n_documents)
    rng = random.Random(RANDOM_SEED)
    memory_size, n_distractors = 5, 5

    all_sentences = [s for doc in documents for s in doc]
    items = []
    for doc in documents:
        if len(doc) <= memory_size:
            continue
        i = len(doc) - 1
        query = doc[i]
        true_context = [doc[i - k] for k in range(1, memory_size + 1)]
        distractors = []
        while len(distractors) < n_distractors:
            cand = rng.choice(all_sentences)
            if cand not in doc:
                distractors.append(cand)
        items.append({"query": query, "context": true_context, "distractors": distractors})

    def mean_pool_embed(sentence: str) -> torch.Tensor:
        h, m = model.backbone.encode_tokens(sentence)
        mf = m.unsqueeze(-1).float()
        return (h * mf).sum(0) / mf.sum(0).clamp(min=1e-9)

    all_scores, all_labels = [], []
    with torch.no_grad():
        for it in items:
            q = mean_pool_embed(it["query"])
            mem = torch.stack([mean_pool_embed(s) for s in it["context"] + it["distractors"]])
            labels = [1] * len(it["context"]) + [0] * len(it["distractors"])
            scores = content_only_score(q, mem)
            all_scores.extend(scores.tolist())
            all_labels.extend(labels)

    auroc = roc_auc_score(all_labels, all_scores)
    return {"memory_content_only_auroc": float(auroc), "n_retrieval_items": len(items)}


def main() -> None:
    start = time.time()
    print("Loading frozen backbone + trained checkpoints (if present)...")
    model = CognitiveEmbeddingModel(BackboneConfig())

    core_ckpt = RESULTS_DIR / "core_model.pt"
    if core_ckpt.exists():
        model.core.load_state_dict(torch.load(core_ckpt))
        print(f"Loaded trained core from {core_ckpt}")
    else:
        print("No trained core checkpoint found -- evaluating UNTRAINED (randomly initialized) core.")

    print("\n=== Evaluating core embedding (EN STS-B + BN paraphrase) ===")
    core_results = evaluate_core(model)
    for k, v in core_results.items():
        print(f"  {k}: {v}")

    print("\n=== Evaluating Memory-Aware Retrieval (content-only, zero-parameter, recommended config) ===")
    memory_results = evaluate_memory_content_only(model)
    for k, v in memory_results.items():
        print(f"  {k}: {v}")

    print("\n=== Zero-shot generalization (STS12-16, Hindi -- never seen during training) ===")
    generalization_results = evaluate_generalization(model)

    print("\n=== LaBSE specialist-baseline comparison (same EN/BN/Hindi test sets) ===")
    labse_results = evaluate_labse_baseline(load_stsb("test"), load_bangla_paraphrase_pairs("test", n_positive=200), load_semrel_hindi("test"))
    for k, v in labse_results.items():
        print(f"  {k}: {v}")

    all_results = {
        **core_results,
        **memory_results,
        **generalization_results,
        **labse_results,
        "runtime_seconds": time.time() - start,
    }
    out_path = RESULTS_DIR / "tables" / "full_pipeline_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nResults written to {out_path}")
    print(f"Total runtime: {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
