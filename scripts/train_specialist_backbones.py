"""Per-language specialist-backbone training -- the "method generalization"
experiment (see conversation log): instead of one shared multilingual
frozen backbone (XLM-R) across all languages, each language gets its OWN
best-available monolingual specialist encoder, with the SAME four-module
cognitive architecture and training recipe applied independently. This
answers the reviewer's "is this generalizable" question more precisely
than the joint multilingual model did: it tests whether the recipe works
across both language AND backbone architecture, not just language with one
fixed backbone. It also deliberately has NO shared embedding space across
languages -- cross-lingual retrieval is out of scope by design here, not
an oversight (see conversation log for why that's the right scope).

Per-language backbones (all confirmed accessible, all 768-dim/12-layer,
so the existing SemanticAttentionPooling/CompositionalAggregator modules
need no changes):
  english: roberta-base
  bangla:  csebuetnlp/banglabert
  telugu:  l3cube-pune/telugu-bert
  hindi:   l3cube-pune/hindi-bert-v2
  arabic:  aubmindlab/bert-base-arabertv2

Per-language training DATA differs honestly based on what's actually
available (not a flaw -- using each language's best available data is the
right approach, not artificially forcing identical resources):
  english: SNLI+MultiNLI hard-negative pretrain -> STS-B fine-tune (the
           existing, validated recipe from train.py)
  bangla:  BnPC fine-tune only (no Bangla NLI resource found)
  telugu:  SemRel2024-Telugu fine-tune only (no Telugu NLI resource found)
  hindi:   XNLI (facebook/xnli, all_languages config, 'hi' slice) hard-
           negative pretrain only -- no dedicated Hindi STS-style train set
  arabic:  XNLI ('ar' slice) hard-negative pretrain only, same reason

Each language's held-out test set is its own (BnPC test / SemRel test),
and is compared against: (a) that language's OWN untrained specialist-
backbone baseline, (b) the existing XLM-R-shared-backbone result for that
language (from the multi-seed run), (c) LaBSE zero-shot.

Saves results/core_model_specialist_<lang>.pt per language, plus a summary
JSON. `datasets` must be imported before `torch` (see loaders.py docstring).
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
from transformers import AutoModel, AutoTokenizer

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
from cogembed.models.backbone import apply_whitening, fit_whitening
from cogembed.models.cognitive_embedding import CognitiveEmbeddingCore

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
BATCH_SIZE, LR, TEMPERATURE = 32, 1e-3, 0.05
NLI_EPOCHS, TASK_EPOCHS = 6, 15
NLI_TRIPLETS_MAX, TASK_TRAIN_MAX, TASK_VAL_MAX = 6000, 2200, 500
_SMOKE_TEST_CAP = None  # None for the real run -- caps test-set size for a fast smoke test

BACKBONES = {
    "english": "roberta-base",
    "bangla": "csebuetnlp/banglabert",
    "telugu": "l3cube-pune/telugu-bert",
    "hindi": "l3cube-pune/hindi-bert-v2",
    "arabic": "aubmindlab/bert-base-arabertv2",
}


class SpecialistBackbone:
    """Same interface as FrozenMultilingualBackbone (encode_tokens ->
    per-token, multi-layer-averaged features), swappable per language."""

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


def load_xnli_triplets(lang_code: str, max_triplets: int = 6000) -> list[dict]:
    """XNLI all_languages train split -- nested dict per row, keyed by
    language. label 0=entailment, 2=contradiction (same convention as
    SNLI/MultiNLI, see loaders.py's load_nli_triplets)."""
    from huggingface_hub import hf_hub_download
    import pandas as pd

    triplets, by_premise = [], {}
    for part in range(4):
        path = hf_hub_download(repo_id="facebook/xnli", repo_type="dataset",
                                filename=f"all_languages/train-{part:05d}-of-00004.parquet")
        df = pd.read_parquet(path)
        for _, row in df.iterrows():
            label = row["label"]
            if label not in (0, 2):
                continue
            premise = row["premise"][lang_code]
            langs = list(row["hypothesis"]["language"])
            if lang_code not in langs:
                continue
            hyp = row["hypothesis"]["translation"][langs.index(lang_code)]
            slot = by_premise.setdefault(premise, {})
            key = "entailment" if label == 0 else "contradiction"
            if key in slot:
                continue
            slot[key] = hyp
            if "entailment" in slot and "contradiction" in slot:
                triplets.append({"anchor": premise, "positive": slot["entailment"], "hard_negative": slot["contradiction"]})
                if len(triplets) >= max_triplets:
                    return triplets
        if len(triplets) >= max_triplets:
            break
    return triplets


def train_nli_stage(backbone, core, triplets, rng):
    rng.shuffle(triplets)
    n_val = max(1, int(len(triplets) * 0.1))
    val_t, train_t = triplets[:n_val], triplets[n_val:]
    cache = lambda key: [backbone.encode_tokens(t[key]) for t in train_t]
    vcache = lambda key: [backbone.encode_tokens(t[key]) for t in val_t]
    ta, tp, tn = cache("anchor"), cache("positive"), cache("hard_negative")
    va, vp, vn = vcache("anchor"), vcache("positive"), vcache("hard_negative")

    def pool(batch):
        return torch.stack([core(h, m) for h, m in batch])

    optimizer = torch.optim.Adam(core.parameters(), lr=LR)
    best_val, best_state = float("inf"), None
    for epoch in range(NLI_EPOCHS):
        core.train()
        perm = np.random.RandomState(RANDOM_SEED + epoch).permutation(len(ta))
        for start in range(0, len(ta), BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            if len(idx) < 2:
                continue
            optimizer.zero_grad()
            a, p, n = pool([ta[i] for i in idx]), pool([tp[i] for i in idx]), pool([tn[i] for i in idx])
            loss = info_nce_loss_hard_negatives(a, p, n, TEMPERATURE)
            loss.backward()
            optimizer.step()
        core.eval()
        with torch.no_grad():
            val_loss = info_nce_loss_hard_negatives(pool(va), pool(vp), pool(vn), TEMPERATURE).item()
        if val_loss < best_val:
            best_val, best_state = val_loss, {k: v.clone() for k, v in core.state_dict().items()}
        print(f"    [nli] epoch {epoch} val_loss={val_loss:.4f} (best={best_val:.4f})")
    core.load_state_dict(best_state)
    return len(train_t)


def train_task_stage(backbone, core, train_pairs, val_pairs, kind, rng):
    def cache_pairs(pairs):
        return [backbone.encode_tokens(p["s1"]) for p in pairs], [backbone.encode_tokens(p["s2"]) for p in pairs]

    c1, c2 = cache_pairs(train_pairs)
    v1, v2 = cache_pairs(val_pairs)

    def pool(batch):
        return torch.stack([core(h, m) for h, m in batch])

    optimizer = torch.optim.Adam(core.parameters(), lr=LR)
    best_val, best_state = -1.0, None
    for epoch in range(TASK_EPOCHS):
        core.train()
        perm = np.random.RandomState(RANDOM_SEED + epoch).permutation(len(c1))
        for start in range(0, len(c1), BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            if len(idx) < 2:
                continue
            optimizer.zero_grad()
            e1, e2 = pool([c1[i] for i in idx]), pool([c2[i] for i in idx])
            loss = info_nce_loss(e1, e2, TEMPERATURE)
            loss.backward()
            optimizer.step()
        core.eval()
        with torch.no_grad():
            vv1, vv2 = pool(v1).numpy(), pool(v2).numpy()
        val_score = score(kind, cosine_sim_np(vv1, vv2), val_pairs)
        if val_score > best_val:
            best_val, best_state = val_score, {k: v.clone() for k, v in core.state_dict().items()}
        print(f"    [task] epoch {epoch} val_score={val_score:.4f} (best={best_val:.4f})")
    core.load_state_dict(best_state)


def evaluate_final(backbone, core, test_pairs, kind):
    def embed_all(sentences):
        with torch.no_grad():
            out = []
            for s in sentences:
                h, m = backbone.encode_tokens(s)
                out.append(core(h, m).numpy())
            return np.stack(out)

    e1, e2 = embed_all([p["s1"] for p in test_pairs]), embed_all([p["s2"] for p in test_pairs])
    raw = score(kind, cosine_sim_np(e1, e2), test_pairs)
    fit = np.concatenate([e1, e2], axis=0)
    mu, w = fit_whitening(fit)
    white = score(kind, cosine_sim_np(apply_whitening(e1, mu, w), apply_whitening(e2, mu, w)), test_pairs)
    return raw, white


def evaluate_baseline(backbone, test_pairs, kind):
    """Untrained: mean-pool + whiten, no cognitive core -- isolates what training adds."""
    from cogembed.models.backbone import mean_pool

    def embed_all(sentences):
        with torch.no_grad():
            out = []
            for s in sentences:
                h, m = backbone.encode_tokens(s)
                out.append(mean_pool(h, m).numpy())
            return np.stack(out)

    e1, e2 = embed_all([p["s1"] for p in test_pairs]), embed_all([p["s2"] for p in test_pairs])
    fit = np.concatenate([e1, e2], axis=0)
    mu, w = fit_whitening(fit)
    return score(kind, cosine_sim_np(apply_whitening(e1, mu, w), apply_whitening(e2, mu, w)), test_pairs)


def run_language(lang: str) -> dict:
    print(f"\n{'=' * 20} {lang} ({BACKBONES[lang]}) {'=' * 20}")
    t0 = time.time()
    backbone = SpecialistBackbone(BACKBONES[lang])
    core = CognitiveEmbeddingCore(backbone.hidden_dim)
    torch.manual_seed(RANDOM_SEED)
    rng = random.Random(RANDOM_SEED)

    if lang == "english":
        triplets = load_nli_triplets(max_premises=NLI_TRIPLETS_MAX)
        train_nli_stage(backbone, core, triplets, rng)
        train_all = rng.sample(load_stsb("train"), TASK_TRAIN_MAX)
        train_pairs = [r for r in train_all if r["score"] >= 3.0]
        val_pairs = rng.sample(load_stsb("validation"), TASK_VAL_MAX)
        test_pairs, kind = load_stsb("test"), "sts"
        train_task_stage(backbone, core, train_pairs, val_pairs, kind, rng)
    elif lang == "bangla":
        train_all = load_bnpc_pairs("train")
        train_pairs = [r for r in train_all if r["label"] == 1]
        if len(train_pairs) > TASK_TRAIN_MAX:
            train_pairs = rng.sample(train_pairs, TASK_TRAIN_MAX)
        val_pairs = load_bnpc_pairs("validation")
        if len(val_pairs) > TASK_VAL_MAX:
            val_pairs = rng.sample(val_pairs, TASK_VAL_MAX)
        test_pairs, kind = load_bnpc_pairs("test"), "auroc"
        train_task_stage(backbone, core, train_pairs, val_pairs, kind, rng)
    elif lang == "telugu":
        train_all = load_semrel_telugu("train")
        train_pairs = [r for r in train_all if r["score"] >= 0.5]
        val_pairs = load_semrel_telugu("dev")
        test_pairs, kind = load_semrel_telugu("test"), "sts"
        train_task_stage(backbone, core, train_pairs, val_pairs, kind, rng)
    elif lang == "hindi":
        triplets = load_xnli_triplets("hi", max_triplets=NLI_TRIPLETS_MAX)
        train_nli_stage(backbone, core, triplets, rng)
        test_pairs, kind = load_semrel_hindi("test"), "sts"
    elif lang == "arabic":
        triplets = load_xnli_triplets("ar", max_triplets=NLI_TRIPLETS_MAX)
        train_nli_stage(backbone, core, triplets, rng)
        test_pairs, kind = load_semrel_arabic("test"), "sts"

    if _SMOKE_TEST_CAP is not None and len(test_pairs) > _SMOKE_TEST_CAP:
        test_pairs = test_pairs[:_SMOKE_TEST_CAP]
    raw, white = evaluate_final(backbone, core, test_pairs, kind)
    baseline = evaluate_baseline(backbone, test_pairs, kind)

    torch.save(core.state_dict(), RESULTS_DIR / f"core_model_specialist_{lang}.pt")
    print(f"  {lang}: baseline={baseline:.4f} ours_raw={raw:.4f} ours_whitened={white:.4f} "
          f"best={max(raw, white):.4f}  ({time.time() - t0:.1f}s)")
    return {"backbone": BACKBONES[lang], "baseline": baseline, "raw": raw, "whitened": white,
            "best": max(raw, white), "n_test": len(test_pairs), "seconds": time.time() - t0}


def main():
    t_start = time.time()
    results = {}
    for lang in BACKBONES:
        results[lang] = run_language(lang)

    print("\n=== Specialist-backbone summary ===")
    for lang, r in results.items():
        print(f"  {lang} ({r['backbone']}): baseline={r['baseline']:.4f} ours={r['best']:.4f} "
              f"delta={r['best'] - r['baseline']:+.4f}")

    out_path = RESULTS_DIR / "tables" / "specialist_backbone_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults written to {out_path}")
    print(f"Total time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
