"""Tests whether Farasa preprocessing (recommended by AraBERT's own
documentation but not applied elsewhere in this project) explains any of
Arabic's anomalous degradation -- training with AraBERT underperforms its
own untrained baseline, the only such case in this study (Discussion and
Limitations, "Arabic"). Requested directly by a pre-submission review.

Installation note: `arabert` (the convenience wrapper providing
ArabertPreprocessor) hard-pins emoji==1.4.2 in its own dependency
metadata. Installing it normally previously downgraded this project's
shared emoji package and broke bnlp-toolkit, used by an unrelated
project. Fixed this time by installing `farasapy` normally (its own
dependencies -- requests, tqdm -- don't touch emoji) and `arabert` with
--no-deps, so pip never resolves or touches emoji at all. Verified
emoji stays at its pre-existing version after both installs.

Recipe: identical to the existing Arabic protocol (train_specialist_backbones.py
/ baseline_linear_head.py's Arabic branch) -- XNLI ('ar' slice) hard-negative
pretrain only, zero-shot evaluation on SemRel2024-Arabic test, no
task-specific fine-tuning stage (there is no dedicated Arabic STS-style
train set). The only change: every sentence (XNLI triplets and test
pairs alike) is run through ArabertPreprocessor.preprocess() -- Farasa
segmentation plus AraBERT's documented normalization -- before tokenization,
for both heads.

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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cogembed.data.loaders import RANDOM_SEED, load_semrel_arabic
from cogembed.losses import info_nce_loss_hard_negatives
from cogembed.models.backbone import apply_whitening, fit_whitening
from cogembed.models.cognitive_embedding import CognitiveEmbeddingCore

from train_specialist_backbones import BACKBONES, SpecialistBackbone, load_xnli_triplets, cosine_sim_np
from baseline_linear_head import LinearProjectionHead

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
LANG = "arabic"
LR, TEMPERATURE, NLI_EPOCHS, BATCH_SIZE = 1e-3, 0.05, 6, 32
NLI_TRIPLETS_MAX = 6000
SEED = RANDOM_SEED

PUBLISHED_NO_FARASA = {"untrained": 0.4498296683744248, "ours": 0.4217035514919273, "linear": 0.4638465810583845}


def score(sims, pairs):
    gold = np.array([p["score"] for p in pairs])
    rho, _ = spearmanr(sims, gold)
    return float(rho)


def make_head(kind, hidden_dim):
    return CognitiveEmbeddingCore(hidden_dim) if kind == "ours" else LinearProjectionHead(hidden_dim)


def pool_batch(core, cache_batch):
    return torch.stack([core(h, m) for h, m in cache_batch])


def train_nli(core, ta, tp, tn, va, vp, vn, seed):
    optimizer = torch.optim.Adam(core.parameters(), lr=LR)
    n_train = len(ta)
    best_val, best_state = float("inf"), None
    for epoch in range(NLI_EPOCHS):
        core.train()
        perm = np.random.RandomState(seed + epoch).permutation(n_train)
        for start in range(0, n_train, BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            if len(idx) < 2:
                continue
            optimizer.zero_grad()
            a, p, n = pool_batch(core, [ta[i] for i in idx]), pool_batch(core, [tp[i] for i in idx]), pool_batch(core, [tn[i] for i in idx])
            loss = info_nce_loss_hard_negatives(a, p, n, TEMPERATURE)
            loss.backward()
            optimizer.step()
        core.eval()
        with torch.no_grad():
            val_loss = info_nce_loss_hard_negatives(pool_batch(core, va), pool_batch(core, vp), pool_batch(core, vn), TEMPERATURE).item()
        if val_loss < best_val:
            best_val, best_state = val_loss, {k: v.clone() for k, v in core.state_dict().items()}
        print(f"    epoch {epoch} val_loss={val_loss:.4f} (best={best_val:.4f})")
    core.load_state_dict(best_state)


def evaluate(core, backbone, test_pairs, prep_fn):
    def embed_all(sentences):
        with torch.no_grad():
            out = []
            for s in sentences:
                h, m = backbone.encode_tokens(prep_fn(s))
                out.append(core(h, m).numpy())
            return np.stack(out)

    e1, e2 = embed_all([p["s1"] for p in test_pairs]), embed_all([p["s2"] for p in test_pairs])
    raw = score(cosine_sim_np(e1, e2), test_pairs)
    fit = np.concatenate([e1, e2], axis=0)
    mu, w = fit_whitening(fit)
    white = score(cosine_sim_np(apply_whitening(e1, mu, w), apply_whitening(e2, mu, w)), test_pairs)
    return max(raw, white), raw, white


def evaluate_untrained(backbone, test_pairs, prep_fn):
    from cogembed.models.backbone import mean_pool

    def embed_all(sentences):
        with torch.no_grad():
            out = []
            for s in sentences:
                h, m = backbone.encode_tokens(prep_fn(s))
                out.append(mean_pool(h, m).numpy())
            return np.stack(out)

    e1, e2 = embed_all([p["s1"] for p in test_pairs]), embed_all([p["s2"] for p in test_pairs])
    fit = np.concatenate([e1, e2], axis=0)
    mu, w = fit_whitening(fit)
    return score(cosine_sim_np(apply_whitening(e1, mu, w), apply_whitening(e2, mu, w)), test_pairs)


def main():
    t_start = time.time()
    print("Loading ArabertPreprocessor (Farasa segmentation + AraBERT normalization)...")
    from arabert.preprocess import ArabertPreprocessor
    prep = ArabertPreprocessor(model_name=BACKBONES[LANG])

    def prep_fn(s: str) -> str:
        return prep.preprocess(s)

    print("Loading AraBERT specialist backbone...")
    backbone = SpecialistBackbone(BACKBONES[LANG])
    rng = random.Random(RANDOM_SEED)

    print("Loading XNLI ('ar') triplets and applying Farasa preprocessing...")
    triplets = load_xnli_triplets("ar", max_triplets=NLI_TRIPLETS_MAX)
    rng.shuffle(triplets)
    n_val = max(1, int(len(triplets) * 0.1))
    val_t, train_t = triplets[:n_val], triplets[n_val:]

    def cache_prepped(sentences):
        return [backbone.encode_tokens(prep_fn(s)) for s in sentences]

    ta, tp, tn = cache_prepped([t["anchor"] for t in train_t]), cache_prepped([t["positive"] for t in train_t]), cache_prepped([t["hard_negative"] for t in train_t])
    va, vp, vn = cache_prepped([t["anchor"] for t in val_t]), cache_prepped([t["positive"] for t in val_t]), cache_prepped([t["hard_negative"] for t in val_t])
    print(f"  NLI train/val: {len(train_t)}/{len(val_t)}")

    test_pairs = load_semrel_arabic("test")
    print(f"  test: {len(test_pairs)}")
    print(f"  caching (with Farasa) done at {time.time() - t_start:.1f}s\n")

    print("Evaluating untrained baseline (with Farasa preprocessing)...")
    untrained_score = evaluate_untrained(backbone, test_pairs, prep_fn)
    print(f"  untrained (Farasa): {untrained_score:.4f}  (published, no Farasa: {PUBLISHED_NO_FARASA['untrained']:.4f})")

    results = {"untrained_farasa": untrained_score, "published_no_farasa": PUBLISHED_NO_FARASA}

    for head_kind in ["ours", "linear"]:
        print(f"\nTraining {head_kind} head on Farasa-preprocessed Arabic XNLI...")
        torch.manual_seed(SEED)
        core = make_head(head_kind, backbone.hidden_dim)
        train_nli(core, ta, tp, tn, va, vp, vn, SEED)
        core.eval()
        best, raw, white = evaluate(core, backbone, test_pairs, prep_fn)
        print(f"  {head_kind} (Farasa): raw={raw:.4f} whitened={white:.4f} best={best:.4f}  "
              f"(published, no Farasa: {PUBLISHED_NO_FARASA[head_kind]:.4f}, delta={best - PUBLISHED_NO_FARASA[head_kind]:+.4f})")
        results[head_kind] = {"raw": raw, "whitened": white, "best": best,
                               "published_no_farasa": PUBLISHED_NO_FARASA[head_kind],
                               "delta_vs_no_farasa": best - PUBLISHED_NO_FARASA[head_kind]}

    print("\n=== Summary: does Farasa preprocessing fix Arabic's anomaly? ===")
    print(f"  untrained baseline: no-Farasa={PUBLISHED_NO_FARASA['untrained']:.4f}  Farasa={untrained_score:.4f}")
    for head_kind in ["ours", "linear"]:
        r = results[head_kind]
        below_baseline_before = PUBLISHED_NO_FARASA[head_kind] < PUBLISHED_NO_FARASA["untrained"]
        below_baseline_after = r["best"] < untrained_score
        print(f"  {head_kind}: no-Farasa={r['published_no_farasa']:.4f} (below baseline: {below_baseline_before})  "
              f"Farasa={r['best']:.4f} (below baseline: {below_baseline_after})")

    out_path = RESULTS_DIR / "tables" / "arabic_farasa_preprocessing.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults written to {out_path}")
    print(f"Total time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
