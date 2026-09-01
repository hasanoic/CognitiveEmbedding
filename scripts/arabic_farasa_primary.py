"""Makes Farasa-preprocessed Arabic the PRINCIPAL Arabic evaluation,
replacing the non-Farasa numbers as primary, rather than treating Farasa
as a side-experiment that "rules out a candidate cause." Requested
directly: "Make Farasa-preprocessed Arabic the principal Arabic
evaluation or provide a compelling justification for not doing so."

Justification for switching, not defending the status quo: verified via
AraBERT's own primary documentation that Farasa segmentation is
mandatory, not merely recommended, for AraBERTv2 (this paper's exact
Arabic backbone) -- the library itself raises an error/warning without
it ("requires Farasa pre-segmentation, but apply_farasa_segmentation was
set to False!"), because the vocabulary was trained specifically on
Farasa-segmented text. The paper's earlier non-Farasa Arabic numbers
were computed by feeding AraBERT out-of-distribution input relative to
its own training format -- not a stylistic preprocessing choice
comparable to what other languages' backbones may or may not recommend.

This script covers what arabic_farasa_preprocessing.py (the earlier,
diagnostic run) did NOT: it saves checkpoints, and it scores under BOTH
raw-vs-test-fit-whitened (matching the original protocol) AND
raw-vs-dev-fit-whitened (matching this paper's leak-free primary
protocol, adopted for Table tab:baseline-head after this Farasa work was
first done) -- so the new Farasa-primary Arabic numbers are consistent
with both decisions made tonight, not just the preprocessing one.

Covers: untrained baseline, linear head (full capacity), structured head
(full capacity). Does not cover: the parameter-matched capacity
experiment, multi-seed grid, or significance testing for Farasa-Arabic
specifically -- reusing the same "flag explicitly, do not silently
leave inconsistent" approach as the whitening-primary-protocol change,
documented in the paper's Limitations rather than left implicit.

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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cogembed.data.loaders import RANDOM_SEED, load_semrel_arabic
from cogembed.losses import info_nce_loss_hard_negatives
from cogembed.models.backbone import apply_whitening, fit_whitening, mean_pool
from cogembed.models.cognitive_embedding import CognitiveEmbeddingCore

from train_specialist_backbones import BACKBONES, SpecialistBackbone, load_xnli_triplets, cosine_sim_np, score
from baseline_linear_head import LinearProjectionHead

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
LANG = "arabic"
LR, TEMPERATURE, NLI_EPOCHS, BATCH_SIZE = 1e-3, 0.05, 6, 32
NLI_TRIPLETS_MAX = 6000
SEED = RANDOM_SEED


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
        print(f"    [nli] epoch {epoch} val_loss={val_loss:.4f} (best={best_val:.4f})")
    core.load_state_dict(best_state)


def embed_all(backbone, core, sentences, prep_fn):
    with torch.no_grad():
        out = []
        for s in sentences:
            h, m = backbone.encode_tokens(prep_fn(s))
            out.append(core(h, m).numpy())
        return np.stack(out)


def embed_all_untrained(backbone, sentences, prep_fn):
    with torch.no_grad():
        out = []
        for s in sentences:
            h, m = backbone.encode_tokens(prep_fn(s))
            out.append(mean_pool(h, m).numpy())
        return np.stack(out)


def score_both_protocols(e1_test, e2_test, e1_dev, e2_dev, test_pairs, kind):
    """Returns (raw, test_fit_whitened, dev_fit_whitened) scores."""
    raw = score(kind, cosine_sim_np(e1_test, e2_test), test_pairs)

    fit_test = np.concatenate([e1_test, e2_test], axis=0)
    mu_t, w_t = fit_whitening(fit_test)
    test_fit_white = score(kind, cosine_sim_np(apply_whitening(e1_test, mu_t, w_t), apply_whitening(e2_test, mu_t, w_t)), test_pairs)

    fit_dev = np.concatenate([e1_dev, e2_dev], axis=0)
    mu_d, w_d = fit_whitening(fit_dev)
    dev_fit_white = score(kind, cosine_sim_np(apply_whitening(e1_test, mu_d, w_d), apply_whitening(e2_test, mu_d, w_d)), test_pairs)

    return raw, test_fit_white, dev_fit_white


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
    dev_pairs = load_semrel_arabic("dev")
    print(f"  test: {len(test_pairs)}  dev: {len(dev_pairs)}")
    print(f"  caching (with Farasa) done at {time.time() - t_start:.1f}s\n")

    out_path = RESULTS_DIR / "tables" / "arabic_farasa_primary.json"
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    if results:
        print(f"Resuming: {list(results.keys())} already scored, skipping.")

    if "untrained" not in results:
        print("Evaluating untrained baseline (Farasa)...")
        e1t = embed_all_untrained(backbone, [p["s1"] for p in test_pairs], prep_fn)
        e2t = embed_all_untrained(backbone, [p["s2"] for p in test_pairs], prep_fn)
        e1d = embed_all_untrained(backbone, [p["s1"] for p in dev_pairs], prep_fn)
        e2d = embed_all_untrained(backbone, [p["s2"] for p in dev_pairs], prep_fn)
        raw, tfw, dfw = score_both_protocols(e1t, e2t, e1d, e2d, test_pairs, "sts")
        print(f"  untrained: raw={raw:.4f} test-fit-whitened={tfw:.4f} dev-fit-whitened={dfw:.4f}")
        results["untrained"] = {"raw": raw, "test_fit_whitened": tfw, "dev_fit_whitened": dfw,
                                 "best_leaky": max(raw, tfw), "best_leakfree": max(raw, dfw)}
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2))

    for head_kind in ["ours", "linear"]:
        if head_kind in results:
            continue
        torch.manual_seed(SEED)
        core = make_head(head_kind, backbone.hidden_dim)
        ckpt_path = RESULTS_DIR / f"farasa_arabic_{head_kind}.pt"
        if ckpt_path.exists():
            print(f"\nResuming: {ckpt_path} already exists, loading it and skipping training.")
            core.load_state_dict(torch.load(ckpt_path))
        else:
            print(f"\nTraining {head_kind} head on Farasa-preprocessed Arabic XNLI...")
            train_nli(core, ta, tp, tn, va, vp, vn, SEED)
            torch.save(core.state_dict(), ckpt_path)
        core.eval()

        e1t = embed_all(backbone, core, [p["s1"] for p in test_pairs], prep_fn)
        e2t = embed_all(backbone, core, [p["s2"] for p in test_pairs], prep_fn)
        e1d = embed_all(backbone, core, [p["s1"] for p in dev_pairs], prep_fn)
        e2d = embed_all(backbone, core, [p["s2"] for p in dev_pairs], prep_fn)
        raw, tfw, dfw = score_both_protocols(e1t, e2t, e1d, e2d, test_pairs, "sts")
        print(f"  {head_kind}: raw={raw:.4f} test-fit-whitened={tfw:.4f} dev-fit-whitened={dfw:.4f}")
        results[head_kind] = {"raw": raw, "test_fit_whitened": tfw, "dev_fit_whitened": dfw,
                               "best_leaky": max(raw, tfw), "best_leakfree": max(raw, dfw)}
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2))

    print("\n=== Summary: Farasa-preprocessed Arabic, leak-free protocol ===")
    for name in ["untrained", "ours", "linear"]:
        r = results[name]
        print(f"  {name}: best_leakfree={r['best_leakfree']:.4f}  (raw={r['raw']:.4f}, dev-fit-whitened={r['dev_fit_whitened']:.4f})")
    print(f"\n  Structured vs untrained: {results['ours']['best_leakfree'] - results['untrained']['best_leakfree']:+.4f}")
    print(f"  Structured vs linear: {results['ours']['best_leakfree'] - results['linear']['best_leakfree']:+.4f}")
    print(f"  Linear vs untrained: {results['linear']['best_leakfree'] - results['untrained']['best_leakfree']:+.4f}")
    print(f"\nResults written to {out_path}")
    print(f"Total time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
