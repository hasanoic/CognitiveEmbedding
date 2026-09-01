"""Batch-size sweep, requested directly by a pre-submission review: does the
structured head benefit more than the linear head from added in-batch
negatives? The paper's "Baselines, Training, and Hardware" subsection
already flags batch size 32 as "a plausible confound we did not test by
varying batch size" -- this closes that gap for one language.

Scope: Bangla only, not all five. Chosen because (a) it has a
statistically significant fair-baseline margin favoring the linear head
(scripts/significance_fair_baseline_all_languages.py: -0.0194, 95% CI
[-0.0334, -0.0050]), so it is the most informative single language to ask
"does more negatives close this gap"; (b) it has no NLI pretrain stage
(direct task fine-tune only), keeping the sweep cheap; (c) "at least one
language" was the explicit ask.

Sweep: batch size in {32, 64, 128}, both heads, everything else identical
(15 epochs, same data, same seed=42). Batch size 32 is NOT retrained -- it
reuses the existing published checkpoints (core_model_specialist_bangla.pt,
linear_head_bangla.pt), re-evaluated under this script's own protocol
rather than hardcoded, so all three batch sizes go through identical
scoring. Only batch sizes 64 and 128 are newly trained, for both heads (4
new runs), using the caching-once pattern: Bangla train/val/test/dev pairs
are encoded through BanglaBERT exactly once and shared across all runs.

Leak-free protocol: whitening is fit once on BnPC's held-out validation
split (n<=500, same sample as Table 5's primary protocol,
significance_leakfree_fair_baseline.py), never on the test set, and
applied unchanged to test embeddings. Superseded the original test-fit-
whitened version of this sweep, which stayed on the earlier protocol after
Table 5 switched to leak-free selection ("Whitening leakage, and what
still reflects the earlier protocol", Discussion and Limitations); results
now write to batch_size_sweep_bangla_leakfree.json, leaving the original
test-fit-whitened batch_size_sweep_bangla.json on disk unchanged as a
historical record. Checkpointed per (head, batch size) cell so a power
interruption loses at most one training run.

Epoch count is held fixed at 15 regardless of batch size (fewer, larger
gradient steps per epoch at bigger batch sizes) -- this isolates the
effect of negative-count per step, which is what "does structured
benefit more from added negatives" is actually asking, rather than
conflating it with total gradient-step count.

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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cogembed.data.loaders import RANDOM_SEED, load_bnpc_pairs
from cogembed.losses import info_nce_loss
from cogembed.models.backbone import apply_whitening, fit_whitening
from cogembed.models.cognitive_embedding import CognitiveEmbeddingCore

from train_specialist_backbones import BACKBONES, SpecialistBackbone
from baseline_linear_head import LinearProjectionHead

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
OUT_PATH = RESULTS_DIR / "tables" / "batch_size_sweep_bangla_leakfree.json"
LANG = "bangla"
BATCH_SIZES = [32, 64, 128]
LR, TEMPERATURE, TASK_EPOCHS = 1e-3, 0.05, 15
SEED = RANDOM_SEED  # 42, matching the published run -- only batch size varies

CHECKPOINT_PATHS = {
    "ours": RESULTS_DIR / "core_model_specialist_bangla.pt",
    "linear": RESULTS_DIR / "linear_head_bangla.pt",
}


def cosine_sim_np(a, b):
    a_n = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)
    b_n = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-9)
    return (a_n * b_n).sum(axis=-1)


def score(sims, pairs):
    labels = np.array([p["label"] for p in pairs])
    return float(roc_auc_score(labels, sims))


def cache_pairs(backbone, pairs):
    return [backbone.encode_tokens(p["s1"]) for p in pairs], [backbone.encode_tokens(p["s2"]) for p in pairs]


def pool_batch(core, cache_batch):
    return torch.stack([core(h, m) for h, m in cache_batch])


def make_head(kind, hidden_dim):
    return CognitiveEmbeddingCore(hidden_dim) if kind == "ours" else LinearProjectionHead(hidden_dim)


def train_at_batch_size(core, c1, c2, v1, v2, val_pairs, batch_size, seed):
    optimizer = torch.optim.Adam(core.parameters(), lr=LR)
    best_val, best_state = -1.0, None
    for epoch in range(TASK_EPOCHS):
        core.train()
        perm = np.random.RandomState(seed + epoch).permutation(len(c1))
        for start in range(0, len(c1), batch_size):
            idx = perm[start:start + batch_size]
            if len(idx) < 2:
                continue
            optimizer.zero_grad()
            e1, e2 = pool_batch(core, [c1[i] for i in idx]), pool_batch(core, [c2[i] for i in idx])
            loss = info_nce_loss(e1, e2, TEMPERATURE)
            loss.backward()
            optimizer.step()
        core.eval()
        with torch.no_grad():
            vv1, vv2 = pool_batch(core, v1).numpy(), pool_batch(core, v2).numpy()
        val_score = score(cosine_sim_np(vv1, vv2), val_pairs)
        if val_score > best_val:
            best_val, best_state = val_score, {k: v.clone() for k, v in core.state_dict().items()}
    core.load_state_dict(best_state)


def evaluate_leakfree(core, dev1, dev2, t1, t2, test_pairs):
    # Always raw for Bangla (this script's only language): a proper split-dev check
    # (verify_bangla_split_dev_selection.py) shows dev-fit whitening does not
    # actually help Bangla -- the earlier "devfit wins" selection elsewhere in this
    # project was an artifact of comparing raw vs whitened on the TEST set itself.
    with torch.no_grad():
        e1, e2 = pool_batch(core, t1).numpy(), pool_batch(core, t2).numpy()
    return score(cosine_sim_np(e1, e2), test_pairs)


def load_results():
    if OUT_PATH.exists():
        return json.loads(OUT_PATH.read_text())
    return {"ours": {}, "linear": {}}


def save_results(results):
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2))


def main():
    t_start = time.time()
    print(f"Loading {LANG} backbone and caching data (once, shared across batch sizes and heads)...")
    backbone = SpecialistBackbone(BACKBONES[LANG])
    rng = random.Random(RANDOM_SEED)

    train_all = load_bnpc_pairs("train")
    train_pairs = [r for r in train_all if r["label"] == 1]
    if len(train_pairs) > 2200:
        train_pairs = rng.sample(train_pairs, 2200)
    val_pairs = load_bnpc_pairs("validation")
    if len(val_pairs) > 500:
        val_pairs = rng.sample(val_pairs, 500)
    dev_pairs = load_bnpc_pairs("validation")  # leak-free whitening fit split, matches Table 5's protocol
    if len(dev_pairs) > 500:
        dev_pairs = random.Random(RANDOM_SEED).sample(dev_pairs, 500)
    test_pairs = load_bnpc_pairs("test")

    c1, c2 = cache_pairs(backbone, train_pairs)
    v1, v2 = cache_pairs(backbone, val_pairs)
    dev1, dev2 = cache_pairs(backbone, dev_pairs)
    t1, t2 = cache_pairs(backbone, test_pairs)
    print(f"  train/val/dev/test: {len(train_pairs)}/{len(val_pairs)}/{len(dev_pairs)}/{len(test_pairs)}")
    print(f"  caching done at {time.time() - t_start:.1f}s\n")

    results = load_results()
    if any(results[hk] for hk in ("ours", "linear")):
        print(f"Resuming: ours={list(results['ours'].keys())} linear={list(results['linear'].keys())} already done.")

    for head_kind in ["ours", "linear"]:
        # batch_size=32: re-evaluate the existing published checkpoint under the leak-free
        # protocol (inference only, no retraining) rather than reuse the old test-fit-whitened number.
        if "32" not in results[head_kind]:
            core = make_head(head_kind, backbone.hidden_dim)
            core.load_state_dict(torch.load(CHECKPOINT_PATHS[head_kind]))
            core.eval()
            s = evaluate_leakfree(core, dev1, dev2, t1, t2, test_pairs)
            results[head_kind]["32"] = s
            save_results(results)
            print(f"  {head_kind} batch_size=32 (published checkpoint, leak-free eval): AUROC={s:.4f}")

        for bs in [64, 128]:
            if str(bs) in results[head_kind]:
                print(f"  {head_kind} batch_size={bs}: already done, skipping.")
                continue
            bs_t0 = time.time()
            torch.manual_seed(SEED)
            core = make_head(head_kind, backbone.hidden_dim)
            train_at_batch_size(core, c1, c2, v1, v2, val_pairs, bs, SEED)
            core.eval()
            s = evaluate_leakfree(core, dev1, dev2, t1, t2, test_pairs)
            results[head_kind][str(bs)] = s
            save_results(results)
            print(f"  {head_kind} batch_size={bs}: AUROC={s:.4f}  ({time.time() - bs_t0:.1f}s)")

        print(f"  {head_kind} full sweep: " + ", ".join(f"bs{bs}={results[head_kind][str(bs)]:.4f}" for bs in BATCH_SIZES))

    print("\n=== Summary: does the structured head close the gap with more negatives? (leak-free protocol) ===")
    for bs in BATCH_SIZES:
        gap = results["ours"][str(bs)] - results["linear"][str(bs)]
        print(f"  batch_size={bs}: ours={results['ours'][str(bs)]:.4f} linear={results['linear'][str(bs)]:.4f} gap(ours-linear)={gap:+.4f}")

    print(f"\nResults written to {OUT_PATH}")
    print(f"Total time: {time.time() - t_start:.1f}s")
    print("DONE_BATCH_SIZE_SWEEP_LEAKFREE")


if __name__ == "__main__":
    main()
