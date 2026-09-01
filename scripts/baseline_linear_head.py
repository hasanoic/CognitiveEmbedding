"""Fair-baseline experiment added in response to peer review: the paper
compared a trained core (Attention+Composition) against LARGE ZERO-SHOT
baselines (LaBSE, mE5), but never against an equally lightweight TRAINED
baseline -- so the gains in Table (specialist backbones) could plausibly
come from "any small trained head," not specifically from the attention-
and-composition design. This closes that gap for the cheapest, most direct
version of the fair-baseline ask: a single trained linear projection on
top of the same frozen backbone's mean-pooled representation, trained with
the identical recipe (same data, same backbones, same InfoNCE objective)
as the main architecture.

A PEFT/LoRA-style adapter baseline is NOT attempted here -- unlike this
linear head (which trains only on cached, frozen backbone output), a LoRA
adapter requires backpropagating through the frozen backbone itself for
every training example, which is a materially larger compute cost on this
project's CPU-only budget. That remains explicitly deferred to future work
(see paper Section 6, eighth limitation).

Reuses SpecialistBackbone, train_nli_stage, train_task_stage,
evaluate_final, and evaluate_baseline directly from
train_specialist_backbones.py -- same backbones, same data, same protocol,
so the only variable that changes is the trainable head itself.

`datasets` must be imported before `torch` (see data/loaders.py docstring).
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import datasets  # noqa: F401

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cogembed.data.loaders import (
    RANDOM_SEED,
    load_bnpc_pairs,
    load_nli_triplets,
    load_semrel_arabic,
    load_semrel_hindi,
    load_semrel_telugu,
    load_stsb,
)
from cogembed.models.backbone import mean_pool

from train_specialist_backbones import (
    BACKBONES,
    SpecialistBackbone,
    evaluate_baseline,
    evaluate_final,
    load_xnli_triplets,
    train_nli_stage,
    train_task_stage,
)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


class LinearProjectionHead(nn.Module):
    """The cheapest possible trained head: mean-pool the frozen backbone's
    token features, then a single linear projection. No attention, no
    composition, no order-sensitivity -- isolates whether "any trained
    head" bootstraps gains, or whether the specific mechanisms matter."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, token_features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        pooled = mean_pool(token_features, mask)
        return self.proj(pooled)


def run_language(lang: str) -> dict:
    print(f"\n{'=' * 20} {lang} ({BACKBONES[lang]}) -- linear-projection baseline {'=' * 20}")
    t0 = time.time()
    backbone = SpecialistBackbone(BACKBONES[lang])
    head = LinearProjectionHead(backbone.hidden_dim)
    torch.manual_seed(RANDOM_SEED)
    rng = random.Random(RANDOM_SEED)

    if lang == "english":
        triplets = load_nli_triplets(max_premises=6000)
        train_nli_stage(backbone, head, triplets, rng)
        train_all = rng.sample(load_stsb("train"), 2200)
        train_pairs = [r for r in train_all if r["score"] >= 3.0]
        val_pairs = rng.sample(load_stsb("validation"), 500)
        test_pairs, kind = load_stsb("test"), "sts"
        train_task_stage(backbone, head, train_pairs, val_pairs, kind, rng)
    elif lang == "bangla":
        train_all = load_bnpc_pairs("train")
        train_pairs = [r for r in train_all if r["label"] == 1]
        if len(train_pairs) > 2200:
            train_pairs = rng.sample(train_pairs, 2200)
        val_pairs = load_bnpc_pairs("validation")
        if len(val_pairs) > 500:
            val_pairs = rng.sample(val_pairs, 500)
        test_pairs, kind = load_bnpc_pairs("test"), "auroc"
        train_task_stage(backbone, head, train_pairs, val_pairs, kind, rng)
    elif lang == "telugu":
        train_all = load_semrel_telugu("train")
        train_pairs = [r for r in train_all if r["score"] >= 0.5]
        val_pairs = load_semrel_telugu("dev")
        test_pairs, kind = load_semrel_telugu("test"), "sts"
        train_task_stage(backbone, head, train_pairs, val_pairs, kind, rng)
    elif lang == "hindi":
        triplets = load_xnli_triplets("hi", max_triplets=6000)
        train_nli_stage(backbone, head, triplets, rng)
        test_pairs, kind = load_semrel_hindi("test"), "sts"
    elif lang == "arabic":
        triplets = load_xnli_triplets("ar", max_triplets=6000)
        train_nli_stage(backbone, head, triplets, rng)
        test_pairs, kind = load_semrel_arabic("test"), "sts"

    raw, white = evaluate_final(backbone, head, test_pairs, kind)
    baseline = evaluate_baseline(backbone, test_pairs, kind)

    torch.save(head.state_dict(), RESULTS_DIR / f"linear_head_{lang}.pt")
    print(f"  {lang}: untrained_baseline={baseline:.4f} linear_head_raw={raw:.4f} linear_head_whitened={white:.4f} "
          f"best={max(raw, white):.4f}  ({time.time() - t0:.1f}s)")
    return {"backbone": BACKBONES[lang], "untrained_baseline": baseline, "raw": raw, "whitened": white,
            "best": max(raw, white), "n_test": len(test_pairs), "seconds": time.time() - t0}


def main():
    t_start = time.time()
    results = {}
    for lang in BACKBONES:
        results[lang] = run_language(lang)

    print("\n=== Linear-projection baseline summary ===")
    for lang, r in results.items():
        print(f"  {lang} ({r['backbone']}): baseline={r['untrained_baseline']:.4f} linear_head={r['best']:.4f} "
              f"delta_vs_baseline={r['best'] - r['untrained_baseline']:+.4f}")

    out_path = RESULTS_DIR / "tables" / "linear_head_baseline_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults written to {out_path}")
    print(f"Total time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
