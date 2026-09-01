"""Paired bootstrap significance test for the one comparison in the paper
where the structured head (ours) beats the linear head: English semantic
similarity, fair-baseline protocol (Table: fair-baseline comparison,
specialist backbones). Margin is +0.0219 Spearman (0.7383 vs 0.7164) on
n=1,379 STS-B test pairs -- small enough that a reviewer reasonably asked
whether it is distinguishable from noise.

No retraining: both checkpoints (results/core_model_specialist_english.pt,
results/linear_head_english.pt) already exist from the original runs. This
script re-embeds the STS-B test set with both trained heads on the frozen
RoBERTa-base backbone (inference only), reproduces the two published
Spearman numbers as a sanity check, then runs a paired bootstrap over the
1,379 test pairs: resample pairs with replacement, recompute both Spearman
correlations on the same resample, record the difference. The distribution
of that difference across resamples gives a 95% CI on the true margin.

`datasets` must be imported before `torch` (see data/loaders.py docstring).
"""

from __future__ import annotations

import sys
from pathlib import Path

import datasets  # noqa: F401

import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cogembed.data.loaders import load_stsb
from cogembed.models.cognitive_embedding import CognitiveEmbeddingCore

from train_specialist_backbones import SpecialistBackbone, cosine_sim_np
from baseline_linear_head import LinearProjectionHead

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
N_BOOTSTRAP = 10000
SEED = 42


def embed_all(backbone, core, sentences):
    with torch.no_grad():
        out = []
        for s in sentences:
            h, m = backbone.encode_tokens(s)
            out.append(core(h, m).numpy())
        return np.stack(out)


def main():
    print("Loading English STS-B test set...")
    test_pairs = load_stsb("test")
    gold = np.array([p["score"] for p in test_pairs])
    print(f"  n = {len(test_pairs)}")

    print("Loading frozen RoBERTa-base backbone...")
    backbone = SpecialistBackbone("roberta-base")

    print("Loading trained checkpoints (inference only, no training)...")
    ours = CognitiveEmbeddingCore(backbone.hidden_dim)
    ours.load_state_dict(torch.load(RESULTS_DIR / "core_model_specialist_english.pt"))
    ours.eval()

    linear = LinearProjectionHead(backbone.hidden_dim)
    linear.load_state_dict(torch.load(RESULTS_DIR / "linear_head_english.pt"))
    linear.eval()

    print("Embedding test set with both heads...")
    e1_ours, e2_ours = embed_all(backbone, ours, [p["s1"] for p in test_pairs]), embed_all(backbone, ours, [p["s2"] for p in test_pairs])
    e1_lin, e2_lin = embed_all(backbone, linear, [p["s1"] for p in test_pairs]), embed_all(backbone, linear, [p["s2"] for p in test_pairs])

    sims_ours = cosine_sim_np(e1_ours, e2_ours)
    sims_lin = cosine_sim_np(e1_lin, e2_lin)

    rho_ours, _ = spearmanr(sims_ours, gold)
    rho_lin, _ = spearmanr(sims_lin, gold)
    print(f"\nReproduced Spearman -- ours: {rho_ours:.4f} (paper: 0.7383)  linear: {rho_lin:.4f} (paper: 0.7164)")
    print(f"Reproduced margin: {rho_ours - rho_lin:+.4f} (paper: +0.0219)")

    print(f"\nPaired bootstrap over {len(test_pairs)} pairs, {N_BOOTSTRAP} resamples...")
    rng = np.random.RandomState(SEED)
    n = len(test_pairs)
    diffs = np.empty(N_BOOTSTRAP)
    for b in range(N_BOOTSTRAP):
        idx = rng.randint(0, n, size=n)
        r_ours, _ = spearmanr(sims_ours[idx], gold[idx])
        r_lin, _ = spearmanr(sims_lin[idx], gold[idx])
        diffs[b] = r_ours - r_lin

    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p_le_zero = float((diffs <= 0).mean())
    print(f"\nBootstrap distribution of (ours - linear):")
    print(f"  mean:   {diffs.mean():+.4f}")
    print(f"  95% CI: [{lo:+.4f}, {hi:+.4f}]")
    print(f"  fraction of resamples with (ours - linear) <= 0: {p_le_zero:.4f}")
    print(f"  {'EXCLUDES zero -> margin is significant at alpha=0.05' if lo > 0 or hi < 0 else 'INCLUDES zero -> margin is NOT significant at alpha=0.05'}")


if __name__ == "__main__":
    main()
