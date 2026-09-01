"""Memory-Aware Retrieval -- VALIDATED, with an important negative result
documented alongside it. Source run: poc_memory_coala.py (and its
predecessor poc_memory_retrieval.py for variants 1-3), copied into
scripts/ as ablation_memory_scoring_variants.py for reproducibility; see
that script, paper Section 5.5 / Table tab:memory-variants, and
logs/memory_coala_full_run.log for the full, logged comparison.

Four formulations were tested for the retrieval scoring function:
  1. content_only:            score = cos(query, slot)
  2. content + fixed decay:   score = cos(query, slot) - 0.3 * distance
  3. content + LEARNED decay: score = cos(query, slot)/tau - lambda*distance
     (tau, lambda trainable)
  4. CoALA-inspired (Sumers et al. 2023, "Cognitive Architectures for
     Language Agents", arXiv:2309.02427 -- conceptually inspired by, NOT
     CoALA's own published equations): learned key/value projections,
     softmax retrieval, gated residual integration (984,577 trainable
     parameters at hidden_dim=768, kv_dim=128), trained with a
     multi-positive NLL loss over the softmax retrieval weights (not
     per-slot BCE, which is a structural mismatch for a softmax read
     mechanism with several valid targets), on WHITENED input features.

Held-out TEST AUROC (181/38/38 train/val/test split of a 257-item English
discourse-retrieval task; distractors ARE distance-matched to the query --
each gets a distance drawn from the same 1..5 range as true-context items,
specifically so a distance-only signal cannot trivially separate the two
classes by construction): (1) 0.700, (2) 0.557, (3) 0.556, (4) 0.637.
**Plain content-only similarity, with ZERO learned parameters, beat every
more sophisticated variant** -- formulations (2)-(4) all showed clear
overfitting signatures on a training set (181 items) too small for their
added capacity; the CoALA-inspired variant's training curve shows this
directly: train AUROC reaches 1.0 by epoch 5 and stays there through
epoch 39, while validation AUROC peaks at epoch 5 (0.6452) and slowly
declines afterward. This is documented as a real, informative negative
result, not silently dropped: the decay/interference idea is not
disproven in principle, it simply doesn't have enough discourse-retrieval
training data in this project yet to earn its added complexity.

DEFAULT / RECOMMENDED CONFIGURATION FOR THE FULL PIPELINE: content-only.
The learned-decay and CoALA-inspired classes are kept here, functional and
documented, for reproducibility and as a starting point if/when a larger
discourse-retrieval training set becomes available -- NOT wired into the
default training script.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def content_only_score(query: torch.Tensor, memory_items: torch.Tensor) -> torch.Tensor:
    """query: [H], memory_items: [N, H] -> similarity scores [N].
    VALIDATED, RECOMMENDED default -- zero learned parameters."""
    q = nn.functional.normalize(query, dim=-1)
    m = nn.functional.normalize(memory_items, dim=-1)
    return m @ q


class LearnedDecayMemory(nn.Module):
    """NOT recommended as-is (see module docstring) -- kept for
    reproducibility. Underperformed content_only in every test run."""

    def __init__(self):
        super().__init__()
        self.log_tau = nn.Parameter(torch.tensor(0.0))
        self.lam = nn.Parameter(torch.tensor(0.0))

    def forward(self, content_sim: torch.Tensor, distance: torch.Tensor) -> torch.Tensor:
        tau = torch.exp(self.log_tau).clamp(min=0.1)
        return content_sim / tau - self.lam * distance


class CoALAInspiredMemory(nn.Module):
    """NOT recommended as-is (see module docstring) -- overfit on the
    available discourse-retrieval training set. Kept for reproducibility
    and as a starting point for future work with more data."""

    def __init__(self, hidden_dim: int, kv_dim: int = 128):
        super().__init__()
        self.W_k = nn.Linear(hidden_dim, kv_dim, bias=False)
        self.W_v = nn.Linear(hidden_dim, kv_dim, bias=False)
        self.W_g = nn.Linear(hidden_dim + kv_dim, hidden_dim)
        self.r_to_h = nn.Linear(kv_dim, hidden_dim)
        self.tau = nn.Parameter(torch.tensor(1.0))

    def forward(self, h_query: torch.Tensor, memory_items: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        k = self.W_k(memory_items)
        v = self.W_v(memory_items)
        h_proj = self.W_k(h_query)
        tau = self.tau.clamp(min=0.05)
        sims = (k @ h_proj) / (tau * (k.norm(dim=-1) * h_proj.norm() + 1e-9))
        alpha = torch.softmax(sims, dim=0)
        r = (alpha.unsqueeze(-1) * v).sum(0)
        g = torch.sigmoid(self.W_g(torch.cat([h_query, r], dim=-1)))
        h_semantic = h_query + g * self.r_to_h(r)
        return alpha, h_semantic
