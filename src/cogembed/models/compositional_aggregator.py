"""Compositional Aggregator -- VALIDATED (see conversation log).

BiGRU over the frozen multi-layer base representation, max-pooled --
InferSent-style (Conneau et al. 2017, "Supervised Learning of Universal
Sentence Representations from Natural Language Inference Data"). Replaces
the original formulation (a small fixed local-window smoothing kernel),
which was found to be largely redundant with the backbone's own 12 layers
of self-attention and showed weak, sign-flipping-between-languages results.

The BiGRU is a genuinely different, ORDER-SENSITIVE mechanism (confirmed
directly: cos(original, word-shuffled) = 0.78 for this module vs. 0.99 for
mean pooling -- mean pooling is nearly order-invariant given fixed-position
frozen features, the BiGRU is not). Validated result: English Spearman 0.682
vs. 0.656 baseline (+0.026); Bangla AUROC 0.987 vs. 0.999 baseline (-0.012,
a small regression worth tracking in the full pipeline's ablations).

Trained the same way as SemanticAttentionPooling (InfoNCE, same recipe) for
direct comparability.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CompositionalAggregator(nn.Module):
    def __init__(self, hidden_dim: int, gru_hidden: int = 128):
        super().__init__()
        self.gru = nn.GRU(hidden_dim, gru_hidden, bidirectional=True, batch_first=True)
        self.project = nn.Linear(gru_hidden * 2, hidden_dim)

    def forward(self, token_features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """token_features: [T, H], mask: [T] -> sentence embedding [H]."""
        length = int(mask.sum().item()) or 1
        seq = token_features[:length].unsqueeze(0)  # [1, T', H]
        out, _ = self.gru(seq)  # [1, T', 2*gru_hidden]
        pooled = out.squeeze(0).max(dim=0).values  # max-pool over time (InferSent-style)
        return self.project(pooled)
