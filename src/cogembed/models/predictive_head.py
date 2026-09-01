"""Predictive Head -- VALIDATED (modest but real, see conversation log).

Discriminative next-sentence prediction (Quick-Thought-style: Logeswaran &
Lee 2018, "An Efficient Framework for Learning Sentence Representations";
CPC-style: van den Oord et al. 2018) via InfoNCE, replacing the original
MSE-regression-to-next-embedding formulation (structurally the older
Skip-Thought approach, Kiros et al. 2015 -- documented in the literature as
no longer competitive against contrastive approaches).

Corpus choice mattered as much as the loss function: tested first on
wikitext-2 (encyclopedic prose, weak local coherence) where training
underperformed the untrained baseline (Recall@1 0.029 vs 0.033); retested on
cnn_dailymail (narrative news text, real sentence-to-sentence coherence)
where training clearly helped (0.0179 vs. 0.0134 untrained, both well above
chance at 0.0022). ROCStories -- the standard dataset for this exact task in
the literature -- returned a 401/gated error on every mirror tried; use it
if HF access becomes available, it is likely a better fit than either
substitute used here.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PredictiveHead(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.pool_proj = nn.Linear(hidden_dim, hidden_dim)
        self.predictor = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, hidden_dim))

    def encode(self, token_features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mf = mask.unsqueeze(-1).float()
        pooled = (token_features * mf).sum(0) / mf.sum(0).clamp(min=1e-9)
        return self.pool_proj(pooled)

    def predict_next(self, current_embedding: torch.Tensor) -> torch.Tensor:
        return self.predictor(current_embedding)
