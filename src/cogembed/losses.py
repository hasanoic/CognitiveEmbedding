"""Shared loss functions. InfoNCE is the single validated loss across every
module in this project -- see individual module docstrings for why MSE
regression (the original spec for the Predictive Head, and implicitly for
the sentence-level objective) was replaced everywhere it was tried.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def info_nce_loss(emb1: torch.Tensor, emb2: torch.Tensor, temperature: float = 0.05) -> torch.Tensor:
    """Symmetric in-batch-negative InfoNCE (SimCSE-style). emb1[i]/emb2[i] is
    the positive pair; every emb2[j], j != i, in the batch is a negative."""
    emb1 = nn.functional.normalize(emb1, dim=-1)
    emb2 = nn.functional.normalize(emb2, dim=-1)
    sims = emb1 @ emb2.T / temperature
    labels = torch.arange(sims.shape[0])
    loss_12 = nn.functional.cross_entropy(sims, labels)
    loss_21 = nn.functional.cross_entropy(sims.T, labels)
    return (loss_12 + loss_21) / 2


def info_nce_loss_hard_negatives(
    anchor: torch.Tensor, positive: torch.Tensor, hard_negative: torch.Tensor, temperature: float = 0.05
) -> torch.Tensor:
    """SimCSE supervised loss (Gao, Yao & Chen 2021, EMNLP, Eq. 6): for anchor
    i, positive[i] is the true match, every positive[j] (j!=i) is an in-batch
    negative, AND hard_negative[i] is an additional explicit negative (e.g. an
    NLI contradiction hypothesis for the same premise) appended to the same
    softmax denominator."""
    anchor = nn.functional.normalize(anchor, dim=-1)
    positive = nn.functional.normalize(positive, dim=-1)
    hard_negative = nn.functional.normalize(hard_negative, dim=-1)
    sim_pos = anchor @ positive.T / temperature
    sim_neg = anchor @ hard_negative.T / temperature
    logits = torch.cat([sim_pos, sim_neg], dim=1)
    labels = torch.arange(sim_pos.shape[0])
    return nn.functional.cross_entropy(logits, labels)


MATRYOSHKA_DIMS = (768, 256, 128, 64)


def matryoshka_info_nce_loss(
    emb1: torch.Tensor, emb2: torch.Tensor, temperature: float = 0.05, dims=MATRYOSHKA_DIMS
) -> torch.Tensor:
    """Matryoshka Representation Learning (Kusupati et al. 2022, NeurIPS):
    applies info_nce_loss independently to each nested PREFIX of the same
    embedding (first 768, first 256, first 128, first 64 dims), then averages.
    No architecture change -- the core module still outputs one 768-dim
    vector; this loss just trains it so that early dimensions alone are
    ALSO a usable embedding, enabling truncation to a smaller size at
    inference time with graceful (not catastrophic) accuracy loss. Equal
    weighting across dims -- the original paper explores weighted variants,
    equal weighting is the simpler, disclosed default here, not tuned."""
    losses = [info_nce_loss(emb1[:, :d], emb2[:, :d], temperature) for d in dims if d <= emb1.shape[-1]]
    return torch.stack(losses).mean()


def matryoshka_info_nce_loss_hard_negatives(
    anchor: torch.Tensor, positive: torch.Tensor, hard_negative: torch.Tensor,
    temperature: float = 0.05, dims=MATRYOSHKA_DIMS
) -> torch.Tensor:
    """Matryoshka variant of info_nce_loss_hard_negatives, same nested-prefix
    averaging as matryoshka_info_nce_loss -- used for the NLI pretrain stage."""
    losses = [
        info_nce_loss_hard_negatives(anchor[:, :d], positive[:, :d], hard_negative[:, :d], temperature)
        for d in dims if d <= anchor.shape[-1]
    ]
    return torch.stack(losses).mean()
