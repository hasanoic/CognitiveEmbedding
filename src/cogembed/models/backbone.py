"""Frozen multilingual backbone + base representation.

Validated finding (see project conversation log / results): raw last-layer
hidden states are anisotropic (Li et al. 2020, "On the Sentence Embeddings
from Pre-trained Language Models") and a poor base for every downstream
module. Averaging across ALL transformer layers before pooling, then
whitening the resulting sentence embedding, is a training-free fix that
alone improved English STS Spearman from 0.407 to 0.656 and Bangla
paraphrase-separation AUROC from 0.969 to 0.999 (see
"BERT Has More to Offer: BERT Layers Combination Yields Better Sentence
Embeddings", ACL Findings EMNLP 2023). Every module in this package
operates on this fixed, validated base representation, not on raw
last-layer output.

The backbone itself is never fine-tuned anywhere in this codebase -- every
module (SemanticAttention, CompositionalAggregator, MemoryRetrieval,
PredictiveHead) is a small trainable head on top of frozen features. This
was a deliberate scope choice validated across this whole project: CPU-scale
compute cannot support fine-tuning a 270M-parameter encoder, and every
positive result found so far came from correcting the *base representation*
and the *pooling head*, not from touching backbone weights.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


@dataclass
class BackboneConfig:
    model_name: str = "xlm-roberta-base"
    max_length: int = 64


class FrozenMultilingualBackbone:
    """Wraps a frozen HF encoder. `encode_tokens` returns per-token features
    averaged across all layers (unweighted -- learned per-layer weights were
    tested and converged to near-uniform, so the simpler fixed average is
    used per Occam's razor, see conversation log)."""

    def __init__(self, config: BackboneConfig | None = None):
        self.config = config or BackboneConfig()
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = AutoModel.from_pretrained(self.config.model_name)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.hidden_dim = self.model.config.hidden_size

    @torch.no_grad()
    def encode_tokens(self, sentence: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (token_features [T, H], attention_mask [T]) -- per-token,
        multi-layer-averaged, NOT yet pooled to a sentence vector. Downstream
        modules (attention pooling, BiGRU composition) need per-token input."""
        enc = self.tokenizer(sentence, return_tensors="pt", truncation=True, max_length=self.config.max_length)
        out = self.model(**enc, output_hidden_states=True)
        layers = torch.stack(out.hidden_states, dim=0)  # [L+1, 1, T, H]
        avg = layers.mean(dim=0)[0]  # [T, H]
        mask = enc["attention_mask"][0]
        return avg, mask

    def encode_batch_tokens(self, sentences: list[str]) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return [self.encode_tokens(s) for s in sentences]


def mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Reference/baseline pooling -- also used as the input to whitening."""
    mf = mask.unsqueeze(-1).float()
    return (hidden * mf).sum(0) / mf.sum(0).clamp(min=1e-9)


def fit_whitening(fit_embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns (mean, whitening_matrix) fit on `fit_embeddings` (dev/train
    set only -- never fit on test data, see project data-leakage discipline).
    Apply via: (embeddings - mean) @ whitening_matrix."""
    mu = fit_embeddings.mean(axis=0, keepdims=True)
    cov = np.cov((fit_embeddings - mu).T)
    u, s, _ = np.linalg.svd(cov)
    w = u @ np.diag(1.0 / np.sqrt(s + 1e-6))
    return mu, w


def apply_whitening(embeddings: np.ndarray, mu: np.ndarray, w: np.ndarray) -> np.ndarray:
    return (embeddings - mu) @ w
