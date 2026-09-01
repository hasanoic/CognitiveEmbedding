"""The joint CognitiveEmbedding model -- combines the four validated
modules, scoped honestly by what was actually tested together.

IMPORTANT design note (see conversation log): Semantic Attention and the
Compositional Aggregator were both validated on SINGLE-SENTENCE tasks
(STS-B, Bangla paraphrase separation) and are combined here into one core
embedding function usable standalone. Memory-Aware Retrieval and the
Predictive Head were both validated only at the DISCOURSE level (using real
preceding/following sentences from a document) -- neither has ever been
tested as part of a single-sentence forward pass, so forcing them into one
here would claim an integration that was never evaluated. They are exposed
as separate, optional components that attach to the same core embedding
when discourse-ordered training data is available (see scripts/train.py),
not as mandatory parts of every forward pass.

Core embedding = project(concat(attention_pooled, composition_pooled)),
then whitened at evaluation time (whitening statistics are fit once on a
dev/train pool, never on test data).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .backbone import FrozenMultilingualBackbone, BackboneConfig
from .compositional_aggregator import CompositionalAggregator
from .semantic_attention import SemanticAttentionPooling
from .predictive_head import PredictiveHead
from .memory_retrieval import content_only_score


class CognitiveEmbeddingCore(nn.Module):
    """Attention + Composition, combined -- the validated, single-sentence-
    capable half of the architecture."""

    def __init__(self, hidden_dim: int, n_heads: int = 4, gru_hidden: int = 128):
        super().__init__()
        self.attention = SemanticAttentionPooling(hidden_dim, n_heads=n_heads)
        self.composition = CompositionalAggregator(hidden_dim, gru_hidden=gru_hidden)
        self.combine = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, token_features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        attn_emb = self.attention(token_features, mask)
        comp_emb = self.composition(token_features, mask)
        return self.combine(torch.cat([attn_emb, comp_emb], dim=-1))


class CognitiveEmbeddingModel:
    """Top-level wrapper: frozen backbone + trainable core + optional
    discourse-level auxiliary heads. The backbone is shared and frozen;
    `core` and the auxiliary heads are the only trainable parameters."""

    def __init__(self, backbone_config: BackboneConfig | None = None, n_heads: int = 4, gru_hidden: int = 128):
        self.backbone = FrozenMultilingualBackbone(backbone_config)
        self.core = CognitiveEmbeddingCore(self.backbone.hidden_dim, n_heads=n_heads, gru_hidden=gru_hidden)
        # Discourse-level auxiliary head -- only meaningful/trained when
        # sequential (sentence_t, sentence_t+1) data is supplied, see train.py.
        self.predictive_head = PredictiveHead(self.backbone.hidden_dim)
        # Memory retrieval: validated default is the zero-parameter content_only
        # scorer (see memory_retrieval.py) -- no trainable module needed here.

    def embed(self, sentence: str) -> torch.Tensor:
        """The main entry point: raw sentence -> cognitive embedding vector."""
        token_features, mask = self.backbone.encode_tokens(sentence)
        return self.core(token_features, mask)

    def embed_tokens(self, token_features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """For use when token features are already cached (training loops)."""
        return self.core(token_features, mask)

    def trainable_parameters(self):
        return list(self.core.parameters()) + list(self.predictive_head.parameters())
