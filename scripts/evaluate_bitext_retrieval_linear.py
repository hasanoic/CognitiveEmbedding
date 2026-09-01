"""Adds the fair linear-head baseline to the bitext retrieval comparison --
reuses evaluate_bitext_retrieval.py's data loading and scoring functions
directly (imported, not duplicated) and adds a third embedding method using
core_model_linear_crosslingual.pt (see train_linear_shared.py /
train_linear_crosslingual.py). Written because a preliminary look at the
linear head's VALIDATION-split bitext recall during its own training (not
the held-out devtest set this script uses) looked unusually high and needed
a genuine, apples-to-apples check on the same devtest set and same protocol
as the published Table tab:bitext numbers, rather than being reported on
the weaker validation-split evidence.

Writes to results/tables/bitext_retrieval_results_linear.json -- does not
touch or overwrite the original bitext_retrieval_results.json.

`datasets` must be imported before `torch` (see data/loaders.py docstring).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import datasets  # noqa: F401

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cogembed.models.backbone import BackboneConfig, mean_pool
from cogembed.models.cognitive_embedding import CognitiveEmbeddingModel

from evaluate_bitext_retrieval import (
    FLORES_COLS,
    N_SAMPLE,
    cosine_sim_matrix,
    embed_labse,
    embed_ours,
    load_flores_subset,
    recall_at_k,
)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


class LinearProjectionHead(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, token_features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.proj(mean_pool(token_features, mask))


def main():
    print("Loading FLORES-200 devtest subset (mteb/flores, ungated mirror)...")
    data = load_flores_subset(N_SAMPLE)
    for lang, sents in data.items():
        print(f"  {lang}: {len(sents)} sentences")

    print("\nLoading trained core checkpoints (cognitive embedding + linear head)...")
    model = CognitiveEmbeddingModel(BackboneConfig())
    model.core.load_state_dict(torch.load(RESULTS_DIR / "core_model_crosslingual.pt"))

    linear_model = CognitiveEmbeddingModel(BackboneConfig())
    linear_model.core = LinearProjectionHead(linear_model.backbone.hidden_dim)
    linear_model.core.load_state_dict(torch.load(RESULTS_DIR / "core_model_linear_crosslingual.pt"))

    print("Loading LaBSE...")
    from transformers import AutoModel, AutoTokenizer

    labse_tok = AutoTokenizer.from_pretrained("sentence-transformers/LaBSE")
    labse_mdl = AutoModel.from_pretrained("sentence-transformers/LaBSE")
    labse_mdl.eval()

    print("\nEmbedding all languages (ours + linear + LaBSE)...")
    ours_emb, linear_emb, labse_emb = {}, {}, {}
    for lang, sents in data.items():
        ours_emb[lang] = embed_ours(model, sents)
        linear_emb[lang] = embed_ours(linear_model, sents)  # same fn -- just calls .embed()
        labse_emb[lang] = embed_labse(labse_tok, labse_mdl, sents)
        print(f"  {lang} done")

    print("\n=== Cross-lingual bitext retrieval: English <-> X, both directions ===")
    results = {}
    for lang in FLORES_COLS:
        if lang == "english":
            continue
        for direction, (src, tgt) in {"en_to_x": ("english", lang), "x_to_en": (lang, "english")}.items():
            ours_sims = cosine_sim_matrix(ours_emb[src], ours_emb[tgt])
            linear_sims = cosine_sim_matrix(linear_emb[src], linear_emb[tgt])
            labse_sims = cosine_sim_matrix(labse_emb[src], labse_emb[tgt])
            ours_r1, ours_r5, ours_r10 = recall_at_k(ours_sims, 1), recall_at_k(ours_sims, 5), recall_at_k(ours_sims, 10)
            linear_r1, linear_r5, linear_r10 = recall_at_k(linear_sims, 1), recall_at_k(linear_sims, 5), recall_at_k(linear_sims, 10)
            labse_r1, labse_r5, labse_r10 = recall_at_k(labse_sims, 1), recall_at_k(labse_sims, 5), recall_at_k(labse_sims, 10)
            key = f"{lang}_{direction}"
            results[key] = {
                "ours_recall@1": ours_r1, "ours_recall@5": ours_r5, "ours_recall@10": ours_r10,
                "linear_recall@1": linear_r1, "linear_recall@5": linear_r5, "linear_recall@10": linear_r10,
                "labse_recall@1": labse_r1, "labse_recall@5": labse_r5, "labse_recall@10": labse_r10,
            }
            print(f"  {key}: ours R@1={ours_r1:.4f} R@5={ours_r5:.4f}  |  linear R@1={linear_r1:.4f} R@5={linear_r5:.4f}  |  labse R@1={labse_r1:.4f} R@5={labse_r5:.4f}")

    out_path = RESULTS_DIR / "tables" / "bitext_retrieval_results_linear.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
