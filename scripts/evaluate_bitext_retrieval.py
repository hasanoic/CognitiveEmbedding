"""Cross-lingual bitext retrieval -- the direct test of what a cross-lingual
embedding is actually FOR in a RAG setting: given a query in one language,
does the correct-meaning document in ANOTHER language rank first among
distractors? Every evaluation so far in this project has been per-language
monolingual (train on X, test on X's own STS/paraphrase/relatedness set) --
this is qualitatively different: it directly measures whether English and
Bangla (etc.) sentences of the same meaning land near each other in the
SAME shared embedding space, which per-language Spearman/AUROC numbers
cannot tell you on their own.

Uses mteb/flores (ungated mirror of FLORES-200 devtest -- facebook/flores
and openlanguagedata/flores_plus are both gated, this one isn't, see
conversation log) -- 1,012 sentences translated in parallel across 204
languages, so row i in column 'eng_Latn' and row i in column 'ben_Beng' are
guaranteed same-meaning translations.

Protocol (standard bitext mining, as used in the LASER/LaBSE papers
themselves): embed all N sentences in language A and all N in language B,
then for each sentence in A, rank all N candidates in B by cosine
similarity -- score is Recall@1 (does the true translation rank first) and
Recall@10, evaluated with English as the pivot (A) against each of the
other four languages (B) in turn, both directions (A->B and B->A, since
retrieval is not always symmetric).

`datasets` must be imported before `torch` (see data/loaders.py docstring).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import datasets  # noqa: F401

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cogembed.models.backbone import BackboneConfig
from cogembed.models.cognitive_embedding import CognitiveEmbeddingModel

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
N_SAMPLE = 500  # subset of the 1,012 devtest sentences -- keeps runtime bounded

FLORES_COLS = {
    "english": "eng_Latn",
    "bangla": "ben_Beng",
    "telugu": "tel_Telu",
    "hindi": "hin_Deva",
    "arabic": "arb_Arab",
}


def load_flores_subset(n: int = N_SAMPLE) -> dict:
    from huggingface_hub import hf_hub_download
    import pandas as pd

    path = hf_hub_download(repo_id="mteb/flores", repo_type="dataset", filename="devtest.parquet")
    df = pd.read_parquet(path)
    if n < len(df):
        df = df.iloc[:n].reset_index(drop=True)
    return {name: df[col].tolist() for name, col in FLORES_COLS.items()}


def cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_n = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)
    b_n = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-9)
    return a_n @ b_n.T


def recall_at_k(sims: np.ndarray, k: int) -> float:
    """sims[i, j] = similarity of query i to candidate j; true match is j==i."""
    n = sims.shape[0]
    topk = np.argpartition(-sims, kth=min(k, n - 1) - 1, axis=1)[:, :k]
    hits = sum(1 for i in range(n) if i in topk[i])
    return hits / n


def embed_ours(model, sentences: list[str]) -> np.ndarray:
    with torch.no_grad():
        return np.stack([model.embed(s).numpy() for s in sentences])


def embed_labse(tok, mdl, sentences: list[str]) -> np.ndarray:
    @torch.no_grad()
    def one(s):
        enc = tok(s, return_tensors="pt", truncation=True, max_length=64)
        out = mdl(**enc)
        mf = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (out.last_hidden_state * mf).sum(1) / mf.sum(1).clamp(min=1e-9)
        return torch.nn.functional.normalize(pooled, dim=-1)[0].numpy()
    return np.stack([one(s) for s in sentences])


def main():
    print("Loading FLORES-200 devtest subset (mteb/flores, ungated mirror)...")
    data = load_flores_subset(N_SAMPLE)
    for lang, sents in data.items():
        print(f"  {lang}: {len(sents)} sentences")

    print("\nLoading trained core checkpoint...")
    model = CognitiveEmbeddingModel(BackboneConfig())
    import os
    ckpt_name = os.environ.get("COGEMBED_CKPT", "core_model.pt")
    model.core.load_state_dict(torch.load(RESULTS_DIR / ckpt_name))
    print(f"Using checkpoint: {ckpt_name}")
    out_name = os.environ.get("COGEMBED_BITEXT_OUT", "bitext_retrieval_results.json")

    print("Loading LaBSE...")
    from transformers import AutoModel, AutoTokenizer

    labse_tok = AutoTokenizer.from_pretrained("sentence-transformers/LaBSE")
    labse_mdl = AutoModel.from_pretrained("sentence-transformers/LaBSE")
    labse_mdl.eval()

    print("\nEmbedding all languages (ours + LaBSE)...")
    ours_emb, labse_emb = {}, {}
    for lang, sents in data.items():
        ours_emb[lang] = embed_ours(model, sents)
        labse_emb[lang] = embed_labse(labse_tok, labse_mdl, sents)
        print(f"  {lang} done")

    print("\n=== Cross-lingual bitext retrieval: English <-> X, both directions ===")
    results = {}
    for lang in FLORES_COLS:
        if lang == "english":
            continue
        for direction, (src, tgt) in {"en_to_x": ("english", lang), "x_to_en": (lang, "english")}.items():
            ours_sims = cosine_sim_matrix(ours_emb[src], ours_emb[tgt])
            labse_sims = cosine_sim_matrix(labse_emb[src], labse_emb[tgt])
            ours_r1, ours_r5, ours_r10 = recall_at_k(ours_sims, 1), recall_at_k(ours_sims, 5), recall_at_k(ours_sims, 10)
            labse_r1, labse_r5, labse_r10 = recall_at_k(labse_sims, 1), recall_at_k(labse_sims, 5), recall_at_k(labse_sims, 10)
            key = f"{lang}_{direction}"
            results[key] = {
                "ours_recall@1": ours_r1, "ours_recall@5": ours_r5, "ours_recall@10": ours_r10,
                "labse_recall@1": labse_r1, "labse_recall@5": labse_r5, "labse_recall@10": labse_r10,
            }
            print(f"  {key}: ours R@1={ours_r1:.4f} R@5={ours_r5:.4f} R@10={ours_r10:.4f}  |  labse R@1={labse_r1:.4f} R@5={labse_r5:.4f} R@10={labse_r10:.4f}")

    out_path = RESULTS_DIR / "tables" / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
