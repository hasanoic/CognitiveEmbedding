"""Re-derives the "Wikipedia failed for the predictive head" claim (paper
Section 5.5) with a fresh, logged run -- the same kind of check that caught
the unverifiable memory-scoring numbers earlier in this project -- and,
using the same run, computes a cheap narrative-coherence proxy for both
corpora tested (wikitext-2 and CNN/DailyMail) to answer a reviewer question
directly: does a measurable property of the corpus predict whether the
predictive head's Recall@1 beats its own untrained baseline?

Coherence proxy: mean cosine similarity between EVERY pair of consecutive
sentences within a document (frozen backbone, raw mean-pooled embeddings,
no whitening), averaged across documents. Higher means the corpus has more
locally predictable, narratively continuous sentence-to-sentence structure
-- the property the predictive head is trying to exploit.

With only two corpora tested, we do NOT compute a formal correlation
coefficient (n=2 is not a meaningful sample for that) -- we report both
values side by side and describe the direction of the relationship
honestly, not as a statistically validated finding.

Uses the specialist English backbone (RoBERTa-base) and the identical
PredictiveHead training recipe as evaluate_memory_predictive_specialist.py,
for direct comparability with that script's CNN/DailyMail result.

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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cogembed.data.loaders import load_cnn_dailymail_documents, load_wikitext_documents
from cogembed.models.backbone import mean_pool

from evaluate_memory_predictive_specialist import SpecialistBackbone, evaluate_predictive

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
BACKBONE_REPO = "roberta-base"


def coherence_proxy(backbone, documents) -> float:
    sims = []
    with torch.no_grad():
        for doc in documents:
            if len(doc) < 2:
                continue
            embs = []
            for s in doc:
                h, m = backbone.encode_tokens(s)
                embs.append(mean_pool(h, m).numpy())
            embs = np.stack(embs)
            norm = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9)
            for i in range(len(norm) - 1):
                sims.append(float(norm[i] @ norm[i + 1]))
    return float(np.mean(sims)), len(sims)


def main():
    backbone = SpecialistBackbone(BACKBONE_REPO)

    results = {}
    for name, loader in [("wikitext2", lambda: load_wikitext_documents(300)),
                          ("cnn_dailymail", lambda: load_cnn_dailymail_documents(300))]:
        print(f"\n=== {name} ===")
        documents = loader()
        coh, n_pairs = coherence_proxy(backbone, documents)
        print(f"  coherence proxy (mean consecutive-sentence cosine, n_pairs={n_pairs}): {coh:.4f}")

        r, head = evaluate_predictive(backbone, documents)
        print(f"  candidates={r['candidate_set_size']}  R@1: trained={r['trained_recall@1']:.4f} "
              f"untrained={r['untrained_recall@1']:.4f} chance={r['chance_recall@1']:.4f}")
        results[name] = {"coherence_proxy": coh, "n_coherence_pairs": n_pairs, **r}

    print("\n=== Summary (specialist English backbone, RoBERTa-base) ===")
    for name, r in results.items():
        print(f"  {name}: coherence={r['coherence_proxy']:.4f}  R@1 trained={r['trained_recall@1']:.4f} "
              f"untrained={r['untrained_recall@1']:.4f}  trained>untrained={r['trained_recall@1'] > r['untrained_recall@1']}")

    out_path = RESULTS_DIR / "tables" / "predictive_corpus_diagnostics.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
