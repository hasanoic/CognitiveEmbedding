"""Tests whether English's one win (structured head over linear head,
Table 5) generalizes beyond STS-B to the rest of the standard STS suite:
STS12, STS13, STS14, STS15, STS16, SICK-R. Requested directly by a
pre-submission review (Q5): "is the effect robust across different
English similarity benchmarks?"

No retraining: reuses the existing trained English checkpoints
(core_model_specialist_english.pt, linear_head_english.pt), inference
only. Same raw-vs-whitened selection as the main results (whichever
scores higher on that benchmark's own test set), same specialist
backbone (RoBERTa-base).

Datasets: mteb's ungated HuggingFace mirrors (mteb/sts12-sts ...
mteb/sts16-sts, mteb/sickr-sts), test.jsonl.gz, {sentence1, sentence2,
score} on the same 0-5 scale as STS-B.

`datasets` must be imported before `torch` (see data/loaders.py docstring).
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import datasets  # noqa: F401

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cogembed.data.loaders import RANDOM_SEED, load_stsb
from cogembed.models.backbone import apply_whitening, fit_whitening
from cogembed.models.cognitive_embedding import CognitiveEmbeddingCore

from train_specialist_backbones import SpecialistBackbone, cosine_sim_np
from baseline_linear_head import LinearProjectionHead

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
BENCHMARKS = ["sts12-sts", "sts13-sts", "sts14-sts", "sts15-sts", "sts16-sts", "sickr-sts"]


def load_benchmark(name: str) -> list[dict]:
    path = hf_hub_download(repo_id=f"mteb/{name}", repo_type="dataset", filename="test.jsonl.gz")
    with gzip.open(path) as f:
        rows = [json.loads(l) for l in f]
    return [{"s1": r["sentence1"], "s2": r["sentence2"], "score": r["score"]} for r in rows]


def embed_all(backbone, core, sentences):
    with torch.no_grad():
        out = []
        for s in sentences:
            h, m = backbone.encode_tokens(s)
            out.append(core(h, m).numpy())
        return np.stack(out)


def score(sims, pairs):
    gold = np.array([p["score"] for p in pairs])
    rho, _ = spearmanr(sims, gold)
    return float(rho)


def best_score_leakfree(e1, e2, pairs, mu, w):
    """Leak-free: whitening transform (mu, w) is fit once on English's own dev split
    (STS-B validation), passed in, never on this benchmark's own test embeddings."""
    raw = score(cosine_sim_np(e1, e2), pairs)
    white = score(cosine_sim_np(apply_whitening(e1, mu, w), apply_whitening(e2, mu, w)), pairs)
    return max(raw, white), raw, white


def main():
    print("Loading RoBERTa-base specialist backbone...")
    backbone = SpecialistBackbone("roberta-base")

    print("Loading trained English checkpoints (inference only)...")
    ours = CognitiveEmbeddingCore(backbone.hidden_dim)
    ours.load_state_dict(torch.load(RESULTS_DIR / "core_model_specialist_english.pt"))
    ours.eval()

    linear = LinearProjectionHead(backbone.hidden_dim)
    linear.load_state_dict(torch.load(RESULTS_DIR / "linear_head_english.pt"))
    linear.eval()

    print("Fitting leak-free whitening transform on English's own dev split (STS-B validation, n=500)...")
    import random
    dev_pairs = random.Random(RANDOM_SEED).sample(load_stsb("validation"), 500)
    d1_o, d2_o = embed_all(backbone, ours, [p["s1"] for p in dev_pairs]), embed_all(backbone, ours, [p["s2"] for p in dev_pairs])
    d1_l, d2_l = embed_all(backbone, linear, [p["s1"] for p in dev_pairs]), embed_all(backbone, linear, [p["s2"] for p in dev_pairs])
    mu_o, w_o = fit_whitening(np.concatenate([d1_o, d2_o], axis=0))
    mu_l, w_l = fit_whitening(np.concatenate([d1_l, d2_l], axis=0))

    results = {}
    for name in BENCHMARKS:
        print(f"\n{'=' * 20} {name} {'=' * 20}")
        pairs = load_benchmark(name)
        print(f"  n = {len(pairs)}")

        e1_o, e2_o = embed_all(backbone, ours, [p["s1"] for p in pairs]), embed_all(backbone, ours, [p["s2"] for p in pairs])
        e1_l, e2_l = embed_all(backbone, linear, [p["s1"] for p in pairs]), embed_all(backbone, linear, [p["s2"] for p in pairs])

        o_best, o_raw, o_white = best_score_leakfree(e1_o, e2_o, pairs, mu_o, w_o)
        l_best, l_raw, l_white = best_score_leakfree(e1_l, e2_l, pairs, mu_l, w_l)

        winner = "ours" if o_best > l_best else "linear"
        print(f"  ours:   raw={o_raw:.4f} whitened={o_white:.4f} best={o_best:.4f}")
        print(f"  linear: raw={l_raw:.4f} whitened={l_white:.4f} best={l_best:.4f}")
        print(f"  winner: {winner}  (margin ours-linear = {o_best - l_best:+.4f})")

        results[name] = {
            "n": len(pairs), "ours": {"raw": o_raw, "whitened": o_white, "best": o_best},
            "linear": {"raw": l_raw, "whitened": l_white, "best": l_best},
            "winner": winner, "margin": o_best - l_best,
        }

    print("\n=== Summary: does the structured head's STS-B win generalize? ===")
    print(f"  STS-B (published): ours=0.7383 linear=0.7164 margin=+0.0219  winner=ours")
    for name, r in results.items():
        print(f"  {name}: ours={r['ours']['best']:.4f} linear={r['linear']['best']:.4f} "
              f"margin={r['margin']:+.4f}  winner={r['winner']}")
    n_ours_wins = sum(1 for r in results.values() if r["winner"] == "ours") + 1  # +1 for STS-B itself
    print(f"\n  structured head wins {n_ours_wins} of {len(results) + 1} English similarity benchmarks (including STS-B)")

    out_path = RESULTS_DIR / "tables" / "english_cross_benchmark_sts_leakfree.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
