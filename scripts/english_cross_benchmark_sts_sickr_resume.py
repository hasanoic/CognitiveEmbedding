"""Resume script: the original english_cross_benchmark_sts.py run was cut
off by a power outage partway through embedding SICK-R (last of six
benchmarks; STS12-16 had already completed and their results are in
logs/english_cross_benchmark_sts_run.log). This re-runs SICK-R only,
reusing the exact same logic, then merges with the already-completed
STS12-16 numbers (transcribed from the pre-outage log, deterministic
given fixed checkpoints and seeds) into the final results JSON.
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

from cogembed.models.cognitive_embedding import CognitiveEmbeddingCore

from train_specialist_backbones import SpecialistBackbone, cosine_sim_np
from baseline_linear_head import LinearProjectionHead

from english_cross_benchmark_sts import load_benchmark, embed_all, best_score

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

# Transcribed from logs/english_cross_benchmark_sts_run.log (pre-outage, completed benchmarks)
COMPLETED = {
    "sts12-sts": {"n": 3108, "ours": {"raw": 0.6246, "whitened": 0.4958, "best": 0.6246},
                  "linear": {"raw": 0.5921, "whitened": 0.4239, "best": 0.5921}, "winner": "ours", "margin": 0.0325},
    "sts13-sts": {"n": 1500, "ours": {"raw": 0.7304, "whitened": 0.7845, "best": 0.7845},
                  "linear": {"raw": 0.7214, "whitened": 0.7547, "best": 0.7547}, "winner": "ours", "margin": 0.0298},
    "sts14-sts": {"n": 3750, "ours": {"raw": 0.6575, "whitened": 0.6872, "best": 0.6872},
                  "linear": {"raw": 0.6365, "whitened": 0.6412, "best": 0.6412}, "winner": "ours", "margin": 0.0460},
    "sts15-sts": {"n": 3000, "ours": {"raw": 0.7417, "whitened": 0.6485, "best": 0.7417},
                  "linear": {"raw": 0.7865, "whitened": 0.6185, "best": 0.7865}, "winner": "linear", "margin": -0.0448},
    "sts16-sts": {"n": 1186, "ours": {"raw": 0.7535, "whitened": 0.7041, "best": 0.7535},
                  "linear": {"raw": 0.7180, "whitened": 0.6518, "best": 0.7180}, "winner": "ours", "margin": 0.0355},
}


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

    print("\n==================== sickr-sts ====================")
    pairs = load_benchmark("sickr-sts")
    print(f"  n = {len(pairs)}")

    e1_o, e2_o = embed_all(backbone, ours, [p["s1"] for p in pairs]), embed_all(backbone, ours, [p["s2"] for p in pairs])
    e1_l, e2_l = embed_all(backbone, linear, [p["s1"] for p in pairs]), embed_all(backbone, linear, [p["s2"] for p in pairs])

    o_best, o_raw, o_white = best_score(e1_o, e2_o, pairs)
    l_best, l_raw, l_white = best_score(e1_l, e2_l, pairs)
    winner = "ours" if o_best > l_best else "linear"
    print(f"  ours:   raw={o_raw:.4f} whitened={o_white:.4f} best={o_best:.4f}")
    print(f"  linear: raw={l_raw:.4f} whitened={l_white:.4f} best={l_best:.4f}")
    print(f"  winner: {winner}  (margin ours-linear = {o_best - l_best:+.4f})")

    results = dict(COMPLETED)
    results["sickr-sts"] = {
        "n": len(pairs), "ours": {"raw": o_raw, "whitened": o_white, "best": o_best},
        "linear": {"raw": l_raw, "whitened": l_white, "best": l_best},
        "winner": winner, "margin": o_best - l_best,
    }

    print("\n=== Summary: does the structured head's STS-B win generalize? ===")
    print(f"  STS-B (published): ours=0.7383 linear=0.7164 margin=+0.0219  winner=ours")
    for name, r in results.items():
        print(f"  {name}: ours={r['ours']['best']:.4f} linear={r['linear']['best']:.4f} "
              f"margin={r['margin']:+.4f}  winner={r['winner']}")
    n_ours_wins = sum(1 for r in results.values() if r["winner"] == "ours") + 1
    print(f"\n  structured head wins {n_ours_wins} of {len(results) + 1} English similarity benchmarks (including STS-B)")

    out_path = RESULTS_DIR / "tables" / "english_cross_benchmark_sts.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
