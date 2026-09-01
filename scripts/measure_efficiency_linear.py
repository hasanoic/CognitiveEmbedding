"""Adds the linear head to the efficiency comparison, measured in the SAME
run as the cognitive embedding and the untrained baseline -- a first attempt
measured the linear head in a separate script invocation and got an
implausible number (faster than the untrained baseline, which does strictly
less computation), which is exactly the kind of uncontrolled-timing mistake
measure_efficiency.py's own docstring warns against ("measured identically
... same CPU/thread configuration"). This script re-measures all three
(cognitive embedding, untrained baseline, linear head) back-to-back, same
process, same run, for both the shared XLM-R backbone and the specialist
RoBERTa backbone, so the comparison is actually controlled.

`datasets` must be imported before `torch` (see data/loaders.py docstring).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import datasets  # noqa: F401

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cogembed.data.loaders import load_stsb
from cogembed.models.backbone import BackboneConfig, FrozenMultilingualBackbone, mean_pool
from cogembed.models.cognitive_embedding import CognitiveEmbeddingCore

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
N_TIMING_SENTENCES = 100
N_WARMUP = 5
N_REPEATS = 5  # repeated timing passes, report mean+std (not just min) -- a bare "best of N" with
# no reported spread hides exactly the measurement noise this comparison's tiny (2-4%) margins need
# disclosed. Matches measure_efficiency.py's N_REPEATS so every row in Table efficiency uses the
# same methodology, not a min-of-3 for some rows and a single uncontrolled pass for others.


class LinearProjectionHead(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, token_features, mask):
        return self.proj(mean_pool(token_features, mask))


def count_params(module) -> int:
    return sum(p.numel() for p in module.parameters())


def count_trainable(module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def time_encoding_stats(embed_fn, sentences: list[str], repeats: int = N_REPEATS) -> dict:
    import statistics

    for s in sentences[:N_WARMUP]:
        embed_fn(s)
    passes_ms = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for s in sentences:
            embed_fn(s)
        elapsed = time.perf_counter() - t0
        passes_ms.append((elapsed / len(sentences)) * 1000)
    mean_ms = statistics.mean(passes_ms)
    std_ms = statistics.stdev(passes_ms) if len(passes_ms) > 1 else 0.0
    return {"mean": mean_ms, "std": std_ms, "min": min(passes_ms), "passes": passes_ms}


def main():
    t_start = time.time()
    print(f"Loading {N_TIMING_SENTENCES} timing sentences (STS-B test)...")
    sentences = [p["s1"] for p in load_stsb("test")[:N_TIMING_SENTENCES]]

    results = {}

    print("\nLoading shared XLM-R backbone + all three shared-backbone heads...")
    backbone = FrozenMultilingualBackbone(BackboneConfig())

    ce_core = CognitiveEmbeddingCore(backbone.hidden_dim)
    ce_core.load_state_dict(torch.load(RESULTS_DIR / "core_model_crosslingual.pt"))
    ce_core.eval()

    lin_core = LinearProjectionHead(backbone.hidden_dim)
    lin_core.load_state_dict(torch.load(RESULTS_DIR / "core_model_linear_crosslingual.pt"))
    lin_core.eval()

    def ce_embed(s):
        with torch.no_grad():
            h, m = backbone.encode_tokens(s)
            return ce_core(h, m)

    def baseline_embed(s):
        with torch.no_grad():
            h, m = backbone.encode_tokens(s)
            return mean_pool(h, m)

    def lin_embed(s):
        with torch.no_grad():
            h, m = backbone.encode_tokens(s)
            return lin_core(h, m)

    print(f"Timing ({N_REPEATS} interleaved passes each, same process, reporting mean+std, not just min)...")
    ce_stats = time_encoding_stats(ce_embed, sentences)
    base_stats = time_encoding_stats(baseline_embed, sentences)
    lin_stats = time_encoding_stats(lin_embed, sentences)

    results["cognitive_xlmr"] = {"ms_per_sentence": ce_stats["mean"], "ms_per_sentence_std": ce_stats["std"],
                                  "ms_per_sentence_min": ce_stats["min"], "ms_per_sentence_passes": ce_stats["passes"],
                                  "total_params": count_params(backbone.model) + count_params(ce_core),
                                  "trainable_params": count_trainable(ce_core)}
    results["baseline_xlmr"] = {"ms_per_sentence": base_stats["mean"], "ms_per_sentence_std": base_stats["std"],
                                 "ms_per_sentence_min": base_stats["min"], "ms_per_sentence_passes": base_stats["passes"],
                                 "total_params": count_params(backbone.model), "trainable_params": 0}
    results["linear_xlmr"] = {"ms_per_sentence": lin_stats["mean"], "ms_per_sentence_std": lin_stats["std"],
                               "ms_per_sentence_min": lin_stats["min"], "ms_per_sentence_passes": lin_stats["passes"],
                               "total_params": count_params(backbone.model) + count_params(lin_core),
                               "trainable_params": count_trainable(lin_core)}

    print(f"\n=== Final ({N_REPEATS}-pass mean +/- std) shared-XLM-R numbers ===")
    print(f"  cognitive embedding: {ce_stats['mean']:.2f} +/- {ce_stats['std']:.2f} ms/sentence, trainable={count_trainable(ce_core):,}")
    print(f"  untrained baseline:  {base_stats['mean']:.2f} +/- {base_stats['std']:.2f} ms/sentence, trainable=0")
    print(f"  linear head:         {lin_stats['mean']:.2f} +/- {lin_stats['std']:.2f} ms/sentence, trainable={count_trainable(lin_core):,}")

    print("\nLoading specialist RoBERTa backbone + both specialist-backbone heads (English)...")
    from transformers import AutoModel, AutoTokenizer

    roberta_tok = AutoTokenizer.from_pretrained("roberta-base")
    roberta_mdl = AutoModel.from_pretrained("roberta-base")
    roberta_mdl.eval()
    for p in roberta_mdl.parameters():
        p.requires_grad = False

    ce_specialist = CognitiveEmbeddingCore(roberta_mdl.config.hidden_size)
    ce_specialist.load_state_dict(torch.load(RESULTS_DIR / "core_model_specialist_english.pt"))
    ce_specialist.eval()

    lin_specialist = LinearProjectionHead(roberta_mdl.config.hidden_size)
    lin_specialist.load_state_dict(torch.load(RESULTS_DIR / "linear_head_english.pt"))
    lin_specialist.eval()

    @torch.no_grad()
    def encode_roberta(s):
        enc = roberta_tok(s, return_tensors="pt", truncation=True, max_length=64)
        out = roberta_mdl(**enc, output_hidden_states=True)
        layers = torch.stack(out.hidden_states, dim=0)
        avg = layers.mean(dim=0)[0]
        mask = enc["attention_mask"][0]
        return avg, mask

    def ce_specialist_embed(s):
        avg, mask = encode_roberta(s)
        with torch.no_grad():
            return ce_specialist(avg, mask)

    def lin_specialist_embed(s):
        avg, mask = encode_roberta(s)
        with torch.no_grad():
            return lin_specialist(avg, mask)

    print(f"Timing ({N_REPEATS} interleaved passes, same pattern as the XLM-R section above)...")
    ce_spec_stats = time_encoding_stats(ce_specialist_embed, sentences)
    lin_spec_stats = time_encoding_stats(lin_specialist_embed, sentences)

    results["cognitive_roberta_specialist"] = {"ms_per_sentence": ce_spec_stats["mean"], "ms_per_sentence_std": ce_spec_stats["std"],
                                                "ms_per_sentence_min": ce_spec_stats["min"], "ms_per_sentence_passes": ce_spec_stats["passes"],
                                                "total_params": count_params(roberta_mdl) + count_params(ce_specialist),
                                                "trainable_params": count_trainable(ce_specialist)}
    results["linear_roberta_specialist"] = {"ms_per_sentence": lin_spec_stats["mean"], "ms_per_sentence_std": lin_spec_stats["std"],
                                             "ms_per_sentence_min": lin_spec_stats["min"], "ms_per_sentence_passes": lin_spec_stats["passes"],
                                             "total_params": count_params(roberta_mdl) + count_params(lin_specialist),
                                             "trainable_params": count_trainable(lin_specialist)}

    print(f"\n=== Final ({N_REPEATS}-pass mean +/- std) specialist-RoBERTa (English) numbers ===")
    print(f"  cognitive embedding: {ce_spec_stats['mean']:.2f} +/- {ce_spec_stats['std']:.2f} ms/sentence, trainable={count_trainable(ce_specialist):,}")
    print(f"  linear head:         {lin_spec_stats['mean']:.2f} +/- {lin_spec_stats['std']:.2f} ms/sentence, trainable={count_trainable(lin_specialist):,}")

    out_path = RESULTS_DIR / "tables" / "efficiency_results_linear.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults written to {out_path}")
    print(f"Total time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
