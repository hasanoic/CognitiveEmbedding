"""Efficiency measurement -- the gap flagged before submission: the whole
positioning of this project rests on an efficiency claim, but until now
zero latency, memory-footprint, or parameter-count numbers existed anywhere
in the project. A reviewer evaluating an efficiency claim without
efficiency numbers will flag it immediately.

Measures, for every model compared this session (ours on shared XLM-R,
ours on specialist RoBERTa, LaBSE, mE5-base, MiniLM-L6-v2, and the raw
frozen backbone with no trained head at all):
  - Total parameters and TRAINABLE parameters (the actual training cost --
    for "ours" this is a few million; for the frozen backbone alone it's 0)
  - Encoding latency: ms/sentence, single-sentence (unbatched) on CPU,
    measured identically for every model -- fairness matters here, since
    this project's own encode_tokens is unbatched, so every comparison
    model is measured the same way rather than given a batching advantage
  - Embedding dimensionality and the resulting memory footprint for
    storing 1M vectors at float32 -- directly connects to the Matryoshka
    compression story (already measured accuracy-side; this adds the
    concrete storage-cost side)

Uses 100 real sentences from the STS-B test set (English) for latency
timing, run on the same CPU/thread configuration as every other
experiment in this project (OMP_NUM_THREADS=1).

`datasets` must be imported before `torch` (see data/loaders.py docstring).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import datasets  # noqa: F401

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cogembed.data.loaders import load_stsb
from cogembed.models.backbone import BackboneConfig, FrozenMultilingualBackbone, mean_pool
from cogembed.models.cognitive_embedding import CognitiveEmbeddingCore, CognitiveEmbeddingModel

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
N_TIMING_SENTENCES = 100
N_WARMUP = 5
N_REPEATS = 5  # repeated timing passes -- a single pass reports no variance at all, and the
# tiny (2-4%) differences this table is used to argue for need it. Matches measure_efficiency_linear.py.


def count_params(module) -> int:
    return sum(p.numel() for p in module.parameters())


def count_trainable(module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def time_encoding(embed_fn, sentences: list[str], repeats: int = N_REPEATS) -> dict:
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
    return {
        "ms_per_sentence": mean_ms, "ms_per_sentence_std": std_ms, "ms_per_sentence_min": min(passes_ms),
        "ms_per_sentence_passes": passes_ms, "sentences_per_sec": 1000.0 / mean_ms,
    }


def memory_footprint_1m(dim: int, dtype_bytes: int = 4) -> dict:
    bytes_per_vector = dim * dtype_bytes
    return {
        "bytes_per_vector": bytes_per_vector,
        "mb_per_1m_vectors": bytes_per_vector * 1_000_000 / (1024 ** 2),
        "gb_per_1m_vectors": bytes_per_vector * 1_000_000 / (1024 ** 3),
    }


def main():
    t_start = time.time()
    print(f"Loading {N_TIMING_SENTENCES} timing sentences (STS-B test)...")
    test_pairs = load_stsb("test")[:N_TIMING_SENTENCES]
    sentences = [p["s1"] for p in test_pairs]

    results = {}

    # ---- 1. Ours: shared XLM-R backbone + trained core ----
    print("\n=== Ours (xlm-roberta-base + cognitive core) ===")
    backbone = FrozenMultilingualBackbone(BackboneConfig())
    core = CognitiveEmbeddingCore(backbone.hidden_dim)
    core.load_state_dict(torch.load(RESULTS_DIR / "core_model.pt"))
    core.eval()

    def ours_embed(s):
        with torch.no_grad():
            h, m = backbone.encode_tokens(s)
            return core(h, m)

    timing = time_encoding(ours_embed, sentences)
    results["ours_xlmr"] = {
        "total_params": count_params(backbone.model) + count_params(core),
        "trainable_params": count_trainable(core),
        "backbone_params": count_params(backbone.model),
        "embedding_dim": backbone.hidden_dim,
        **timing,
        **memory_footprint_1m(backbone.hidden_dim),
    }
    print(f"  {results['ours_xlmr']['ms_per_sentence']:.2f} ms/sentence, "
          f"trainable={results['ours_xlmr']['trainable_params']:,} / total={results['ours_xlmr']['total_params']:,}")

    # ---- 2. Raw frozen backbone, no trained head (mean-pool baseline) ----
    print("\n=== Baseline: xlm-roberta-base, raw mean-pool, no trained head ===")

    def baseline_embed(s):
        with torch.no_grad():
            h, m = backbone.encode_tokens(s)
            return mean_pool(h, m)

    timing = time_encoding(baseline_embed, sentences)
    results["baseline_xlmr_meanpool"] = {
        "total_params": count_params(backbone.model),
        "trainable_params": 0,
        "embedding_dim": backbone.hidden_dim,
        **timing,
        **memory_footprint_1m(backbone.hidden_dim),
    }
    print(f"  {results['baseline_xlmr_meanpool']['ms_per_sentence']:.2f} ms/sentence, trainable=0")

    # ---- 3. Ours: specialist RoBERTa backbone (English) ----
    print("\n=== Ours (roberta-base specialist + cognitive core) ===")
    from transformers import AutoModel, AutoTokenizer

    roberta_tok = AutoTokenizer.from_pretrained("roberta-base")
    roberta_mdl = AutoModel.from_pretrained("roberta-base")
    roberta_mdl.eval()
    for p in roberta_mdl.parameters():
        p.requires_grad = False
    specialist_core = CognitiveEmbeddingCore(roberta_mdl.config.hidden_size)
    specialist_core.load_state_dict(torch.load(RESULTS_DIR / "core_model_specialist_english.pt"))
    specialist_core.eval()

    @torch.no_grad()
    def specialist_embed(s):
        enc = roberta_tok(s, return_tensors="pt", truncation=True, max_length=64)
        out = roberta_mdl(**enc, output_hidden_states=True)
        layers = torch.stack(out.hidden_states, dim=0)
        avg = layers.mean(dim=0)[0]
        mask = enc["attention_mask"][0]
        return specialist_core(avg, mask)

    timing = time_encoding(specialist_embed, sentences)
    results["ours_roberta_specialist"] = {
        "total_params": count_params(roberta_mdl) + count_params(specialist_core),
        "trainable_params": count_trainable(specialist_core),
        "backbone_params": count_params(roberta_mdl),
        "embedding_dim": roberta_mdl.config.hidden_size,
        **timing,
        **memory_footprint_1m(roberta_mdl.config.hidden_size),
    }
    print(f"  {results['ours_roberta_specialist']['ms_per_sentence']:.2f} ms/sentence, "
          f"trainable={results['ours_roberta_specialist']['trainable_params']:,}")

    # ---- 4-6. Comparison models ----
    comparison_models = {
        "labse": "sentence-transformers/LaBSE",
        "me5_base": "intfloat/multilingual-e5-base",
        "minilm_l6_v2": "sentence-transformers/all-MiniLM-L6-v2",
    }
    for name, repo in comparison_models.items():
        print(f"\n=== {name} ({repo}) ===")
        tok = AutoTokenizer.from_pretrained(repo)
        mdl = AutoModel.from_pretrained(repo)
        mdl.eval()

        @torch.no_grad()
        def embed(s, tok=tok, mdl=mdl):
            enc = tok(s, return_tensors="pt", truncation=True, max_length=64)
            out = mdl(**enc)
            mf = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (out.last_hidden_state * mf).sum(1) / mf.sum(1).clamp(min=1e-9)
            return torch.nn.functional.normalize(pooled, dim=-1)[0]

        timing = time_encoding(embed, sentences)
        dim = mdl.config.hidden_size
        results[name] = {
            "total_params": count_params(mdl),
            "trainable_params": 0,
            "embedding_dim": dim,
            **timing,
            **memory_footprint_1m(dim),
        }
        print(f"  {results[name]['ms_per_sentence']:.2f} ms/sentence, params={results[name]['total_params']:,}, dim={dim}")

    # ---- 7. Matryoshka tiers (storage-cost side of the already-measured accuracy) ----
    print("\n=== Memory footprint by Matryoshka tier (storage cost, not re-timed) ===")
    matryoshka_footprint = {}
    for dim in [768, 256, 128, 64]:
        matryoshka_footprint[dim] = memory_footprint_1m(dim)
        print(f"  {dim}-dim: {matryoshka_footprint[dim]['mb_per_1m_vectors']:.1f} MB / 1M vectors")
    results["matryoshka_footprint"] = matryoshka_footprint

    print("\n=== Summary ===")
    print(f"{'model':26}{'ms/sent':>10}{'sent/sec':>10}{'total_params':>15}{'trainable':>12}{'dim':>6}{'MB/1M vecs':>12}")
    for name, r in results.items():
        if name == "matryoshka_footprint":
            continue
        print(f"{name:26}{r['ms_per_sentence']:>10.2f}{r['sentences_per_sec']:>10.1f}"
              f"{r['total_params']:>15,}{r['trainable_params']:>12,}{r['embedding_dim']:>6}{r['mb_per_1m_vectors']:>12.1f}")

    out_path = RESULTS_DIR / "tables" / "efficiency_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults written to {out_path}")
    print(f"Total time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
