"""Multilingual generalization report -- the direct test of the claim "this
is a language-general cognitive embedding method, not an English-specific
one" (see conversation log). Two tiers, evaluated with the SAME protocol:

  TRAINED tier (native supervision during Stage 1 of train.py, three
  typologically unrelated families): English (Germanic, STS-B), Bangla
  (Indo-Aryan, BnPC gold), Telugu (Dravidian, SemRel2024).

  ZERO-SHOT tier (the architecture never sees these languages in ANY form
  during training -- the real transfer test): Hindi (Indo-Aryan, within-
  family check against Bangla), Arabic (Semitic, out-of-family check,
  different script and morphology).

For every language, reports three numbers so relative improvement (not just
absolute score) can be judged -- absolute scores are not comparable across
languages because backbone quality varies per language, but relative
improvement over the SAME language's own untrained baseline is:
  1. untrained baseline: frozen backbone + mean-pool + whitening, NO
     trained core module at all (isolates what the trained core adds).
  2. ours: the trained CognitiveEmbeddingCore (this run's checkpoint).
  3. LaBSE: specialist multilingual baseline, zero-shot on every language
     here (LaBSE was never fine-tuned on any of this project's data).

Final summary reports the macro-average and cross-language STANDARD
DEVIATION of (ours - untrained) relative improvement -- low variance across
unrelated language families is the actual evidence for "language-general
mechanism," not any single language's absolute number.

`datasets` must be imported before `torch` (see data/loaders.py docstring).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import datasets  # noqa: F401 -- import-order fix, must precede torch

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cogembed.data.loaders import load_bnpc_pairs, load_semrel_arabic, load_semrel_hindi, load_semrel_telugu, load_stsb
from cogembed.models.backbone import BackboneConfig, FrozenMultilingualBackbone, apply_whitening, fit_whitening, mean_pool
from cogembed.models.cognitive_embedding import CognitiveEmbeddingModel

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

LANGUAGES = {
    "english": {"tier": "trained", "family": "Germanic", "kind": "sts", "loader": lambda: load_stsb("test")},
    "bangla": {"tier": "trained", "family": "Indo-Aryan", "kind": "auroc", "loader": lambda: load_bnpc_pairs("test")},
    "telugu": {"tier": "trained", "family": "Dravidian", "kind": "sts", "loader": lambda: load_semrel_telugu("test")},
    "hindi": {"tier": "zero-shot", "family": "Indo-Aryan", "kind": "sts", "loader": lambda: load_semrel_hindi("test")},
    "arabic": {"tier": "zero-shot", "family": "Semitic", "kind": "sts", "loader": lambda: load_semrel_arabic("test")},
}


def cosine_sim_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_n = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)
    b_n = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-9)
    return (a_n * b_n).sum(axis=-1)


def score(kind: str, e1: np.ndarray, e2: np.ndarray, pairs: list[dict]) -> float:
    sims = cosine_sim_np(e1, e2)
    if kind == "sts":
        gold = np.array([p["score"] for p in pairs])
        rho, _ = spearmanr(sims, gold)
        return float(rho)
    labels = np.array([p["label"] for p in pairs])
    return float(roc_auc_score(labels, sims))


def embed_backbone_whitened(backbone: FrozenMultilingualBackbone, pairs: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Untrained baseline: frozen backbone, mean-pool, whitening fit on this eval pool -- no core module."""
    def embed_all(sentences):
        out = []
        with torch.no_grad():
            for s in sentences:
                h, m = backbone.encode_tokens(s)
                out.append(mean_pool(h, m).numpy())
        return np.stack(out)

    e1, e2 = embed_all([p["s1"] for p in pairs]), embed_all([p["s2"] for p in pairs])
    fit = np.concatenate([e1, e2], axis=0)
    mu, w = fit_whitening(fit)
    return apply_whitening(e1, mu, w), apply_whitening(e2, mu, w)


def embed_ours(model: CognitiveEmbeddingModel, pairs: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    def embed_all(sentences):
        with torch.no_grad():
            return np.stack([model.embed(s).numpy() for s in sentences])

    e1, e2 = embed_all([p["s1"] for p in pairs]), embed_all([p["s2"] for p in pairs])
    fit = np.concatenate([e1, e2], axis=0)
    mu, w = fit_whitening(fit)
    return apply_whitening(e1, mu, w), apply_whitening(e2, mu, w)


def embed_labse(tok, mdl, pairs: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    @torch.no_grad()
    def labse_embed(sentence: str) -> np.ndarray:
        enc = tok(sentence, return_tensors="pt", truncation=True, max_length=64)
        out = mdl(**enc)
        mf = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (out.last_hidden_state * mf).sum(1) / mf.sum(1).clamp(min=1e-9)
        return torch.nn.functional.normalize(pooled, dim=-1)[0].numpy()

    def embed_all(sentences):
        return np.stack([labse_embed(s) for s in sentences])

    return embed_all([p["s1"] for p in pairs]), embed_all([p["s2"] for p in pairs])


def main() -> None:
    print("Loading frozen backbone + trained core checkpoint...")
    model = CognitiveEmbeddingModel(BackboneConfig())
    core_ckpt = RESULTS_DIR / "core_model.pt"
    if core_ckpt.exists():
        model.core.load_state_dict(torch.load(core_ckpt))
        print(f"Loaded trained core from {core_ckpt}")
    else:
        print("No trained core checkpoint found -- evaluating UNTRAINED core (will match baseline).")

    print("Loading LaBSE (sentence-transformers/LaBSE)...")
    from transformers import AutoModel, AutoTokenizer

    labse_tok = AutoTokenizer.from_pretrained("sentence-transformers/LaBSE")
    labse_mdl = AutoModel.from_pretrained("sentence-transformers/LaBSE")
    labse_mdl.eval()

    results = {}
    for lang, cfg in LANGUAGES.items():
        pairs = cfg["loader"]()
        kind = cfg["kind"]
        print(f"\n=== {lang} ({cfg['family']}, {cfg['tier']}, n={len(pairs)}) ===")

        b1, b2 = embed_backbone_whitened(model.backbone, pairs)
        baseline = score(kind, b1, b2, pairs)

        o1, o2 = embed_ours(model, pairs)
        ours = score(kind, o1, o2, pairs)

        l1, l2 = embed_labse(labse_tok, labse_mdl, pairs)
        labse = score(kind, l1, l2, pairs)

        rel_improvement = ours - baseline
        print(f"  untrained_baseline={baseline:.4f}  ours={ours:.4f} (delta={rel_improvement:+.4f})  labse={labse:.4f}")

        results[lang] = {
            "family": cfg["family"],
            "tier": cfg["tier"],
            "metric": "spearman" if kind == "sts" else "auroc",
            "n_pairs": len(pairs),
            "untrained_baseline": baseline,
            "ours": ours,
            "relative_improvement": rel_improvement,
            "labse": labse,
        }

    trained_deltas = [r["relative_improvement"] for r in results.values() if r["tier"] == "trained"]
    zeroshot_deltas = [r["relative_improvement"] for r in results.values() if r["tier"] == "zero-shot"]
    all_deltas = [r["relative_improvement"] for r in results.values()]

    print("\n=== Language-generality summary ===")
    print(f"  Trained tier   -- mean delta={np.mean(trained_deltas):+.4f}  std={np.std(trained_deltas):.4f}  (n_languages={len(trained_deltas)})")
    print(f"  Zero-shot tier -- mean delta={np.mean(zeroshot_deltas):+.4f}  std={np.std(zeroshot_deltas):.4f}  (n_languages={len(zeroshot_deltas)})")
    print(f"  All languages  -- mean delta={np.mean(all_deltas):+.4f}  std={np.std(all_deltas):.4f}")
    print("  Low std across typologically unrelated families = evidence of a language-general mechanism,")
    print("  not proof of it -- five languages is a small sample for a variance claim; report accordingly.")

    summary = {
        "per_language": results,
        "trained_tier_mean_delta": float(np.mean(trained_deltas)),
        "trained_tier_std_delta": float(np.std(trained_deltas)),
        "zeroshot_tier_mean_delta": float(np.mean(zeroshot_deltas)),
        "zeroshot_tier_std_delta": float(np.std(zeroshot_deltas)),
        "all_mean_delta": float(np.mean(all_deltas)),
        "all_std_delta": float(np.std(all_deltas)),
    }
    out_path = RESULTS_DIR / "tables" / "multilingual_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
