"""Paired bootstrap significance test for every language in the fair-baseline
comparison (Table 5 / tab:baseline-head: structured head vs. linear head,
specialist backbones). Generalizes significance_english_fair_baseline.py,
which covered English only, to all five languages, in response to a
pre-submission review asking for CIs on Table 5 as a whole.

No retraining: all ten checkpoints already exist from the original runs
(core_model_specialist_<lang>.pt, linear_head_<lang>.pt). This script
re-embeds each language's test set with both trained heads (inference
only), reproduces the published metric (Spearman for sts languages, AUROC
for Bangla) as a sanity check against results/tables/specialist_backbone_results.json
and linear_head_baseline_results.json, then runs a paired bootstrap: 10,000
resamples of the test set, metric recomputed on each resample for both
heads, recording the difference.

`datasets` must be imported before `torch` (see data/loaders.py docstring).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import datasets  # noqa: F401

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cogembed.data.loaders import load_bnpc_pairs, load_semrel_arabic, load_semrel_hindi, load_semrel_telugu, load_stsb
from cogembed.models.backbone import apply_whitening, fit_whitening
from cogembed.models.cognitive_embedding import CognitiveEmbeddingCore

from train_specialist_backbones import BACKBONES, SpecialistBackbone, cosine_sim_np
from baseline_linear_head import LinearProjectionHead

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
N_BOOTSTRAP = 10000
SEED = 42

LANGUAGES = {
    "english": ("sts", lambda: load_stsb("test")),
    "bangla": ("auroc", lambda: load_bnpc_pairs("test")),
    "telugu": ("sts", lambda: load_semrel_telugu("test")),
    "hindi": ("sts", lambda: load_semrel_hindi("test")),
    "arabic": ("sts", lambda: load_semrel_arabic("test")),
}

PUBLISHED = {
    # Matches Table tab:baseline-head exactly (best of raw/whitened per cell)
    "english": (0.7383, 0.7164),
    "bangla": (0.8452, 0.8647),
    "telugu": (0.7768, 0.7912),
    "hindi": (0.6980, 0.7061),
    "arabic": (0.4217, 0.4638),
}


def embed_all(backbone, core, sentences):
    with torch.no_grad():
        out = []
        for s in sentences:
            h, m = backbone.encode_tokens(s)
            out.append(core(h, m).numpy())
        return np.stack(out)


def metric(kind, sims, pairs):
    if kind == "sts":
        gold = np.array([p["score"] for p in pairs])
        rho, _ = spearmanr(sims, gold)
        return float(rho)
    labels = np.array([p["label"] for p in pairs])
    return float(roc_auc_score(labels, sims))


def best_sims(e1, e2, kind, pairs):
    """Reproduces evaluate_final's raw-vs-whitened selection: whitening is
    fit on this test set's own embeddings (the paper's disclosed whitening
    leakage caveat), and whichever of {raw, whitened} scores higher on the
    FULL test set is the published number. That choice is fixed here, not
    re-made per bootstrap resample, to avoid a multiple-comparisons bias
    from picking whichever looks better on each resample."""
    raw_sims = cosine_sim_np(e1, e2)
    raw_score = metric(kind, raw_sims, pairs)
    fit = np.concatenate([e1, e2], axis=0)
    mu, w = fit_whitening(fit)
    white_sims = cosine_sim_np(apply_whitening(e1, mu, w), apply_whitening(e2, mu, w))
    white_score = metric(kind, white_sims, pairs)
    if white_score > raw_score:
        return white_sims, white_score, "whitened"
    return raw_sims, raw_score, "raw"


def bootstrap_metric(kind, sims, pairs, idx):
    if kind == "sts":
        gold = np.array([p["score"] for p in pairs])
        rho, _ = spearmanr(sims[idx], gold[idx])
        return float(rho)
    labels = np.array([p["label"] for p in pairs])
    # AUROC needs both classes present in the resample; retry on degenerate draws
    if len(set(labels[idx])) < 2:
        return None
    return float(roc_auc_score(labels[idx], sims[idx]))


def main():
    rng_master = np.random.RandomState(SEED)
    all_results = {}

    for lang, (kind, loader) in LANGUAGES.items():
        print(f"\n{'=' * 20} {lang} {'=' * 20}")
        test_pairs = loader()
        n = len(test_pairs)
        print(f"  n = {n}")

        backbone = SpecialistBackbone(BACKBONES[lang])

        ours = CognitiveEmbeddingCore(backbone.hidden_dim)
        ours.load_state_dict(torch.load(RESULTS_DIR / f"core_model_specialist_{lang}.pt"))
        ours.eval()

        linear = LinearProjectionHead(backbone.hidden_dim)
        linear.load_state_dict(torch.load(RESULTS_DIR / f"linear_head_{lang}.pt"))
        linear.eval()

        e1_ours = embed_all(backbone, ours, [p["s1"] for p in test_pairs])
        e2_ours = embed_all(backbone, ours, [p["s2"] for p in test_pairs])
        e1_lin = embed_all(backbone, linear, [p["s1"] for p in test_pairs])
        e2_lin = embed_all(backbone, linear, [p["s2"] for p in test_pairs])

        sims_ours, m_ours, which_ours = best_sims(e1_ours, e2_ours, kind, test_pairs)
        sims_lin, m_lin, which_lin = best_sims(e1_lin, e2_lin, kind, test_pairs)
        pub_ours, pub_lin = PUBLISHED[lang]
        print(f"  reproduced: ours={m_ours:.4f} [{which_ours}] (paper: {pub_ours:.4f})  "
              f"linear={m_lin:.4f} [{which_lin}] (paper: {pub_lin:.4f})")
        print(f"  reproduced margin (ours - linear): {m_ours - m_lin:+.4f} (paper: {pub_ours - pub_lin:+.4f})")

        rng = np.random.RandomState(SEED + hash(lang) % 1000)
        diffs = []
        attempts = 0
        while len(diffs) < N_BOOTSTRAP and attempts < N_BOOTSTRAP * 3:
            attempts += 1
            idx = rng.randint(0, n, size=n)
            r_ours = bootstrap_metric(kind, sims_ours, test_pairs, idx)
            r_lin = bootstrap_metric(kind, sims_lin, test_pairs, idx)
            if r_ours is None or r_lin is None:
                continue
            diffs.append(r_ours - r_lin)
        diffs = np.array(diffs)

        lo, hi = float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))
        significant = bool(lo > 0 or hi < 0)
        print(f"  bootstrap ({len(diffs)} resamples): mean={diffs.mean():+.4f}  95% CI=[{lo:+.4f}, {hi:+.4f}]  "
              f"{'SIGNIFICANT' if significant else 'not significant'} at alpha=0.05")

        all_results[lang] = {
            "kind": kind, "n_test": n, "which_ours": which_ours, "which_linear": which_lin,
            "ours": m_ours, "linear": m_lin, "margin": m_ours - m_lin,
            "bootstrap_mean": float(diffs.mean()), "ci_lo": lo, "ci_hi": hi,
            "n_resamples": len(diffs), "significant_at_0.05": significant,
        }

    out_path = RESULTS_DIR / "tables" / "significance_fair_baseline_all_languages.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
