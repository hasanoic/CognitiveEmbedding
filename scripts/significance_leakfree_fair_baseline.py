"""Recomputes the paired bootstrap significance test for Table 5
(tab:baseline-head) under the leak-free, dev-fit-whitened protocol now
used as primary (see "Whitening leakage, and what still reflects the
earlier protocol", Discussion and Limitations). The earlier bootstrap
CIs (significance_fair_baseline_all_languages.py) were computed under
the original test-fit-whitened selection, which stopped being primary
once Table 5 switched to dev-fit whitening -- those CIs no longer
describe the numbers actually printed in the table. This closes that
gap directly, requested explicitly: "Report confidence intervals
wherever possible, particularly for the central Table 5 comparison."

No retraining: reuses the existing trained checkpoints (inference
only). For each language and both heads: embed the test set AND the
dev set (same dev splits as whitening_dev_only.py), fit whitening on
dev, apply to test, and pick whichever of {raw, dev-fit-whitened}
matches the language's published Table 5 selection (fixed once against
the full sample, not re-selected per bootstrap resample, to avoid a
multiple-comparisons bias -- same discipline as every other bootstrap
in this project). Bootstrap: 10,000 resamples of the test set.

`datasets` must be imported before `torch` (see data/loaders.py docstring).
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import datasets  # noqa: F401

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cogembed.data.loaders import (
    RANDOM_SEED,
    load_bnpc_pairs,
    load_semrel_arabic,
    load_semrel_hindi,
    load_semrel_telugu,
    load_stsb,
)
from cogembed.models.backbone import apply_whitening, fit_whitening
from cogembed.models.cognitive_embedding import CognitiveEmbeddingCore

from train_specialist_backbones import BACKBONES, SpecialistBackbone, cosine_sim_np
from baseline_linear_head import LinearProjectionHead

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
N_BOOTSTRAP = 10000
SEED = 42

# Which selection (raw vs dev-fit-whitened) matches Table 5's current published value,
# fixed from whitening_dev_only.py's already-computed results (not re-derived here).
LANGUAGES = {
    "english": ("sts", lambda rng: rng.sample(load_stsb("validation"), 500), lambda: load_stsb("test"), {"ours": "raw", "linear": "raw"}),
    # Bangla's dev-fit-whitened selection was chosen because it scored higher on the
    # TEST set -- itself a mild form of test-set peeking. A proper split-dev check
    # (verify_bangla_split_dev_selection.py: fit whitening on one held-out half of
    # dev, decide raw-vs-whitened on the OTHER half, never touching test) shows
    # whitening does not actually help Bangla (selection-half AUROC ~0.84 raw vs
    # ~0.69 whitened, not a close call) -- raw is the correct, leak-free selection.
    "bangla": ("auroc", lambda rng: (lambda d: rng.sample(d, 500) if len(d) > 500 else d)(load_bnpc_pairs("validation")), lambda: load_bnpc_pairs("test"), {"ours": "raw", "linear": "raw"}),
    "telugu": ("sts", lambda rng: load_semrel_telugu("dev"), lambda: load_semrel_telugu("test"), {"ours": "raw", "linear": "raw"}),
    "hindi": ("sts", lambda rng: load_semrel_hindi("dev"), lambda: load_semrel_hindi("test"), {"ours": "raw", "linear": "raw"}),
    "arabic": ("sts", lambda rng: load_semrel_arabic("dev"), lambda: load_semrel_arabic("test"), {"ours": "raw", "linear": "raw"}),
}

PUBLISHED_LEAKFREE = {
    "english": {"ours": 0.7383, "linear": 0.7164},
    "bangla": {"ours": 0.8452, "linear": 0.8647},
    "telugu": {"ours": 0.7768, "linear": 0.7912},
    "hindi": {"ours": 0.6777, "linear": 0.7061},
    "arabic": {"ours": 0.4149, "linear": 0.4526},
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


def bootstrap_metric(kind, sims, pairs, idx):
    if kind == "sts":
        gold = np.array([p["score"] for p in pairs])
        rho, _ = spearmanr(sims[idx], gold[idx])
        return float(rho)
    labels = np.array([p["label"] for p in pairs])
    if len(set(labels[idx])) < 2:
        return None
    return float(roc_auc_score(labels[idx], sims[idx]))


def main():
    out_path = RESULTS_DIR / "tables" / "significance_leakfree_fair_baseline.json"
    all_results = json.loads(out_path.read_text()) if out_path.exists() else {}
    if all_results:
        print(f"Resuming: {list(all_results.keys())} already done, skipping.")

    for lang, (kind, dev_loader, test_loader, selection) in LANGUAGES.items():
        if lang in all_results:
            continue
        print(f"\n{'=' * 20} {lang} {'=' * 20}")
        rng = random.Random(RANDOM_SEED)
        dev_pairs = dev_loader(rng)
        test_pairs = test_loader()
        n = len(test_pairs)
        print(f"  dev n={len(dev_pairs)}  test n={n}")

        backbone = SpecialistBackbone(BACKBONES[lang])
        lang_results = {}

        for head_kind in ["ours", "linear"]:
            if head_kind == "ours":
                core = CognitiveEmbeddingCore(backbone.hidden_dim)
                core.load_state_dict(torch.load(RESULTS_DIR / f"core_model_specialist_{lang}.pt"))
            else:
                core = LinearProjectionHead(backbone.hidden_dim)
                core.load_state_dict(torch.load(RESULTS_DIR / f"linear_head_{lang}.pt"))
            core.eval()

            d1, d2 = embed_all(backbone, core, [p["s1"] for p in dev_pairs]), embed_all(backbone, core, [p["s2"] for p in dev_pairs])
            t1, t2 = embed_all(backbone, core, [p["s1"] for p in test_pairs]), embed_all(backbone, core, [p["s2"] for p in test_pairs])

            raw_sims = cosine_sim_np(t1, t2)
            fit_dev = np.concatenate([d1, d2], axis=0)
            mu, w = fit_whitening(fit_dev)
            devfit_sims = cosine_sim_np(apply_whitening(t1, mu, w), apply_whitening(t2, mu, w))

            which = selection[head_kind]
            sims = devfit_sims if which == "devfit" else raw_sims
            m = metric(kind, sims, test_pairs)
            pub = PUBLISHED_LEAKFREE[lang][head_kind]
            print(f"  {head_kind} [{which}]: reproduced={m:.4f} (published={pub:.4f})")

            rng_boot = np.random.RandomState(SEED + hash(lang + head_kind) % 1000)
            boots = []
            attempts = 0
            while len(boots) < N_BOOTSTRAP and attempts < N_BOOTSTRAP * 3:
                attempts += 1
                idx = rng_boot.randint(0, n, size=n)
                v = bootstrap_metric(kind, sims, test_pairs, idx)
                if v is not None:
                    boots.append(v)
            boots = np.array(boots)
            lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
            lang_results[head_kind] = {"which": which, "value": m, "ci_lo": lo, "ci_hi": hi, "boot_mean": float(boots.mean())}
            print(f"    95% CI: [{lo:+.4f}, {hi:+.4f}]")

        margin_boots_lo, margin_boots_hi = None, None
        # Paired margin CI: resample same indices for both heads' chosen sims jointly
        core_ours = CognitiveEmbeddingCore(backbone.hidden_dim)
        core_ours.load_state_dict(torch.load(RESULTS_DIR / f"core_model_specialist_{lang}.pt"))
        core_ours.eval()
        core_lin = LinearProjectionHead(backbone.hidden_dim)
        core_lin.load_state_dict(torch.load(RESULTS_DIR / f"linear_head_{lang}.pt"))
        core_lin.eval()

        d1o, d2o = embed_all(backbone, core_ours, [p["s1"] for p in dev_pairs]), embed_all(backbone, core_ours, [p["s2"] for p in dev_pairs])
        t1o, t2o = embed_all(backbone, core_ours, [p["s1"] for p in test_pairs]), embed_all(backbone, core_ours, [p["s2"] for p in test_pairs])
        d1l, d2l = embed_all(backbone, core_lin, [p["s1"] for p in dev_pairs]), embed_all(backbone, core_lin, [p["s2"] for p in dev_pairs])
        t1l, t2l = embed_all(backbone, core_lin, [p["s1"] for p in test_pairs]), embed_all(backbone, core_lin, [p["s2"] for p in test_pairs])

        raw_o, raw_l = cosine_sim_np(t1o, t2o), cosine_sim_np(t1l, t2l)
        mu_o, w_o = fit_whitening(np.concatenate([d1o, d2o], axis=0))
        mu_l, w_l = fit_whitening(np.concatenate([d1l, d2l], axis=0))
        devfit_o = cosine_sim_np(apply_whitening(t1o, mu_o, w_o), apply_whitening(t2o, mu_o, w_o))
        devfit_l = cosine_sim_np(apply_whitening(t1l, mu_l, w_l), apply_whitening(t2l, mu_l, w_l))

        sims_o = devfit_o if selection["ours"] == "devfit" else raw_o
        sims_l = devfit_l if selection["linear"] == "devfit" else raw_l

        rng_boot = np.random.RandomState(SEED + hash(lang + "margin") % 1000)
        margin_boots = []
        attempts = 0
        while len(margin_boots) < N_BOOTSTRAP and attempts < N_BOOTSTRAP * 3:
            attempts += 1
            idx = rng_boot.randint(0, n, size=n)
            vo = bootstrap_metric(kind, sims_o, test_pairs, idx)
            vl = bootstrap_metric(kind, sims_l, test_pairs, idx)
            if vo is not None and vl is not None:
                margin_boots.append(vo - vl)
        margin_boots = np.array(margin_boots)
        m_lo, m_hi = float(np.percentile(margin_boots, 2.5)), float(np.percentile(margin_boots, 97.5))
        significant = bool(m_lo > 0 or m_hi < 0)
        print(f"  margin (ours-linear) 95% CI: [{m_lo:+.4f}, {m_hi:+.4f}]  {'SIGNIFICANT' if significant else 'not significant'}")

        lang_results["margin_ci_lo"] = m_lo
        lang_results["margin_ci_hi"] = m_hi
        lang_results["margin_significant"] = significant
        all_results[lang] = lang_results

        out_path = RESULTS_DIR / "tables" / "significance_leakfree_fair_baseline.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(all_results, indent=2))

    print("\n=== Summary (leak-free protocol, matches current Table 5) ===")
    for lang, r in all_results.items():
        print(f"  {lang}: ours CI=[{r['ours']['ci_lo']:+.4f},{r['ours']['ci_hi']:+.4f}]  "
              f"linear CI=[{r['linear']['ci_lo']:+.4f},{r['linear']['ci_hi']:+.4f}]  "
              f"margin CI=[{r['margin_ci_lo']:+.4f},{r['margin_ci_hi']:+.4f}]  "
              f"{'SIG' if r['margin_significant'] else 'ns'}")


if __name__ == "__main__":
    main()
