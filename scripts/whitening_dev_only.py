"""Re-scores the fair-baseline comparison (Table 5) with whitening fit on a
held-out DEV subset instead of the test set's own embeddings, closing the
leakage gap the paper already discloses (Section 3.1: "This is fit on the
evaluation set's own embeddings... not a truly held-out fitting pool").
Requested directly by a pre-submission review.

No retraining: reuses the existing trained checkpoints for both heads,
all five languages (inference only). For each language, the dev split is
genuinely unseen by that language's task-training stage for Hindi and
Arabic (which skip task-specific fine-tuning entirely, XNLI-pretrain-only)
and was used only for model SELECTION, not whitening, for English/Bangla/
Telugu -- so in every case this dev-fit whitening is a stricter, leakage-
free alternative to the original protocol.

Dev splits: English (STS-B validation, 500), Bangla (BnPC validation,
<=500), Telugu (SemRel dev), Hindi (SemRel dev, 288 -- unused during
training), Arabic (SemRel dev, 32 -- unused during training, small but
sufficient to fit a 768-dim mean/covariance).

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

LANGUAGES = {
    "english": ("sts", lambda rng: rng.sample(load_stsb("validation"), 500), lambda: load_stsb("test")),
    "bangla": ("auroc", lambda rng: (lambda d: rng.sample(d, 500) if len(d) > 500 else d)(load_bnpc_pairs("validation")), lambda: load_bnpc_pairs("test")),
    "telugu": ("sts", lambda rng: load_semrel_telugu("dev"), lambda: load_semrel_telugu("test")),
    "hindi": ("sts", lambda rng: load_semrel_hindi("dev"), lambda: load_semrel_hindi("test")),
    "arabic": ("sts", lambda rng: load_semrel_arabic("dev"), lambda: load_semrel_arabic("test")),
}

PUBLISHED = {
    "english": {"ours": (0.7383233475152673, "raw"), "linear": (0.7163951507422928, "raw")},
    "bangla": {"ours": (0.8452311286928136, "raw"), "linear": (0.864655174248244, "raw")},
    "telugu": {"ours": (0.7767991540693114, "raw"), "linear": (0.7912023658050895, "raw")},
    "hindi": {"ours": (0.6980004661954695, "whitened"), "linear": (0.7060653171855318, "raw")},
    "arabic": {"ours": (0.4217035514919273, "whitened"), "linear": (0.4638465810583845, "whitened")},
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


def main():
    all_results = {}

    for lang, (kind, dev_loader, test_loader) in LANGUAGES.items():
        print(f"\n{'=' * 20} {lang} {'=' * 20}")
        rng = random.Random(RANDOM_SEED)
        dev_pairs = dev_loader(rng)
        test_pairs = test_loader()
        print(f"  dev n={len(dev_pairs)}  test n={len(test_pairs)}")

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

            d1 = embed_all(backbone, core, [p["s1"] for p in dev_pairs])
            d2 = embed_all(backbone, core, [p["s2"] for p in dev_pairs])
            t1 = embed_all(backbone, core, [p["s1"] for p in test_pairs])
            t2 = embed_all(backbone, core, [p["s2"] for p in test_pairs])

            raw_score = metric(kind, cosine_sim_np(t1, t2), test_pairs)

            # Original (leakage) protocol: whitening fit on the test set's own embeddings
            fit_test = np.concatenate([t1, t2], axis=0)
            mu_test, w_test = fit_whitening(fit_test)
            leak_white_score = metric(kind, cosine_sim_np(apply_whitening(t1, mu_test, w_test), apply_whitening(t2, mu_test, w_test)), test_pairs)

            # Held-out protocol: whitening fit on the DEV set, applied to test
            fit_dev = np.concatenate([d1, d2], axis=0)
            mu_dev, w_dev = fit_whitening(fit_dev)
            devfit_white_score = metric(kind, cosine_sim_np(apply_whitening(t1, mu_dev, w_dev), apply_whitening(t2, mu_dev, w_dev)), test_pairs)

            pub_val, pub_which = PUBLISHED[lang][head_kind]
            print(f"  {head_kind}: raw={raw_score:.4f}  test-fit-whitened(leakage)={leak_white_score:.4f}  "
                  f"dev-fit-whitened(held-out)={devfit_white_score:.4f}  (published best={pub_val:.4f} [{pub_which}])")

            lang_results[head_kind] = {
                "raw": raw_score, "test_fit_whitened_leakage": leak_white_score,
                "dev_fit_whitened_held_out": devfit_white_score,
                "published_best": pub_val, "published_which": pub_which,
            }

        all_results[lang] = lang_results
        out_path = RESULTS_DIR / "tables" / "whitening_dev_only_results.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(all_results, indent=2))

    print("\n=== Summary: does held-out (dev-fit) whitening change which head wins? ===")
    for lang, r in all_results.items():
        o_best = max(r["ours"]["raw"], r["ours"]["dev_fit_whitened_held_out"])
        l_best = max(r["linear"]["raw"], r["linear"]["dev_fit_whitened_held_out"])
        print(f"  {lang}: ours(dev-fit best)={o_best:.4f}  linear(dev-fit best)={l_best:.4f}  "
              f"winner={'ours' if o_best > l_best else 'linear'}  (published winner={'ours' if PUBLISHED[lang]['ours'][0] > PUBLISHED[lang]['linear'][0] else 'linear'})")

    print(f"\nResults written to {RESULTS_DIR / 'tables' / 'whitening_dev_only_results.json'}")


if __name__ == "__main__":
    main()
