"""Checks a real methodological concern flagged in pre-submission review: Table 5's
raw-vs-whitened selection currently picks whichever scores higher ON THE TEST SET
(significance_leakfree_fair_baseline.py / whitening_dev_only.py's own published
numbers), which is itself a mild form of test-set peeking -- comparing two candidate
scoring methods on test and reporting the winner can optimistically bias the reported
number, even though the whitening TRANSFORM itself is fit only on dev (already fixed
earlier this project).

This script checks whether switching to a strictly dev-based selection rule (fit
whitening on dev, evaluate both raw and whitened on that SAME dev split, pick
whichever wins there, then apply that fixed choice to test and report the test score)
changes any of Table 5's currently-published selections. This mirrors exactly how
every training loop in this project already selects its best epoch (best_val on a
held-out validation split, never test) -- the same principle, applied to whitening
selection instead of epoch selection.

No retraining: reuses existing checkpoints, inference only, all 5 languages, both
heads. Reports both the current (test-based) and the dev-based selection side by
side, plus what the reported test score would be under each rule, so a genuine
mismatch (if any) is visible directly rather than assumed away.

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

from train_specialist_backbones import BACKBONES, SpecialistBackbone
from baseline_linear_head import LinearProjectionHead

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

LANGUAGES = {
    "english": ("sts", lambda rng: rng.sample(load_stsb("validation"), 500), lambda: load_stsb("test")),
    "bangla": ("auroc", lambda rng: (lambda d: rng.sample(d, 500) if len(d) > 500 else d)(load_bnpc_pairs("validation")), lambda: load_bnpc_pairs("test")),
    "telugu": ("sts", lambda rng: load_semrel_telugu("dev"), lambda: load_semrel_telugu("test")),
    "hindi": ("sts", lambda rng: load_semrel_hindi("dev"), lambda: load_semrel_hindi("test")),
    "arabic": ("sts", lambda rng: load_semrel_arabic("dev"), lambda: load_semrel_arabic("test")),
}

# Current, currently-published selection (test-based), from significance_leakfree_fair_baseline.py
CURRENT_SELECTION = {
    "english": {"ours": "raw", "linear": "raw"},
    "bangla": {"ours": "devfit", "linear": "devfit"},
    "telugu": {"ours": "raw", "linear": "raw"},
    "hindi": {"ours": "raw", "linear": "raw"},
    "arabic": {"ours": "raw", "linear": "raw"},
}


def embed_all(backbone, core, sentences):
    with torch.no_grad():
        out = []
        for s in sentences:
            h, m = backbone.encode_tokens(s)
            out.append(core(h, m).numpy())
        return np.stack(out)


def cosine_sim_np(a, b):
    a_n = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)
    b_n = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-9)
    return (a_n * b_n).sum(axis=-1)


def metric(kind, sims, pairs):
    if kind == "sts":
        gold = np.array([p["score"] for p in pairs])
        rho, _ = spearmanr(sims, gold)
        return float(rho)
    labels = np.array([p["label"] for p in pairs])
    return float(roc_auc_score(labels, sims))


def main():
    results = {}
    for lang, (kind, dev_loader, test_loader) in LANGUAGES.items():
        print(f"\n{'=' * 20} {lang} {'=' * 20}")
        rng = random.Random(RANDOM_SEED)
        dev_pairs = dev_loader(rng)
        test_pairs = test_loader()
        backbone = SpecialistBackbone(BACKBONES[lang])
        results[lang] = {}

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

            mu, w = fit_whitening(np.concatenate([d1, d2], axis=0))

            dev_raw = metric(kind, cosine_sim_np(d1, d2), dev_pairs)
            dev_white = metric(kind, cosine_sim_np(apply_whitening(d1, mu, w), apply_whitening(d2, mu, w)), dev_pairs)
            test_raw = metric(kind, cosine_sim_np(t1, t2), test_pairs)
            test_white = metric(kind, cosine_sim_np(apply_whitening(t1, mu, w), apply_whitening(t2, mu, w)), test_pairs)

            devbased_choice = "devfit" if dev_white > dev_raw else "raw"
            devbased_test_score = test_white if devbased_choice == "devfit" else test_raw
            current_choice = CURRENT_SELECTION[lang][head_kind]
            current_test_score = test_white if current_choice == "devfit" else test_raw

            match = devbased_choice == current_choice
            print(f"  {head_kind}: dev raw={dev_raw:.4f} dev whitened={dev_white:.4f} -> dev-based choice={devbased_choice}")
            print(f"           test raw={test_raw:.4f} test whitened={test_white:.4f} -> current (test-based) choice={current_choice}")
            print(f"           dev-based test score={devbased_test_score:.4f}  current published score={current_test_score:.4f}  "
                  f"{'MATCH' if match else '*** MISMATCH ***'}")

            results[lang][head_kind] = {
                "dev_raw": dev_raw, "dev_whitened": dev_white, "devbased_choice": devbased_choice,
                "test_raw": test_raw, "test_whitened": test_white, "current_choice": current_choice,
                "devbased_test_score": devbased_test_score, "current_test_score": current_test_score,
                "selections_match": match,
            }

    out_path = RESULTS_DIR / "tables" / "devbased_selection_check.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))

    print("\n=== Summary: does dev-based selection agree with the currently-published test-based selection? ===")
    all_match = True
    for lang, heads in results.items():
        for head_kind, r in heads.items():
            status = "MATCH" if r["selections_match"] else "MISMATCH"
            if not r["selections_match"]:
                all_match = False
            print(f"  {lang} {head_kind}: {status}  (dev-based={r['devbased_choice']}, current={r['current_choice']}, "
                  f"score diff={r['devbased_test_score'] - r['current_test_score']:+.4f})")
    print(f"\nAll selections match: {all_match}")
    print(f"Results written to {out_path}")
    print("DONE_DEVBASED_SELECTION_CHECK")


if __name__ == "__main__":
    main()
