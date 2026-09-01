"""Follow-up to verify_devbased_selection.py. That check found that naive dev-based
selection (fit whitening on the full 500-example dev split, then compare raw vs
whitened evaluated on THOSE SAME 500 points) disagrees with the currently-published
test-based selection for Bangla: dev-based picks raw, test-based picks dev-fit-
whitened. But evaluating whitening in-sample on the exact points it was fit on has
its own circularity (the transform's mean/decorrelation matrix is a direct function
of those points' embeddings), so that naive check doesn't cleanly settle the question.

This script does the proper 3-way split for Bangla specifically (the only language
where the two rules disagree; n=500 dev is large enough to split, unlike Telugu/
Hindi/Arabic's much smaller dev splits): fit whitening on one random half of BnPC's
dev split (250 examples), decide raw-vs-whitened using the OTHER half (250 examples,
never touched by the whitening fit), then apply that fixed choice's fitted transform
to the test set and report the test score. No test-set peeking, no in-sample
whitening-fit-vs-eval circularity.

No retraining: reuses existing checkpoints, inference only.

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
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cogembed.data.loaders import RANDOM_SEED, load_bnpc_pairs
from cogembed.models.backbone import apply_whitening, fit_whitening
from cogembed.models.cognitive_embedding import CognitiveEmbeddingCore

from train_specialist_backbones import BACKBONES, SpecialistBackbone
from baseline_linear_head import LinearProjectionHead

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


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


def auroc(sims, pairs):
    labels = np.array([p["label"] for p in pairs])
    return float(roc_auc_score(labels, sims))


def main():
    rng = random.Random(RANDOM_SEED)
    dev_all = load_bnpc_pairs("validation")
    if len(dev_all) > 500:
        dev_all = rng.sample(dev_all, 500)
    shuffled = dev_all[:]
    random.Random(RANDOM_SEED).shuffle(shuffled)
    half = len(shuffled) // 2
    dev_fit_half, dev_select_half = shuffled[:half], shuffled[half:]
    test_pairs = load_bnpc_pairs("test")
    print(f"Bangla dev split: {len(dev_fit_half)} for whitening fit, {len(dev_select_half)} for selection, "
          f"{len(test_pairs)} test (untouched by either).")

    backbone = SpecialistBackbone(BACKBONES["bangla"])
    results = {}

    for head_kind in ["ours", "linear"]:
        if head_kind == "ours":
            core = CognitiveEmbeddingCore(backbone.hidden_dim)
            core.load_state_dict(torch.load(RESULTS_DIR / "core_model_specialist_bangla.pt"))
        else:
            core = LinearProjectionHead(backbone.hidden_dim)
            core.load_state_dict(torch.load(RESULTS_DIR / "linear_head_bangla.pt"))
        core.eval()

        f1, f2 = embed_all(backbone, core, [p["s1"] for p in dev_fit_half]), embed_all(backbone, core, [p["s2"] for p in dev_fit_half])
        s1, s2 = embed_all(backbone, core, [p["s1"] for p in dev_select_half]), embed_all(backbone, core, [p["s2"] for p in dev_select_half])
        t1, t2 = embed_all(backbone, core, [p["s1"] for p in test_pairs]), embed_all(backbone, core, [p["s2"] for p in test_pairs])

        mu, w = fit_whitening(np.concatenate([f1, f2], axis=0))

        select_raw = auroc(cosine_sim_np(s1, s2), dev_select_half)
        select_white = auroc(cosine_sim_np(apply_whitening(s1, mu, w), apply_whitening(s2, mu, w)), dev_select_half)
        test_raw = auroc(cosine_sim_np(t1, t2), test_pairs)
        test_white = auroc(cosine_sim_np(apply_whitening(t1, mu, w), apply_whitening(t2, mu, w)), test_pairs)

        choice = "devfit" if select_white > select_raw else "raw"
        reported_score = test_white if choice == "devfit" else test_raw

        print(f"\n{head_kind}:")
        print(f"  selection-half: raw={select_raw:.4f} whitened={select_white:.4f} -> choice={choice}")
        print(f"  test: raw={test_raw:.4f} whitened={test_white:.4f} -> reported (split-dev protocol)={reported_score:.4f}")

        results[head_kind] = {
            "select_raw": select_raw, "select_whitened": select_white, "choice": choice,
            "test_raw": test_raw, "test_whitened": test_white, "reported_score": reported_score,
        }

    out_path = RESULTS_DIR / "tables" / "bangla_split_dev_selection_check.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults written to {out_path}")
    print("DONE_BANGLA_SPLIT_DEV_CHECK")


if __name__ == "__main__":
    main()
