"""Paired bootstrap significance test for Table 7 (tab:classification):
MASSIVE intent-classification accuracy, monolingual and zero-shot transfer
from English, structured head vs. linear head. Added in response to a
pre-submission review asking for CIs on Table 7.

No retraining of the embedding models: both cross-lingual checkpoints
already exist (core_model_crosslingual.pt, core_model_linear_crosslingual.pt).
This re-embeds MASSIVE train/test for en/bn/ar/hi with both heads (the
expensive part, same cost as the original evaluate_classification.py run,
minus the LaBSE third of that cost since this comparison only needs
ours-vs-linear), fits the same LogisticRegression classifiers used in the
original evaluation (monolingual: native classifier per language;
zero-shot: English-trained classifier applied to bn/ar/hi), and reproduces
the published accuracies as a sanity check.

For significance, the classifier fit is held fixed (exactly as reported)
and the TEST SET is bootstrap-resampled 10,000 times: accuracy is
recomputed for both heads on the same resampled indices each time, and the
distribution of (ours_acc - linear_acc) across resamples gives a 95% CI on
each cell's margin. This is standard practice for testing whether an
already-fit classifier's accuracy difference between two embedding spaces
is distinguishable from noise, and is far cheaper than refitting per
resample since LogisticRegression only depends on the (unresampled)
training embeddings.

`datasets` must be imported before `torch` (see data/loaders.py docstring).
"""

from __future__ import annotations

import gzip
import json
import random
import sys
import time
from pathlib import Path

import datasets  # noqa: F401

import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cogembed.models.backbone import BackboneConfig, mean_pool
from cogembed.models.cognitive_embedding import CognitiveEmbeddingModel

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
LANGUAGES = ["en", "bn", "ar", "hi", "te"]
TRAIN_CAP = 3000
N_BOOTSTRAP = 10000
SEED = 42

PUBLISHED = {
    # lang: (ours_mono, linear_mono, ours_zs, linear_zs)   zs=None for en (trivial/omitted)
    # Values are shuffled-train-cap (post-bug-fix) numbers from classification_results.json,
    # used only for a console sanity-check print, not for correctness.
    "en": (0.6866, 0.7518, None, None),
    "bn": (0.5965, 0.6762, 0.2535, 0.3699),
    "ar": (0.5612, 0.6200, 0.2962, 0.3554),
    "hi": (0.6443, 0.6944, 0.3433, 0.4472),
    "te": (0.6106, 0.6654, 0.2787, 0.3396),
}


class LinearProjectionHead(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, token_features, mask):
        return self.proj(mean_pool(token_features, mask))


def load_massive(lang: str, split: str) -> list[dict]:
    path = hf_hub_download(repo_id="mteb/amazon_massive_intent", repo_type="dataset",
                            filename=f"{split}/{lang}.json.gz")
    with gzip.open(path) as f:
        return [json.loads(l) for l in f]


def embed_all(embed_fn, sentences: list[str]) -> np.ndarray:
    with torch.no_grad():
        return np.stack([embed_fn(s) for s in sentences])


def bootstrap_acc_diff(correct_a: np.ndarray, correct_b: np.ndarray, n_boot: int, seed: int):
    n = len(correct_a)
    rng = np.random.RandomState(seed)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.randint(0, n, size=n)
        diffs[b] = correct_a[idx].mean() - correct_b[idx].mean()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return diffs, float(lo), float(hi)


def bootstrap_macrof1_diff(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray,
                            labels: list, n_boot: int, seed: int):
    """Macro-F1 does not decompose into a per-sample average like accuracy, so each
    resample recomputes f1_score directly. `labels` is fixed to the full test set's
    class set (not re-derived per resample) so every resample's macro average is over
    the same denominator, even if a rare class's few examples are absent from a given
    resample -- zero_division=0 scores that class 0 for that resample rather than
    silently dropping it from the average."""
    n = len(y_true)
    rng = np.random.RandomState(seed)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.randint(0, n, size=n)
        f1_a = f1_score(y_true[idx], pred_a[idx], average="macro", labels=labels, zero_division=0)
        f1_b = f1_score(y_true[idx], pred_b[idx], average="macro", labels=labels, zero_division=0)
        diffs[b] = f1_a - f1_b
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return diffs, float(lo), float(hi)


def main():
    t_start = time.time()
    print("Loading MASSIVE intent classification (en/bn/ar/hi)...")
    data = {}
    for lang in LANGUAGES:
        train_rows = load_massive(lang, "train")
        random.Random(SEED).shuffle(train_rows)  # see evaluate_classification.py: the file's
        # first TRAIN_CAP rows in on-disk order omit 21 of 60 classes entirely; shuffling first
        # drops that to 1 too-rare-to-guarantee class.
        train_rows = train_rows[:TRAIN_CAP]
        test_rows = load_massive(lang, "test")
        data[lang] = {"train": train_rows, "test": test_rows}
        print(f"  {lang}: train={len(train_rows)} test={len(test_rows)}")

    print("\nLoading trained cross-lingual checkpoints (inference only, no training)...")
    ours_model = CognitiveEmbeddingModel(BackboneConfig())
    ours_model.core.load_state_dict(torch.load(RESULTS_DIR / "core_model_crosslingual.pt"))

    def ours_embed(s):
        return ours_model.embed(s).numpy()

    linear_model = CognitiveEmbeddingModel(BackboneConfig())
    linear_model.core = LinearProjectionHead(linear_model.backbone.hidden_dim)
    linear_model.core.load_state_dict(torch.load(RESULTS_DIR / "core_model_linear_crosslingual.pt"))

    def linear_embed(s):
        return linear_model.embed(s).numpy()

    embeddings = {}
    for name, embed_fn in [("ours", ours_embed), ("linear", linear_embed)]:
        print(f"\n{'=' * 10} embedding with {name} {'=' * 10}")
        embeddings[name] = {}
        for lang in LANGUAGES:
            print(f"  {lang}...")
            tr, te = data[lang]["train"], data[lang]["test"]
            embeddings[name][lang] = {
                "train_X": embed_all(embed_fn, [r["text"] for r in tr]),
                "train_y": np.array([r["label"] for r in tr]),
                "test_X": embed_all(embed_fn, [r["text"] for r in te]),
                "test_y": np.array([r["label"] for r in te]),
            }

    out_path = RESULTS_DIR / "tables" / "significance_classification.json"
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    if results:
        print(f"Resuming: {list(results.keys())} already done, skipping.")

    for lang in LANGUAGES:
        if lang in results:
            continue
        print(f"\n{'=' * 20} {lang}: significance {'=' * 20}")
        results[lang] = {}

        # Monolingual: native classifier per language, per head
        correct, preds = {}, {}
        test_y = embeddings["ours"][lang]["test_y"]
        labels = sorted(set(test_y.tolist()))
        for name in ["ours", "linear"]:
            e = embeddings[name][lang]
            clf = LogisticRegression(max_iter=1000)
            clf.fit(e["train_X"], e["train_y"])
            pred = clf.predict(e["test_X"])
            preds[name] = pred
            correct[name] = (pred == e["test_y"]).astype(float)
            acc = correct[name].mean()
            macro_f1 = f1_score(e["test_y"], pred, average="macro", labels=labels, zero_division=0)
            pub = PUBLISHED[lang][0] if name == "ours" else PUBLISHED[lang][1]
            print(f"  monolingual {name}: acc={acc:.4f} macro_f1={macro_f1:.4f} (paper acc: {pub:.4f})")

        diffs, lo, hi = bootstrap_acc_diff(correct["ours"], correct["linear"], N_BOOTSTRAP, SEED)
        sig = lo > 0 or hi < 0
        print(f"  monolingual acc margin: {diffs.mean():+.4f}  95% CI=[{lo:+.4f}, {hi:+.4f}]  "
              f"{'SIGNIFICANT' if sig else 'not significant'}")

        f1_diffs, f1_lo, f1_hi = bootstrap_macrof1_diff(test_y, preds["ours"], preds["linear"], labels, N_BOOTSTRAP, SEED)
        f1_sig = f1_lo > 0 or f1_hi < 0
        print(f"  monolingual macro-F1 margin: {f1_diffs.mean():+.4f}  95% CI=[{f1_lo:+.4f}, {f1_hi:+.4f}]  "
              f"{'SIGNIFICANT' if f1_sig else 'not significant'}")

        results[lang]["monolingual"] = {
            "ours_acc": float(correct["ours"].mean()), "linear_acc": float(correct["linear"].mean()),
            "margin": float(diffs.mean()), "ci_lo": lo, "ci_hi": hi, "significant_at_0.05": sig,
            "ours_macro_f1": float(f1_score(test_y, preds["ours"], average="macro", labels=labels, zero_division=0)),
            "linear_macro_f1": float(f1_score(test_y, preds["linear"], average="macro", labels=labels, zero_division=0)),
            "f1_margin": float(f1_diffs.mean()), "f1_ci_lo": f1_lo, "f1_ci_hi": f1_hi, "f1_significant_at_0.05": f1_sig,
        }

        if lang == "en":
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(results, indent=2))
            continue  # zero-shot from English onto English is trivial, omitted from Table 7 too

        # Zero-shot: English-trained classifier applied to this language's test set
        zs_correct, zs_preds = {}, {}
        zs_test_y = embeddings["ours"][lang]["test_y"]
        zs_labels = sorted(set(zs_test_y.tolist()))
        for name in ["ours", "linear"]:
            en_e = embeddings[name]["en"]
            clf = LogisticRegression(max_iter=1000)
            clf.fit(en_e["train_X"], en_e["train_y"])
            e = embeddings[name][lang]
            pred = clf.predict(e["test_X"])
            zs_preds[name] = pred
            zs_correct[name] = (pred == e["test_y"]).astype(float)
            acc = zs_correct[name].mean()
            macro_f1 = f1_score(e["test_y"], pred, average="macro", labels=zs_labels, zero_division=0)
            pub = PUBLISHED[lang][2] if name == "ours" else PUBLISHED[lang][3]
            print(f"  zero-shot {name}: acc={acc:.4f} macro_f1={macro_f1:.4f} (paper acc: {pub:.4f})")

        zdiffs, zlo, zhi = bootstrap_acc_diff(zs_correct["ours"], zs_correct["linear"], N_BOOTSTRAP, SEED + 1)
        zsig = zlo > 0 or zhi < 0
        print(f"  zero-shot acc margin: {zdiffs.mean():+.4f}  95% CI=[{zlo:+.4f}, {zhi:+.4f}]  "
              f"{'SIGNIFICANT' if zsig else 'not significant'}")

        zf1_diffs, zf1_lo, zf1_hi = bootstrap_macrof1_diff(zs_test_y, zs_preds["ours"], zs_preds["linear"], zs_labels, N_BOOTSTRAP, SEED + 1)
        zf1_sig = zf1_lo > 0 or zf1_hi < 0
        print(f"  zero-shot macro-F1 margin: {zf1_diffs.mean():+.4f}  95% CI=[{zf1_lo:+.4f}, {zf1_hi:+.4f}]  "
              f"{'SIGNIFICANT' if zf1_sig else 'not significant'}")

        results[lang]["zero_shot"] = {
            "ours_acc": float(zs_correct["ours"].mean()), "linear_acc": float(zs_correct["linear"].mean()),
            "margin": float(zdiffs.mean()), "ci_lo": zlo, "ci_hi": zhi, "significant_at_0.05": zsig,
            "ours_macro_f1": float(f1_score(zs_test_y, zs_preds["ours"], average="macro", labels=zs_labels, zero_division=0)),
            "linear_macro_f1": float(f1_score(zs_test_y, zs_preds["linear"], average="macro", labels=zs_labels, zero_division=0)),
            "f1_margin": float(zf1_diffs.mean()), "f1_ci_lo": zf1_lo, "f1_ci_hi": zf1_hi, "f1_significant_at_0.05": zf1_sig,
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2))

    print(f"\nResults written to {out_path}")
    print(f"Total time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
