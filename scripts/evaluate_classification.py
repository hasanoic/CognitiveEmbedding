"""Classification probing + cross-lingual zero-shot transfer -- a genuinely
missing task type until now (everything else this project has tested is
similarity/retrieval-shaped). Uses MASSIVE intent classification (via the
ungated mteb/amazon_massive_intent mirror), 59-way intent labels, same
taxonomy across English/Bangla/Arabic/Hindi -- lets us run the standard
protocol directly: fit a classifier on FROZEN embeddings of English training
data, then (a) test in-language, and (b) apply that SAME classifier
zero-shot to Bangla/Arabic/Hindi test embeddings with NO retraining. High
zero-shot accuracy is evidence the shared embedding space genuinely aligns
same-intent sentences across languages -- a different, complementary signal
to the bitext retrieval test (this tests downstream task transfer, not
direct sentence-pair alignment).

MUST use the shared cross-lingual checkpoint (core_model_crosslingual.pt,
the one trained with the FLORES alignment fix) -- the specialist per-
language backbones live in disjoint embedding spaces by construction, so a
classifier trained on one could not meaningfully transfer to another.

Also fits NATIVE per-language classifiers (train BN -> test BN, etc.) for
the monolingual comparison, and runs LaBSE through the same protocol as a
reference baseline.

Also runs the fair linear-head baseline through this SAME protocol -- a gap
a reviewer identified: the linear-head comparison (Table tab:baseline-head)
was previously only run on the similarity/paraphrase task, never on this
downstream classification-transfer task. Uses core_model_linear_crosslingual.pt
(see train_linear_shared.py / train_linear_crosslingual.py), trained through
the identical two-stage pipeline as core_model_crosslingual.pt with only the
head architecture swapped, so this is the same fair-comparison logic as
Table tab:baseline-head, extended to this task.

Reports macro-F1 alongside accuracy: MASSIVE's 59 intent classes are
severely imbalanced (test set per language: 1 to 209 examples per class,
mean 50), so accuracy alone is dominated by the large classes. Fixed a
correctness bug found while adding this: MASSIVE's train file is not
pre-shuffled, so capping to the first TRAIN_CAP rows in file order left 21
of 60 classes with ZERO training examples -- the classifier could never
predict them, regardless of embedding quality. Now shuffled with a fixed
seed before capping, which drops that to 1 class too rare (across the
full 11.5K-row train split) for any TRAIN_CAP-sized sample to guarantee
covering. This changes the previously-published accuracy numbers too, not
just adds a metric -- both scripts that touch this data
(evaluate_classification.py, significance_classification.py) were fixed
together, and Table tab:classification's numbers need to be regenerated
from this corrected run, not amended in place.

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
from huggingface_hub import hf_hub_download
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cogembed.models.backbone import BackboneConfig
from cogembed.models.cognitive_embedding import CognitiveEmbeddingModel

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
LANGUAGES = ["en", "bn", "ar", "hi", "te"]
TRAIN_CAP = 3000  # MASSIVE train splits are ~11.5K/language; cap for CPU feasibility
RANDOM_SEED = 42
_SMOKE_TEST_CAP = None  # None for the real run -- caps train/test size for a fast smoke test


def load_massive(lang: str, split: str) -> list[dict]:
    path = hf_hub_download(repo_id="mteb/amazon_massive_intent", repo_type="dataset",
                            filename=f"{split}/{lang}.json.gz")
    with gzip.open(path) as f:
        return [json.loads(l) for l in f]


def embed_all(embed_fn, sentences: list[str]) -> np.ndarray:
    with torch.no_grad():
        return np.stack([embed_fn(s) for s in sentences])


def main():
    t_start = time.time()
    print("Loading MASSIVE intent classification (en/bn/ar/hi)...")
    data = {}
    for lang in LANGUAGES:
        train_rows = load_massive(lang, "train")
        random.Random(RANDOM_SEED).shuffle(train_rows)  # MASSIVE's train file is not pre-shuffled:
        # taking the first TRAIN_CAP rows in file order left 21 of 60 intent classes entirely
        # absent from training data (classifier could never predict them). Shuffling first drops
        # that to 1 missing class (too rare -- fewer than TRAIN_CAP/full_train_size copies exist
        # in the full split -- for any fixed-size sample to guarantee covering it).
        train_rows = train_rows[:TRAIN_CAP]
        test_rows = load_massive(lang, "test")
        if _SMOKE_TEST_CAP is not None:
            train_rows, test_rows = train_rows[:_SMOKE_TEST_CAP], test_rows[:_SMOKE_TEST_CAP]
        data[lang] = {"train": train_rows, "test": test_rows}
        print(f"  {lang}: train={len(train_rows)} test={len(test_rows)}")

    print("\nLoading our trained cross-lingual core model...")
    model = CognitiveEmbeddingModel(BackboneConfig())
    model.core.load_state_dict(torch.load(RESULTS_DIR / "core_model_crosslingual.pt"))

    def ours_embed(s):
        return model.embed(s).numpy()

    print("Loading the fair linear-head cross-lingual model...")
    import torch.nn as nn
    from cogembed.models.backbone import mean_pool as _mean_pool

    class LinearProjectionHead(nn.Module):
        def __init__(self, hidden_dim: int):
            super().__init__()
            self.proj = nn.Linear(hidden_dim, hidden_dim)

        def forward(self, token_features, mask):
            return self.proj(_mean_pool(token_features, mask))

    linear_model = CognitiveEmbeddingModel(BackboneConfig())
    linear_model.core = LinearProjectionHead(linear_model.backbone.hidden_dim)
    linear_model.core.load_state_dict(torch.load(RESULTS_DIR / "core_model_linear_crosslingual.pt"))

    def linear_embed(s):
        return linear_model.embed(s).numpy()

    from transformers import AutoModel, AutoTokenizer
    print("Loading LaBSE (reference baseline)...")
    labse_tok = AutoTokenizer.from_pretrained("sentence-transformers/LaBSE")
    labse_mdl = AutoModel.from_pretrained("sentence-transformers/LaBSE")
    labse_mdl.eval()

    @torch.no_grad()
    def labse_embed(s):
        enc = labse_tok(s, return_tensors="pt", truncation=True, max_length=64)
        out = labse_mdl(**enc)
        mf = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (out.last_hidden_state * mf).sum(1) / mf.sum(1).clamp(min=1e-9)
        return torch.nn.functional.normalize(pooled, dim=-1)[0].numpy()

    def method_done(name):
        # Not just "name in results": a prior run with fewer LANGUAGES (e.g. before Telugu was
        # added) can leave a method "complete" for its old language set but missing new ones.
        return name in results and all(lang in results[name]["monolingual"] for lang in LANGUAGES)

    out_path = RESULTS_DIR / "tables" / "classification_results.json"
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    if results:
        done_now = [n for n in ("ours", "linear", "labse") if method_done(n)]
        print(f"Resuming: {done_now} fully done for all {len(LANGUAGES)} languages, skipping.")

    for name, embed_fn in [("ours", ours_embed), ("linear", linear_embed), ("labse", labse_embed)]:
        if method_done(name):
            continue
        print(f"\n{'=' * 15} {name} {'=' * 15}")
        embeddings = {}
        for lang in LANGUAGES:
            print(f"  embedding {lang}...")
            tr, te = data[lang]["train"], data[lang]["test"]
            embeddings[lang] = {
                "train_X": embed_all(embed_fn, [r["text"] for r in tr]),
                "train_y": [r["label"] for r in tr],
                "test_X": embed_all(embed_fn, [r["text"] for r in te]),
                "test_y": [r["label"] for r in te],
            }

        # Monolingual: native classifier per language
        monolingual = {}
        for lang in LANGUAGES:
            clf = LogisticRegression(max_iter=1000)
            clf.fit(embeddings[lang]["train_X"], embeddings[lang]["train_y"])
            pred = clf.predict(embeddings[lang]["test_X"])
            acc = accuracy_score(embeddings[lang]["test_y"], pred)
            macro_f1 = f1_score(embeddings[lang]["test_y"], pred, average="macro", zero_division=0)
            monolingual[lang] = {"acc": float(acc), "macro_f1": float(macro_f1)}
            print(f"    monolingual {lang}: acc={acc:.4f} macro_f1={macro_f1:.4f}")

        # Cross-lingual zero-shot: classifier trained on English ONLY, applied to others
        en_clf = LogisticRegression(max_iter=1000)
        en_clf.fit(embeddings["en"]["train_X"], embeddings["en"]["train_y"])
        zero_shot = {}
        for lang in LANGUAGES:
            pred = en_clf.predict(embeddings[lang]["test_X"])
            acc = accuracy_score(embeddings[lang]["test_y"], pred)
            macro_f1 = f1_score(embeddings[lang]["test_y"], pred, average="macro", zero_division=0)
            zero_shot[lang] = {"acc": float(acc), "macro_f1": float(macro_f1)}
            print(f"    zero-shot (EN classifier) -> {lang}: acc={acc:.4f} macro_f1={macro_f1:.4f}")

        results[name] = {"monolingual": monolingual, "zero_shot_from_english": zero_shot}
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2))
        print(f"  (partial results written to {out_path})")

    print("\n=== Summary (acc / macro-F1) ===")
    print(f"{'lang':6}{'ours_mono':18}{'ours_zs':18}{'linear_mono':18}{'linear_zs':18}{'labse_mono':18}{'labse_zs':18}")
    for lang in LANGUAGES:
        def fmt(d):
            return f"{d['acc']:.4f}/{d['macro_f1']:.4f}"
        row = f"{lang:6}{fmt(results['ours']['monolingual'][lang]):<18}"
        row += f"{fmt(results['ours']['zero_shot_from_english'][lang]):<18}"
        row += f"{fmt(results['linear']['monolingual'][lang]):<18}"
        row += f"{fmt(results['linear']['zero_shot_from_english'][lang]):<18}"
        row += f"{fmt(results['labse']['monolingual'][lang]):<18}"
        row += f"{fmt(results['labse']['zero_shot_from_english'][lang]):<18}"
        print(row)

    print(f"\nResults written to {out_path}")
    print(f"Total time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
