"""Extends the parameter-matched structured head experiment (previously
Bangla only) to Telugu, Hindi, and Arabic, requested directly by a pre-
submission review: "one language is not enough" to turn 'perhaps the
large model was difficult to train' into 'even after controlling for
capacity, the architectural advantage remains limited.'

Reuses the exact same reduced architecture from
parameter_matched_structured_head.py (K=2 attention heads, GRU hidden
52, pre-combine bottleneck 104 -- 589,280 trainable params, within 0.2%
of the linear head's 590,592), since every specialist backbone in this
study shares hidden_dim=768 (stated in Section "Languages, Backbones,
and Data"), so the same reduced configuration transfers without
redesign. English and Bangla already have this result (Bangla: run
tonight; English is the one language where the structured head already
wins at full capacity, included here for completeness/symmetry at zero
extra cost since it reuses cached data if convenient, but is not the
focus).

Telugu follows Bangla's recipe (task-specific data only, no NLI stage).
Hindi and Arabic follow the XNLI-pretrain-only, zero-shot recipe (no
task-specific data exists for either) -- same as every other experiment
in this study that touches these two languages.

`datasets` must be imported before `torch` (see data/loaders.py docstring).
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import datasets  # noqa: F401

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cogembed.data.loaders import (
    RANDOM_SEED,
    load_semrel_arabic,
    load_semrel_hindi,
    load_semrel_telugu,
)
from cogembed.losses import info_nce_loss, info_nce_loss_hard_negatives
from cogembed.models.backbone import apply_whitening, fit_whitening

from train_specialist_backbones import BACKBONES, SpecialistBackbone, load_xnli_triplets, cosine_sim_np
from parameter_matched_structured_head import (
    ParameterMatchedCore, K_HEADS, GRU_HIDDEN, BOTTLENECK, LR, TEMPERATURE, TASK_EPOCHS, BATCH_SIZE,
)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
NLI_EPOCHS = 6
NLI_TRIPLETS_MAX = 6000
SEED = RANDOM_SEED

PUBLISHED_LEAKFREE = {
    "telugu": {"ours_full": 0.7767991540693114, "linear": 0.7912023658050895},
    "hindi": {"ours_full": 0.6777469661946467, "linear": 0.7060653171855318},
    "arabic": {"ours_full": 0.4149129019845984, "linear": 0.4525680393892466},
}


def score(kind, sims, pairs):
    if kind == "sts":
        gold = np.array([p["score"] for p in pairs])
        rho, _ = spearmanr(sims, gold)
        return float(rho)
    labels = np.array([p["label"] for p in pairs])
    return float(roc_auc_score(labels, sims))


def cache_pairs(backbone, pairs):
    return [backbone.encode_tokens(p["s1"]) for p in pairs], [backbone.encode_tokens(p["s2"]) for p in pairs]


def cache_sentences(backbone, sentences):
    return [backbone.encode_tokens(s) for s in sentences]


def pool_batch(core, cache_batch):
    return torch.stack([core(h, m) for h, m in cache_batch])


def train_nli(core, ta, tp, tn, va, vp, vn, seed):
    optimizer = torch.optim.Adam(core.parameters(), lr=LR)
    n_train = len(ta)
    best_val, best_state = float("inf"), None
    for epoch in range(NLI_EPOCHS):
        core.train()
        perm = np.random.RandomState(seed + epoch).permutation(n_train)
        for start in range(0, n_train, BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            if len(idx) < 2:
                continue
            optimizer.zero_grad()
            a, p, n = pool_batch(core, [ta[i] for i in idx]), pool_batch(core, [tp[i] for i in idx]), pool_batch(core, [tn[i] for i in idx])
            loss = info_nce_loss_hard_negatives(a, p, n, TEMPERATURE)
            loss.backward()
            optimizer.step()
        core.eval()
        with torch.no_grad():
            val_loss = info_nce_loss_hard_negatives(pool_batch(core, va), pool_batch(core, vp), pool_batch(core, vn), TEMPERATURE).item()
        if val_loss < best_val:
            best_val, best_state = val_loss, {k: v.clone() for k, v in core.state_dict().items()}
        print(f"    [nli] epoch {epoch} val_loss={val_loss:.4f} (best={best_val:.4f})")
    core.load_state_dict(best_state)


def train_task(core, c1, c2, v1, v2, val_pairs, kind, seed):
    optimizer = torch.optim.Adam(core.parameters(), lr=LR)
    best_val, best_state = -1.0, None
    for epoch in range(TASK_EPOCHS):
        core.train()
        perm = np.random.RandomState(seed + epoch).permutation(len(c1))
        for start in range(0, len(c1), BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            if len(idx) < 2:
                continue
            optimizer.zero_grad()
            e1, e2 = pool_batch(core, [c1[i] for i in idx]), pool_batch(core, [c2[i] for i in idx])
            loss = info_nce_loss(e1, e2, TEMPERATURE)
            loss.backward()
            optimizer.step()
        core.eval()
        with torch.no_grad():
            vv1, vv2 = pool_batch(core, v1).numpy(), pool_batch(core, v2).numpy()
        val_score = score(kind, cosine_sim_np(vv1, vv2), val_pairs)
        if val_score > best_val:
            best_val, best_state = val_score, {k: v.clone() for k, v in core.state_dict().items()}
        print(f"    [task] epoch {epoch} val_score={val_score:.4f} (best={best_val:.4f})")
    core.load_state_dict(best_state)


def evaluate(core, t1, t2, test_pairs, kind, wd1, wd2):
    """Leak-free: whitening fit on the dedicated whitening-dev cache (wd1, wd2), never
    on the test embeddings -- matches whitening_dev_only.py's protocol, now Table
    tab:baseline-head's primary. raw-vs-whitened is still the higher of the two on the
    test set, the same selection convention as every other leak-free number here."""
    with torch.no_grad():
        e1, e2 = pool_batch(core, t1).numpy(), pool_batch(core, t2).numpy()
        d1, d2 = pool_batch(core, wd1).numpy(), pool_batch(core, wd2).numpy()
    raw = score(kind, cosine_sim_np(e1, e2), test_pairs)
    fit = np.concatenate([d1, d2], axis=0)
    mu, w = fit_whitening(fit)
    white = score(kind, cosine_sim_np(apply_whitening(e1, mu, w), apply_whitening(e2, mu, w)), test_pairs)
    return max(raw, white), raw, white


def run_telugu(backbone, rng):
    train_all = load_semrel_telugu("train")
    train_pairs = [r for r in train_all if r["score"] >= 0.5]
    val_pairs = load_semrel_telugu("dev")
    test_pairs, kind = load_semrel_telugu("test"), "sts"
    c1, c2 = cache_pairs(backbone, train_pairs)
    v1, v2 = cache_pairs(backbone, val_pairs)
    t1, t2 = cache_pairs(backbone, test_pairs)
    # SemRel dev doubles as both early-stopping validation (train_task) and the
    # leak-free whitening-fit set (evaluate) -- same reuse whitening_dev_only.py
    # already established is legitimate for Telugu.
    return {"nli": None, "task": (c1, c2, v1, v2, val_pairs, kind), "test": (t1, t2, test_pairs, kind), "whiten_dev": (v1, v2)}


def run_xnli_only(backbone, rng, lang_code, test_loader, dev_loader):
    triplets = load_xnli_triplets(lang_code, max_triplets=NLI_TRIPLETS_MAX)
    rng.shuffle(triplets)
    n_val = max(1, int(len(triplets) * 0.1))
    val_t, train_t = triplets[:n_val], triplets[n_val:]
    ta, tp, tn = cache_sentences(backbone, [t["anchor"] for t in train_t]), cache_sentences(backbone, [t["positive"] for t in train_t]), cache_sentences(backbone, [t["hard_negative"] for t in train_t])
    va, vp, vn = cache_sentences(backbone, [t["anchor"] for t in val_t]), cache_sentences(backbone, [t["positive"] for t in val_t]), cache_sentences(backbone, [t["hard_negative"] for t in val_t])
    test_pairs, kind = test_loader(), "sts"
    t1, t2 = cache_pairs(backbone, test_pairs)
    # Hindi/Arabic have no task-specific training data at all, so unlike Telugu there is
    # no existing dev cache to reuse for whitening -- load SemRel dev fresh (unused
    # anywhere else in either language's training), same set whitening_dev_only.py uses.
    dev_pairs = dev_loader("dev")
    wd1, wd2 = cache_pairs(backbone, dev_pairs)
    return {"nli": (ta, tp, tn, va, vp, vn), "task": None, "test": (t1, t2, test_pairs, kind), "whiten_dev": (wd1, wd2)}


LANGUAGE_SETUP = {
    "telugu": run_telugu,
    "hindi": lambda backbone, rng: run_xnli_only(backbone, rng, "hi", load_semrel_hindi, load_semrel_hindi),
    "arabic": lambda backbone, rng: run_xnli_only(backbone, rng, "ar", load_semrel_arabic, load_semrel_arabic),
}


def main():
    t_start = time.time()
    out_path = RESULTS_DIR / "tables" / "parameter_matched_all_languages_leakfree.json"
    all_results = json.loads(out_path.read_text()) if out_path.exists() else {}
    if all_results:
        print(f"Resuming: {list(all_results.keys())} already done, skipping.")

    for lang, setup_fn in LANGUAGE_SETUP.items():
        if lang in all_results:
            continue
        print(f"\n{'=' * 20} {lang} (leak-free) {'=' * 20}")
        lang_t0 = time.time()
        backbone = SpecialistBackbone(BACKBONES[lang])
        rng = random.Random(RANDOM_SEED)
        cached = setup_fn(backbone, rng)
        print(f"  caching done at {time.time() - lang_t0:.1f}s")

        torch.manual_seed(SEED)
        core = ParameterMatchedCore(backbone.hidden_dim)
        n_params = sum(p.numel() for p in core.parameters() if p.requires_grad)
        print(f"  ParameterMatchedCore trainable params: {n_params:,}")

        ckpt_path = RESULTS_DIR / f"parameter_matched_{lang}_leakfree.pt"
        if ckpt_path.exists():
            print(f"  Resuming: {ckpt_path} already exists, loading it and skipping training.")
            core.load_state_dict(torch.load(ckpt_path))
        else:
            if cached["nli"] is not None:
                ta, tp, tn, va, vp, vn = cached["nli"]
                train_nli(core, ta, tp, tn, va, vp, vn, SEED)
            if cached["task"] is not None:
                c1, c2, v1, v2, val_pairs, task_kind = cached["task"]
                train_task(core, c1, c2, v1, v2, val_pairs, task_kind, SEED)
            torch.save(core.state_dict(), ckpt_path)

        core.eval()
        t1, t2, test_pairs, kind = cached["test"]
        wd1, wd2 = cached["whiten_dev"]
        best, raw, white = evaluate(core, t1, t2, test_pairs, kind, wd1, wd2)

        pub = PUBLISHED_LEAKFREE[lang]
        print(f"  {lang}: parameter-matched raw={raw:.4f} whitened={white:.4f} best={best:.4f}  n_params={n_params:,}  ({time.time() - lang_t0:.1f}s)")
        print(f"  published leak-free: full-capacity structured={pub['ours_full']:.4f} linear={pub['linear']:.4f}")
        print(f"  parameter-matched vs full-capacity structured: {best - pub['ours_full']:+.4f}")
        print(f"  parameter-matched vs linear: {best - pub['linear']:+.4f}")

        all_results[lang] = {
            "protocol": "leak-free (whitening fit on dev split, never test)",
            "n_params_matched": n_params, "n_params_linear": 590592, "n_params_full_structured": 4430592,
            "parameter_matched": {"raw": raw, "whitened": white, "best": best},
            "full_capacity_structured_published": pub["ours_full"],
            "linear_published": pub["linear"],
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(all_results, indent=2))
        print(f"  (partial results written to {out_path})")

    print("\n=== Summary: does capacity-matching close the gap everywhere? ===")
    for lang, r in all_results.items():
        m, pub_ours, pub_lin = r["parameter_matched"]["best"], r["full_capacity_structured_published"], r["linear_published"]
        print(f"  {lang}: matched={m:.4f}  full-capacity={pub_ours:.4f} (delta {m - pub_ours:+.4f})  "
              f"linear={pub_lin:.4f} (delta {m - pub_lin:+.4f})")

    print(f"\nTotal time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
