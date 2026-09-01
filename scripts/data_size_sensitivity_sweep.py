"""Data-size sensitivity sweep, requested directly by a pre-submission
review (W4): does the linear-vs-structured margin hold across data
regimes, or would the structured head win with more data? This is the
direct test the "Why English is the exception" argument (data
availability) names as its own missing piece.

Design note on why this isn't identical across languages: Hindi and
Arabic have NO task-specific fine-tuning data at all in this study --
they train on XNLI triplets only, then evaluate zero-shot (Section
"Languages, Backbones, and Data"). There is no task-specific pool to
subsample for them. So the fraction swept is:
  - English, Bangla, Telugu: fraction of TASK-SPECIFIC fine-tuning
    pairs (2200, 2200, 552 at 100%). NLI pretrain (English only) stays
    at full size -- it is a generic corpus, not the scarce resource
    that varies by language the way task data does.
  - Hindi, Arabic: fraction of XNLI PRETRAIN triplets (6000 at 100%),
    their only trainable signal.
This is not a perfectly uniform sweep, but a uniform one is not
available: it follows the same "each language's best available data,
honestly" principle already used throughout this paper (Section
"Languages, Backbones, and Data").

Caching-once optimization (same pattern as multi_seed_fair_baseline.py):
each language's full data pool and test set are encoded through its
specialist backbone exactly once; fractions are subsamples of the
SAME cached features, so no re-encoding happens per fraction. The 100%
point is not retrained -- it reuses the existing published checkpoint
(core_model_specialist_{lang}.pt / linear_head_{lang}.pt), re-evaluated
under this script's own leak-free protocol (inference only). Only
10%/25%/50% are newly trained, for both heads, across all five
languages: 30 new runs.

Leak-free protocol: whitening is fit once on a held-out dev split (never
the test set) and applied unchanged to the test embeddings, matching
Table 5's primary protocol (significance_leakfree_fair_baseline.py).
English/Bangla/Telugu reuse the validation pairs already cached for
early stopping as the whitening-fit dev split; Hindi/Arabic (no task-
specific training stage) cache their SemRel "dev" split separately,
purely for whitening. This supersedes the original test-fit-whitened
version of this sweep, which stayed on the earlier protocol after
Table 5 switched to leak-free selection ("Whitening leakage, and what
still reflects the earlier protocol", Discussion and Limitations).
Results now write to data_size_sensitivity_sweep_leakfree.json, leaving
the original test-fit-whitened data_size_sensitivity_sweep.json on disk
unchanged as a historical record.

Checkpointed at (language, head) granularity: results are written
incrementally after each head's four fractions complete, and re-running
this script skips every (language, head) pair already present in the
results JSON. The original version of this sweep was interrupted by a
session restart partway through Arabic and needed a hand-written,
one-off resume script (data_size_sensitivity_sweep_arabic_resume.py);
this version's built-in per-head resume replaces the need for that.

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
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cogembed.data.loaders import (
    RANDOM_SEED,
    load_bnpc_pairs,
    load_nli_triplets,
    load_semrel_arabic,
    load_semrel_hindi,
    load_semrel_telugu,
    load_stsb,
)
from cogembed.losses import info_nce_loss, info_nce_loss_hard_negatives
from cogembed.models.backbone import apply_whitening, fit_whitening
from cogembed.models.cognitive_embedding import CognitiveEmbeddingCore

from train_specialist_backbones import BACKBONES, SpecialistBackbone, load_xnli_triplets
from baseline_linear_head import LinearProjectionHead

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
OUT_PATH = RESULTS_DIR / "tables" / "data_size_sensitivity_sweep_leakfree.json"
FRACTIONS = [0.10, 0.25, 0.50, 1.00]
LR, TEMPERATURE, BATCH_SIZE = 1e-3, 0.05, 32
NLI_EPOCHS, TASK_EPOCHS = 6, 15
NLI_TRIPLETS_MAX = 6000
SEED = RANDOM_SEED

CHECKPOINT_PATHS = {
    "ours": lambda lang: RESULTS_DIR / f"core_model_specialist_{lang}.pt",
    "linear": lambda lang: RESULTS_DIR / f"linear_head_{lang}.pt",
}

# Table 5's current leak-free published numbers (significance_leakfree_fair_baseline.py /
# tab:baseline-head) -- the 100%-fraction anchor for this sweep.
PUBLISHED_LEAKFREE = {
    "english": {"ours": 0.7383233475152673, "linear": 0.7163951507422928},
    "bangla": {"ours": 0.8452311286928136, "linear": 0.864655174248244},
    "telugu": {"ours": 0.7767991540693114, "linear": 0.7912023658050895},
    "hindi": {"ours": 0.6777469661946467, "linear": 0.7060653171855318},
    "arabic": {"ours": 0.4149129019845984, "linear": 0.4525680393892466},
}


def cosine_sim_np(a, b):
    a_n = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)
    b_n = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-9)
    return (a_n * b_n).sum(axis=-1)


def score(kind, sims, pairs):
    if kind == "sts":
        gold = np.array([p["score"] for p in pairs])
        rho, _ = spearmanr(sims, gold)
        return float(rho)
    labels = np.array([p["label"] for p in pairs])
    return float(roc_auc_score(labels, sims))


def cache_sentences(backbone, sentences):
    return [backbone.encode_tokens(s) for s in sentences]


def cache_pairs(backbone, pairs):
    return [backbone.encode_tokens(p["s1"]) for p in pairs], [backbone.encode_tokens(p["s2"]) for p in pairs]


def pool_batch(core, cache_batch):
    return torch.stack([core(h, m) for h, m in cache_batch])


def make_head(kind, hidden_dim):
    return CognitiveEmbeddingCore(hidden_dim) if kind == "ours" else LinearProjectionHead(hidden_dim)


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
    core.load_state_dict(best_state)


def evaluate_leakfree(core, dev1, dev2, t1, t2, test_pairs, kind, force_raw=False):
    with torch.no_grad():
        e1, e2 = pool_batch(core, t1).numpy(), pool_batch(core, t2).numpy()
    raw = score(kind, cosine_sim_np(e1, e2), test_pairs)
    if force_raw:
        # Bangla: a proper split-dev check (verify_bangla_split_dev_selection.py)
        # shows dev-fit whitening does not actually help Bangla -- the earlier
        # "devfit wins" selection was an artifact of comparing raw vs whitened on
        # the TEST set itself. Raw is the correct, leak-free selection here.
        return raw
    with torch.no_grad():
        d1e, d2e = pool_batch(core, dev1).numpy(), pool_batch(core, dev2).numpy()
    mu, w = fit_whitening(np.concatenate([d1e, d2e], axis=0))
    white = score(kind, cosine_sim_np(apply_whitening(e1, mu, w), apply_whitening(e2, mu, w)), test_pairs)
    return max(raw, white)


# ---- Per-language setup: cache full pool once, return what varies-by-fraction ----

def setup_task_data_language(backbone, rng, train_pairs, val_pairs, test_pairs, kind, nli_cached=None):
    c1, c2 = cache_pairs(backbone, train_pairs)
    v1, v2 = cache_pairs(backbone, val_pairs)
    t1, t2 = cache_pairs(backbone, test_pairs)
    return {"vary": "task", "c1": c1, "c2": c2, "v1": v1, "v2": v2, "val_pairs": val_pairs,
            "kind": kind, "t1": t1, "t2": t2, "test_pairs": test_pairs, "nli": nli_cached,
            "dev1": v1, "dev2": v2, "dev_pairs": val_pairs}


def setup_english(backbone, rng):
    triplets = load_nli_triplets(max_premises=NLI_TRIPLETS_MAX)
    rng.shuffle(triplets)
    n_val = max(1, int(len(triplets) * 0.1))
    val_t, train_t = triplets[:n_val], triplets[n_val:]
    ta, tp, tn = cache_sentences(backbone, [t["anchor"] for t in train_t]), cache_sentences(backbone, [t["positive"] for t in train_t]), cache_sentences(backbone, [t["hard_negative"] for t in train_t])
    va, vp, vn = cache_sentences(backbone, [t["anchor"] for t in val_t]), cache_sentences(backbone, [t["positive"] for t in val_t]), cache_sentences(backbone, [t["hard_negative"] for t in val_t])
    nli_cached = (ta, tp, tn, va, vp, vn)

    train_all = rng.sample(load_stsb("train"), 2200)
    train_pairs = [r for r in train_all if r["score"] >= 3.0]
    val_pairs = rng.sample(load_stsb("validation"), 500)
    test_pairs, kind = load_stsb("test"), "sts"
    return setup_task_data_language(backbone, rng, train_pairs, val_pairs, test_pairs, kind, nli_cached)


def setup_bangla(backbone, rng):
    train_all = load_bnpc_pairs("train")
    train_pairs = [r for r in train_all if r["label"] == 1]
    if len(train_pairs) > 2200:
        train_pairs = rng.sample(train_pairs, 2200)
    val_pairs = load_bnpc_pairs("validation")
    if len(val_pairs) > 500:
        val_pairs = rng.sample(val_pairs, 500)
    test_pairs, kind = load_bnpc_pairs("test"), "auroc"
    cached = setup_task_data_language(backbone, rng, train_pairs, val_pairs, test_pairs, kind)
    # Whitening dev split, kept for structural consistency with other languages'
    # setup functions -- unused in practice, since Bangla's evaluate_leakfree call
    # is forced to raw scoring (a proper split-dev check showed whitening does not
    # actually help Bangla; see the force_raw docstring on evaluate_leakfree).
    whiten_dev_pairs = load_bnpc_pairs("validation")
    if len(whiten_dev_pairs) > 500:
        whiten_dev_pairs = random.Random(RANDOM_SEED).sample(whiten_dev_pairs, 500)
    dv1, dv2 = cache_pairs(backbone, whiten_dev_pairs)
    cached["dev1"], cached["dev2"], cached["dev_pairs"] = dv1, dv2, whiten_dev_pairs
    return cached


def setup_telugu(backbone, rng):
    train_all = load_semrel_telugu("train")
    train_pairs = [r for r in train_all if r["score"] >= 0.5]
    val_pairs = load_semrel_telugu("dev")
    test_pairs, kind = load_semrel_telugu("test"), "sts"
    return setup_task_data_language(backbone, rng, train_pairs, val_pairs, test_pairs, kind)


def setup_xnli_only(backbone, rng, lang_code, test_loader, dev_loader):
    triplets = load_xnli_triplets(lang_code, max_triplets=NLI_TRIPLETS_MAX)
    rng.shuffle(triplets)
    n_val = max(1, int(len(triplets) * 0.1))
    val_t, train_t = triplets[:n_val], triplets[n_val:]
    ta, tp, tn = cache_sentences(backbone, [t["anchor"] for t in train_t]), cache_sentences(backbone, [t["positive"] for t in train_t]), cache_sentences(backbone, [t["hard_negative"] for t in train_t])
    va, vp, vn = cache_sentences(backbone, [t["anchor"] for t in val_t]), cache_sentences(backbone, [t["positive"] for t in val_t]), cache_sentences(backbone, [t["hard_negative"] for t in val_t])
    test_pairs, kind = test_loader(), "sts"
    t1, t2 = cache_pairs(backbone, test_pairs)
    dev_pairs = dev_loader()
    dv1, dv2 = cache_pairs(backbone, dev_pairs)
    return {"vary": "nli", "ta": ta, "tp": tp, "tn": tn, "va": va, "vp": vp, "vn": vn,
            "kind": kind, "t1": t1, "t2": t2, "test_pairs": test_pairs, "n_full": len(ta),
            "dev1": dv1, "dev2": dv2, "dev_pairs": dev_pairs}


LANGUAGE_SETUP = {
    "english": setup_english,
    "bangla": setup_bangla,
    "telugu": setup_telugu,
    "hindi": lambda backbone, rng: setup_xnli_only(backbone, rng, "hi", load_semrel_hindi, lambda: load_semrel_hindi("dev")),
    "arabic": lambda backbone, rng: setup_xnli_only(backbone, rng, "ar", load_semrel_arabic, lambda: load_semrel_arabic("dev")),
}


def run_fraction(cached, head_kind, hidden_dim, fraction, seed, force_raw=False):
    torch.manual_seed(seed)
    core = make_head(head_kind, hidden_dim)
    frac_rng = np.random.RandomState(seed)

    if cached["vary"] == "task":
        if cached["nli"] is not None:
            ta, tp, tn, va, vp, vn = cached["nli"]
            train_nli(core, ta, tp, tn, va, vp, vn, seed)
        n_full = len(cached["c1"])
        n_use = max(2, int(round(n_full * fraction)))
        idx = frac_rng.choice(n_full, size=n_use, replace=False) if n_use < n_full else np.arange(n_full)
        c1 = [cached["c1"][i] for i in idx]
        c2 = [cached["c2"][i] for i in idx]
        train_task(core, c1, c2, cached["v1"], cached["v2"], cached["val_pairs"], cached["kind"], seed)
        n_train_used = n_use
    else:  # vary == "nli"
        n_full = cached["n_full"]
        n_use = max(2, int(round(n_full * fraction)))
        idx = frac_rng.choice(n_full, size=n_use, replace=False) if n_use < n_full else np.arange(n_full)
        ta = [cached["ta"][i] for i in idx]
        tp = [cached["tp"][i] for i in idx]
        tn = [cached["tn"][i] for i in idx]
        train_nli(core, ta, tp, tn, cached["va"], cached["vp"], cached["vn"], seed)
        n_train_used = n_use

    core.eval()
    best = evaluate_leakfree(core, cached["dev1"], cached["dev2"], cached["t1"], cached["t2"], cached["test_pairs"], cached["kind"], force_raw=force_raw)
    return best, n_train_used


def load_results():
    if OUT_PATH.exists():
        return json.loads(OUT_PATH.read_text())
    return {}


def save_results(results):
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2))


def head_done(all_results, lang, head_kind):
    return lang in all_results and head_kind in all_results[lang] and len(all_results[lang][head_kind]) == len(FRACTIONS)


def main():
    t_start = time.time()
    all_results = load_results()
    if all_results:
        done = {lang: [hk for hk in ("ours", "linear") if head_done(all_results, lang, hk)] for lang in all_results}
        print(f"Resuming: {done}")

    for lang, setup_fn in LANGUAGE_SETUP.items():
        if head_done(all_results, lang, "ours") and head_done(all_results, lang, "linear"):
            print(f"{lang}: both heads already done, skipping.")
            continue

        print(f"\n{'=' * 20} {lang}: caching (once, shared across heads and fractions) {'=' * 20}")
        lang_t0 = time.time()
        backbone = SpecialistBackbone(BACKBONES[lang])
        rng = random.Random(RANDOM_SEED)
        cached = setup_fn(backbone, rng)
        print(f"  varying: {cached['vary']}-data  caching done at {time.time() - lang_t0:.1f}s")

        lang_results = all_results.get(lang, {"vary": cached["vary"]})

        for head_kind in ["ours", "linear"]:
            if head_done(all_results, lang, head_kind):
                print(f"  {lang} {head_kind}: already done, skipping.")
                continue

            head_t0 = time.time()
            lang_results[head_kind] = {}
            for frac in FRACTIONS:
                frac_t0 = time.time()
                if frac == 1.00:
                    # 100%: re-evaluate the existing published checkpoint under the
                    # leak-free protocol (inference only, no retraining).
                    core = make_head(head_kind, backbone.hidden_dim)
                    core.load_state_dict(torch.load(CHECKPOINT_PATHS[head_kind](lang)))
                    core.eval()
                    s = evaluate_leakfree(core, cached["dev1"], cached["dev2"], cached["t1"], cached["t2"], cached["test_pairs"], cached["kind"], force_raw=(lang == "bangla"))
                    n_used = "published checkpoint, leak-free eval"
                else:
                    s, n_used = run_fraction(cached, head_kind, backbone.hidden_dim, frac, SEED, force_raw=(lang == "bangla"))
                lang_results[head_kind][str(frac)] = {"score": s, "n_train": n_used}
                print(f"  {lang} {head_kind} frac={frac:.2f}: score={s:.4f} n_train={n_used}  ({time.time() - frac_t0:.1f}s)")

            print(f"  {lang} {head_kind}: full sweep done  ({time.time() - head_t0:.1f}s)")
            all_results[lang] = lang_results
            save_results(all_results)
            print(f"  (partial results written to {OUT_PATH})")

        print(f"  {lang} total: {time.time() - lang_t0:.1f}s")

    print("\n=== Summary: does the margin shrink or reverse as data grows? (leak-free protocol) ===")
    for lang, r in all_results.items():
        print(f"  {lang} (varying {r['vary']}-data):")
        for frac in FRACTIONS:
            o = r["ours"][str(frac)]["score"]
            l = r["linear"][str(frac)]["score"]
            print(f"    frac={frac:.2f}: ours={o:.4f} linear={l:.4f} margin(ours-linear)={o - l:+.4f}")

    print(f"\nTotal time: {time.time() - t_start:.1f}s")
    print("DONE_DATA_SIZE_SWEEP_LEAKFREE")


if __name__ == "__main__":
    main()
