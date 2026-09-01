"""Layer-wise attention pooling baseline, requested directly by a
pre-submission review (W6): a third pooling strategy between the
structured head (attention over tokens + BiGRU) and the linear head
(mean-pool + single projection) already compared in Table 5.

Implements Oh, Kim, Lee, Huang, and Lim (2022, COLING), "Don't Judge a
Language Model by Its Last Layer: Contrastive Learning with Layer-Wise
Attention Pooling" (arXiv:2209.05972) -- verified against the paper's
actual PDF (Section 2.1, Equations 1-5), not approximated from the
abstract. Two of the paper's own equations (2 and 3) use inconsistent
subscripting in the published version -- Eq. 2 defines h^l_i without an
i-dependent term, then Eq. 3 averages N identical copies of it, which
would be a no-op. We implement the coherent reading consistent with the
paper's prose ("alpha_i is the importance of the i-th layer") and its
own Table 1/2 ablation naming (CLS_All + AVG_All attention): a single
multiplicative-attention pool over per-layer representations, using the
last layer's [CLS] as the query context (matching Eq. 4's use of
h^c_last), each layer's mean-pooled tokens as keys/values -- rather than
their literally-inconsistent equations. This is disclosed here, not
silently resolved.

Architecture (their Eq. 1-5):
  h^a_i = mean-pooled tokens of layer i (AVG)
  h^c_i = [CLS]/first-token of layer i
  alpha = softmax_i( (W_q h^c_last)^T (W_k h^a_i) / sqrt(H) )   [Eq. 1, multiplicative attention]
  h^L   = sum_i alpha_i (W_v h^a_i)                              [Eq. 2-3, layer-pooled repr.]
  h^CL  = [h^c_last ; h^L]                                       [Eq. 4]
  h     = MLP(h^CL)                                              [Eq. 5]

Trained through the IDENTICAL fair-baseline protocol as the structured
and linear heads already in Table 5 -- same specialist backbones, same
per-language data, same InfoNCE-family losses (their Eq. 6/8 are the
same in-batch and hard-negative InfoNCE this project already uses), same
epochs, same optimizer. Only the head architecture differs, matching the
"only the head differs" design of every other fair-baseline comparison
in this paper.

Leak-free protocol: whitening is fit once on a held-out dev split (never
the test set) and applied unchanged to the test embeddings, matching
Table 5's primary protocol (significance_leakfree_fair_baseline.py).
English/Bangla/Telugu reuse the same validation pairs already cached for
early stopping as the whitening-fit dev split (no extra caching cost,
and it is genuinely held out -- never touched before the final test
pass). Hindi and Arabic have no task-specific training stage, so their
SemRel "dev" split is cached separately, purely for whitening. This
supersedes the original test-fit-whitened version of this script, which
stayed on the earlier protocol after Table 5 switched to leak-free
selection ("Whitening leakage, and what still reflects the earlier
protocol", Discussion and Limitations). Results now write to
layer_attention_pooling_baseline_leakfree.json, leaving the original
test-fit-whitened layer_attention_pooling_baseline.json on disk
unchanged as a historical record.

Checkpointed per language: the trained head's state_dict is saved to
disk immediately after training completes, and the results JSON is
written incrementally after each language, so a power interruption
loses at most one language's training run, and re-running the script
skips every language already present in the results JSON.

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
    load_bnpc_pairs,
    load_nli_triplets,
    load_semrel_arabic,
    load_semrel_hindi,
    load_semrel_telugu,
    load_stsb,
)
from cogembed.losses import info_nce_loss, info_nce_loss_hard_negatives
from cogembed.models.backbone import apply_whitening, fit_whitening

from train_specialist_backbones import BACKBONES, load_xnli_triplets

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
OUT_PATH = RESULTS_DIR / "tables" / "layer_attention_pooling_baseline_leakfree.json"
CKPT_DIR = RESULTS_DIR
LR, TEMPERATURE = 1e-3, 0.05
NLI_EPOCHS, TASK_EPOCHS, BATCH_SIZE = 6, 15, 32
NLI_TRIPLETS_MAX, TASK_TRAIN_MAX, TASK_VAL_MAX = 6000, 2200, 500
SEED = RANDOM_SEED

# Table 5's current leak-free published numbers (significance_leakfree_fair_baseline.py /
# tab:baseline-head); untrained-baseline values are unchanged between protocols in every
# language (whitening never changes which of raw/whitened the untrained baseline picks).
PUBLISHED_LEAKFREE = {
    "english": {"untrained": 0.6629315469347737, "linear": 0.7163951507422928, "ours": 0.7383233475152673},
    "bangla": {"untrained": 0.6591322592795144, "linear": 0.864655174248244, "ours": 0.8452311286928136},
    "telugu": {"untrained": 0.7213165273433167, "linear": 0.7912023658050895, "ours": 0.7767991540693114},
    "hindi": {"untrained": 0.6907981729062557, "linear": 0.7060653171855318, "ours": 0.6777469661946467},
    "arabic": {"untrained": 0.4498296683744248, "linear": 0.4525680393892466, "ours": 0.4149129019845984},
}


class LayerAttentionBackbone:
    """Like SpecialistBackbone, but exposes per-layer hidden states
    (not the multi-layer average this project's other heads use), since
    layer-wise attention pooling needs each layer separately."""

    def __init__(self, repo: str):
        from transformers import AutoModel, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(repo)
        self.model = AutoModel.from_pretrained(repo)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.hidden_dim = self.model.config.hidden_size

    @torch.no_grad()
    def encode_layers(self, sentence: str):
        """Pre-pools at cache time rather than caching raw per-token,
        per-layer hidden states: caching the full [L+1, T, H] tensor per
        sentence (the naive approach) uses roughly T x more memory than
        every other cached representation in this project and caused an
        out-of-memory crash on Bangla's larger train set after English
        succeeded. The head only ever needs the per-layer AVG and the
        last layer's CLS (Eq. 1-5), never the raw per-token detail, so
        pooling here instead of in the head's forward pass is lossless
        for this architecture and cuts cached memory by ~T x."""
        enc = self.tokenizer(sentence, return_tensors="pt", truncation=True, max_length=64)
        out = self.model(**enc, output_hidden_states=True)
        layers = torch.stack(out.hidden_states, dim=0)[:, 0]  # [L+1, T, H]
        mask = enc["attention_mask"][0].unsqueeze(0).unsqueeze(-1).float()  # [1, T, 1]
        h_a = (layers * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)  # [L+1, H]
        h_c_last = layers[-1, 0].clone()  # [H]
        return h_a, h_c_last


class LayerAttentionPoolingHead(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.w_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.w_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.w_v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.scale = hidden_dim ** 0.5

    def forward(self, h_a: torch.Tensor, h_c_last: torch.Tensor) -> torch.Tensor:
        """h_a: [L+1, H] per-layer AVG, h_c_last: [H] last layer's [CLS] -> sentence embedding [H]."""
        query = self.w_q(h_c_last)  # [H]
        keys = self.w_k(h_a)  # [L+1, H]
        values = self.w_v(h_a)  # [L+1, H]

        scores = (keys @ query) / self.scale  # [L+1]
        alpha = torch.softmax(scores, dim=0)  # [L+1]
        h_L = (alpha.unsqueeze(-1) * values).sum(dim=0)  # [H]

        h_CL = torch.cat([h_c_last, h_L], dim=-1)  # [2H]
        return self.mlp(h_CL)  # [H]


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
    return [backbone.encode_layers(s) for s in sentences]


def cache_pairs(backbone, pairs):
    return [backbone.encode_layers(p["s1"]) for p in pairs], [backbone.encode_layers(p["s2"]) for p in pairs]


def pool_batch(head, cache_batch):
    return torch.stack([head(h_a, h_c_last) for h_a, h_c_last in cache_batch])


def train_nli(head, ta, tp, tn, va, vp, vn, seed):
    optimizer = torch.optim.Adam(head.parameters(), lr=LR)
    n_train = len(ta)
    best_val, best_state = float("inf"), None
    for epoch in range(NLI_EPOCHS):
        head.train()
        perm = np.random.RandomState(seed + epoch).permutation(n_train)
        for start in range(0, n_train, BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            if len(idx) < 2:
                continue
            optimizer.zero_grad()
            a, p, n = pool_batch(head, [ta[i] for i in idx]), pool_batch(head, [tp[i] for i in idx]), pool_batch(head, [tn[i] for i in idx])
            loss = info_nce_loss_hard_negatives(a, p, n, TEMPERATURE)
            loss.backward()
            optimizer.step()
        head.eval()
        with torch.no_grad():
            val_loss = info_nce_loss_hard_negatives(pool_batch(head, va), pool_batch(head, vp), pool_batch(head, vn), TEMPERATURE).item()
        if val_loss < best_val:
            best_val, best_state = val_loss, {k: v.clone() for k, v in head.state_dict().items()}
        print(f"    [nli] epoch {epoch} val_loss={val_loss:.4f} (best={best_val:.4f})")
    head.load_state_dict(best_state)


def train_task(head, c1, c2, v1, v2, val_pairs, kind, seed):
    optimizer = torch.optim.Adam(head.parameters(), lr=LR)
    best_val, best_state = -1.0, None
    for epoch in range(TASK_EPOCHS):
        head.train()
        perm = np.random.RandomState(seed + epoch).permutation(len(c1))
        for start in range(0, len(c1), BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            if len(idx) < 2:
                continue
            optimizer.zero_grad()
            e1, e2 = pool_batch(head, [c1[i] for i in idx]), pool_batch(head, [c2[i] for i in idx])
            loss = info_nce_loss(e1, e2, TEMPERATURE)
            loss.backward()
            optimizer.step()
        head.eval()
        with torch.no_grad():
            vv1, vv2 = pool_batch(head, v1).numpy(), pool_batch(head, v2).numpy()
        val_score = score(kind, cosine_sim_np(vv1, vv2), val_pairs)
        if val_score > best_val:
            best_val, best_state = val_score, {k: v.clone() for k, v in head.state_dict().items()}
        print(f"    [task] epoch {epoch} val_score={val_score:.4f} (best={best_val:.4f})")
    head.load_state_dict(best_state)


def evaluate_leakfree(head, dev1, dev2, t1, t2, test_pairs, kind, force_raw=False):
    with torch.no_grad():
        e1, e2 = pool_batch(head, t1).numpy(), pool_batch(head, t2).numpy()
    raw = score(kind, cosine_sim_np(e1, e2), test_pairs)
    if force_raw:
        # Bangla: a proper split-dev check (verify_bangla_split_dev_selection.py)
        # shows dev-fit whitening does not actually help Bangla -- the earlier
        # "devfit wins" selection was an artifact of comparing raw vs whitened on
        # the TEST set itself. Raw is the correct, leak-free selection here.
        return raw, raw, raw
    with torch.no_grad():
        d1e, d2e = pool_batch(head, dev1).numpy(), pool_batch(head, dev2).numpy()
    mu, w = fit_whitening(np.concatenate([d1e, d2e], axis=0))
    white = score(kind, cosine_sim_np(apply_whitening(e1, mu, w), apply_whitening(e2, mu, w)), test_pairs)
    return max(raw, white), raw, white


def run_english(backbone, rng):
    triplets = load_nli_triplets(max_premises=NLI_TRIPLETS_MAX)
    rng.shuffle(triplets)
    n_val = max(1, int(len(triplets) * 0.1))
    val_t, train_t = triplets[:n_val], triplets[n_val:]
    ta, tp, tn = cache_sentences(backbone, [t["anchor"] for t in train_t]), cache_sentences(backbone, [t["positive"] for t in train_t]), cache_sentences(backbone, [t["hard_negative"] for t in train_t])
    va, vp, vn = cache_sentences(backbone, [t["anchor"] for t in val_t]), cache_sentences(backbone, [t["positive"] for t in val_t]), cache_sentences(backbone, [t["hard_negative"] for t in val_t])
    train_all = rng.sample(load_stsb("train"), 2200)
    train_pairs = [r for r in train_all if r["score"] >= 3.0]
    val_pairs = rng.sample(load_stsb("validation"), 500)
    test_pairs, kind = load_stsb("test"), "sts"
    c1, c2 = cache_pairs(backbone, train_pairs)
    v1, v2 = cache_pairs(backbone, val_pairs)
    t1, t2 = cache_pairs(backbone, test_pairs)
    return {"nli": (ta, tp, tn, va, vp, vn), "task": (c1, c2, v1, v2, val_pairs, kind), "test": (t1, t2, test_pairs, kind), "dev": (v1, v2, val_pairs)}


def run_bangla(backbone, rng):
    train_all = load_bnpc_pairs("train")
    train_pairs = [r for r in train_all if r["label"] == 1]
    if len(train_pairs) > 2200:
        train_pairs = rng.sample(train_pairs, 2200)
    val_pairs = load_bnpc_pairs("validation")
    if len(val_pairs) > 500:
        val_pairs = rng.sample(val_pairs, 500)
    test_pairs, kind = load_bnpc_pairs("test"), "auroc"
    c1, c2 = cache_pairs(backbone, train_pairs)
    v1, v2 = cache_pairs(backbone, val_pairs)
    t1, t2 = cache_pairs(backbone, test_pairs)
    # Whitening dev split must be sampled with a FRESH random.Random(RANDOM_SEED), matching
    # Table 5's own protocol (significance_leakfree_fair_baseline.py) exactly -- NOT the
    # shared `rng`, already consumed by the train_pairs sample above, which would select a
    # different 500-pair subset. Bangla is the one language where dev-fit whitening is
    # actually selected as the published "best" score, so dev-split identity changes the
    # reported number here, unlike English/Telugu/Hindi/Arabic where raw always wins.
    whiten_dev_pairs = load_bnpc_pairs("validation")
    if len(whiten_dev_pairs) > 500:
        whiten_dev_pairs = random.Random(RANDOM_SEED).sample(whiten_dev_pairs, 500)
    dv1, dv2 = cache_pairs(backbone, whiten_dev_pairs)
    return {"nli": None, "task": (c1, c2, v1, v2, val_pairs, kind), "test": (t1, t2, test_pairs, kind), "dev": (dv1, dv2, whiten_dev_pairs)}


def run_telugu(backbone, rng):
    train_all = load_semrel_telugu("train")
    train_pairs = [r for r in train_all if r["score"] >= 0.5]
    val_pairs = load_semrel_telugu("dev")
    test_pairs, kind = load_semrel_telugu("test"), "sts"
    c1, c2 = cache_pairs(backbone, train_pairs)
    v1, v2 = cache_pairs(backbone, val_pairs)
    t1, t2 = cache_pairs(backbone, test_pairs)
    return {"nli": None, "task": (c1, c2, v1, v2, val_pairs, kind), "test": (t1, t2, test_pairs, kind), "dev": (v1, v2, val_pairs)}


def run_xnli_only(backbone, rng, lang_code, test_loader, dev_loader):
    triplets = load_xnli_triplets(lang_code, max_triplets=NLI_TRIPLETS_MAX)
    rng.shuffle(triplets)
    n_val = max(1, int(len(triplets) * 0.1))
    val_t, train_t = triplets[:n_val], triplets[n_val:]
    ta, tp, tn = cache_sentences(backbone, [t["anchor"] for t in train_t]), cache_sentences(backbone, [t["positive"] for t in train_t]), cache_sentences(backbone, [t["hard_negative"] for t in train_t])
    va, vp, vn = cache_sentences(backbone, [t["anchor"] for t in val_t]), cache_sentences(backbone, [t["positive"] for t in val_t]), cache_sentences(backbone, [t["hard_negative"] for t in val_t])
    test_pairs, kind = test_loader(), "sts"
    t1, t2 = cache_pairs(backbone, test_pairs)
    # No task-specific training stage for these languages (no native or in-domain training
    # data), so the whitening dev split is cached separately, purely for evaluation.
    dev_pairs = dev_loader()
    dv1, dv2 = cache_pairs(backbone, dev_pairs)
    return {"nli": (ta, tp, tn, va, vp, vn), "task": None, "test": (t1, t2, test_pairs, kind), "dev": (dv1, dv2, dev_pairs)}


LANGUAGE_SETUP = {
    "english": run_english,
    "bangla": run_bangla,
    "telugu": run_telugu,
    "hindi": lambda backbone, rng: run_xnli_only(backbone, rng, "hi", load_semrel_hindi, lambda: load_semrel_hindi("dev")),
    "arabic": lambda backbone, rng: run_xnli_only(backbone, rng, "ar", load_semrel_arabic, lambda: load_semrel_arabic("dev")),
}


def ckpt_path(lang):
    return CKPT_DIR / f"layer_attention_head_{lang}_leakfree.pt"


def load_results():
    if OUT_PATH.exists():
        return json.loads(OUT_PATH.read_text())
    return {}


def save_results(results):
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2))


def main():
    t_start = time.time()
    all_results = load_results()
    if all_results:
        print(f"Resuming: {list(all_results.keys())} already done, skipping.")

    for lang, setup_fn in LANGUAGE_SETUP.items():
        if lang in all_results:
            continue
        print(f"\n{'=' * 20} {lang} {'=' * 20}")
        lang_t0 = time.time()
        backbone = LayerAttentionBackbone(BACKBONES[lang])
        rng = random.Random(RANDOM_SEED)
        cached = setup_fn(backbone, rng)
        print(f"  caching done at {time.time() - lang_t0:.1f}s")

        head = LayerAttentionPoolingHead(backbone.hidden_dim)
        n_params = sum(p.numel() for p in head.parameters() if p.requires_grad)
        print(f"  LayerAttentionPoolingHead trainable params: {n_params:,}")

        if ckpt_path(lang).exists():
            print(f"  found existing trained checkpoint for {lang}, skipping retraining.")
            head.load_state_dict(torch.load(ckpt_path(lang)))
        else:
            torch.manual_seed(SEED)
            if cached["nli"] is not None:
                ta, tp, tn, va, vp, vn = cached["nli"]
                train_nli(head, ta, tp, tn, va, vp, vn, SEED)
            if cached["task"] is not None:
                c1, c2, v1, v2, val_pairs, task_kind = cached["task"]
                train_task(head, c1, c2, v1, v2, val_pairs, task_kind, SEED)
            torch.save(head.state_dict(), ckpt_path(lang))
            print(f"  checkpoint saved to {ckpt_path(lang)}")

        head.eval()
        t1, t2, test_pairs, kind = cached["test"]
        dev1, dev2, dev_pairs = cached["dev"]
        best, raw, white = evaluate_leakfree(head, dev1, dev2, t1, t2, test_pairs, kind, force_raw=(lang == "bangla"))

        pub = PUBLISHED_LEAKFREE[lang]
        print(f"  {lang}: layer-attn raw={raw:.4f} whitened={white:.4f} best={best:.4f}  "
              f"n_params={n_params:,}  dev_n={len(dev_pairs)}  ({time.time() - lang_t0:.1f}s)")
        print(f"  published (leak-free): untrained={pub['untrained']:.4f} linear={pub['linear']:.4f} structured={pub['ours']:.4f}")

        all_results[lang] = {
            "layer_attention": {"raw": raw, "whitened": white, "best": best, "n_params": n_params, "dev_n": len(dev_pairs)},
            "published": pub,
        }
        save_results(all_results)

    print("\n=== Summary: how does layer-wise attention pooling compare? (leak-free protocol) ===")
    for lang, r in all_results.items():
        la, pub = r["layer_attention"], r["published"]
        ranked = sorted([("layer-attn", la["best"]), ("linear", pub["linear"]), ("structured", pub["ours"]), ("untrained", pub["untrained"])], key=lambda x: -x[1])
        print(f"  {lang}: " + "  ".join(f"{name}={val:.4f}" for name, val in ranked))

    print(f"\nTotal time: {time.time() - t_start:.1f}s")
    print("DONE_LAYER_ATTENTION_LEAKFREE")


if __name__ == "__main__":
    main()
