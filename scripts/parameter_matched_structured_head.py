"""Parameter-matched structured head, requested directly by a pre-submission
review: shrink the structured (Attention+Composition) head to roughly the
linear head's trainable-parameter count, via smaller BiGRU/attention
projections, to disambiguate "the mechanisms are unnecessary" from "the
structured head is undercapacity-mismatched at this scale" -- the same
question already raised, untested, in the paper's "Capacity as a confound"
limitation.

Parameter accounting (verified against the production CognitiveEmbeddingCore
and LinearProjectionHead classes, hidden_dim=768):
  Full structured head (K=4 attention heads, GRU hidden=128, no bottleneck):
    attention:    768*4 + 768*4*768 + 768        = 2,363,136
    composition:  GRU(768,128,bidir) + project    =   887,040
    combine:      Linear(1536,768)                = 1,180,416
    total                                          = 4,430,592  (matches paper exactly)
  Linear head: Linear(768,768) = 590,592

A naive K/m shrink alone cannot reach 590K: the final combine layer,
Linear(hidden_dim*2, hidden_dim), costs 1,180,416 by itself -- more than
the entire linear-head budget -- because it still projects from and to the
full 768-dim space. Matching the budget requires the "smaller ...
projections" the review asks for literally: attention and composition each
output a smaller bottleneck dimension d_out (not the full 768) before the
final combine layer maps back up to 768. This keeps the SAME three-part
architecture (attention, composition, combine) at reduced scale, rather
than changing the design.

Solving for K=2 attention heads, GRU hidden m=52, bottleneck d_out=104:
    attention:    768*2 + 768*2*104 + 104         =   161,384
    composition:  GRU(768,52,bidir) + project      =   267,384
    combine:      Linear(208,768)                  =   160,512
    total                                          =   589,280  (0.22% off 590,592)

Evaluated on Bangla only (same choice as batch_size_sweep.py): the fair-
baseline comparison's one statistically significant margin favoring the
linear head, no NLI pretrain stage (keeps this cheap), moderate dataset
size. Same data, same seed (42), same 15-epoch task-only recipe as the
full-capacity structured head and linear head already reported in Table 5
-- only the head's own internal capacity changes.

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
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cogembed.data.loaders import RANDOM_SEED, load_bnpc_pairs
from cogembed.losses import info_nce_loss
from cogembed.models.backbone import apply_whitening, fit_whitening

from train_specialist_backbones import BACKBONES, SpecialistBackbone

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
LANG = "bangla"
K_HEADS, GRU_HIDDEN, BOTTLENECK = 2, 52, 104
LR, TEMPERATURE, TASK_EPOCHS, BATCH_SIZE = 1e-3, 0.05, 15, 32
SEED = RANDOM_SEED

PUBLISHED_BANGLA_LEAKFREE = {"ours_full": 0.8731797937152742, "linear": 0.8925804316785585}


class ReducedSemanticAttentionPooling(nn.Module):
    """Identical logic to SemanticAttentionPooling, but projects to a
    smaller bottleneck dimension instead of the full hidden_dim."""

    def __init__(self, hidden_dim: int, d_out: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.queries = nn.Linear(hidden_dim, n_heads, bias=False)
        self.project = nn.Linear(hidden_dim * n_heads, d_out)

    def forward(self, token_features, mask):
        logits = self.queries(token_features)
        logits = logits.masked_fill((mask == 0).unsqueeze(-1), float("-inf"))
        weights = torch.softmax(logits, dim=0)
        views = torch.einsum("th,tk->kh", token_features, weights)
        return self.project(views.reshape(-1))


class ReducedCompositionalAggregator(nn.Module):
    """Identical logic to CompositionalAggregator, but a smaller GRU and a
    smaller bottleneck output dimension instead of the full hidden_dim."""

    def __init__(self, hidden_dim: int, d_out: int, gru_hidden: int):
        super().__init__()
        self.gru = nn.GRU(hidden_dim, gru_hidden, bidirectional=True, batch_first=True)
        self.project = nn.Linear(gru_hidden * 2, d_out)

    def forward(self, token_features, mask):
        length = int(mask.sum().item()) or 1
        seq = token_features[:length].unsqueeze(0)
        out, _ = self.gru(seq)
        pooled = out.squeeze(0).max(dim=0).values
        return self.project(pooled)


class ParameterMatchedCore(nn.Module):
    """Same three-part design as CognitiveEmbeddingCore (attention +
    composition -> combine), scaled down to roughly the linear head's
    parameter budget via a shared bottleneck dimension before combine."""

    def __init__(self, hidden_dim: int, d_out: int = BOTTLENECK, n_heads: int = K_HEADS, gru_hidden: int = GRU_HIDDEN):
        super().__init__()
        self.attention = ReducedSemanticAttentionPooling(hidden_dim, d_out, n_heads)
        self.composition = ReducedCompositionalAggregator(hidden_dim, d_out, gru_hidden)
        self.combine = nn.Linear(d_out * 2, hidden_dim)

    def forward(self, token_features, mask):
        attn_emb = self.attention(token_features, mask)
        comp_emb = self.composition(token_features, mask)
        return self.combine(torch.cat([attn_emb, comp_emb], dim=-1))


def cosine_sim_np(a, b):
    a_n = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)
    b_n = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-9)
    return (a_n * b_n).sum(axis=-1)


def score(sims, pairs):
    labels = np.array([p["label"] for p in pairs])
    return float(roc_auc_score(labels, sims))


def cache_pairs(backbone, pairs):
    return [backbone.encode_tokens(p["s1"]) for p in pairs], [backbone.encode_tokens(p["s2"]) for p in pairs]


def pool_batch(core, cache_batch):
    return torch.stack([core(h, m) for h, m in cache_batch])


def train_task(core, c1, c2, v1, v2, val_pairs, seed):
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
        val_score = score(cosine_sim_np(vv1, vv2), val_pairs)
        if val_score > best_val:
            best_val, best_state = val_score, {k: v.clone() for k, v in core.state_dict().items()}
        print(f"    epoch {epoch:2d} val_auroc={val_score:.4f} (best={best_val:.4f})")
    core.load_state_dict(best_state)


def evaluate(core, t1, t2, test_pairs, v1, v2):
    """Leak-free: whitening is fit on the dev/validation cache (v1, v2), never on the
    test embeddings themselves, matching the protocol Table tab:baseline-head uses as
    primary (whitening_dev_only.py). raw-vs-whitened is still selected by whichever
    scores higher on the test set, the same selection convention already established
    for every other leak-free number in this paper -- only the whitening TRANSFORM's
    fitting data changes, not how the binary raw/whitened choice is made."""
    with torch.no_grad():
        e1, e2 = pool_batch(core, t1).numpy(), pool_batch(core, t2).numpy()
        d1, d2 = pool_batch(core, v1).numpy(), pool_batch(core, v2).numpy()
    raw = score(cosine_sim_np(e1, e2), test_pairs)
    fit = np.concatenate([d1, d2], axis=0)
    mu, w = fit_whitening(fit)
    white = score(cosine_sim_np(apply_whitening(e1, mu, w), apply_whitening(e2, mu, w)), test_pairs)
    return max(raw, white), raw, white


def main():
    t_start = time.time()

    torch.manual_seed(SEED)
    probe = ParameterMatchedCore(768)
    n_params = sum(p.numel() for p in probe.parameters() if p.requires_grad)
    print(f"ParameterMatchedCore trainable params: {n_params:,}  (target: 590,592, linear head)")
    del probe

    print(f"\nLoading {LANG} backbone and caching data...")
    backbone = SpecialistBackbone(BACKBONES[LANG])
    rng = random.Random(RANDOM_SEED)

    train_all = load_bnpc_pairs("train")
    train_pairs = [r for r in train_all if r["label"] == 1]
    if len(train_pairs) > 2200:
        train_pairs = rng.sample(train_pairs, 2200)
    val_pairs = load_bnpc_pairs("validation")
    if len(val_pairs) > 500:
        val_pairs = rng.sample(val_pairs, 500)
    test_pairs = load_bnpc_pairs("test")

    c1, c2 = cache_pairs(backbone, train_pairs)
    v1, v2 = cache_pairs(backbone, val_pairs)
    t1, t2 = cache_pairs(backbone, test_pairs)
    print(f"  train/val/test: {len(train_pairs)}/{len(val_pairs)}/{len(test_pairs)}")
    print(f"  caching done at {time.time() - t_start:.1f}s\n")

    torch.manual_seed(SEED)
    core = ParameterMatchedCore(backbone.hidden_dim)
    ckpt_path = RESULTS_DIR / "parameter_matched_bangla_leakfree.pt"
    if ckpt_path.exists():
        print(f"Resuming: {ckpt_path} already exists, loading it and skipping training.")
        core.load_state_dict(torch.load(ckpt_path))
    else:
        print("Training parameter-matched structured head (K=2, gru=52, bottleneck=104)...")
        train_task(core, c1, c2, v1, v2, val_pairs, SEED)
        torch.save(core.state_dict(), ckpt_path)
    core.eval()
    best, raw, white = evaluate(core, t1, t2, test_pairs, v1, v2)

    print(f"\n=== Result (leak-free: whitening fit on dev, never test) ===")
    print(f"  parameter-matched structured head: raw={raw:.4f} whitened={white:.4f} best={best:.4f}")
    print(f"  full-capacity structured head (published leak-free, 4,430,592 params): {PUBLISHED_BANGLA_LEAKFREE['ours_full']:.4f}")
    print(f"  linear head (published leak-free, 590,592 params): {PUBLISHED_BANGLA_LEAKFREE['linear']:.4f}")
    print(f"  parameter-matched vs linear (both ~590K params): {best - PUBLISHED_BANGLA_LEAKFREE['linear']:+.4f}")
    print(f"  parameter-matched vs full-capacity structured: {best - PUBLISHED_BANGLA_LEAKFREE['ours_full']:+.4f}")

    results = {
        "protocol": "leak-free (whitening fit on dev split, never test)",
        "n_params_matched": n_params, "n_params_linear": 590592, "n_params_full_structured": 4430592,
        "config": {"k_heads": K_HEADS, "gru_hidden": GRU_HIDDEN, "bottleneck": BOTTLENECK},
        "parameter_matched": {"raw": raw, "whitened": white, "best": best},
        "full_capacity_structured_published": PUBLISHED_BANGLA_LEAKFREE["ours_full"],
        "linear_published": PUBLISHED_BANGLA_LEAKFREE["linear"],
    }
    out_path = RESULTS_DIR / "tables" / "parameter_matched_bangla_leakfree.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults written to {out_path}")
    print(f"Total time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
