# Does Structured Pooling Earn Its Complexity?

Code and experiments for the paper *"Does Structured Pooling Earn Its Complexity? A Component-Wise Attribution Study for Low-Resource Sentence Embeddings"* (Hasan Mahmud, submitted to ACM TALLIP). Full manuscript source: [`paper/cognitive_embedding_tallip_v7.tex`](paper/cognitive_embedding_tallip_v7.tex) ([compiled PDF](paper/cognitive_embedding_tallip_v7.pdf)).

This is a **controlled attribution study**, not a new-architecture paper. It asks a narrower question: given a frozen, publicly available encoder, does a small pooling head built from mechanisms inspired by human sentence processing (selective attention, compositional integration) actually earn its extra complexity over the simplest trained alternative — a single linear projection — under an identical, fair training protocol? Four mechanisms are evaluated: attention and composition form the core embedding; memory-aware retrieval and predictive next-sentence modeling are evaluated separately, as downstream capabilities rather than components fused into the embedding itself.

## Headline results

At the one fixed, untuned capacity evaluated throughout (the structured head has 7.5x more trainable parameters than the linear head and trains 2.7-2.8x slower), the linear head has the higher point estimate in **21 of 22** task-language comparisons across similarity, classification transfer, and bitext retrieval — a descriptive count, not an independent-trials statistic, since several comparisons reuse the same trained checkpoints. Formal significance holds for only a subset: Bangla and Hindi significantly favor the linear head, English significantly favors the structured head, and Telugu and Arabic are statistically inconclusive.

That is not the full story. A **parameter-matched control** (structured head scaled down to within 0.2% of the linear head's own parameter count, run under the same leak-free evaluation protocol) shows capacity is a substantial, language-dependent confound: it **fully explains Hindi's reversal** (the structured head wins outright once matched), is real but too weak to establish confidently in Arabic, **partially explains** Bangla's margin (narrows by roughly two-thirds without closing), and does not explain Telugu's margin at all. The negative headline result is therefore a property of the specific, fixed, untuned configuration tested throughout, not a general verdict on structured or cognitively-motivated pooling.

Beyond the attribution result itself: memory-aware retrieval and predictive next-sentence modeling are real, usable capabilities once kept separate from the core embedding rather than fused into it (fusing either was tested directly and does not help); Matryoshka compression preserves or slightly improves accuracy at a 12x dimensional reduction; bitext alignment and downstream classifier transfer are empirically distinct properties of a cross-lingual embedding space; and the fair-baseline, capacity-matched protocol used throughout is a reusable template for testing whether any proposed pooling mechanism earns its complexity before publication.

**Not established:** a reliable inference-latency advantage for either head. An initial single-timing-pass measurement suggested one, but repeated interleaved measurements showed the standard deviations were as large as the differences being compared — the paper explicitly reports this as not established, rather than as a result, once that was found.

All numbers in the paper are from real, completed experimental runs — the JSON result tables in `results/tables/` and the raw run logs in `logs/` are the underlying evidence for every reported number.

## What's actually in this architecture

Two mechanisms are fused into the core sentence embedding (semantic attention + compositional aggregation, via a BiGRU). Two more (memory-aware retrieval, predictive next-sentence head) are validated as **separate downstream capabilities** — fusing either into the embedding was tested directly and did not help, and that negative result is reported rather than hidden. See the paper's ablation section (Table 4 / `tab:ablation`) for exactly which mechanism contributes where, and Section 5.4 for the memory/predictive fusion test.

## Repository structure

```
src/cogembed/          Core package: model architecture, losses, data loaders
  models/               Semantic attention, compositional aggregator,
                         memory-aware retrieval, predictive head, the fused
                         core embedding, and the frozen-backbone wrapper
  data/                 Dataset loaders (STS-B, BnPC, SemRel2024, FLORES-200,
                         MASSIVE, SNLI/MultiNLI, XNLI, discourse corpora) and
                         a registry documenting dataset accessibility/status
  losses.py             InfoNCE, hard-negative InfoNCE, Matryoshka nested loss

scripts/                Training and evaluation entry points (see below)
results/tables/         JSON result tables backing every number in the paper.
                         Files ending in `_leakfree` (or `_leakfree.json`) are
                         the final, primary leak-free evaluation protocol
                         (whitening fit only on a held-out development split,
                         never the test set); files without that suffix are
                         earlier, test-fit-whitened runs kept on disk as a
                         historical record, not the numbers reported as
                         primary in the paper.
logs/                   Full stdout logs from every training/evaluation run
paper/                  LaTeX source for the manuscript, `cognitive_embedding_
                         tallip_v7.tex`, the final version submitted to TALLIP.
```

## Reproducing the results

### Setup

```bash
pip install -r requirements.txt
```

Windows/CPU note: `datasets` must be imported before `torch` in every entry point (a real, reproduced native-library conflict on this platform) — every script here already does this; preserve the import order if you extend them. Set `KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1` as environment variables before running anything, or scripts will segfault on Windows/CPU. All experiments were run on a single CPU machine, no GPU (exact package versions, OS, and hardware: paper Section 6, "Reproducibility").

### Data

Everything downloads automatically from Hugging Face on first run via `src/cogembed/data/loaders.py`, **except BnPC** (the gold Bangla paraphrase corpus), which is Kaggle-hosted and must be obtained manually. Place `BnPC_train.csv`, `BnPC_val.csv`, `BnPC_test.csv` in `data/raw/BnPC/`. See `src/cogembed/data/registry.py` for the full accounting of every dataset used, its accessibility, and its gold/silver status.

### Training

```bash
# Shared multilingual backbone: NLI pretrain + EN/BN/TE joint fine-tune + predictive head
python scripts/train.py

# Per-language specialist backbones (RoBERTa/BanglaBERT/Telugu-BERT/Hindi-BERT/AraBERT)
python scripts/train_specialist_backbones.py

# Cross-lingual alignment fine-tune (fixes the bitext retrieval gap, paper Section 5.6)
python scripts/train_crosslingual.py

# Matryoshka compression training
python scripts/train_matryoshka.py
```

### Primary evaluation (leak-free protocol, Table 5 and Table 2)

```bash
python scripts/whitening_dev_only.py                       # primary fair-baseline scores
python scripts/significance_leakfree_fair_baseline.py       # bootstrap significance + CIs
```

### Secondary/robustness evaluation (all rerun under the same leak-free protocol)

```bash
python scripts/multi_seed_fair_baseline.py                  # Table 6: 3-seed robustness
python scripts/data_size_sensitivity_sweep.py                # Table 15: data-size sweep
python scripts/batch_size_sweep.py                          # batch-size sweep, Bangla
python scripts/layer_attention_pooling_baseline.py           # Table 16: third pooling design
python scripts/parameter_matched_all_languages.py            # capacity-matched control
python scripts/arabic_farasa_preprocessing.py                 # Farasa-preprocessed Arabic rerun
python scripts/verify_devbased_selection.py                  # checks raw-vs-whitened selection bias
python scripts/verify_bangla_split_dev_selection.py           # held-out split-dev check for Bangla
```

### Downstream tasks and other analyses

```bash
python scripts/evaluate_bitext_retrieval.py                 # cross-lingual retrieval
python scripts/evaluate_classification.py                    # MASSIVE intent classification + zero-shot transfer
python scripts/evaluate_extra_baselines.py                    # mE5-base, MiniLM comparison
python scripts/evaluate_matryoshka.py                        # accuracy-vs-dimension curve
python scripts/ablation_core_components.py                    # attention-only vs. composition-only vs. combined
python scripts/ablation_predictive_fusion.py                  # does fusing the predictive head help?
python scripts/measure_efficiency.py                         # latency, parameters, memory footprint
python scripts/qualitative_analysis.py                       # concrete success/failure examples
```

Every script caches the frozen backbone's forward passes where possible and is CPU-feasible. Long-running sweeps (multi-seed, data-size, batch-size) checkpoint incrementally and resume automatically if interrupted.

## Honest limitations

Stated explicitly in the paper (Section 6, "Discussion and Limitations"), not left for a reader to discover: no human cognitive-behavioral validation exists for these mechanisms (they are cognitively *inspired*, not cognitively *validated* — see "Cognitive validity"); the 21/22 comparison count is a descriptive tally, not an independent-trials statistic, and a formal significance test supports only a subset of the margins ("Statistical rigor"); capacity is a substantial but language-dependent confound whose effect was directly tested, not assumed ("Capacity as a confound"); this architecture does not claim to match current state-of-the-art multilingual embedding models (Table 13 ranks it against seven strong baselines including fine-tuned LaBSE and multilingual-E5); and every language/head combination in this study uses one specialist encoder, not multiple alternative backbones.

## Citation

If you use this code or the accompanying paper, please cite the manuscript in [`paper/cognitive_embedding_tallip_v7.tex`](paper/cognitive_embedding_tallip_v7.tex).
