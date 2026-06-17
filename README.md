# Same Architecture, Different Capacity: Optimizer-Induced Spectral Scaling Laws

[![CI](https://github.com/optimizer-scaling-laws/spectral-scaling-laws/actions/workflows/ci.yml/badge.svg)](https://github.com/optimizer-scaling-laws/spectral-scaling-laws/actions/workflows/ci.yml)
[![arXiv](https://img.shields.io/badge/arXiv-2605.21803-b31b1b.svg)](https://arxiv.org/abs/2605.21803)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Code and reproducible artifacts for our paper **[Same Architecture, Different Capacity: Optimizer-Induced Spectral Scaling Laws](https://arxiv.org/abs/2605.21803)** — Nandan Kumar Jha and Brandon Reagen.
[📄 Paper](https://arxiv.org/abs/2605.21803) · [🌐 Project page](https://optimizer-scaling-laws.github.io/) · [▶️ Reproduce in Colab](https://colab.research.google.com/github/optimizer-scaling-laws/spectral-scaling-laws/blob/main/notebooks/reproduce_main_figures.ipynb)

<p align="center">
  <img src="assets/teaser.png" width="900" alt="Spectral scaling exponents depend on optimizer choice (soft and hard spectral rank vs FFN width)"><br>
  <em><strong>Figure 1.</strong> Spectral scaling exponents depend on optimizer choice: soft (left) and hard (right) spectral rank as a function of FFN width in GPT-2 160M. AdamW exhibits the largest hard–soft asymmetry (Δ = β<sub>soft</sub> − β<sub>hard</sub> = 0.37), indicating concentrated eigenspectra. Muon and Dion (1/2) reduce this asymmetry to Δ ≈ 0.14. Moreover, hard-rank scaling exhibits stronger dependence on optimizer choice.</em>
</p>

## TL;DR

**Same architecture, different optimizer, different capacity.** Realized representation capacity is not architecture-only — it emerges from the architecture–optimizer interaction. Holding the Transformer architecture and width schedule fixed, different optimizers turn added FFN width into usable capacity at different rates, changing the *scaling exponents* of the FFN representation spectrum. The separation is sharpest in **hard** spectral rank: globally, AdamW scales at **β ≈ 0.29** while Muon and NorMuon reach **β ≈ 0.80**, even though soft rank grows with width across all optimizers. On rare (TAIL) tokens the gap widens further — AdamW **β ≈ 0.44** vs. Muon **β ≈ 1.02**, a **2.3× larger exponent**. These optimizer-induced shifts **exceed** architectural interventions (attention rank, positional encoding), and **matched pretraining loss does not imply matched representation geometry**.

## Reproduce the headline result in 60 seconds

No GPU, no training, no raw logs. The Colab notebook refits the scaling exponents and regenerates the main figures directly from the committed processed CSVs:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/optimizer-scaling-laws/spectral-scaling-laws/blob/main/notebooks/reproduce_main_figures.ipynb)

To regenerate every committed PDF figure locally:

```bash
pip install -e ".[dev]"
make figures      # writes all figures to results/figures/
```

## Installation

```bash
pip install -e ".[dev]"        # full stack + tests
pip install -e ".[metrics]"    # just the spectral diagnostics (NumPy + PyTorch, CPU-friendly)
```

## Run the diagnostic on your own model

The spectral-rank metrics are not tied to this training stack — point them at any model's FFN activations to measure the spectral capacity *your* optimizer is realizing:

```python
from optimizer_ssl.probe import spectral_rank

metrics = spectral_rank(activations)          # soft rank, hard rank, spectral entropy
# pass token_freq=... to split HEAD / MID / TAIL
```

See [`docs/diagnostic_api.md`](docs/diagnostic_api.md) and runnable CPU examples in [`examples/`](examples/).

## What's in here

```text
optimizer_ssl/            training stack, spectral telemetry, scaling fits, and the standalone diagnostic
  ├── spectra/            covariance spectra, soft/hard rank, frequency-bucketed (HEAD/MID/TAIL) metrics
  ├── analysis/           log parsing, rank aggregation, power-law fits with confidence intervals
  └── probe.py            model-agnostic spectral-rank diagnostic
configs/                  full launch-config grids for every paper experiment (+ reusable components)
scripts/
  ├── train/              single-run and sweep launchers
  ├── analysis/           raw-log → normalized CSV → rank-scaling beta tables
  ├── reproduce/          one-command figure reproduction wrappers
  └── preprocess/         FineWeb10B download and token-frequency preprocessing
results/
  ├── processed/          lightweight released CSVs + token_frequencies.npy
  ├── figures/            publication-quality PDF figures
  └── figure_manifest.csv per-figure provenance: inputs, command, raw-log coverage
notebooks/                Colab-ready headline reproduction
third_party/dion/         vendored Dion / Muon / NorMuon implementations (upstream notices preserved)
docs/                     method, metrics, reproduction, compute, and optimizer notes
tests/                    CPU-safe sanity tests (metrics, fits, parsing, configs, artifacts)
```

## Reproduce the figures

All figure families ship with their processed CSVs, so every committed PDF regenerates without training:

```bash
make figures
# equivalently:
bash scripts/reproduce/reproduce_main_results_from_processed.sh results/processed results/figures
```

To rebuild the 160M processed CSVs from external raw logs, fill in a run manifest based on `results/processed/run_metadata_template.csv`, then:

```bash
bash scripts/reproduce/reproduce_main_results_from_logs.sh \
  results/processed/run_metadata.csv \
  results/processed
```

The raw-log path normalizes the paper's legacy telemetry (`SE_post`, `PR_post`) into the public vocabulary (`soft_rank`, `hard_rank`, `spectral_entropy`). Every figure maps to its inputs, exact command, and raw-log coverage status in [`results/figure_manifest.csv`](results/figure_manifest.csv); see [`docs/release_audit.md`](docs/release_audit.md) for the reproduction tiers (figures and the main parser path reproduce from this repo; the special-family raw logs are external).

## Train from scratch

Released configs cover every experiment family. The table below gives the launch matrix at a glance; optimizer-specific hyperparameters are summarized in [`docs/optimizer_hyperparameters.md`](docs/optimizer_hyperparameters.md) and encoded in [`configs/components/optimizers/`](configs/components/optimizers/).

| Family | Model | Widths | Optimizers / variants | Heads | Steps | GPUs | Config path |
|---|---:|---:|---|---:|---:|---:|---|
| Main width sweep | GPT-2 160M | 1×–8× | AdamW, Muon, NorMuon, Dion r=1/2, Dion r=1/16 | 12 | 6000 | 4 | `configs/paper_runs/main_160m_width_sweep/` |
| 350M TAIL sweep | GPT-2 350M | 1×–4× | AdamW, Muon, NorMuon, Dion r=1/16 | 32 | 8000 | 8 | `configs/paper_runs/main_350m_width_sweep/` |
| Dion rank sweep | GPT-2 160M | 1×–8× | AdamW, Dion r=1/2, r=1/4, r=1/8, r=1/16 | 12 | 6000 | 4 | `configs/paper_runs/dion_rank_sweep/160m/` |
| Matched-loss / extended AdamW | GPT-2 160M | 1×–8× | AdamW 6K, AdamW 12K, Dion r=1/16 | 12 | 6000 / 12000 | 4 | `configs/paper_runs/matched_loss/160m/` |
| Architecture vs optimizer | GPT-2 160M | 1×–8× | AdamW, Muon, NorMuon, Dion r=1/2, Dion r=1/16 | 12 vs 6 | 6000 | 4 | `configs/paper_runs/architecture_vs_optimizer/160m/` |

Each launcher takes an experiment-family argument, for example:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train/run_width_sweep_160m.sh muon
```

The full set of launchers lives in [`scripts/train/`](scripts/train/). Logs and eigen metrics are written under `outputs/` (git-ignored). The figures reproduce CPU-only in seconds.

## Data and token-frequency buckets

The raw FineWeb10B shards are not committed. To download the pretokenized cache and recompute the released frequency table:

```bash
bash scripts/preprocess/prepare_fineweb10b_token_buckets.sh
# smoke test on two shards:
NUM_TRAIN_SHARDS=2 bash scripts/preprocess/prepare_fineweb10b_token_buckets.sh
```

You don't need the raw shards to assign HEAD/MID/TAIL buckets — the released `results/processed/token_frequencies.npy` is sufficient on its own:

```text
results/processed/token_frequencies.npy
results/processed/token_frequency_stats.json
```

## Reproducibility notes

Released configs include `seed: 1337` and an explicit `spectral_error_policy`. The paper used one run per configuration rather than multi-seed sweeps; reported scaling exponents include confidence intervals from the log–log fits, and the seed field can be varied for multi-seed reproductions when compute permits. See [`docs/reproducibility.md`](docs/reproducibility.md).

## Acknowledgments

The GPT training stack and data loader derive from [modded-nanoGPT](https://github.com/KellerJordan/modded-nanogpt); the Muon, NorMuon, and [Dion](https://github.com/microsoft/dion/) optimizers are vendored under [`third_party/dion/`](third_party/dion/) with their upstream READMEs and notices preserved. These optimizers are prior work used as-is, not contributions of this repository. See [`NOTICE.md`](NOTICE.md).

## Citation

If you use this code or its findings, please cite this paper:

```bibtex
@article{jha2026optimizer,
  title   = {Same Architecture, Different Capacity: Optimizer-Induced Spectral Scaling Laws},
  author  = {Nandan Kumar Jha and Brandon Reagen},
  year    = {2026},
  url     = {https://arxiv.org/abs/2605.21803}
}
```

The spectral scaling-laws framework was introduced in our earlier work (EMNLP 2025), which you may also wish to cite:

```bibtex
@inproceedings{jha2025spectral,
  title     = {Spectral Scaling Laws in Language Models: How Effectively Do Feed-Forward Networks Use Their Latent Space?},
  author    = {Jha, Nandan Kumar and Reagen, Brandon},
  booktitle = {Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year      = {2025}
}
```
