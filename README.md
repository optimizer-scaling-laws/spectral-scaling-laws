# Optimizer-Induced Spectral Scaling Laws

[![CI](https://github.com/optimizer-scaling-laws/spectral-scaling-laws/actions/workflows/ci.yml/badge.svg)](https://github.com/optimizer-scaling-laws/spectral-scaling-laws/actions/workflows/ci.yml)
[![arXiv](https://img.shields.io/badge/arXiv-2605.21803-b31b1b.svg)](https://arxiv.org/abs/2605.21803)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official code for paper **[Same Architecture, Different Capacity: Optimizer-Induced Spectral Scaling Laws](https://arxiv.org/abs/2605.21803)**.

<div align="center">
  <a href="https://arxiv.org/abs/2605.21803"><img src="https://arxiv.org/static/browse/0.3.4/images/icons/favicon-32x32.png" height="16" width="16" style="vertical-align:middle"> <b>arXiv</b></a> &nbsp;|&nbsp;
  <a href="https://optimizer-scaling-laws.github.io/">🌐 <b>Project Website</b></a> &nbsp;|&nbsp;
  <a href="https://colab.research.google.com/github/optimizer-scaling-laws/spectral-scaling-laws/blob/main/notebooks/reproduce_main_figures.ipynb"><img src="https://colab.research.google.com/img/colab_favicon_256px.png" height="16" width="16" style="vertical-align:middle"> <b>Reproduce in Colab</b></a>
</div>

<br>

<p align="center">
  <img src="assets/teaser.png" width="900" alt="Spectral scaling exponents depend on optimizer choice (soft and hard spectral rank vs FFN width)"><br>
  <em><strong>Figure 1.</strong> Spectral scaling exponents depend on optimizer choice: soft (left) and hard (right) spectral rank as a function of FFN width in GPT-2 160M. AdamW exhibits the largest hard–soft asymmetry (Δ = β<sub>soft</sub> − β<sub>hard</sub> = 0.37), indicating concentrated eigenspectra. Muon and Dion (1/2) reduce this asymmetry to Δ ≈ 0.14. Moreover, hard-rank scaling exhibits stronger dependence on optimizer choice, compared to soft-rank scaling.</em>
</p>

## TL;DR --> Same architecture, different optimizer, different capacity

Realized representation capacity is not architecture-only, rather it emerges from the architecture–optimizer interaction. Holding the Transformer architecture and width schedule fixed, different optimizers turn added FFN width into usable capacity at different rates, changing the *scaling exponents* of the realized capacity. This distinction is more visible in **hard** spectral rank:  AdamW scales at **β ≈ 0.29** while Muon and NorMuon reach **β ≈ 0.80**, even though soft rank grows with width across all optimizers. On rare (TAIL) tokens the gap widens further-- AdamW **β ≈ 0.44** vs. Muon **β ≈ 1.02**, a **2.3× larger exponent**. These optimizer-induced shifts can **exceed** the effect of architectural interventions (e.g., attention rank intervention).

## Reproduce main results

The Colab notebook re-fits the scaling exponents and produce the main figures directly from the raw logs generated at pre-trianing:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/optimizer-scaling-laws/spectral-scaling-laws/blob/main/notebooks/reproduce_main_figures.ipynb)

To re-generate other plots:

```bash
pip install -e ".[metrics,analysis]"
make figures      # writes all figures to results/figures/
```

## Installation

`pyproject.toml` contents all the dependencies needed for all the trianing runs and analyzing the spectral metrics:

| Extra | Command | For |
|---|---|---|
| `metrics` | `pip install -e ".[metrics]"` | CPU diagnostics (NumPy + PyTorch) |
| `analysis` | `pip install -e ".[analysis]"` | figure scripts and Colab notebook |
| `train` | `pip install -e ".[train]"` | training configurations |
| `dev` | `pip install -e ".[dev]"` | tests |
| `metrics,analysis` | `pip install -e ".[metrics,analysis]"` | figure reproduction without training runs|

## Run the diagnostic on your own model

The spectral-rank metrics can be reused across other training stack:

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
configs/                  full launch-config grids
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
docs/                     data, metrics, reproduction, training, compute, and optimizer notes
tests/                    CPU-safe sanity tests (metrics, fits, parsing, configs, artifacts)
```

## Reproduce the plots

All figures are accompanied with their logs in CSVs, so they can be re-generated without training:

```bash
make figures
# equivalently:
bash scripts/reproduce/reproduce_main_results_from_processed.sh results/processed results/figures
```

To generate the 160M processed CSVs from external raw logs, get the data in `results/processed/run_metadata_template.csv`, then:

```bash
bash scripts/reproduce/reproduce_main_results_from_logs.sh \
  results/processed/run_metadata.csv \
  results/processed
```

The raw-log path normalizes the paper-run telemetry schema (`SE_post`, `PR_post`) into the public vocabulary (`soft_rank`, `hard_rank`, `spectral_entropy`). Every figure maps to its inputs, exact command, and raw-log coverage status in [`results/figure_manifest.csv`](results/figure_manifest.csv); see [`docs/reproduction.md`](docs/reproduction.md) for the reproduction tiers and release boundary.

## Train from scratch

Released configs cover every experiment family. Per-optimizer hyperparameters are summarized in [`docs/optimizers.md`](docs/optimizers.md) and encoded in [`configs/components/optimizers/`](configs/components/optimizers/); the configs themselves live under `configs/paper_runs/<family>/`, and full grid details (model dims, steps, token budget) are in [`docs/compute_budget.md`](docs/compute_budget.md). Every launcher lives in [`scripts/train/`](scripts/train/) and runs as `CUDA_VISIBLE_DEVICES=… bash scripts/train/<script> [optimizer]`.

| Family | Scale | Optimizers | Widths | GPUs | Launch |
|---|:--:|---|:--:|:--:|---|
| Main width sweep | 160M | AdamW · Muon · NorMuon · Dion(1/2, 1/16) | 1×–8× | 4 | `run_width_sweep_160m.sh [opt]` |
| Main width sweep | 350M | AdamW · Muon · NorMuon · Dion(1/16) | 1×–4× | 8 | `run_width_sweep_350m.sh [opt]` |
| Dion rank sweep | 160M | AdamW · Dion(1/2, 1/4, 1/8, 1/16) | 1×–8× | 4 | `run_dion_rank_sweep_160m.sh` |
| Matched-loss | 160M | AdamW(6K · 12K) · Dion(1/16) | 1×–8× | 4 | `run_matched_loss_160m.sh` |
| Architecture × optimizer | 160M | 5 optimizers × {12-head · 6-head} | 1×–8× | 4 | `run_architecture_vs_optimizer_160m.sh` |

For example, the AdamW arm of the 160M sweep on four GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train/run_width_sweep_160m.sh adamw
```

Logs and eigen metrics are written under `outputs/` (git-ignored); the figures themselves reproduce CPU-only in seconds.

## Data and token-frequency buckets

To download the pretokenized cache and recompute the released frequency table:

```bash
bash scripts/preprocess/prepare_fineweb10b_token_buckets.sh
# smoke test on two shards:
NUM_TRAIN_SHARDS=2 bash scripts/preprocess/prepare_fineweb10b_token_buckets.sh
```

Raw shards are not needed to assign HEAD/MID/TAIL buckets, the released `results/processed/token_frequencies.npy` can be used:

```text
results/processed/token_frequencies.npy
results/processed/token_frequency_stats.json
```

## Reproducibility notes

Released configs include `seed: 1337` and an explicit `spectral_error_policy`. In the paper, reported scaling exponents include confidence intervals from the log–log fits. See [`docs/reproduction.md`](docs/reproduction.md).

## Acknowledgments

The GPT training stack and data loader are used from [modded-nanoGPT](https://github.com/KellerJordan/modded-nanogpt), and the Muon, NorMuon, and [Dion](https://github.com/microsoft/dion/) optimizers are used from [`third_party/dion/`](third_party/dion/). These optimizers are prior work used as-is, not contributions of this repository. See [`NOTICE.md`](NOTICE.md).

## Citation

If you find this code or insights useful, please consider citing this paper:

```bibtex
@article{jha2026optimizer,
  title   = {Same Architecture, Different Capacity: Optimizer-Induced Spectral Scaling Laws},
  author    = {Jha, Nandan Kumar and Reagen, Brandon},
  year    = {2026},
  url     = {https://arxiv.org/abs/2605.21803}
}
```

The original spectral scaling-laws was introduced in our earlier work (EMNLP 2025):

```bibtex
@inproceedings{jha2025spectral,
  title     = {Spectral Scaling Laws in Language Models: How Effectively Do Feed-Forward Networks Use Their Latent Space?},
  author    = {Jha, Nandan Kumar and Reagen, Brandon},
  booktitle = {Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year      = {2025}
}
```
