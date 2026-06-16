# Optimizer-Induced Spectral Scaling Laws

[![CI](https://github.com/optimizer-scaling-laws/spectral-scaling-laws/actions/workflows/ci.yml/badge.svg)](https://github.com/optimizer-scaling-laws/spectral-scaling-laws/actions/workflows/ci.yml)
[![arXiv](https://img.shields.io/badge/arXiv-2605.21803-b31b1b.svg)](https://arxiv.org/abs/2605.21803)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Code, configs, and lightweight reproducibility artifacts for **Same Architecture, Different Capacity: Optimizer-Induced Spectral Scaling Laws**.

- Paper: https://arxiv.org/abs/2605.21803
- Project website: https://optimizer-scaling-laws.github.io/
- Repository: https://github.com/optimizer-scaling-laws/spectral-scaling-laws
- Authors: Nandan Kumar Jha and Brandon Reagen

This repository studies how optimizer geometry changes the realized representation capacity of language models. The central observation is that models with the same architecture and comparable pretraining loss can exhibit different spectral scaling behavior in their FFN representations.

## What is included

- GPT training stack adapted from the Dion codebase.
- Vendored Muon/NorMuon/Dion optimizer support through `third_party/dion/`.
- Spectral telemetry for covariance spectra, soft rank, hard rank/participation ratio, spectral asymmetry, and HEAD/MID/TAIL frequency-bucketed metrics.
- FineWeb10B download and token-frequency preprocessing utilities.
- Released `results/processed/token_frequencies.npy` token-frequency artifact for reproducible token-frequency buckets.
- Full launch-config grids for the 40-run 160M width sweep, the 16-run 350M width sweep, the 40-run 160M Dion rank sweep, the 24-run matched-loss grid, and the 80-run 12-head vs 6-head architecture-vs-optimizer grid.
- Shell scripts for downloading data, recomputing frequency buckets, and launching training runs.
- Lightweight tests for spectral metrics, standalone diagnostics, power-law fitting, config loading, token buckets, and raw-log analysis parsing.
- Analysis scripts that parse submitted-run spectral logs into normalized CSVs, aggregate rank-scaling beta tables, and regenerate all committed PDF figures from processed CSVs.
- `results/figure_manifest.csv`, an audit table mapping each figure to its processed inputs, reproduction command, and raw-log coverage status.

## Repository map

```text
configs/                  paper-run configs and reusable config components
optimizer_ssl/            paper-specific training, spectra, scaling, and utility code
third_party/dion/         vendored Dion/Muon/NorMuon optimizer implementation
scripts/preprocess/       FineWeb10B download and token-frequency preprocessing
scripts/train/            single-run and sweep launch scripts
scripts/validation/       token-frequency audit scripts
scripts/analysis/         raw-log parsing and rank-scaling aggregation
scripts/reproduce/        one-command processed-data reproduction wrappers
results/processed/        lightweight released artifacts and generated CSVs
results/figures/          publication-quality PDF figures only
results/figure_manifest.csv figure provenance and reproduction status
examples/                 CPU-safe standalone diagnostic examples
.github/workflows/        CPU-safe CI checks
docs/                     method, metrics, reproduction, compute, optimizer notes
tests/                    lightweight sanity tests
```

## Quickstart

```bash
pip install -e ".[dev]"
pytest tests/
```

For CPU-only spectral diagnostics without the training stack:

```bash
pip install -e ".[metrics]"
python examples/probe_synthetic_activations.py
```


## One-minute notebook reproduction

Run the headline 160M spectral-scaling analysis directly in Colab, using only the committed processed CSVs:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/optimizer-scaling-laws/spectral-scaling-laws/blob/main/notebooks/reproduce_main_figures.ipynb)

The notebook regenerates the global hard/soft scaling view, the HEAD/MID/TAIL frequency-bucket grid, and the live beta table without training, checkpoints, raw logs, or a GPU.

## Data and token-frequency buckets

The raw FineWeb10B token shards are not committed. To download the full pretokenized FineWeb10B cache and recompute the released frequency table:

```bash
bash scripts/preprocess/prepare_fineweb10b_token_buckets.sh
```

For a smoke test with two train shards:

```bash
NUM_TRAIN_SHARDS=2 bash scripts/preprocess/prepare_fineweb10b_token_buckets.sh
```

The repository also includes the small released artifact:

```text
results/processed/token_frequencies.npy
results/processed/token_frequency_stats.json
```

which is sufficient for HEAD/MID/TAIL bucket assignment without downloading FineWeb10B.

## Training launch examples

Run one 160M example on 4 GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train/train_160m_example.sh
```

Run the full 160M width sweep for one optimizer on 4 GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train/run_width_sweep_160m.sh adamw
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train/run_width_sweep_160m.sh muon
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train/run_width_sweep_160m.sh normuon
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train/run_width_sweep_160m.sh dion_r1_2
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train/run_width_sweep_160m.sh dion_r1_16
```

Run the full 350M width sweep for one optimizer on 8 GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/train/run_width_sweep_350m.sh adamw
```

Run the full Dion rank sweep at 160M on 4 GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train/run_dion_rank_sweep_160m.sh
```

Run the matched-loss / extended-AdamW grid at 160M on 4 GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train/run_matched_loss_160m.sh
```

Run the 12-head vs 6-head architecture-vs-optimizer grid at 160M on 4 GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train/run_architecture_vs_optimizer_160m.sh
```

Logs and eigen metrics are written under `outputs/`, which is ignored by Git.

## Reproduce main rank-scaling figures

The repository includes processed CSVs for the submitted-run figure families: 160M global soft/hard scaling, HEAD/MID/TAIL bucket panels, the Dion TAIL-token rank sweep, the matched-loss / extended-AdamW scaling-breakdown family, the GPT2-350M TAIL-token confirmation plots, and the architecture-vs-optimizer comparison. Regenerate all committed PDFs with:

```bash
make figures
```

or directly with:

```bash
bash scripts/reproduce/reproduce_main_results_from_processed.sh \
  results/processed \
  results/figures
```

To rebuild the main 160M processed CSVs from external raw logs, prepare a run manifest from `results/processed/run_metadata_template.csv`, then run:

```bash
bash scripts/reproduce/reproduce_main_results_from_logs.sh \
  results/processed/run_metadata.csv \
  results/processed
```

The raw-log path normalizes submitted-run legacy telemetry (`SE_post`, `PR_post`) into the public vocabulary (`soft_rank`, `hard_rank`, `spectral_entropy`). All committed figure families now have full launch-config grid coverage; submitted raw logs for the special figure families remain external unless supplied separately. See `results/figure_manifest.csv` and `docs/release_audit.md`.

## Reproducibility notes

Released configs now include `seed: 1337` and an explicit `spectral_error_policy`.
The submitted paper used one run per configuration rather than multi-seed sweeps;
future reproductions can vary the seed field when compute permits. See
`docs/reproducibility.md` for details.

## Standalone diagnostics

The metrics can be used outside this GPT training stack:

```python
from optimizer_ssl.probe import spectral_rank
metrics = spectral_rank(activations)
```

See `docs/diagnostic_api.md` and `examples/`.

## Attribution

This repository builds on the Dion optimizer codebase. Vendored optimizer code is kept under `third_party/dion/` with its upstream README and notices.

## Citation

See `CITATION.cff`. The preferred citation is the arXiv paper:

```bibtex
@misc{jha2026samearchitecturedifferentcapacity,
  title         = {Same Architecture, Different Capacity: Optimizer-Induced Spectral Scaling Laws},
  author        = {Nandan Kumar Jha and Brandon Reagen},
  year          = {2026},
  eprint        = {2605.21803},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2605.21803}
}
```
