# Optimizer-Induced Spectral Scaling Laws

[![CI](https://github.com/optimizer-scaling-laws/spectral-scaling-laws/actions/workflows/ci.yml/badge.svg)](https://github.com/optimizer-scaling-laws/spectral-scaling-laws/actions/workflows/ci.yml)

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
- Released `results/processed/token_frequencies.npy` and `.pt` token-frequency artifacts for reproducible token-frequency buckets.
- Launchable configs for 160M and 350M width sweeps and Dion rank sweeps.
- Shell scripts for downloading data, recomputing frequency buckets, and launching training runs.
- Lightweight tests for spectral metrics, standalone diagnostics, power-law fitting, config loading, and token buckets.

The paper figure/table analysis pipeline will be integrated in the next analysis pass after the submitted-run log parsers and plotting scripts are cleaned.

## Repository map

```text
configs/                  paper-run configs and reusable config components
optimizer_ssl/            paper-specific training, spectra, scaling, and utility code
third_party/dion/         vendored Dion/Muon/NorMuon optimizer implementation
scripts/preprocess/       FineWeb10B download and token-frequency preprocessing
scripts/train/            single-run and sweep launch scripts
scripts/validation/       token-frequency audit scripts
results/processed/        lightweight released artifacts
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
results/processed/token_frequencies.pt
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
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train/run_width_sweep_160m.sh dion_r1_16
```

Run the full 350M width sweep for one optimizer on 8 GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/train/run_width_sweep_350m.sh adamw
```

Run Dion rank sweep at 160M on 4 GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train/run_dion_rank_sweep_160m.sh
```

Logs and eigen metrics are written under `outputs/`, which is ignored by Git.

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
