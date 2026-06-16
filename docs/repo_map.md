# Repository map

## Top level

- `train.py`: compatibility entrypoint for `torchrun train.py --config ...` style launches.
- `pyproject.toml`, `requirements.txt`, `environment.yml`: installation metadata.
- `Makefile`: convenience targets for tests, formatting, data preparation, and example launches.
- `LICENSE`, `NOTICE.md`, `CITATION.cff`: licensing, attribution, and citation metadata.

## `configs/`

Frozen launch configs live under `configs/paper_runs/`. Reusable reference snippets live under `configs/components/`. Example configs that are not paper-scale live under `configs/examples/`.

## `data/`

Local data mount point. Raw `.bin` shards are intentionally ignored by Git. See `docs/data.md`.

## `optimizer_ssl/`

The reusable Python package for this project:

- `data/`: headered shard loading and token-frequency bucket utilities.
- `models/`: GPT model and dataloader utilities.
- `optimizers/`: thin registry layer for AdamW and vendored Dion-family optimizers.
- `train/`: training entrypoint.
- `spectra/`: modular spectral telemetry code:
  - `covariance.py`: activation covariance and eigenvalue utilities.
  - `effective_rank.py`: spectral entropy, soft rank, and hard rank.
  - `frequency_metrics.py`: HEAD/MID/TAIL token-frequency buckets and Tail Integrity Index helpers.
  - `spectral_asymmetry.py`: soft-minus-hard asymmetry helpers.
  - `tracker.py`: `GPTEigenMetricsTracker`, the training-time hook/logging orchestrator.
  - `eigen_metrics_gpt.py`: compatibility shim for older imports.
- `scaling/`: power-law fitting helpers.
- `analysis/`: reserved for cleaned paper-analysis code.
- `utils/`: small shared utilities.

## `third_party/dion/`

Vendored Dion/Muon/NorMuon implementation with upstream attribution preserved.

## `scripts/`

User-facing commands:

- `preprocess/`: download/create data and compute token frequencies.
- `validation/`: audit released token-frequency buckets against raw shards.
- `train/`: launch single runs, sweeps, and Slurm examples.
- `analysis/`: reserved for cleaned analysis entrypoints.
- `reproduce/`: reserved for one-command reproduction wrappers.

## `results/`

Small released artifacts and generated paper outputs. Raw logs/checkpoints are not committed.

## `docs/`

Human-facing explanations for method, metrics, data, training, reproduction, and troubleshooting.

## `tests/`

CPU-safe tests designed to run quickly in CI.

### `optimizer_ssl/train/` module layout

The training package is intentionally modularized around the main loop without turning the repo into a large framework:

- `train_lm.py`: owns the main training loop and remains the implementation behind top-level `train.py`.
- `config.py`: `Hyperparameters`, CLI parsing, YAML merging, and lightweight validation.
- `distributed.py`: DDP/DeviceMesh initialization and distributed cleanup helpers.
- `optimizer_factory.py`: construction of AdamW, Muon, NorMuon, Dion, and related optimizer variants.
- `checkpointing.py`: distributed save/load wrapper around `torch.distributed.checkpoint`.
- `evaluation.py`: validation-loss estimation.
- `experiment_logging.py`: rank-zero printing, run-name construction, optional WandB setup, and metric logging.

The public launch surface is unchanged: use `torchrun ... train.py --config ...` or the shell scripts under `scripts/train/`.

## Citation and project links

See `docs/citation.md` and `CITATION.cff` for the paper, project website, repository URL, and citation metadata.

## Pre-analysis hardening additions

- `optimizer_ssl/probe.py` provides the model-agnostic CPU-safe spectral diagnostic API.
- `examples/` contains small scripts that run the metric code without the GPT training stack.
- `.github/workflows/ci.yml` runs CPU-safe tests on pull requests.
- `docs/reproducibility.md` documents seeds, frequency-bucket reduction modes, and artifact formats.
