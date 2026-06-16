# Repository map

## Top level

- `train.py`: compatibility entrypoint for `torchrun train.py --config ...` style launches.
- `pyproject.toml`, `requirements.txt`, `environment.yml`: installation metadata.
- `Makefile`: convenience targets for tests, formatting, data preparation, and example launches.
- `LICENSE`, `NOTICE.md`, `CITATION.cff`: licensing, attribution, and citation metadata.

## `configs/`

Frozen launch configs live under `configs/paper_runs/`, including the main width sweeps, Dion rank sweep, matched-loss grid, and architecture-vs-optimizer head-count grid. Reusable reference snippets live under `configs/components/`. Example configs that are not paper-scale live under `configs/examples/`.

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
- `analysis/`: log-schema normalization, raw eigen-log parsers, rank aggregation, and figure helpers.
- `utils/`: small shared utilities.

## `third_party/dion/`

Vendored Dion/Muon/NorMuon implementation with upstream attribution preserved.

## `scripts/`

User-facing commands:

- `preprocess/`: download/create data and compute token frequencies.
- `validation/`: audit released token-frequency buckets against raw shards.
- `train/`: launch single runs, sweeps, and Slurm examples.
- `analysis/`: parse raw eigen logs, aggregate rank-scaling points, and regenerate figure PDFs.
- `reproduce/`: wrappers for processed-CSV figure reproduction and the main raw-log-to-CSV path.

## `results/`

Small released artifacts and generated paper outputs. Raw logs/checkpoints are not committed.

Important files:

- `results/figure_manifest.csv`: figure provenance and reproduction-status audit.
- `results/processed/`: compact processed CSV/NumPy artifacts.
- `results/figures/`: publication-quality PDF figures only.
- `results/external_artifacts.md`: where external raw-log/checkpoint bundles can be recorded.

## `docs/`

Human-facing explanations for method, metrics, data, training, reproduction, compute, artifact formats, and release status.

## `examples/`

CPU-safe standalone diagnostic examples that run metric code without the GPT training stack.

## `tests/`

CPU-safe tests designed to run quickly in CI.

## `notebooks/`

Colab-ready, executed notebooks for lightweight public reproduction. The current notebook, `notebooks/reproduce_main_figures.ipynb`, regenerates the headline 160M global and frequency-bucket scaling views from committed processed CSVs only. It does not train models, load checkpoints, or require a GPU.

## Removed placeholder folders

The empty `assets/` placeholder remains removed. Add it only if it contains polished reusable artifacts.

## `optimizer_ssl/train/` module layout

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
