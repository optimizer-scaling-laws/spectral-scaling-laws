# Training

Training is launched with `torchrun` through shell wrappers in `scripts/train/`.

## Tiny debug run

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/train/train_tiny_debug.sh
```

This is the recommended first GPU smoke test. It creates synthetic data and runs a 2-layer GPT for 10 steps.

## Single run

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
bash scripts/train/run_single.sh configs/paper_runs/main_160m_width_sweep/adamw/1x.yaml
```

## Optimizer hyperparameters

Optimizer-specific settings are centralized in [`configs/components/optimizers/`](../configs/components/optimizers/) and summarized in [`optimizers.md`](optimizers.md). The paper-run YAMLs compose these optimizer components with model-size, width, and telemetry settings.

## 160M width sweep

Paper runs used 4 GPUs.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train/run_width_sweep_160m.sh adamw
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train/run_width_sweep_160m.sh muon
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train/run_width_sweep_160m.sh normuon
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train/run_width_sweep_160m.sh dion_r1_2
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train/run_width_sweep_160m.sh dion_r1_16
```

To run only selected widths:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train/run_width_sweep_160m.sh adamw 1x 4x 8x
```

## 350M width sweep

Paper runs used 8 GPUs.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/train/run_width_sweep_350m.sh adamw
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/train/run_width_sweep_350m.sh muon
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/train/run_width_sweep_350m.sh normuon
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/train/run_width_sweep_350m.sh dion_r1_16
```

## Dion rank sweep

The full GPT2-160M Dion rank-sweep grid contains 40 launch configs: AdamW baseline plus Dion rank fractions `1/2`, `1/4`, `1/8`, and `1/16`, each over widths `1x` through `8x`.

Run the full grid:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train/run_dion_rank_sweep_160m.sh
```

Run one rank family or selected widths:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train/run_dion_rank_sweep_160m.sh dion_r1_2 1x 4x 8x
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train/run_dion_rank_sweep_160m.sh adamw 1x
```

## Matched-loss / extended AdamW grid

The full GPT2-160M matched-loss grid contains 24 launch configs: `adamw_6k`, `adamw_12k`, and `dion_r1_16`, each over widths `1x` through `8x`. `adamw_12k` is identical to `adamw_6k` except for `num_iterations: 12000`; the other two families use `num_iterations: 6000`.

Run the full grid:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train/run_matched_loss_160m.sh
```

Run one family or selected widths:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train/run_matched_loss_160m.sh adamw_12k 1x 4x 8x
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train/run_matched_loss_160m.sh dion_r1_16 1x
```

## Architecture-vs-optimizer head-count grid

The full GPT2-160M architecture-vs-optimizer grid contains 80 launch configs: `heads_12` and `heads_6`, each over AdamW, Muon, NorMuon, Dion rank-1/2, and Dion rank-1/16 across widths `1x` through `8x`. The 6-head ablation changes only `n_head: 6`; all other training, optimizer, data, and telemetry settings match the corresponding 12-head run.

Run the full grid:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train/run_architecture_vs_optimizer_160m.sh
```

Run one head group / optimizer / selected widths:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train/run_architecture_vs_optimizer_160m.sh heads_6 adamw 1x 4x 8x
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train/run_architecture_vs_optimizer_160m.sh heads_12 dion_r1_2 1x
```

## Frequency-bucketed telemetry mode

All frozen paper configs that enable HEAD/MID/TAIL bucket telemetry explicitly use:

```yaml
track_by_frequency: true
frequency_bucket_reduction: rank0_local
token_freq_file: results/processed/token_frequencies.npy
```

`rank0_local` preserves the paper's instrumentation path: pooled spectra are globally reduced across ranks, while bucketed spectra are measured from the rank-0 local shard at each logging step.

For new experiments, users can switch to:

```yaml
frequency_bucket_reduction: distributed_covariance
```

This reduces bucket-specific covariance sufficient statistics across all data-parallel ranks before computing bucket soft/hard ranks.


## Outputs

By default, logs and spectral telemetry are written under:

```text
outputs/
```

This directory is ignored by Git. Use `LOG_DIR=/path/to/logs` to redirect shell-script logs.

## Training-code layout

The top-level `train.py` remains the compatibility entrypoint for all launch commands. Internally, the training package is split into small modules:

```text
optimizer_ssl/train/
├── train_lm.py              # main training loop
├── config.py                # CLI/YAML config loading
├── distributed.py           # DDP / DeviceMesh setup
├── optimizer_factory.py     # AdamW, Muon, NorMuon, Dion construction
├── checkpointing.py         # distributed checkpoint save/load
├── evaluation.py            # validation loss
└── experiment_logging.py    # rank-zero printing and optional WandB logging
```

This refactor is organizational only. It does not change the launch commands, config files, spectral telemetry names, dataset paths, or optimizer settings.

## Seeds and telemetry error policy

Every released config includes:

```yaml
seed: 1337
spectral_error_policy: warn
```

`seed` is applied before model, dataloader, and optimizer construction. The paper
used one run per configuration; this field is exposed so future reproductions can
run seed sweeps when compute permits.

`spectral_error_policy` controls how the spectral tracker handles recoverable
telemetry failures:

- `warn`: log an explicit missing marker and continue.
- `raise`: fail fast; recommended for debugging.
- `nan`: reserve missing/NaN-style markers where possible.
