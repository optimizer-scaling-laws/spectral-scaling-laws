# Reproduction

This repo supports four practical reproduction levels before the final paper-analysis scripts are integrated.


## Tier -1: tiny GPU smoke test

This verifies the local training/dataloader/eigen-telemetry path without downloading FineWeb10B or running paper-scale models.

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/train/train_tiny_debug.sh
```

This creates synthetic headered shards under `data/tiny_debug/` and runs a 2-layer GPT for 10 steps. It is not a paper-scale experiment.

## Tier 0: inspect released token-frequency buckets

```bash
python -m pytest tests/test_token_frequency_buckets.py
```

This uses the released `results/processed/token_frequencies.npy` artifact and does not require raw FineWeb shards.

## Tier 1: download FineWeb10B and recompute token-frequency buckets

Full paper setting:

```bash
bash scripts/preprocess/prepare_fineweb10b_token_buckets.sh
```

Smoke-test setting:

```bash
NUM_TRAIN_SHARDS=2 bash scripts/preprocess/prepare_fineweb10b_token_buckets.sh
```

Equivalent manual commands:

```bash
python scripts/preprocess/download_fineweb10b.py   --output_dir data/fineweb10B   --num_train_shards 103

python scripts/preprocess/compute_token_frequencies.py   --data_dir data/fineweb10B   --output results/processed/token_frequencies.npy   --json_out results/processed/token_frequency_stats.json

python scripts/validation/audit_token_frequency_buckets.py   --freq_file results/processed/token_frequencies.npy   --data_dir data/fineweb10B   --format headered   --json_out results/processed/token_frequency_audit.json
```

## Tier 2: launch training runs

The paper used DDP-style launches with 4 GPUs for 160M-family runs and 8 GPUs for 350M-family runs.

Single 160M example:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train/train_160m_example.sh
```

160M width sweep for one optimizer:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train/run_width_sweep_160m.sh adamw
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train/run_width_sweep_160m.sh muon
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train/run_width_sweep_160m.sh normuon
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train/run_width_sweep_160m.sh dion_r1_16
```

350M width sweep for one optimizer:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/train/run_width_sweep_350m.sh adamw
```

Dion rank sweep:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train/run_dion_rank_sweep_160m.sh
```

All outputs are written under `outputs/`.

## Tier 3: paper figures and tables

The final figure/table reproduction pipeline will be connected after integrating the cleaned submitted-run analysis scripts.
