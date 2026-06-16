# Getting started

This page gives the shortest path from a fresh clone to a working local checkout.

## 1. Install

```bash
conda create -n optimizer-ssl python=3.10 -y
conda activate optimizer-ssl
pip install -e ".[dev]"
```

A CUDA-enabled PyTorch installation is required for training. The CPU-only tests below do not launch training.

## 2. Run lightweight checks

```bash
pytest tests/
```

These tests verify metric definitions, token-frequency bucket construction, config loading, binary shard parsing, and basic scaling-law fitting.

## 3. Run the tiny GPU smoke test

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/train/train_tiny_debug.sh
```

This creates synthetic headered shards under `data/tiny_debug/`, then launches a 2-layer GPT run for 10 steps. It is useful for checking that `torchrun`, the dataloader, AdamW, and eigen telemetry are wired correctly.

## 4. Use the released token-frequency table

The repository includes:

```text
results/processed/token_frequencies.npy
```

This artifact is enough to assign tokens to HEAD/MID/TAIL frequency buckets.

## 5. Recompute the token-frequency table from FineWeb10B

```bash
bash scripts/preprocess/prepare_fineweb10b_token_buckets.sh
```

For a quicker download/path smoke test:

```bash
NUM_TRAIN_SHARDS=2 bash scripts/preprocess/prepare_fineweb10b_token_buckets.sh
```

## 6. Launch paper-scale runs

160M-family runs use 4 GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train/run_width_sweep_160m.sh adamw
```

350M-family runs use 8 GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/train/run_width_sweep_350m.sh adamw
```

See `docs/training.md` for the full launch matrix.
