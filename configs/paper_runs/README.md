# Paper-run launch configs

This directory contains self-contained YAML configs for launching Optimizer-SSL training and telemetry runs.

## Main width sweeps

- `main_160m_width_sweep/`: 160M-family GPT runs, FFN multipliers `1x` through `8x`. Launch scripts use 4 GPUs.
- `main_350m_width_sweep/`: 350M-family GPT runs, FFN multipliers `1x` through `4x`. Launch scripts use 8 GPUs.

The main sweep configs enable frequency-bucketed eigen telemetry with `results/processed/token_frequencies.npy`, producing HEAD/MID/TAIL soft-rank and hard-rank metrics.

## Dion rank sweep configs

- `dion_rank_sweep/160m/`: full 40-run GPT2-160M launch-config grid for the Dion TAIL-token rank-sweep figure family.

The grid contains:

```text
dion_rank_sweep/160m/adamw/1x.yaml ... 8x.yaml
dion_rank_sweep/160m/dion_r1_2/1x.yaml ... 8x.yaml
dion_rank_sweep/160m/dion_r1_4/1x.yaml ... 8x.yaml
dion_rank_sweep/160m/dion_r1_8/1x.yaml ... 8x.yaml
dion_rank_sweep/160m/dion_r1_16/1x.yaml ... 8x.yaml
```

All Dion rank-fraction runs use the same training configuration and differ only in the Dion `rank_fraction` setting. Their optimizer hyperparameters are intentionally different from AdamW/Muon/NorMuon, as encoded in each YAML. All runs use `num_iterations: 6000`, `eigen_log_steps: 200`, 4 GPUs, and `frequency_bucket_reduction: rank0_local`.

## Matched-loss configs

- `matched_loss/160m/`: full 24-run GPT2-160M launch-config grid for the matched-loss / extended-AdamW figure family.

The grid contains:

```text
matched_loss/160m/adamw_6k/1x.yaml ... 8x.yaml
matched_loss/160m/adamw_12k/1x.yaml ... 8x.yaml
matched_loss/160m/dion_r1_16/1x.yaml ... 8x.yaml
```

`adamw_12k` is identical to `adamw_6k` except for `num_iterations: 12000`. `dion_r1_16` uses the Dion rank-1/16 optimizer hyperparameters and `num_iterations: 6000`. Paper-run raw logs remain external, but the full launch-config grid is committed.

## Architecture-vs-optimizer head-count configs

- `architecture_vs_optimizer/160m/`: full 80-run GPT2-160M launch-config grid for the 12-head vs 6-head architecture comparison.

The grid contains both `heads_12` and `heads_6`, each over:

```text
adamw/1x.yaml ... 8x.yaml
muon/1x.yaml ... 8x.yaml
normuon/1x.yaml ... 8x.yaml
dion_r1_2/1x.yaml ... 8x.yaml
dion_r1_16/1x.yaml ... 8x.yaml
```

The 6-head configs are identical to the corresponding 12-head configs except for `n_head: 6`. Paper-run raw logs remain external, but the full launch-config grid is committed.

## Metadata fields

Each frozen config includes human-readable metadata:

```yaml
run_name:
paper_experiment:
model_scale:
width_multiplier:
optimizer_name:
optimizer_variant:
num_gpus_used:
```

The training script ignores metadata it does not need; the fields exist to make configs easier to inspect and easier for future analysis scripts to parse.

All configs disable WandB by default (`no_wandb: true`) and write eigen metrics under `outputs/eigen_metrics/...`.
