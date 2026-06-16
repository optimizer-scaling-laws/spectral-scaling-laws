# Paper-run launch configs

This directory contains self-contained YAML configs for launching the Optimizer-SSL training and telemetry runs.

## Main width sweeps

- `main_160m_width_sweep/`: 160M-family GPT runs, FFN multipliers `1x` through `8x`. Launch scripts use 4 GPUs.
- `main_350m_width_sweep/`: 350M-family GPT runs, FFN multipliers `1x` through `4x`. Launch scripts use 8 GPUs.

The main sweep configs enable frequency-bucketed eigen telemetry with `results/processed/token_frequencies.npy`, producing HEAD/MID/TAIL soft-rank and hard-rank metrics.

## Dion rank sweep

- `dion_rank_sweep/160m/`: 160M, 4x FFN, Dion rank fractions `1/2`, `1/4`, `1/8`, and `1/16`.

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
