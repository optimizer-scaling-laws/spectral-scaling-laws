# Configs

This directory contains frozen paper-run configs, reusable config components, and tiny examples.

## `paper_runs/`

Launchable YAML files for the paper experiments:

- `main_160m_width_sweep/`: 160M-family width sweep, 4-GPU launch scripts, FFN multipliers `1x` to `8x`.
- `main_350m_width_sweep/`: 350M-family width sweep, 8-GPU launch scripts, FFN multipliers `1x` to `4x`.
- `dion_rank_sweep/160m/`: Dion rank-fraction sweep at 160M.

## `components/`

Reusable model, optimizer, data, and analysis snippets. These are documentation/reference components; the public paper-run YAMLs are self-contained and do not require config composition at runtime.

## `examples/`

Small configs for smoke tests. `examples/tiny_debug.yaml` is not a paper-scale experiment.
## Frequency-bucket reduction

Paper configs with token-frequency telemetry use:

```yaml
track_by_frequency: true
frequency_bucket_reduction: rank0_local
```

`rank0_local` reproduces the submitted-paper bucket telemetry. The code also supports `distributed_covariance` for new experiments that should reduce HEAD/MID/TAIL covariance statistics across all data-parallel ranks before computing soft/hard ranks.


## Cleanup rules

- `save_every` is represented as `checkpoint_freq`.
- Dion `oversample` is represented as `rcqr_oversample`.
- token-frequency paths point to `results/processed/token_frequencies.npy`.
- private WandB logging is disabled by default with `no_wandb: true`.
- outputs are written under `outputs/`, which is ignored by Git.
- public metrics use `soft_rank` and `hard_rank` terminology.
