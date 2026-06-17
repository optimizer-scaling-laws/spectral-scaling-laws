# Matched-loss / extended-AdamW launch configs

This directory contains the full GPT2-160M launch-config grid for the matched-loss / extended-AdamW scaling-breakdown figure family.

The grid covers three run families over FFN width multipliers `1x` through `8x`:

```text
matched_loss/160m/adamw_6k/1x.yaml ... 8x.yaml
matched_loss/160m/adamw_12k/1x.yaml ... 8x.yaml
matched_loss/160m/dion_r1_16/1x.yaml ... 8x.yaml
```

Run-family semantics:

- `adamw_6k`: AdamW baseline, same launch hyperparameters as the main GPT2-160M AdamW width sweep, with `num_iterations: 6000`.
- `adamw_12k`: extended AdamW run, identical to `adamw_6k` except `num_iterations: 12000`.
- `dion_r1_16`: Dion rank `1/16` comparison run, using the Dion optimizer hyperparameters from the Dion rank-sweep configs and `num_iterations: 6000`.

All configs use 4 GPUs, `eigen_log_steps: 200`, frequency-bucket telemetry, and `frequency_bucket_reduction: rank0_local` to match the paper-run telemetry convention.

The paper raw logs remain external. The committed processed CSVs under `results/processed/` reproduce the committed matched-loss PDF figures.
