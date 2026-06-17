# Optimizer hyperparameters

The frozen launch configs compose model-size, width, telemetry, and optimizer components. This page summarizes the optimizer-specific surface so the main README can stay short while the exact YAML files remain the source of truth.

## Optimizer component files

| Optimizer family | Component file | Scalar optimizer | Learning rate | Momentum / `mu` | Weight decay | Rank fraction | Notes |
|---|---|---:|---:|---:|---:|---:|---|
| AdamW | [`adamw.yaml`](../configs/components/optimizers/adamw.yaml) | AdamW | `0.003` | `0.95` | `0.01` | — | Baseline optimizer family |
| Muon | [`muon.yaml`](../configs/components/optimizers/muon.yaml) | Lion | `0.02` | `0.95` | `0.01` | — | Uses `adjust_lr: spectral_norm` |
| NorMuon | [`normuon.yaml`](../configs/components/optimizers/normuon.yaml) | Lion | `0.02` | `0.95` | `0.01` | — | Muon-family optimizer with the normalized variant used in the paper runs |
| Dion r=1/2 | [`dion_r1_2.yaml`](../configs/components/optimizers/dion_r1_2.yaml) | Lion | `0.02` | `0.95` | `0.01` | `0.5` | `rcqr_oversample: 1.25`, mixed precision enabled |
| Dion r=1/4 | [`dion_r1_4.yaml`](../configs/components/optimizers/dion_r1_4.yaml) | Lion | `0.02` | `0.95` | `0.01` | `0.25` | Used in the Dion rank-fraction sweep |
| Dion r=1/8 | [`dion_r1_8.yaml`](../configs/components/optimizers/dion_r1_8.yaml) | Lion | `0.02` | `0.95` | `0.01` | `0.125` | Used in the Dion rank-fraction sweep |
| Dion r=1/16 | [`dion_r1_16.yaml`](../configs/components/optimizers/dion_r1_16.yaml) | Lion | `0.02` | `0.95` | `0.01` | `0.0625` | Used in the main 160M/350M sweeps and matched-loss comparison |

All optimizer components include:

```yaml
seed: 1337
spectral_error_policy: warn
```

Those fields are repeated in the composed launch configs so reproducibility and telemetry behavior remain explicit at the run level.

## Experiment-family usage

| Experiment family | Optimizer variants used |
|---|---|
| Main 160M width sweep | AdamW, Muon, NorMuon, Dion r=1/2, Dion r=1/16 |
| Main 350M TAIL sweep | AdamW, Muon, NorMuon, Dion r=1/16 |
| Dion rank sweep | AdamW, Dion r=1/2, Dion r=1/4, Dion r=1/8, Dion r=1/16 |
| Matched-loss / extended AdamW | AdamW 6K, AdamW 12K, Dion r=1/16 |
| Architecture vs optimizer | AdamW, Muon, NorMuon, Dion r=1/2, Dion r=1/16, each under 12-head and 6-head configurations |

The exact per-run settings live under [`configs/paper_runs/`](../configs/paper_runs/). When in doubt, treat the run YAMLs as authoritative.
