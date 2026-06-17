# Optimizers

This repository studies how optimizer geometry changes realized spectral capacity. AdamW, Muon, NorMuon, and Dion-family optimizers are used as experimental conditions; this repo does not claim Muon, NorMuon, or Dion as new optimizer contributions.

## Optimizer component configs

Optimizer-specific settings are centralized under `configs/components/optimizers/` and repeated in composed launch configs so each historical run is self-contained.

| Optimizer family | Component config | Key fields |
|---|---|---|
| AdamW | `configs/components/optimizers/adamw.yaml` | `optimizer: adamw`, `scalar_opt: adamw`, `lr: 0.003`, `mu: 0.95`, `weight_decay: 0.01` |
| Muon | `configs/components/optimizers/muon.yaml` | `optimizer: muon`, `scalar_opt: lion`, `lr: 0.02`, `mu: 0.95`, `weight_decay: 0.01`, `adjust_lr: spectral_norm` |
| NorMuon | `configs/components/optimizers/normuon.yaml` | `optimizer: normuon`, Muon-style settings with the normalized Muon variant |
| Dion r=1/2 | `configs/components/optimizers/dion_r1_2.yaml` | `optimizer: dion`, `rank_fraction: 0.5`, `rcqr_oversample: 1.25`, mixed precision enabled |
| Dion r=1/4 | `configs/components/optimizers/dion_r1_4.yaml` | `optimizer: dion`, `rank_fraction: 0.25`, `rcqr_oversample: 1.25`, mixed precision enabled |
| Dion r=1/8 | `configs/components/optimizers/dion_r1_8.yaml` | `optimizer: dion`, `rank_fraction: 0.125`, `rcqr_oversample: 1.25`, mixed precision enabled |
| Dion r=1/16 | `configs/components/optimizers/dion_r1_16.yaml` | `optimizer: dion`, `rank_fraction: 0.0625`, `rcqr_oversample: 1.25`, mixed precision enabled |

All component configs also include:

```yaml
seed: 1337
spectral_error_policy: warn
```

The seed is included for future reproducibility. The paper experiments used one training run per configuration, so reported scaling-exponent intervals are log-log fit intervals over width points, not multi-seed confidence intervals.

## Implementation attribution

The repository vendors the Dion/Muon/NorMuon implementation under `third_party/dion/`. Upstream attribution is preserved in `third_party/dion/README.md`, `third_party/dion/NOTICE.md`, and the top-level `NOTICE.md`.

## Adding a new optimizer

1. Add or vendor the optimizer implementation.
2. Register the optimizer in `optimizer_ssl/optimizers/registry.py`.
3. Add a reusable component config under `configs/components/optimizers/`.
4. Create launch configs under `configs/paper_runs/` or `configs/examples/`.
5. Run training with spectral telemetry enabled.
6. Aggregate logs and fit spectral scaling laws with the analysis pipeline in `scripts/analysis/` and `scripts/reproduce/`.

For a new paper figure, also add compact processed CSVs, a plotting entrypoint, a reproduction wrapper, and rows in `results/figure_manifest.csv`.
