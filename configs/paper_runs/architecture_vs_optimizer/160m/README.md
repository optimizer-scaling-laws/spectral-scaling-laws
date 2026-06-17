# Architecture-vs-optimizer configs — GPT2-160M

This directory contains the full 80-run launch-config grid for the head-count architecture comparison used in the architecture-vs-optimizer figure.

```text
architecture_vs_optimizer/160m/
├── heads_12/
│   ├── adamw/1x.yaml ... 8x.yaml
│   ├── muon/1x.yaml ... 8x.yaml
│   ├── normuon/1x.yaml ... 8x.yaml
│   ├── dion_r1_2/1x.yaml ... 8x.yaml
│   └── dion_r1_16/1x.yaml ... 8x.yaml
└── heads_6/
    ├── adamw/1x.yaml ... 8x.yaml
    ├── muon/1x.yaml ... 8x.yaml
    ├── normuon/1x.yaml ... 8x.yaml
    ├── dion_r1_2/1x.yaml ... 8x.yaml
    └── dion_r1_16/1x.yaml ... 8x.yaml
```

The 6-head ablations keep the GPT2-160M model width, depth, FFN multiplier, optimizer hyperparameters, data, token-frequency telemetry, and 6000-step training schedule fixed. The only architectural field changed relative to the corresponding 12-head run is:

```yaml
n_head: 6
```

All configs use:

```yaml
model_scale: 160m
model_dim: 768
n_layer: 12
num_iterations: 6000
eigen_log_steps: 200
num_gpus_used: 4
track_by_frequency: true
frequency_bucket_reduction: rank0_local
```

The committed figure is regenerated from processed beta values in `results/processed/architecture_vs_optimizer_*.csv`. These configs provide the historical launch grid; paper raw logs remain external.
