# Adding a new optimizer

1. Add or vendor the optimizer implementation.
2. Register it in `optimizer_ssl/optimizers/registry.py`.
3. Add a config under `configs/components/optimizers/`.
4. Create frozen run configs under `configs/paper_runs/`.
5. Run training with eigen telemetry enabled.
6. Aggregate logs and fit spectral scaling laws once the analysis pipeline is integrated.
