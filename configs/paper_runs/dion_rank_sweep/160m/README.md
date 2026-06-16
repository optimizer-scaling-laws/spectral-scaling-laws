# GPT2-160M Dion rank-sweep configs

This directory contains the frozen launch-config grid for the Dion TAIL-token rank-sweep figure family.

The committed config grid covers 40 runs:

- `adamw/1x.yaml` ... `adamw/8x.yaml` for the dashed AdamW baseline.
- `dion_r1_2/1x.yaml` ... `dion_r1_2/8x.yaml` for Dion rank fraction `1/2`.
- `dion_r1_4/1x.yaml` ... `dion_r1_4/8x.yaml` for Dion rank fraction `1/4`.
- `dion_r1_8/1x.yaml` ... `dion_r1_8/8x.yaml` for Dion rank fraction `1/8`.
- `dion_r1_16/1x.yaml` ... `dion_r1_16/8x.yaml` for Dion rank fraction `1/16`.

All runs use the GPT2-160M setup, 12 layers, 12 attention heads, 4 GPUs,
`num_iterations: 6000`, `eigen_log_steps: 200`, and rank-0-local
frequency-bucket telemetry. Dion rank-fraction runs share the same training
configuration and differ only in the Dion `rank_fraction` setting. Their Dion
optimizer hyperparameters differ from AdamW/Muon/NorMuon as specified in each YAML.
