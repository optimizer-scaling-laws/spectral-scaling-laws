# Experiments

## Main width sweeps

The main experiments vary the FFN width multiplier while holding the transformer depth, embedding dimension, sequence length, and dataset fixed.

| Family | Layers | Model dim | Heads | FFN multipliers | Launch GPUs |
|---|---:|---:|---:|---:|---:|
| 160M | 12 | 768 | 12 | 1x–8x | 4 |
| 350M | 24 | 1024 | 32 | 1x–4x | 8 |

Main sweep configs enable frequency-bucketed eigen telemetry using `results/processed/token_frequencies.npy`.

## Optimizers

The launch configs include AdamW, Muon, NorMuon, and Dion. The main Dion comparison uses the low-rank `r=1/16` setting. Additional Dion rank-sweep configs cover `r=1/2`, `1/4`, `1/8`, and `1/16`.



## Analysis scripts

Final plotting and table-generation scripts will be added after the cleaned submitted-run analysis scripts are integrated.
