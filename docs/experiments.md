# Experiments

## Main width sweeps

The main experiments vary the FFN width multiplier while holding the transformer depth, embedding dimension, sequence length, and dataset fixed.

| Family | Layers | Model dim | Heads | FFN multipliers | Launch GPUs |
|---|---:|---:|---:|---:|---:|
| 160M | 12 | 768 | 12 | 1x–8x | 4 |
| 350M | 24 | 1024 | 32 | 1x–4x | 8 |

Main sweep configs enable frequency-bucketed eigen telemetry using `results/processed/token_frequencies.npy`.

## Optimizers

The launch configs include AdamW, Muon, NorMuon, and Dion. The main 160M width sweep includes Dion rank fractions `r=1/2` and `r=1/16`; the 350M TAIL confirmation includes Dion `r=1/16`. The Dion rank-sweep config grid covers AdamW baseline plus Dion rank fractions `r=1/2`, `1/4`, `1/8`, and `1/16` over all 160M width multipliers `1x`–`8x`. All rank-sweep runs use 6000 training steps.

## Matched-loss / extended AdamW

The matched-loss grid covers `adamw_6k`, `adamw_12k`, and `dion_r1_16` over 160M width multipliers `1x`–`8x`. `adamw_12k` is identical to `adamw_6k` except that `num_iterations` is changed from 6000 to 12000.




## Architecture-vs-optimizer head-count comparison

The architecture-vs-optimizer grid compares the optimizer-induced beta gain at the default 12-head architecture against the per-optimizer beta shift induced by reducing the model to 6 attention heads. The full launch grid covers `heads_12` and `heads_6`, AdamW/Muon/NorMuon/Dion rank-1/2/Dion rank-1/16, and 160M width multipliers `1x`--`8x`. The 6-head ablation changes only `n_head: 6`; all other training and optimizer settings match the corresponding 12-head configuration.


## Analysis scripts

Processed CSVs and figure-generation scripts for the 160M global and HEAD/MID/TAIL rank-scaling figures are included under `results/processed/`, `results/figures/`, and `scripts/analysis/`.
