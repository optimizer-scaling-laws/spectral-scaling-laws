# Compute budget

The full paper experiments require multi-GPU training; they are not laptop reproductions. The repository is tiered (see [`release_audit.md`](release_audit.md)) so that figures and processed results can be verified without rerunning training: every committed figure regenerates from compact CSVs **CPU-only in seconds**, while the runs below produce the underlying data.

## Models

| Scale | d_model | Layers | Heads | Seq len | FFN width sweep (D) | Batch (seq) | Steps | Tokens/run |
|---|---:|---:|---:|---:|---|---:|---:|---:|
| GPT-2 160M | 768 | 12 | 12 | 512 | 768 → 6144 (×1–8) | 1024 | 6000 | ≈3.1B |
| GPT-2 350M | 1024 | 24 | 32 | 512 | 1024 → 4096 (×1–4) | 1024 | 8000 | ≈4.2B |

All runs train on FineWeb10B. FFN width is swept via the FFN multiplier (`ffn_mult`); the model is otherwise fixed within each scale.

## Hardware

We use **4×NVIDIA RTX 3090 (24 GB each)** for the 160M-scale experiments and **8×NVIDIA RTX 3090** for the 350M-scale experiments.

## Experiments

The main FFN-width sweep comprises **40 training runs at 160M** (five optimizers × eight widths) and **16 runs at 350M** (four optimizers × four widths). The 160M sweep covers AdamW, Muon, NorMuon, Dion(1/2), and Dion(1/16); the 350M sweep covers the same set without Dion(1/2). These are complemented by the Dion rank-fraction sweep, matched-loss / extended-AdamW comparison, GPT-2 350M TAIL-token confirmation, and 12-head vs. 6-head architecture-vs-optimizer comparison. All committed figure families regenerate from compact processed CSVs; full raw logs and checkpoints remain external.

Regenerating any committed figure from the processed CSVs needs no GPU and finishes in seconds.
