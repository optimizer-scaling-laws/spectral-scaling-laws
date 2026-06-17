# Reproduction

This repo supports several reproduction levels. The most complete in-repo path is **processed-CSV figure reproduction**: every committed PDF under `results/figures/` can be regenerated from committed CSVs under `results/processed/`.

## Reproduction tiers

| Tier | Scope | Requires external artifacts? |
|---|---|---|
| Tiny smoke test | CPU/GPU sanity checks for code paths | No |
| Token buckets | Inspect or recompute token-frequency buckets | FineWeb10B only if recomputing |
| Training launches | Launch new 160M/350M-style runs | Training compute and data |
| Processed-CSV figures | Regenerate all committed PDFs | No |
| Raw-log parser path | Rebuild main 160M processed CSVs from `eigen_metrics_logs/` | Yes, external raw logs |
| Special figure raw provenance | Dion rank sweep, matched-loss, 350M TAIL, architecture-vs-optimizer | Yes, external raw logs; processed CSVs and full launch-config grids are committed |

The figure-by-figure source of truth is `results/figure_manifest.csv`.

## Tier -1: tiny GPU smoke test

This verifies the local training/dataloader/eigen-telemetry path without downloading FineWeb10B or running paper-scale models.

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/train/train_tiny_debug.sh
```

This creates synthetic headered shards under `data/tiny_debug/` and runs a 2-layer GPT for 10 steps. It is not a paper-scale experiment.

## Tier 0: inspect released token-frequency buckets

```bash
python -m pytest tests/test_token_frequency_buckets.py
```

This uses the released `results/processed/token_frequencies.npy` artifact and does not require raw FineWeb shards.

## Tier 1: download FineWeb10B and recompute token-frequency buckets

Full paper setting:

```bash
bash scripts/preprocess/prepare_fineweb10b_token_buckets.sh
```

Smoke-test setting:

```bash
NUM_TRAIN_SHARDS=2 bash scripts/preprocess/prepare_fineweb10b_token_buckets.sh
```

Equivalent manual commands:

```bash
python scripts/preprocess/download_fineweb10b.py \
  --output_dir data/fineweb10B \
  --num_train_shards 103

python scripts/preprocess/compute_token_frequencies.py \
  --data_dir data/fineweb10B \
  --output results/processed/token_frequencies.npy \
  --json_out results/processed/token_frequency_stats.json

python scripts/validation/audit_token_frequency_buckets.py \
  --freq_file results/processed/token_frequencies.npy \
  --data_dir data/fineweb10B \
  --format headered \
  --json_out results/processed/token_frequency_audit.json
```

## Tier 2: launch training runs

The paper used DDP-style launches with 4 GPUs for 160M-family runs and 8 GPUs for 350M-family runs.

Single 160M example:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train/train_160m_example.sh
```

160M width sweep for one optimizer:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train/run_width_sweep_160m.sh adamw
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train/run_width_sweep_160m.sh muon
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train/run_width_sweep_160m.sh normuon
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train/run_width_sweep_160m.sh dion_r1_2
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train/run_width_sweep_160m.sh dion_r1_16
```

350M width sweep for one optimizer:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/train/run_width_sweep_350m.sh adamw
```

Full Dion rank-fraction config grid, including AdamW baseline:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train/run_dion_rank_sweep_160m.sh
```

Full matched-loss / extended-AdamW config grid:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train/run_matched_loss_160m.sh
```

All outputs are written under `outputs/`, which is ignored by Git.

## Tier 3: reproduce all committed figures from processed CSVs

Regenerate the included PDFs with:

```bash
make figures
```

or directly:

```bash
bash scripts/reproduce/reproduce_main_results_from_processed.sh \
  results/processed \
  results/figures
```

This produces:

```text
results/figures/global_hard_rank_scaling.pdf
results/figures/global_soft_rank_scaling.pdf
results/figures/frequency_bucket_rank_grid.pdf
results/figures/dion_tail_hard_rank_sweep.pdf
results/figures/dion_tail_soft_rank_sweep.pdf
results/figures/matched_loss_scaling_breakdown.pdf
results/figures/matched_loss_beta_dynamics.pdf
results/figures/matched_loss_pr_trajectories_by_width.pdf
results/figures/tail_350m_hard_rank_scaling.pdf
results/figures/tail_350m_soft_rank_scaling.pdf
results/figures/architecture_vs_optimizer.pdf
```

Focused wrappers are also available:

```bash
bash scripts/reproduce/reproduce_dion_rank_sweep.sh results/processed results/figures
bash scripts/reproduce/reproduce_matched_loss.sh results/processed results/figures
bash scripts/reproduce/reproduce_350m_tail.sh results/processed results/figures
bash scripts/reproduce/reproduce_architecture_vs_optimizer.sh results/processed results/figures
```

Only PDF figures are tracked.

## Tier 4: rebuild main processed CSVs from raw logs

The paper's full raw logs are external artifacts. After downloading them, create a manifest from `results/processed/run_metadata_template.csv` and run:

```bash
bash scripts/reproduce/reproduce_main_results_from_logs.sh \
  results/processed/run_metadata.csv \
  results/processed
```

This parses raw legacy or released eigen logs into `layer_metrics.csv`, then rebuilds the main global and HEAD/MID/TAIL scaling-point CSVs and beta tables.

This raw-log wrapper covers the main parser/aggregation path. All committed figure families have full launch-config grid coverage, but the paper's raw logs for the special figure families remain external. Their committed processed inputs are:

- Dion rank sweep: configs under `configs/paper_runs/dion_rank_sweep/160m/`; processed inputs `dion_tail_rank_sweep_points.csv`, `dion_tail_rank_sweep_beta_table.csv`.
- Matched-loss / extended AdamW: configs under `configs/paper_runs/matched_loss/160m/`; processed inputs `matched_loss_beta_dynamics.csv`, `matched_loss_pr_trajectories.csv`, and `matched_loss_terminal_beta_table.csv`.
- GPT2-350M TAIL: configs under `configs/paper_runs/main_350m_width_sweep/`; processed inputs `tail_350m_rank_scaling_points.csv`, `tail_350m_beta_table.csv`.
- Architecture-vs-optimizer: configs under `configs/paper_runs/architecture_vs_optimizer/160m/`; processed inputs `architecture_vs_optimizer_beta_values.csv`, `architecture_vs_optimizer_comparison.csv`.

External raw-log bundles can be recorded in `results/external_artifacts.md`.
