# Reproduction

This repo supports several reproduction levels. The most complete in-repo path is **processed-CSV figure reproduction**: every committed PDF under `results/figures/` can be regenerated from committed artifacts under `results/processed/`.

## Reproduction tiers

| Tier | Scope | Requires external artifacts? |
|---|---|---|
| Tiny smoke test | CPU/GPU sanity checks for code paths | No |
| Token buckets | Inspect or recompute token-frequency buckets | FineWeb10B only if recomputing |
| Training launches | Launch new 160M/350M-style runs from frozen configs | Training compute and data |
| Processed-CSV figures | Regenerate all committed PDFs | No |
| Raw-log parser path | Rebuild main 160M processed CSVs from `eigen_metrics_logs/` | Yes, external raw logs |
| Special figure raw provenance | Dion rank sweep, matched-loss, 350M TAIL, architecture-vs-optimizer | Yes, external raw logs; processed CSVs and full launch-config grids are committed |

The figure-by-figure source of truth is `results/figure_manifest.csv`.

## Tiny smoke test

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/train/train_tiny_debug.sh
```

This creates synthetic headered shards under `data/tiny_debug/` and runs a 2-layer GPT for 10 steps. It is not a paper-scale experiment.

## Token-frequency buckets

Inspect the released bucket artifact without raw FineWeb shards:

```bash
python -m pytest tests/test_token_frequency_buckets.py
```

Recompute buckets from FineWeb10B if the raw shards are available:

```bash
bash scripts/preprocess/prepare_fineweb10b_token_buckets.sh
```

For manual download, audit commands, and the shard format, see [`docs/data.md`](data.md).

## Training launches

Frozen launch-config grids cover all paper experiment families. The training scripts write outputs under `outputs/`, which is ignored by Git. A single 160M example is:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train/train_160m_example.sh
```

For full width sweeps, Dion rank sweeps, matched-loss runs, architecture-vs-optimizer runs, and the 350M launcher, see [`docs/training.md`](training.md).

## Regenerate committed PDFs from processed CSVs

```bash
make figures
```

This calls the wrappers in `scripts/reproduce/` and regenerates all committed PDF figures under `results/figures/`. Only PDF figures are tracked; PNG previews are intentionally omitted. The exact processed inputs and command for each figure are listed in `results/figure_manifest.csv`.

## Rebuild main processed CSVs from raw logs

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
