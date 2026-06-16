# Analysis inventory

This document tracks how raw telemetry logs, processed CSVs, and public figure scripts relate to the paper results.

## Current analysis coverage

| Figure family | Public reproduction script | Public input level | Processed outputs | Figure outputs | Status |
|---|---|---|---|---|---|
| Global soft/hard rank scaling | `scripts/reproduce/reproduce_main_results_from_processed.sh` | raw logs + processed CSV | `global_rank_scaling_points.csv`, `main_beta_table.csv` | `global_hard_rank_scaling.pdf`, `global_soft_rank_scaling.pdf` | Integrated |
| HEAD/MID/TAIL bucket scaling | `scripts/reproduce/reproduce_main_results_from_processed.sh` | raw logs + processed CSV | `frequency_bucket_rank_scaling_points.csv`, `frequency_bucket_beta_table.csv` | `frequency_bucket_rank_grid.pdf` | Integrated |
| Dion TAIL rank sweep | `scripts/reproduce/reproduce_dion_rank_sweep.sh` | full launch configs + processed CSV | `dion_tail_rank_sweep_points.csv`, `dion_tail_rank_sweep_beta_table.csv` | `dion_tail_hard_rank_sweep.pdf`, `dion_tail_soft_rank_sweep.pdf` | Integrated |
| Matched-loss / extended AdamW breakdown | `scripts/reproduce/reproduce_matched_loss.sh` | full launch configs + processed CSV | `matched_loss_beta_dynamics.csv`, `matched_loss_pr_trajectories.csv`, `matched_loss_terminal_beta_table.csv` | `matched_loss_scaling_breakdown.pdf`, component PDFs | Integrated |
| GPT2-350M TAIL scaling | `scripts/reproduce/reproduce_350m_tail.sh` | full launch configs + processed CSV | `tail_350m_rank_scaling_points.csv`, `tail_350m_beta_table.csv` | `tail_350m_hard_rank_scaling.pdf`, `tail_350m_soft_rank_scaling.pdf` | Integrated |
| Architecture vs optimizer comparison | `scripts/reproduce/reproduce_architecture_vs_optimizer.sh` | full launch configs + processed beta values | `architecture_vs_optimizer_beta_values.csv`, `architecture_vs_optimizer_comparison.csv` | `architecture_vs_optimizer.pdf` | Integrated |

The submitted/legacy scripts that originally produced these figures had names such as `rank_plots_global.py`, `rank_plots_multi.py`, `tail_rank_plots_multi.py`, `beta_dynamics_plot.py`, `pr_trajectories_by_width.py`, `tail_rank_plots_350m.py`, and `plot_optimizer_vs_architecture.py`. The public repo entrypoints above replace those scripts and read normalized CSVs from `results/processed/`.

The legacy plotting scripts parsed `SE_post` and `PR_post`; the cleaned analysis pipeline normalizes these to `spectral_entropy`, `soft_rank`, and `hard_rank` before aggregation.

## Figure manifest

For a compact audit table of committed figures, processed inputs, commands, and raw-log coverage, see:

```text
results/figure_manifest.csv
```

## Raw log layouts

Global logs are expected at:

```text
<run>/eigen_metrics_logs/layer_<layer>_eigen.txt
```

Frequency-bucket logs are expected at:

```text
<run>/eigen_metrics_logs/frequency_tertiles/layer_<layer>_eigen_freq.txt
```

A run manifest points directly to the `eigen_metrics_logs` directory for each run, so the parser does not depend on a particular optimizer/width folder name.

## Main parser/aggregation path

```text
run_metadata.csv
  ↓
parse_eigen_logs_to_csv.py
  ↓
layer_metrics.csv
  ↓
aggregate_rank_scaling.py
  ↓
global_rank_scaling_points.csv
frequency_bucket_rank_scaling_points.csv
main_beta_table.csv
frequency_bucket_beta_table.csv
```

This path is the public raw-log-to-processed path for the main 160M global and HEAD/MID/TAIL rank-scaling figures, assuming external raw logs are supplied.

## Submitted-run compatibility choices

- Submitted logs used legacy metric names: `SE_*`, `PR_*`, `EEE_*`, and `JS`.
- `SE_post` is raw spectral entropy; the public analysis converts it to `soft_rank = exp(SE_post)`.
- `PR_post` maps to `hard_rank`.
- `EEE` and `JS` are ignored because they are not part of the Optimizer-SSL public metric vocabulary.
- Frequency-bucket runs from the submitted version should be labeled `frequency_bucket_reduction=rank0_local` in the manifest.
- Historical submitted logs did not record explicit seeds; use `seed=not_recorded` rather than inventing one.

## Processed-CSV-only / external-log result families

All committed PDF figures are reproduced from committed processed CSVs. Rebuilding processed CSVs for special figure families from submitted raw logs requires external raw-log bundles, even though their full launch-config grids are now committed.

Special families with full launch-config coverage but external submitted logs:

- Dion TAIL rank sweep.
- Matched-loss / extended-AdamW breakdown.
- GPT2-350M TAIL confirmation.
- Architecture-vs-optimizer 12-head vs 6-head comparison.

For these families, the committed processed CSVs are the public source of truth for figure reproduction, and the committed configs define the full launch grids for future re-runs.

## Deferred analysis families

No main-paper figure family currently remains intentionally deferred. Potential future additions should be treated as new result families with their own processed CSVs, figure scripts, reproduction wrapper, manifest rows, and tests.

Appendix-only exploratory plots, per-layer trajectories, or per-checkpoint diagnostics should not be added unless they clarify a paper claim and can be documented cleanly.
