# Results artifacts

This directory contains lightweight released artifacts, processed analysis CSVs, figure provenance, and generated paper-style PDF figures.

## Figure manifest

```text
figure_manifest.csv
```

This manifest is the audit table for committed figures. It records each PDF, the processed CSVs used to generate it, the reproduction command, and whether raw-log reproduction is represented in-repo or requires external artifacts.

## Included processed artifacts

```text
processed/token_frequencies.npy
processed/token_frequency_stats.json
processed/run_metadata.csv
processed/global_rank_scaling_points.csv
processed/frequency_bucket_rank_scaling_points.csv
processed/main_beta_table.csv
processed/frequency_bucket_beta_table.csv
processed/dion_rank_sweep_run_metadata.csv
processed/dion_tail_rank_sweep_points.csv
processed/dion_tail_rank_sweep_beta_table.csv
processed/dion_rank_sweep_summary.json
processed/matched_loss_run_metadata.csv
processed/matched_loss_terminal_beta_table.csv
processed/matched_loss_beta_dynamics.csv
processed/matched_loss_pr_trajectories.csv
processed/matched_loss_summary.json
processed/tail_350m_run_metadata.csv
processed/tail_350m_rank_scaling_points.csv
processed/tail_350m_beta_table.csv
processed/tail_350m_summary.json
processed/architecture_vs_optimizer_beta_values.csv
processed/architecture_vs_optimizer_comparison.csv
processed/analysis_summary.json
```

`token_frequencies.npy` is the preferred public token-frequency artifact. The older PyTorch `.pt` artifact is intentionally not included in this release.

## Included figures

```text
figures/global_hard_rank_scaling.pdf
figures/global_soft_rank_scaling.pdf
figures/frequency_bucket_rank_grid.pdf
figures/dion_tail_hard_rank_sweep.pdf
figures/dion_tail_soft_rank_sweep.pdf
figures/matched_loss_scaling_breakdown.pdf
figures/matched_loss_beta_dynamics.pdf
figures/matched_loss_pr_trajectories_by_width.pdf
figures/tail_350m_hard_rank_scaling.pdf
figures/tail_350m_soft_rank_scaling.pdf
figures/architecture_vs_optimizer.pdf
```

Only publication-quality PDF figures are tracked in this repository.

## Reproduction status

All committed PDFs can be regenerated from committed processed CSVs via:

```bash
bash scripts/reproduce/reproduce_main_results_from_processed.sh \
  results/processed \
  results/figures
```

The main 160M global and HEAD/MID/TAIL processed CSVs can be rebuilt from raw logs if external raw logs are supplied. Dion rank-sweep, matched-loss, GPT2-350M TAIL, and architecture-vs-optimizer all have full launch-config grid coverage, but their submitted raw logs remain external unless supplied separately.

## Large artifacts not included

`layer_metrics.csv`, `matched_loss_layer_metrics.csv`, `tail_350m_layer_metrics.csv`, full submitted-run raw logs, and checkpoints are not committed because they are large and can be regenerated or distributed as external artifacts. Use `results/external_artifacts.md` to record external raw-log locations, sizes, and checksums.
