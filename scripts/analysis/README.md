# Analysis scripts

These scripts convert raw eigen telemetry logs into processed CSVs and regenerate figure PDFs.

## Reproduce all figures from processed CSVs

```bash
bash scripts/reproduce/reproduce_main_results_from_processed.sh \
  results/processed \
  results/figures
```

This calls:

```text
scripts/analysis/make_global_rank_figures.py
scripts/analysis/make_frequency_bucket_figures.py
scripts/analysis/make_dion_rank_sweep_figures.py
scripts/analysis/make_matched_loss_figures.py
scripts/analysis/make_350m_tail_figures.py
scripts/analysis/make_architecture_vs_optimizer_figure.py
```

Each figure script defaults to PDF output. PNGs are not tracked.

## Rebuild main processed CSVs from raw logs

Start from:

```text
results/processed/run_metadata_template.csv
```

Create `results/processed/run_metadata.csv` with one row per run. Each row should point `log_dir` to that run's `eigen_metrics_logs` directory.

Then run:

```bash
bash scripts/reproduce/reproduce_main_results_from_logs.sh \
  results/processed/run_metadata.csv \
  results/processed
```

This calls:

```text
scripts/analysis/parse_eigen_logs_to_csv.py
scripts/analysis/aggregate_rank_scaling.py
```

The parser supports the paper's legacy logs (`SE_post`, `PR_post`) and the released log vocabulary (`soft_rank`, `hard_rank`, `spectral_entropy`). This raw-log path is the maintained in-repo path for the main 160M global and HEAD/MID/TAIL rank-scaling analyses. Special figure families are reproduced from committed processed CSVs unless their external raw logs are supplied.

## Dion rank-sweep figures

```bash
python scripts/analysis/make_dion_rank_sweep_figures.py \
  --points results/processed/dion_tail_rank_sweep_points.csv \
  --betas results/processed/dion_tail_rank_sweep_beta_table.csv \
  --out-dir results/figures \
  --formats pdf
```

## Matched-loss / extended-AdamW figures

```bash
python scripts/analysis/make_matched_loss_figures.py \
  --beta-dynamics results/processed/matched_loss_beta_dynamics.csv \
  --pr-trajectories results/processed/matched_loss_pr_trajectories.csv \
  --out-dir results/figures \
  --formats pdf
```

## GPT2-350M TAIL-token figures

```bash
python scripts/analysis/make_350m_tail_figures.py \
  --points results/processed/tail_350m_rank_scaling_points.csv \
  --betas results/processed/tail_350m_beta_table.csv \
  --out-dir results/figures \
  --formats pdf
```

## Architecture-vs-optimizer figure

```bash
python scripts/analysis/make_architecture_vs_optimizer_figure.py \
  --comparison results/processed/architecture_vs_optimizer_comparison.csv \
  --out-dir results/figures \
  --formats pdf
```
