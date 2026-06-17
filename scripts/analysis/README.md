# Analysis scripts

These scripts convert raw eigen telemetry logs into processed CSVs and regenerate figure PDFs. For the public one-command path, use:

```bash
make figures
```

`make figures` calls the committed figure builders for global rank scaling, frequency buckets, Dion rank sweep, matched loss, GPT2-350M TAIL, and architecture-vs-optimizer. Each figure script defaults to PDF output; PNGs are not tracked.

## Rebuild main processed CSVs from raw logs

Start from `results/processed/run_metadata_template.csv`, create `results/processed/run_metadata.csv` with one row per run, and run:

```bash
bash scripts/reproduce/reproduce_main_results_from_logs.sh \
  results/processed/run_metadata.csv \
  results/processed
```

This calls `parse_eigen_logs_to_csv.py` and `aggregate_rank_scaling.py`. The parser supports the paper's legacy logs (`SE_post`, `PR_post`) and the released vocabulary (`soft_rank`, `hard_rank`, `spectral_entropy`).

## Individual figure builders

- `make_global_rank_figures.py`
- `make_frequency_bucket_figures.py`
- `make_dion_rank_sweep_figures.py`
- `make_matched_loss_figures.py`
- `make_350m_tail_figures.py`
- `make_architecture_vs_optimizer_figure.py`

Use `results/figure_manifest.csv` for the exact processed inputs and reproduction command for each committed PDF.
