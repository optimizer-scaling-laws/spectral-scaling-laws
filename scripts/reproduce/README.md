# Reproduction scripts

## From processed CSVs

Regenerate all committed PDF figures, including the main 160M rank-scaling figures, Dion TAIL-token rank sweep, matched-loss figures, GPT2-350M TAIL plots, and architecture-vs-optimizer comparison:

```bash
make figures
```

or directly:

```bash
bash scripts/reproduce/reproduce_main_results_from_processed.sh \
  results/processed \
  results/figures
```

Focused wrappers:

```bash
bash scripts/reproduce/reproduce_dion_rank_sweep.sh results/processed results/figures
bash scripts/reproduce/reproduce_matched_loss.sh results/processed results/figures
bash scripts/reproduce/reproduce_350m_tail.sh results/processed results/figures
bash scripts/reproduce/reproduce_architecture_vs_optimizer.sh results/processed results/figures
```

## From raw logs

After downloading the external raw logs and creating a run manifest:

```bash
bash scripts/reproduce/reproduce_main_results_from_logs.sh \
  results/processed/run_metadata.csv \
  results/processed
```

This raw-log wrapper covers the main parser/aggregation path. Full raw logs should be distributed as external artifacts. The repository keeps small sample logs only for parser tests and examples.

For figure-by-figure status, see `results/figure_manifest.csv`.
