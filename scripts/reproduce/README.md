# Reproduction scripts

These wrappers are the script-level entrypoints behind `make figures` and the raw-log parser path.

## Processed CSVs → committed PDFs

```bash
make figures
```

Focused wrappers are available for individual figure families: Dion rank sweep, matched loss, GPT2-350M TAIL, and architecture-vs-optimizer.

## Raw logs → main 160M processed CSVs

After downloading external raw logs and creating `results/processed/run_metadata.csv` from the template:

```bash
bash scripts/reproduce/reproduce_main_results_from_logs.sh \
  results/processed/run_metadata.csv \
  results/processed
```

This raw-log wrapper covers the main parser/aggregation path. The repository keeps small sample logs only for parser tests and examples. For figure-by-figure status, see `results/figure_manifest.csv`.
