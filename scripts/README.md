# Scripts

User-facing command wrappers live here.

- `preprocess/`: download FineWeb10B shards, create tiny debug shards, and compute token frequencies.
- `validation/`: audit released token-frequency buckets against raw shards.
- `train/`: launch tiny debug runs, single runs, width sweeps, rank sweeps, matched-loss grids, and Slurm examples.
- `analysis/`: parse raw eigen logs, aggregate rank-scaling tables, and regenerate figure PDFs from processed CSVs.
- `reproduce/`: one-command wrappers for processed-CSV figure reproduction and the main raw-log-to-CSV pipeline.

The shell scripts are intentionally thin wrappers around explicit Python commands and `torchrun` launches. The most common public reproduction command is:

```bash
make figures
```

or directly:

```bash
bash scripts/reproduce/reproduce_main_results_from_processed.sh \
  results/processed \
  results/figures
```
