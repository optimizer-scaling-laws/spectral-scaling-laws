# Scripts

User-facing command wrappers live here.

- `preprocess/`: download FineWeb10B shards, create tiny debug shards, and compute token frequencies.
- `validation/`: audit released token-frequency buckets against raw shards.
- `train/`: launch tiny debug runs, single runs, width sweeps, rank sweeps, matched-loss grids, architecture grids, and Slurm examples.
- `analysis/`: parse eigen logs, aggregate rank-scaling tables, and regenerate figure PDFs from processed CSVs.
- `reproduce/`: one-command wrappers for processed-CSV figure reproduction and the main raw-log-to-CSV path.

Common entrypoints:

```bash
make figures      # regenerate committed PDFs from processed CSVs
make check        # lint + CPU-safe tests
```

For the full reproduction ladder, see [`docs/reproduction.md`](../docs/reproduction.md).
