# Scripts

User-facing command wrappers live here.

- `preprocess/`: download FineWeb10B shards, create tiny debug shards, compute token frequencies.
- `validation/`: audit token-frequency buckets against raw shards.
- `train/`: launch tiny debug runs, single runs, width sweeps, rank sweeps, and Slurm examples.
- `analysis/`: reserved for cleaned analysis entrypoints after the paper plotting scripts are integrated.
- `reproduce/`: reserved for one-command reproduction wrappers after processed result generation is integrated.

The shell scripts are intentionally thin wrappers around explicit Python commands and `torchrun` launches.
