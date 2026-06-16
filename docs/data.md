# Data

The training code expects GPT-2-tokenized FineWeb-style binary shards. Raw shards are too large for Git and are intentionally ignored.

## Expected layout

```text
data/fineweb10B/
├── fineweb_val_000000.bin
├── fineweb_train_000001.bin
├── ...
└── fineweb_train_000103.bin
```

Each shard has:

```text
256 int32 header entries
followed by uint16 token IDs
```

The expected header values are:

```text
header[0] = 20240520
header[1] = 1
header[2] = number of tokens in the shard
```

## Download FineWeb10B shards

```bash
python scripts/preprocess/download_fineweb10b.py \
  --output_dir data/fineweb10B \
  --num_train_shards 103
```

## Compute token frequencies

```bash
python scripts/preprocess/compute_token_frequencies.py \
  --data_dir data/fineweb10B \
  --output results/processed/token_frequencies.npy \
  --json_out results/processed/token_frequency_stats.json
```

## Audit buckets against raw shards

```bash
python scripts/validation/audit_token_frequency_buckets.py \
  --freq_file results/processed/token_frequencies.npy \
  --data_dir data/fineweb10B \
  --format headered \
  --json_out results/processed/token_frequency_audit.json
```

## Tiny debug data

For a local smoke test that does not download FineWeb10B:

```bash
python scripts/preprocess/create_tiny_debug_data.py --output_dir data/tiny_debug
```

This generates one tiny train shard and one tiny validation shard in the same headered binary format.

## Token-frequency artifact formats

The preferred public artifact is `results/processed/token_frequencies.npy`, an
integer vector with one entry per token ID. `token_frequencies.pt` is retained for
compatibility with earlier training code. Public loaders support `.npy`, `.npz`,
and `.pt`; `.pt` loading uses `weights_only=True` when supported by PyTorch.

`results/processed/token_frequencies.pt` is also included for compatibility with earlier scripts.
