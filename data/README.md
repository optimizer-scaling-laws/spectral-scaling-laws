# Data directory

Raw FineWeb10B token shards are not committed to this repository.

Download the GPT-2-tokenized shards used by the training scripts with:

```bash
python scripts/preprocess/download_fineweb10b.py --output_dir data/fineweb10B --num_train_shards 103
```

The expected layout is:

```text
data/fineweb10B/
├── fineweb_val_000000.bin
├── fineweb_train_000001.bin
├── ...
└── fineweb_train_000103.bin
```

Each `.bin` file is a headered token shard: 256 `int32` header entries followed by `uint16` GPT-2 token IDs.

For a no-download smoke test, create synthetic tiny shards with:

```bash
python scripts/preprocess/create_tiny_debug_data.py --output_dir data/tiny_debug
```
