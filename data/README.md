# Data directory

Raw FineWeb10B token shards are not committed to this repository. This directory is a local mount point for downloaded or synthetic data used by the training scripts.

For the full FineWeb10B layout, download commands, token-frequency computation, and bucket-audit commands, see [`docs/data.md`](../docs/data.md).

For a no-download smoke test, create tiny synthetic shards with:

```bash
python scripts/preprocess/create_tiny_debug_data.py --output_dir data/tiny_debug
```
