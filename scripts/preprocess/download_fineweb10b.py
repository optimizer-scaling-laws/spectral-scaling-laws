#!/usr/bin/env python3
"""Download GPT-2-tokenized FineWeb10B shards used by the training scripts.

The training code expects headered binary shards under ``data/fineweb10B`` with
names like ``fineweb_train_000001.bin`` and ``fineweb_val_000000.bin``.

Examples
--------
Download the full FineWeb10B cache (103 train shards + validation shard):

    python scripts/preprocess/download_fineweb10b.py --num_train_shards 103

Download a small smoke-test subset:

    python scripts/preprocess/download_fineweb10b.py --num_train_shards 2
"""

from __future__ import annotations

import argparse
from pathlib import Path
from huggingface_hub import hf_hub_download


def download_file(filename: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / filename
    if target.exists():
        print(f"[skip] {target}")
        return
    print(f"[download] {filename} -> {output_dir}")
    hf_hub_download(
        repo_id="kjj0/fineweb10B-gpt2",
        filename=filename,
        repo_type="dataset",
        local_dir=str(output_dir),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download pretokenized FineWeb10B shards.")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/fineweb10B"),
        help="Directory where .bin shards are written. Default: data/fineweb10B",
    )
    parser.add_argument(
        "--num_train_shards",
        type=int,
        default=103,
        help="Number of train shards to download. Full FineWeb10B uses 103.",
    )
    parser.add_argument(
        "--skip_val",
        action="store_true",
        help="Do not download fineweb_val_000000.bin.",
    )
    args = parser.parse_args()

    if not args.skip_val:
        download_file("fineweb_val_000000.bin", args.output_dir)
    for idx in range(1, args.num_train_shards + 1):
        download_file(f"fineweb_train_{idx:06d}.bin", args.output_dir)

    print(f"Done. Shards are available in: {args.output_dir}")


if __name__ == "__main__":
    main()
