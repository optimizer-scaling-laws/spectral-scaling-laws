#!/usr/bin/env python3
"""Create tiny headered token shards for smoke-testing the training pipeline.

The generated shards match the FineWeb binary format expected by the training
loader: 256 int32 header entries followed by uint16 token IDs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

MAGIC = 20240520
VERSION = 1
HEADER_SIZE_INTS = 256
VOCAB_SIZE = 50304


def write_shard(path: Path, num_tokens: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = np.zeros(HEADER_SIZE_INTS, dtype=np.int32)
    header[0] = MAGIC
    header[1] = VERSION
    header[2] = num_tokens
    # A deterministic but non-constant stream keeps the dataloader/model path realistic.
    tokens = rng.integers(0, VOCAB_SIZE, size=num_tokens, dtype=np.uint16)
    with path.open("wb") as f:
        header.tofile(f)
        tokens.tofile(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="data/tiny_debug")
    parser.add_argument("--train_tokens", type=int, default=131_072)
    parser.add_argument("--val_tokens", type=int, default=8_192)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    out = Path(args.output_dir)
    write_shard(out / "fineweb_train_000001.bin", args.train_tokens, args.seed)
    write_shard(out / "fineweb_val_000000.bin", args.val_tokens, args.seed + 1)
    print(f"Wrote tiny debug shards to {out}")


if __name__ == "__main__":
    main()
