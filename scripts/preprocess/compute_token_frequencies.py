#!/usr/bin/env python3
"""Compute token frequencies from headered FineWeb-style binary token shards."""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from optimizer_ssl.data.binary_shards import load_headered_token_shard
from optimizer_ssl.data.frequency_table import compute_bucket_stats


def find_files(data_dir=None, data_glob=None):
    if (data_dir is None) == (data_glob is None):
        raise ValueError("Provide exactly one of --data_dir or --data_glob")
    if data_glob:
        return sorted(glob.glob(data_glob))
    return sorted(glob.glob(str(Path(data_dir) / "fineweb_train_*.bin")))


def compute_frequencies(files, vocab_size=50304, chunk_size=10_000_000):
    freq = torch.zeros(vocab_size, dtype=torch.long)
    total = 0
    for file in tqdm(files, desc="Counting token frequencies"):
        tokens = load_headered_token_shard(file)
        total += len(tokens)
        for start in range(0, len(tokens), chunk_size):
            chunk = np.asarray(tokens[start:start + chunk_size])
            counts = np.bincount(chunk[chunk < vocab_size], minlength=vocab_size)[:vocab_size]
            freq += torch.from_numpy(counts.astype(np.int64))
    return freq, total


def main():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--data_dir")
    g.add_argument("--data_glob")
    p.add_argument("--output", required=True)
    p.add_argument("--vocab_size", type=int, default=50304)
    p.add_argument("--chunk_size", type=int, default=10_000_000)
    p.add_argument("--json_out")
    p.add_argument("--npy_out", help="Optional .npy copy of the frequency vector")
    args = p.parse_args()

    files = find_files(args.data_dir, args.data_glob)
    if not files:
        raise SystemExit("No input files found")
    freq, total = compute_frequencies(files, args.vocab_size, args.chunk_size)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    if str(args.output).endswith(".npy"):
        np.save(args.output, freq.cpu().numpy().astype(np.int64))
    else:
        torch.save(freq, args.output)
    if args.npy_out:
        Path(args.npy_out).parent.mkdir(parents=True, exist_ok=True)
        np.save(args.npy_out, freq.cpu().numpy().astype(np.int64))
    stats = compute_bucket_stats(freq)
    stats["raw_tokens_scanned"] = int(total)
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
