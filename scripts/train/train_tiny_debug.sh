#!/usr/bin/env bash
set -euo pipefail

# GPU smoke test for the full training/eigen-metric path.
# This creates tiny local headered shards and runs a 2-layer GPT for 10 steps.

python scripts/preprocess/create_tiny_debug_data.py --output_dir data/tiny_debug
NPROC_PER_NODE="${NPROC_PER_NODE:-1}" bash scripts/train/run_single.sh configs/examples/tiny_debug.yaml
