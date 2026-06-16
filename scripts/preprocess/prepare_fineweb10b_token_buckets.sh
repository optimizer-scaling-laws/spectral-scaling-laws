#!/usr/bin/env bash
set -euo pipefail

# End-to-end data preparation for token-frequency buckets.
# Full paper setting: 103 train shards (~10B tokens). For a smoke test, pass a
# smaller number, e.g. `NUM_TRAIN_SHARDS=2 bash scripts/preprocess/prepare_fineweb10b_token_buckets.sh`.

DATA_DIR="${DATA_DIR:-data/fineweb10B}"
NUM_TRAIN_SHARDS="${NUM_TRAIN_SHARDS:-103}"
FREQ_FILE="${FREQ_FILE:-results/processed/token_frequencies.npy}"
STATS_FILE="${STATS_FILE:-results/processed/token_frequency_stats.json}"
AUDIT_FILE="${AUDIT_FILE:-results/processed/token_frequency_audit.json}"

python scripts/preprocess/download_fineweb10b.py   --output_dir "${DATA_DIR}"   --num_train_shards "${NUM_TRAIN_SHARDS}"

python scripts/preprocess/compute_token_frequencies.py   --data_dir "${DATA_DIR}"   --output "${FREQ_FILE}"   --json_out "${STATS_FILE}"

python scripts/validation/audit_token_frequency_buckets.py   --freq_file "${FREQ_FILE}"   --data_dir "${DATA_DIR}"   --format headered   --json_out "${AUDIT_FILE}"
