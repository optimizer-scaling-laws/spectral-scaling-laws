#!/usr/bin/env bash
set -euo pipefail

MANIFEST=${1:-results/processed/run_metadata.csv}
OUT_DIR=${2:-results/processed}
LAYER_METRICS="$OUT_DIR/layer_metrics.csv"

python scripts/analysis/parse_eigen_logs_to_csv.py \
  --manifest "$MANIFEST" \
  --output "$LAYER_METRICS" \
  --require-all-buckets-per-step

python scripts/analysis/aggregate_rank_scaling.py \
  --layer-metrics "$LAYER_METRICS" \
  --output-dir "$OUT_DIR" \
  --final-samples 5
