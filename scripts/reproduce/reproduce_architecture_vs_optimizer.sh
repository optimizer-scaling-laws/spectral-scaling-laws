#!/usr/bin/env bash
set -euo pipefail

PROCESSED_DIR=${1:-results/processed}
FIGURE_DIR=${2:-results/figures}

python scripts/analysis/make_architecture_vs_optimizer_figure.py \
  --comparison "${PROCESSED_DIR}/architecture_vs_optimizer_comparison.csv" \
  --out-dir "${FIGURE_DIR}" \
  --formats pdf
