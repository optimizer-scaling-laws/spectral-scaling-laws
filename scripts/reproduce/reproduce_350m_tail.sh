#!/usr/bin/env bash
set -euo pipefail

PROCESSED_DIR="${1:-results/processed}"
FIGURE_DIR="${2:-results/figures}"

python scripts/analysis/make_350m_tail_figures.py \
  --points "${PROCESSED_DIR}/tail_350m_rank_scaling_points.csv" \
  --betas "${PROCESSED_DIR}/tail_350m_beta_table.csv" \
  --out-dir "${FIGURE_DIR}" \
  --formats pdf
