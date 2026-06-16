#!/usr/bin/env bash
set -euo pipefail

PROCESSED_DIR=${1:-results/processed}
FIGURE_DIR=${2:-results/figures}
MODEL_SCALE=${MODEL_SCALE:-160m}

python scripts/analysis/make_dion_rank_sweep_figures.py \
  --points "${PROCESSED_DIR}/dion_tail_rank_sweep_points.csv" \
  --betas "${PROCESSED_DIR}/dion_tail_rank_sweep_beta_table.csv" \
  --out-dir "${FIGURE_DIR}" \
  --model-scale "${MODEL_SCALE}" \
  --formats pdf
