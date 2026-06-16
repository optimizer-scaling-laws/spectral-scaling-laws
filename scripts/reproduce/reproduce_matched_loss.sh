#!/usr/bin/env bash
set -euo pipefail

PROCESSED_DIR=${1:-results/processed}
FIGURE_DIR=${2:-results/figures}

python scripts/analysis/make_matched_loss_figures.py \
  --beta-dynamics "${PROCESSED_DIR}/matched_loss_beta_dynamics.csv" \
  --pr-trajectories "${PROCESSED_DIR}/matched_loss_pr_trajectories.csv" \
  --out-dir "${FIGURE_DIR}" \
  --formats pdf
