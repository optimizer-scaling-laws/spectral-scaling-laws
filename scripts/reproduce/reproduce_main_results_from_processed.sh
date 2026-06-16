#!/usr/bin/env bash
set -euo pipefail

PROCESSED_DIR=${1:-results/processed}
FIGURE_DIR=${2:-results/figures}
MODEL_SCALE=${MODEL_SCALE:-160m}

python scripts/analysis/make_global_rank_figures.py \
  --points "${PROCESSED_DIR}/global_rank_scaling_points.csv" \
  --betas "${PROCESSED_DIR}/main_beta_table.csv" \
  --out-dir "${FIGURE_DIR}" \
  --model-scale "${MODEL_SCALE}" \
  --formats pdf

python scripts/analysis/make_frequency_bucket_figures.py \
  --points "${PROCESSED_DIR}/frequency_bucket_rank_scaling_points.csv" \
  --betas "${PROCESSED_DIR}/frequency_bucket_beta_table.csv" \
  --out-dir "${FIGURE_DIR}" \
  --model-scale "${MODEL_SCALE}" \
  --formats pdf

python scripts/analysis/make_dion_rank_sweep_figures.py \
  --points "${PROCESSED_DIR}/dion_tail_rank_sweep_points.csv" \
  --betas "${PROCESSED_DIR}/dion_tail_rank_sweep_beta_table.csv" \
  --out-dir "${FIGURE_DIR}" \
  --model-scale "${MODEL_SCALE}" \
  --formats pdf

python scripts/analysis/make_matched_loss_figures.py \
  --beta-dynamics "${PROCESSED_DIR}/matched_loss_beta_dynamics.csv" \
  --pr-trajectories "${PROCESSED_DIR}/matched_loss_pr_trajectories.csv" \
  --out-dir "${FIGURE_DIR}" \
  --formats pdf

python scripts/analysis/make_350m_tail_figures.py \
  --points "${PROCESSED_DIR}/tail_350m_rank_scaling_points.csv" \
  --betas "${PROCESSED_DIR}/tail_350m_beta_table.csv" \
  --out-dir "${FIGURE_DIR}" \
  --model-scale 350m \
  --formats pdf
python scripts/analysis/make_architecture_vs_optimizer_figure.py \
  --comparison "${PROCESSED_DIR}/architecture_vs_optimizer_comparison.csv" \
  --out-dir "${FIGURE_DIR}" \
  --formats pdf

