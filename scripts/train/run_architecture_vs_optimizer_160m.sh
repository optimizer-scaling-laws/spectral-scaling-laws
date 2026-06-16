#!/usr/bin/env bash
set -euo pipefail

# Launch the full GPT2-160M architecture-vs-optimizer head-count grid.
# Paper runs used 4 GPUs.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train/run_architecture_vs_optimizer_160m.sh
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train/run_architecture_vs_optimizer_160m.sh heads_6 adamw 1x 4x 8x
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train/run_architecture_vs_optimizer_160m.sh heads_12 dion_r1_2 1x
#
# Head groups:
#   heads_12 = default GPT2-160M 12-head runs
#   heads_6  = 6-head ablation runs; all other training settings match heads_12
#
# Optimizer groups:
#   adamw, muon, normuon, dion_r1_2, dion_r1_16

HEAD_GROUPS=(heads_12 heads_6)
OPTIMIZERS=(adamw muon normuon dion_r1_2 dion_r1_16)
WIDTHS=(1x 2x 3x 4x 5x 6x 7x 8x)

if [[ $# -gt 0 ]]; then
  HEAD_GROUPS=("$1")
  shift
fi

if [[ $# -gt 0 ]]; then
  OPTIMIZERS=("$1")
  shift
fi

if [[ $# -gt 0 ]]; then
  WIDTHS=("$@")
fi

NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export NPROC_PER_NODE

for HEAD_GROUP in "${HEAD_GROUPS[@]}"; do
  for OPTIMIZER in "${OPTIMIZERS[@]}"; do
    for WIDTH in "${WIDTHS[@]}"; do
      CONFIG="configs/paper_runs/architecture_vs_optimizer/160m/${HEAD_GROUP}/${OPTIMIZER}/${WIDTH}.yaml"
      if [[ ! -f "${CONFIG}" ]]; then
        echo "Missing config: ${CONFIG}" >&2
        exit 2
      fi
      bash scripts/train/run_single.sh "${CONFIG}"
    done
  done
done
