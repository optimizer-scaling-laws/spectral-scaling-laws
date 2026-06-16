#!/usr/bin/env bash
set -euo pipefail

# Launch the 160M-family FFN-width sweep. Paper runs used 4 GPUs.
# Usage:
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train/run_width_sweep_160m.sh adamw
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train/run_width_sweep_160m.sh dion_r1_16 1x 4x 8x

OPTIMIZER="${1:-adamw}"
shift || true
if [[ $# -eq 0 ]]; then
  WIDTHS=(1x 2x 3x 4x 5x 6x 7x 8x)
else
  WIDTHS=("$@")
fi
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export NPROC_PER_NODE

for WIDTH in ${WIDTHS[@]}; do
  CONFIG="configs/paper_runs/main_160m_width_sweep/${OPTIMIZER}/${WIDTH}.yaml"
  if [[ ! -f "${CONFIG}" ]]; then
    echo "Missing config: ${CONFIG}" >&2
    exit 2
  fi
  bash scripts/train/run_single.sh "${CONFIG}"
done
