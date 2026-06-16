#!/usr/bin/env bash
set -euo pipefail

# Launch the 350M-family FFN-width sweep. Paper runs used 8 GPUs.
# Usage:
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/train/run_width_sweep_350m.sh muon

OPTIMIZER="${1:-adamw}"
shift || true
if [[ $# -eq 0 ]]; then
  WIDTHS=(1x 2x 3x 4x)
else
  WIDTHS=("$@")
fi
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export NPROC_PER_NODE

for WIDTH in ${WIDTHS[@]}; do
  CONFIG="configs/paper_runs/main_350m_width_sweep/${OPTIMIZER}/${WIDTH}.yaml"
  if [[ ! -f "${CONFIG}" ]]; then
    echo "Missing config: ${CONFIG}" >&2
    exit 2
  fi
  bash scripts/train/run_single.sh "${CONFIG}"
done
