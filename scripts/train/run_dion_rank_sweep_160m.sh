#!/usr/bin/env bash
set -euo pipefail

# Launch the full GPT2-160M Dion rank-fraction sweep.
# Paper runs used 4 GPUs.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train/run_dion_rank_sweep_160m.sh
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train/run_dion_rank_sweep_160m.sh dion_r1_2 1x 4x 8x
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train/run_dion_rank_sweep_160m.sh adamw 1x
#
# Groups:
#   adamw  = dashed baseline used in the Dion rank-sweep figure
#   dion_r1_2   = Dion rank fraction 1/2
#   dion_r1_4   = Dion rank fraction 1/4
#   dion_r1_8   = Dion rank fraction 1/8
#   dion_r1_16  = Dion rank fraction 1/16

GROUPS=(adamw dion_r1_2 dion_r1_4 dion_r1_8 dion_r1_16)
WIDTHS=(1x 2x 3x 4x 5x 6x 7x 8x)

if [[ $# -gt 0 ]]; then
  GROUPS=("$1")
  shift
fi

if [[ $# -gt 0 ]]; then
  WIDTHS=("$@")
fi

NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export NPROC_PER_NODE

for GROUP in "${GROUPS[@]}"; do
  for WIDTH in "${WIDTHS[@]}"; do
    CONFIG="configs/paper_runs/dion_rank_sweep/160m/${GROUP}/${WIDTH}.yaml"
    if [[ ! -f "${CONFIG}" ]]; then
      echo "Missing config: ${CONFIG}" >&2
      exit 2
    fi
    bash scripts/train/run_single.sh "${CONFIG}"
  done
done
