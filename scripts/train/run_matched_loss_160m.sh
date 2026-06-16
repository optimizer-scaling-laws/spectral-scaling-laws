#!/usr/bin/env bash
set -euo pipefail

# Launch the GPT2-160M matched-loss / extended-AdamW config grid.
# Paper runs used 4 GPUs.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train/run_matched_loss_160m.sh
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train/run_matched_loss_160m.sh adamw_12k 1x 4x 8x
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train/run_matched_loss_160m.sh dion_r1_16 1x
#
# Groups:
#   adamw_6k    = AdamW baseline, 6000 iterations
#   adamw_12k   = extended AdamW run, identical except num_iterations=12000
#   dion_r1_16  = Dion rank fraction 1/16, 6000 iterations

GROUPS=(adamw_6k adamw_12k dion_r1_16)
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
    CONFIG="configs/paper_runs/matched_loss/160m/${GROUP}/${WIDTH}.yaml"
    if [[ ! -f "${CONFIG}" ]]; then
      echo "Missing config: ${CONFIG}" >&2
      exit 2
    fi
    bash scripts/train/run_single.sh "${CONFIG}"
  done
done
