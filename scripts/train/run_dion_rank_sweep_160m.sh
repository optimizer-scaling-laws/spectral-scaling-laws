#!/usr/bin/env bash
set -euo pipefail

# Launch the 160M Dion rank-fraction sweep. Paper runs used 4 GPUs.
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export NPROC_PER_NODE

for RANK in r1_2 r1_4 r1_8 r1_16; do
  CONFIG="configs/paper_runs/dion_rank_sweep/160m/${RANK}.yaml"
  bash scripts/train/run_single.sh "${CONFIG}"
done
