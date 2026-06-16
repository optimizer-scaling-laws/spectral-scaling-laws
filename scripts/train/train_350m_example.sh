#!/usr/bin/env bash
set -euo pipefail
NPROC_PER_NODE="${NPROC_PER_NODE:-8}" bash scripts/train/run_single.sh   configs/paper_runs/main_350m_width_sweep/adamw/1x.yaml
