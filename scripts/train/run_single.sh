#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   NPROC_PER_NODE=4 bash scripts/train/run_single.sh configs/paper_runs/main_160m_width_sweep/adamw/1x.yaml
#   CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 bash scripts/train/run_single.sh <config>

CONFIG="${1:?Usage: bash scripts/train/run_single.sh <config.yaml>}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
LOG_DIR="${LOG_DIR:-outputs/logs}"
mkdir -p "${LOG_DIR}"

NAME="$(echo "${CONFIG}" | sed 's#configs/paper_runs/##; s#/#_#g; s#.yaml$##')"
LOG_FILE="${LOG_DIR}/${NAME}.log"

echo "[run] ${CONFIG}"
echo "[log] ${LOG_FILE}"
PYTHONPATH="${PWD}:${PWD}/third_party/dion:${PYTHONPATH:-}" PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}"   optimizer_ssl/train/train_lm.py   --config "${CONFIG}" 2>&1 | tee "${LOG_FILE}"
