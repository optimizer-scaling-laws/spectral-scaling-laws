#!/usr/bin/env bash
#SBATCH --job-name=optimizer-ssl-160m
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=outputs/slurm/%x-%j.out

set -euo pipefail
mkdir -p outputs/slurm
bash scripts/train/run_width_sweep_160m.sh adamw
