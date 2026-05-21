#!/bin/bash
#SBATCH -J grpo-rl
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100l:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --partition=unkillable
#SBATCH --time=6:00:00
#SBATCH --output=slurm_logs/grpo_%j.out
#SBATCH --error=slurm_logs/grpo_%j.err

set -euo pipefail

# Line-buffer stdout/stderr in Slurm (otherwise the first [grpo] step can appear late).
export PYTHONUNBUFFERED=1

# Run from the directory where `sbatch` was invoked.
cd "${SLURM_SUBMIT_DIR:-$PWD}"

export PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
mkdir -p slurm_logs

source .env
source .venv/bin/activate

echo "host=$(hostname)  gpus=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L || true

uv run --active --no-sync --env-file .env python -m chemflow.rl.run_grpo \
    data=qm9 \
    model=dit \
    cfg=uncond \
    representation=pointcloud \
    'rl.ckpt="${oc.env:PROJECT_ROOT}/.pretrained_model/epoch=206-step=4968.ckpt"' \
    rl.reward.apply_validity_gate=false \
    rl.reward.scaffold_diversity=false \
    "$@"
