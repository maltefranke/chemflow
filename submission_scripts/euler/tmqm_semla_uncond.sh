#!/bin/bash

#SBATCH --job-name=semla-tmqm-uncond
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=14:00:00
#SBATCH --mem-per-cpu=4GB
#SBATCH --gpus=rtx_4090:4
#SBATCH --output=SLURM_OUTPUT-%j.log

set -euo pipefail

cd /cluster/project/krause/agoldszal/chemflow

source .env
source .venv/bin/activate

export CHEMFLOW_RUN_ID="$(uv run python -c "import wandb; print(wandb.util.generate_id())")"
export WANDB_RUN_ID="${CHEMFLOW_RUN_ID}"

uv run --active --env-file .env \
  torchrun --standalone --nproc_per_node=4 run.py \
  data=tmqm \
  model=semla \
  cfg=uncond \
  representation=pointcloud