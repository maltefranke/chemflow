#!/bin/bash

#SBATCH --job-name=semla-qm9-linear
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=14:00:00
#SBATCH --mem-per-cpu=4GB
#SBATCH --gpus=rtx_4090:4
#SBATCH --output=SLURM_OUTPUT-%j.log

set -euo pipefail

cd /cluster/project/krause/frankem/chemflow

source .env
source .venv/bin/activate

# Shared run id for Hydra outputs, wandb, and all DDP ranks.
export CHEMFLOW_RUN_ID="$(uv run python -c "import wandb; print(wandb.util.generate_id())")"
export WANDB_RUN_ID="${CHEMFLOW_RUN_ID}"

uv run --active --env-file .env \
  torchrun --standalone --nproc_per_node=4 run.py \
  data=qm9 \
  model=semla \
  cfg=uncond \
  schedule=linear \
  'hydra.run.dir=${hydra:runtime.cwd}/outputs/${hydra:runtime.choices.data}/${hydra:runtime.choices.cfg}/linear/${now:%Y-%m-%d}/${oc.env:CHEMFLOW_RUN_ID}'