#!/bin/bash

#SBATCH --job-name=chemflow-rates
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=8GB
#SBATCH --gpus=rtx_4090:4
##SBATCH --gpus=rtx_2080_ti:1
#SBATCH --output=SLURM_OUTPUT-%j.log

cd /cluster/project/krause/frankem/chemflow

source .env
source .venv/bin/activate

export CHEMFLOW_RUN_ID="$(uv run python -c "import wandb; print(wandb.util.generate_id())")"
export WANDB_RUN_ID="${CHEMFLOW_RUN_ID}"

uv run --active --env-file .env torchrun --standalone --nproc_per_node=4 run.py \
  trainer.trainer.max_epochs=20 \
  model.integrator.num_integration_steps=100 \
  data=geom