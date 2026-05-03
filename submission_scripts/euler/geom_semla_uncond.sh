#!/bin/bash

#SBATCH --job-name=semla-geom-uncond
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=72:00:00
#SBATCH --mem-per-cpu=8GB
#SBATCH --gpus=rtx_4090:4
#SBATCH --output=SLURM_OUTPUT-%j.log

set -euo pipefail

cd /cluster/project/krause/frankem/chemflow

source .env
source .venv/bin/activate

export CHEMFLOW_RUN_ID="$(uv run python -c "import wandb; print(wandb.util.generate_id())")"
export WANDB_RUN_ID="${CHEMFLOW_RUN_ID}"

uv run --active --env-file .env \
  torchrun --standalone --nproc_per_node=4 run.py \
  data=geom \
  model=semla \
  cfg=uncond \
  trainer.trainer.max_epochs=30 \
  trainer.trainer.val_check_interval=3000 \
  +model.module.optimizer_config.optimizer._target_=torch.optim.AdamW \
  ~model.module.optimizer_config.optimizer.momentum \
  model.module.optimizer_config.optimizer.lr=1e-4 \
  +model.module.optimizer_config.optimizer.betas=[0.9,0.95] \
  +model.module.optimizer_config.optimizer.eps=1e-8 \
  model.module.optimizer_config.optimizer.weight_decay=0.0
