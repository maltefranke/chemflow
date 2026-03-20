#!/bin/bash

#SBATCH --job-name=chemflow-rates
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=4GB
#SBATCH --gpus=rtx_4090:4
##SBATCH --gpus=rtx_2080_ti:1
#SBATCH --output=SLURM_OUTPUT-%j.log

cd /cluster/project/jorner/schmiste/flexflow/chemflow

source .env
source .venv/bin/activate

uv run --active --env-file .env run.py data=qm9_scaffold trainer.trainer.max_epochs=30 model.integrator.num_integration_steps=300 model.integrator.time_strategy=log model=dit logging.offline=true # model=transformer dit