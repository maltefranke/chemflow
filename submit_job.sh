#!/bin/bash

#SBATCH --job-name=chemflow-rates
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=18:00:00
#SBATCH --mem-per-cpu=4GB
#SBATCH --gpus=rtx_4090:1
#SBATCH --output=SLURM_OUTPUT-%j.log

cd /cluster/project/krause/frankem/chemflow

source .env
source .venv/bin/activate

uv run --active --env-file .env run.py trainer.trainer.max_epochs=500 model=egnn_qm9_rates model.module.loss_weights.l_x=10.0 # data.n_atoms_strategy=fixed