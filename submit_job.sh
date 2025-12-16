#!/bin/bash

#SBATCH --job-name=semlaflow
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --mem-per-cpu=2GB
#SBATCH --gpus=1
#SBATCH --output=SLURM_OUTPUT-%j.log

source .env
source .venv/bin/activate

python run.py