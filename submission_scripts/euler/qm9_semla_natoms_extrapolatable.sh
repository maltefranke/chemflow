#!/bin/bash

#SBATCH --job-name=semla-qm9-natoms-extrap
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=32:00:00
#SBATCH --mem-per-cpu=4GB
#SBATCH --gpus=rtx_4090:4
#SBATCH --output=SLURM_OUTPUT-%j.log

set -euo pipefail

cd /cluster/project/krause/frankem/chemflow

source .env
source .venv/bin/activate

# Shared run id for Hydra outputs, wandb, and all DDP ranks (must match hydra.run.dir).
export CHEMFLOW_RUN_ID="$(uv run python -c "import wandb; print(wandb.util.generate_id())")"
export WANDB_RUN_ID="${CHEMFLOW_RUN_ID}"

# ------------------------------------------------------------------------------
# QM9 + SemlaBB + natoms CFG, using ExtrapolatableCountEmbedding for both count
# embeddings (backbone node_count + CFG natoms encoder).
#
# Each encoder lives as a Hydra `_target_` sub-block, so we only rewrite
# `_target_` and append the extra hyperparameters (`++` = add-or-override).
#
# Defaults tuned for QM9 (max training count ~29):
#   min_period=64    shortest Fourier period ≥ 2 × training range  (smooth)
#   max_period=512   longest Fourier period                        (slow trend)
#   max_count=32     normaliser for the raw scalar path            (= ~train max)
# ------------------------------------------------------------------------------

uv run --active --env-file .env \
  torchrun --standalone --nproc_per_node=4 run.py \
  data=qm9 \
  model=semla \
  cfg=natoms \
  trainer.trainer.max_epochs=2000 \
  ~trainer.trainer.val_check_interval \
  +trainer.trainer.check_val_every_n_epoch=25 \
  model.node_count_embedding._target_=chemflow.model.embedding.ExtrapolatableCountEmbedding \
  ++model.node_count_embedding.min_period=64.0 \
  ++model.node_count_embedding.max_period=512.0 \
  ++model.node_count_embedding.max_count=32.0 \
  cfg.cfg.natoms_encoder._target_=chemflow.model.embedding.ExtrapolatableCountEmbedding \
  ++cfg.cfg.natoms_encoder.min_period=64.0 \
  ++cfg.cfg.natoms_encoder.max_period=512.0 \
  ++cfg.cfg.natoms_encoder.max_count=32.0 \
  'hydra.run.dir=${hydra:runtime.cwd}/outputs/${hydra:runtime.choices.data}/${hydra:runtime.choices.cfg}/extrapolatable/${now:%Y-%m-%d}/${oc.env:CHEMFLOW_RUN_ID}'
