#!/bin/bash

#SBATCH --job-name=natoms-cfg-eval
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --mem-per-cpu=4GB
#SBATCH --gpus=rtx_4090:1
#SBATCH --output=SLURM_OUTPUT-%j.log

set -euo pipefail

cd /cluster/project/krause/frankem/chemflow

source .env
source .venv/bin/activate

# Eval a checkpoint trained with ExtrapolatableCountEmbedding for both the
# backbone `node_count_embedding` and the CFG `natoms_encoder`.
#
# The --overrides list MUST mirror the training-time Hydra overrides used in
# qm9_semla_natoms_extrapolatable.sh so the composed config (and therefore the
# instantiated model graph) exactly matches the checkpoint.

uv run --active --env-file .env python eval_scripts/eval_natoms_cfg.py \
  --checkpoint /cluster/project/krause/frankem/chemflow/outputs/qm9/natoms/extrapolatable/2026-04-23/x1bxecj8/morph-qm9/x1bxecj8/checkpoints/epoch=499-step=48500.ckpt \
  --output-dir eval_outputs/natoms_cfg_extrapolate \
  --targets 50 \
  --guidance-scales 1.0 \
  --n-mols 500 \
  --predict-batch-size 100 \
  --overrides \
    data=qm9 \
    model=semla \
    cfg=natoms \
    model.node_count_embedding._target_=chemflow.model.embedding.ExtrapolatableCountEmbedding \
    ++model.node_count_embedding.min_period=64.0 \
    ++model.node_count_embedding.max_period=512.0 \
    ++model.node_count_embedding.max_count=32.0 \
    cfg.cfg.natoms_encoder._target_=chemflow.model.embedding.ExtrapolatableCountEmbedding \
    ++cfg.cfg.natoms_encoder.min_period=64.0 \
    ++cfg.cfg.natoms_encoder.max_period=512.0 \
    ++cfg.cfg.natoms_encoder.max_count=32.0
