#!/bin/bash

# Multi-seed evaluation: 10k generations × 3 seeds → all metrics + mean/std.
#
# Edit CHECKPOINT (and the --overrides block if your training run used
# non-default Hydra overrides, e.g. ExtrapolatableCountEmbedding) before
# submitting.

#SBATCH --job-name=chemflow-eval-full
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time=10:00:00
#SBATCH --mem-per-cpu=8GB
#SBATCH --gpus=rtx_4090:1
#SBATCH --output=SLURM_OUTPUT-%j.log

set -euo pipefail

cd /cluster/project/krause/frankem/chemflow

source .env
source .venv/bin/activate

# === EDIT ME =============================================================
# CHECKPOINT="/cluster/project/krause/frankem/chemflow/outputs/geom_e133.ckpt"
CHECKPOINT="/cluster/project/krause/frankem/chemflow/outputs/qm9/uncond/2026-04-25/h4ldkelz/morph-qm9/h4ldkelz/checkpoints/epoch=1999-step=134600.ckpt"
OUTPUT_DIR="/cluster/scratch/frankem/morph/qm9/uncond/eval_outputs_10k_100steps_w_semla_metrics/full_metrics_$(date +%Y%m%d_%H%M%S)"
N_MOLS=10000
N_SEEDS=3
SEEDS="42,123,2026"
PREDICT_BATCH_SIZE=64
# =========================================================================

mkdir -p "${OUTPUT_DIR}"

uv run --active --env-file .env python eval_scripts/eval_full_metrics.py \
  --checkpoint "${CHECKPOINT}" \
  --output-dir "${OUTPUT_DIR}" \
  --n-mols "${N_MOLS}" \
  --n-seeds "${N_SEEDS}" \
  --seeds "${SEEDS}" \
  --predict-batch-size "${PREDICT_BATCH_SIZE}" \
  --overrides \
    data=qm9 \
    model=semla \
    integrator.integrator.num_integration_steps=100 \
    #cfg=natoms \
    #model.node_count_embedding._target_=chemflow.model.embedding.ExtrapolatableCountEmbedding \
    #++model.node_count_embedding.min_period=64.0 \
    #++model.node_count_embedding.max_period=512.0 \
    #++model.node_count_embedding.max_count=32.0 \
    #cfg.cfg.natoms_encoder._target_=chemflow.model.embedding.ExtrapolatableCountEmbedding \
    #++cfg.cfg.natoms_encoder.min_period=64.0 \
    #++cfg.cfg.natoms_encoder.max_period=512.0 \
    #++cfg.cfg.natoms_encoder.max_count=32.0
