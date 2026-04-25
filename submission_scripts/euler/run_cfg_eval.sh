#!/bin/bash

# Eval a checkpoint trained with ExtrapolatableCountEmbedding for both the
# backbone `node_count_embedding` and the CFG `natoms_encoder`.
#
# The --overrides list MUST mirror the training-time Hydra overrides used in
# qm9_semla_natoms_extrapolatable.sh so the composed config (and therefore the
# instantiated model graph) exactly matches the checkpoint.

uvr eval_scripts/eval_natoms_cfg.py \
  --checkpoint /cluster/project/krause/frankem/chemflow/outputs/qm9/natoms/extrapolatable/2026-04-23/x1bxecj8/morph-qm9/x1bxecj8/checkpoints/epoch=249-step=24250.ckpt \
  --output-dir eval_outputs/natoms_cfg_extrapolate \
  --targets 33,35,37,40\
  --guidance-scales 1.0 \
  --n-mols 200 \
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
