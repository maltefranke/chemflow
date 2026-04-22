#!/bin/bash
set -euo pipefail

# Launch Phase-1 GRPO fine-tuning (fixed atom count: no insertions, no deletions).
# Run from the repo root.

cd "$(dirname "$0")/.."

uv run --active --env-file .env python -m rl.run_grpo \
    --ckpt .pretrained_model/epoch=499-step=48500.ckpt \
    --n_updates 200 --num_steps 200 --a_sde 0 \
    --reward qed \
    --wandb --wandb_project chemflow-grpo --wandb_name phase1-qed-seed0 \
    data.n_atoms_strategy=fixed \
    data.datamodule.batch_size.test=128
