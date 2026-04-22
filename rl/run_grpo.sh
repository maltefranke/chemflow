#!/bin/bash
set -euo pipefail

# Launch Phase-2 GRPO fine-tuning (full variable-atom regime: substitutions,
# insertions, and deletions).  Drop `data.n_atoms_strategy=fixed` to let the
# integrator change topology each step; keep `--a_sde 0` so position updates
# are deterministic (ODE) while the discrete channels carry the policy
# gradient.

cd "$(dirname "$0")/.."

uv run --active --env-file .env python -m rl.run_grpo \
    --ckpt .pretrained_model/epoch=499-step=48500.ckpt \
    --n_updates 200 --num_steps 200 --a_sde 0 \
    --reward qed \
    --wandb --wandb_project chemflow-grpo --wandb_name phase2-qed-seed0 \
    data.datamodule.batch_size.test=128
