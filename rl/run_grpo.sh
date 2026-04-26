#!/bin/bash
set -euo pipefail

# Launch Phase-2 GRPO fine-tuning (full variable-atom regime: substitutions,
# insertions, and deletions).  `a_sde > 0` is required for the position
# channel to carry a well-scaled policy gradient -- with `a_sde=0` the
# position variance floors at 1e-6 and position log-ratios blow up / get
# clamped, so positions train nothing.  Post-RL we'd switch back to ODE
# (a_sde=0) for deterministic inference.

cd "$(dirname "$0")/.."

uv run --active --env-file .env python -m rl.run_grpo \
    --ckpt .pretrained_model/epoch=499-step=48500.ckpt \
    --n_updates 50 --num_steps 100 --a_sde 0.05 --lr 1e-4 \
    --update_passes 1 \
    --max_grad_norm 1.0 --kl_coef 0.05 \
    --per_element_logp_mean \
    --reward n_atoms \
    --wandb --wandb_project chemflow-grpo --wandb_name phase2-natoms-seed0_alpha0.05_elementmean \
    --save .rl_ckpts/grpo_natoms_seed0_alpha0.05_elementmean.pt \
    --save_best .rl_ckpts/grpo_natoms_seed0_alpha0.05_best_elementmean.pt \
    data.datamodule.batch_size.test=128
