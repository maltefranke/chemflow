#!/bin/bash
set -euo pipefail

# Launch Phase-2 GRPO fine-tuning (full variable-atom regime: substitutions,
# insertions, and deletions).  Positions use ReinFlow-style exploration:
# π(x'|x) = N(x + v_θ·dt, σ² I) with --sigma_explore (= σ).

cd "$(dirname "$0")/.."

uv run --active --env-file .env python -m rl.run_grpo \
    --ckpt .pretrained_model/epoch=499-step=48500.ckpt \
    --n_updates 50 --num_steps 100 --sigma_explore 0.05 --lr 1e-4 \
    --update_passes 1 \
    --max_grad_norm 1.0 --kl_coef 0.05 \
    --per_element_logp_mean \
    --reward n_atoms \
    --wandb --wandb_project chemflow-grpo --wandb_name phase2-natoms-seed0_sig0.05_elementmean \
    --save .rl_ckpts/grpo_natoms_seed0_sig0.05_elementmean.pt \
    --save_best .rl_ckpts/grpo_natoms_seed0_sig0.05_best_elementmean.pt \
    data.datamodule.batch_size.test=128
