#!/bin/bash
#SBATCH -J grpo-natoms
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100l:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --partition=unkillable
#SBATCH --time=4:00:00
#SBATCH --output=slurm_logs/grpo_%j.out
#SBATCH --error=slurm_logs/grpo_%j.err

# SLURM wrapper for `rl/run_grpo.sh`.  Parameterises `a_sde`, `SEED`, `GROUP_SIZE`,
# `KL_COEF` (0 = no KL), `LR`, `VAR_FLOOR` (position log-prob variance floor),
# `UPDATE_PASSES` (PPO-style passes over sampled trajectory), optional
# `PER_ELEMENT_LOGP_MEAN`, and W&B overrides
# `GRPO_WANDB_PROJECT` / `GRPO_WANDB_GROUP` (same env vars as `sweep_grpo_unkillable.sh`).
#
# Examples:
#   for a in 0.05 0.1; do for k in 0.02 0.05 0.1; do A_SDE=$a KL_COEF=$k sbatch rl/run_grpo_slurm.sh; done; done
#   for lr in 3e-5 1e-4; do LR=$lr sbatch rl/run_grpo_slurm.sh; done
#   # Same W&B project as a prior sweep + per-element log-prob mean:
#   GRPO_WANDB_PROJECT=chemflow-grpo-sweep-20260423_164156 GRPO_WANDB_GROUP=20260423_164156 \
#     PER_ELEMENT_LOGP_MEAN=1 A_SDE=0.05 KL_COEF=0.05 sbatch rl/run_grpo_slurm.sh
#   VAR_FLOOR=1e-2 A_SDE=0.1 sbatch rl/run_grpo_slurm.sh
#
#   GROUP_SIZE=1 is recommended for stable batch-normalized advantages (see grpo.py).
#
# Default matches the last KL run tag shape: seed0, a_sde=0.1, g1, mu4, kl=0.05, lr=1e-4.
#
# Policy updates: N_UPDATES (default 100; matches sweep_grpo_unkillable.sh). Other
# hyperparameters align with rl/run_grpo.sh (lr from env, reward=n_atoms,
# max_grad_norm 1.0, batch 128) unless you override here.

set -euo pipefail

# Run from the directory where `sbatch` was invoked.
cd "${SLURM_SUBMIT_DIR:-$PWD}"

mkdir -p slurm_logs .rl_ckpts

A_SDE="${A_SDE:-0.05}"
SEED="${SEED:-0}"
GROUP_SIZE="${GROUP_SIZE:-1}"
KL_COEF="${KL_COEF:-0.05}"
LR="${LR:-1e-4}"
# Position log-prob variance floor (matches grpo.GRPOConfig.var_floor / DEFAULT_VAR_FLOOR).
VAR_FLOOR="${VAR_FLOOR:-1e-2}"
N_UPDATES="${N_UPDATES:-100}"
PER_ELEMENT_LOGP_MEAN="${PER_ELEMENT_LOGP_MEAN:-0}"
UPDATE_PASSES="${UPDATE_PASSES:-4}"
ELEM_SUFFIX=""
PER_ELEM_FLAG=()
if [[ "$PER_ELEMENT_LOGP_MEAN" =~ ^(1|true|yes|on)$ ]]; then
  ELEM_SUFFIX="_elementmean"
  PER_ELEM_FLAG=(--per_element_logp_mean)
fi

RUN_TAG="phase2-natoms-seed${SEED}_alpha${A_SDE}_g${GROUP_SIZE}_mu${UPDATE_PASSES}_kl${KL_COEF}_lr${LR}_vf${VAR_FLOOR}${ELEM_SUFFIX}"
CKPT_TAG="seed${SEED}_alpha${A_SDE}_g${GROUP_SIZE}_mu${UPDATE_PASSES}_kl${KL_COEF}_lr${LR}_vf${VAR_FLOOR}${ELEM_SUFFIX}"

GRPO_WANDB_PROJECT="${GRPO_WANDB_PROJECT:-chemflow-grpo-sweep-20260424_111439}"
GRPO_WANDB_GROUP="${GRPO_WANDB_GROUP:-}"

echo "host=$(hostname)  gpus=${CUDA_VISIBLE_DEVICES:-unset}  a_sde=${A_SDE}  seed=${SEED}  group_size=${GROUP_SIZE}  update_passes=${UPDATE_PASSES}  kl_coef=${KL_COEF}  lr=${LR}  var_floor=${VAR_FLOOR}  n_updates=${N_UPDATES}  per_element_logp_mean=${PER_ELEMENT_LOGP_MEAN}  run=${RUN_TAG}  wandb_project=${GRPO_WANDB_PROJECT}  wandb_group=${GRPO_WANDB_GROUP:-<none>}"
nvidia-smi -L || true

WANDB_EXTR=()
if [[ -n "$GRPO_WANDB_GROUP" ]]; then
  WANDB_EXTR=(--wandb_group "$GRPO_WANDB_GROUP")
fi

uv run --env-file .env python -m rl.run_grpo \
    --ckpt .pretrained_model/epoch=499-step=48500.ckpt \
    --n_updates "$N_UPDATES" --num_steps 100 --a_sde "$A_SDE" --lr "$LR" \
    --var_floor "$VAR_FLOOR" \
    --max_grad_norm 1.0 \
    --seed "$SEED" --group_size "$GROUP_SIZE" --update_passes "$UPDATE_PASSES" --kl_coef "$KL_COEF" \
    "${PER_ELEM_FLAG[@]}" \
    --reward n_atoms \
    --wandb --wandb_project "$GRPO_WANDB_PROJECT" --wandb_name "$RUN_TAG" \
    "${WANDB_EXTR[@]}" \
    --save ".rl_ckpts/grpo_natoms_${CKPT_TAG}.pt" \
    --save_best ".rl_ckpts/grpo_natoms_${CKPT_TAG}_best.pt" \
    data.datamodule.batch_size.test=128

echo "done: ${RUN_TAG}"
