#!/bin/bash
set -euo pipefail

export PYTHONUNBUFFERED=1

# Easy local launcher for the current GRPO default setup. Override any knob with
# env vars, e.g. `N_UPDATES=20 GRPO_WANDB_PROJECT=my-proj bash rl/run_grpo.sh`.

cd "$(dirname "$0")/.."
mkdir -p .rl_ckpts

SIGMA_EXPLORE="${SIGMA_EXPLORE:-0.05}"
SEED="${SEED:-0}"
GROUP_SIZE="${GROUP_SIZE:-8}"
KL_COEF="${KL_COEF:-0.02}"
LR="${LR:-1e-4}"
N_UPDATES="${N_UPDATES:-200}"
MAX_ATOMS="${MAX_ATOMS:-100}"
PER_ELEMENT_LOGP_MEAN="${PER_ELEMENT_LOGP_MEAN:-0}"
UPDATE_PASSES="${UPDATE_PASSES:-2}"
REWARD="${REWARD:-n_atoms}"
REWARD_SLUG="${REWARD//_/}"

SCAFFOLD_DIVERSITY="${SCAFFOLD_DIVERSITY:-1}"
SCAFFOLD_BUCKET_SIZE="${SCAFFOLD_BUCKET_SIZE:-10}"
SCAFFOLD_PENALTY="${SCAFFOLD_PENALTY:-0.5}"
SCAFFOLD_WINDOW_BATCHES="${SCAFFOLD_WINDOW_BATCHES:-50}"
SCAFFOLD_LABELED="${SCAFFOLD_LABELED:-0}"
SCAFFOLD_DIVERSITY_KEY="${SCAFFOLD_DIVERSITY_KEY:-canonical_smiles}"

SCAFFOLD_ENABLED="false"
SCAFFOLD_SUFFIX=""
if [[ "$SCAFFOLD_DIVERSITY" =~ ^(1|true|yes|on)$ ]]; then
  SCAFFOLD_ENABLED="true"
  PEN_TAG="${SCAFFOLD_PENALTY//./p}"
  WIN_TAG="${SCAFFOLD_WINDOW_BATCHES//-/m}"
  SCAFFOLD_SUFFIX="_scaff_b${SCAFFOLD_BUCKET_SIZE}_p${PEN_TAG}_w${WIN_TAG}"
  if [[ "$SCAFFOLD_DIVERSITY_KEY" == "canonical_smiles" ]]; then
    SCAFFOLD_SUFFIX+="_canonsmi"
  fi
  if [[ "$SCAFFOLD_LABELED" =~ ^(1|true|yes|on)$ ]]; then
    SCAFFOLD_SUFFIX+="_labeled"
  fi
fi

KL_OMIT_POS="${KL_OMIT_POS:-1}"
KL_OMIT_ENABLED="false"
KL_OMIT_SUFFIX="fullposkl"
if [[ "$KL_OMIT_POS" =~ ^(1|true|yes|on)$ ]]; then
  KL_OMIT_ENABLED="true"
  KL_OMIT_SUFFIX="omitposkl"
fi

ELEM_SUFFIX=""
PER_ELEM_ENABLED="false"
if [[ "$PER_ELEMENT_LOGP_MEAN" =~ ^(1|true|yes|on)$ ]]; then
  ELEM_SUFFIX="_elementmean"
  PER_ELEM_ENABLED="true"
fi

SIG_TAG="${SIGMA_EXPLORE//./p}"
RUN_TAG="${REWARD_SLUG}-seed${SEED}_sig${SIG_TAG}_g${GROUP_SIZE}_mu${UPDATE_PASSES}_kl${KL_COEF}_lr${LR}${ELEM_SUFFIX}_maxa${MAX_ATOMS}_${KL_OMIT_SUFFIX}${SCAFFOLD_SUFFIX}"
CKPT_TAG="${REWARD_SLUG}_seed${SEED}_sig${SIG_TAG}_g${GROUP_SIZE}_mu${UPDATE_PASSES}_kl${KL_COEF}_lr${LR}${ELEM_SUFFIX}_maxa${MAX_ATOMS}-${KL_OMIT_SUFFIX}${SCAFFOLD_SUFFIX}"

GRPO_WANDB_PROJECT="${GRPO_WANDB_PROJECT:-chemflow-grpo-sweep-20260424_111439}"
GRPO_WANDB_GROUP="${GRPO_WANDB_GROUP:-}"

WANDB_EXTR=()
if [[ -n "$GRPO_WANDB_GROUP" ]]; then
  WANDB_EXTR=("rl.wandb.group=$GRPO_WANDB_GROUP")
fi

echo "reward=${REWARD} sigma_explore=${SIGMA_EXPLORE} seed=${SEED} group_size=${GROUP_SIZE} update_passes=${UPDATE_PASSES} kl_coef=${KL_COEF} kl_omit_pos=${KL_OMIT_POS} lr=${LR} n_updates=${N_UPDATES} max_atoms=${MAX_ATOMS} run=${RUN_TAG}"

uv run --env-file .env python -m rl.run_grpo \
    'rl.ckpt=".pretrained_model/epoch=499-step=48500.ckpt"' \
    "rl.n_updates=$N_UPDATES" \
    rl.grpo.num_integration_steps=100 \
    "rl.max_atoms=$MAX_ATOMS" \
    "rl.grpo.sigma_explore=$SIGMA_EXPLORE" \
    "rl.lr=$LR" \
    rl.grpo.max_grad_norm=1.0 \
    "rl.seed=$SEED" \
    "rl.grpo.group_size=$GROUP_SIZE" \
    "rl.grpo.update_passes=$UPDATE_PASSES" \
    "rl.grpo.kl_coef=$KL_COEF" \
    "rl.grpo.per_element_logp_mean=$PER_ELEM_ENABLED" \
    "rl.reward.name=$REWARD" \
    "rl.reward.scaffold_diversity=$SCAFFOLD_ENABLED" \
    "rl.reward.scaffold_diversity_key=$SCAFFOLD_DIVERSITY_KEY" \
    "rl.reward.scaffold_bucket_size=$SCAFFOLD_BUCKET_SIZE" \
    "rl.reward.scaffold_penalty=$SCAFFOLD_PENALTY" \
    "rl.reward.scaffold_window_batches=$SCAFFOLD_WINDOW_BATCHES" \
    "rl.reward.scaffold_labeled=$SCAFFOLD_LABELED" \
    "rl.grpo.kl_omit_pos=$KL_OMIT_ENABLED" \
    rl.wandb.enabled=true \
    "rl.wandb.project=$GRPO_WANDB_PROJECT" \
    "rl.wandb.name=$RUN_TAG" \
    "${WANDB_EXTR[@]}" \
    "rl.save=.rl_ckpts/grpo_${CKPT_TAG}.pt" \
    "rl.save_best=.rl_ckpts/grpo_${CKPT_TAG}_best.pt" \
    data.datamodule.batch_size.test=128

echo "done: ${RUN_TAG}"
