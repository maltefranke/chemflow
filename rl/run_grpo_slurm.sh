#!/bin/bash
#SBATCH -J grpo-rl
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100l:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --partition=unkillable
#SBATCH --time=6:00:00
#SBATCH --output=slurm_logs/grpo_%j.out
#SBATCH --error=slurm_logs/grpo_%j.err

set -euo pipefail

# Run from the directory where `sbatch` was invoked.
cd "${SLURM_SUBMIT_DIR:-$PWD}"

mkdir -p slurm_logs .rl_ckpts

SIGMA_EXPLORE="${SIGMA_EXPLORE:-0.05}"
SEED="${SEED:-0}"
GROUP_SIZE="${GROUP_SIZE:-8}"
KL_COEF="${KL_COEF:-0.02}"
LR="${LR:-1e-4}"
N_UPDATES="${N_UPDATES:-200}"
MAX_ATOMS="${MAX_ATOMS:-100}"
PER_ELEMENT_LOGP_MEAN="${PER_ELEMENT_LOGP_MEAN:-0}"
UPDATE_PASSES="${UPDATE_PASSES:-2}"
REWARD="${REWARD:-tanimoto}"
# Filename-safe short name (n_atoms → natoms).
REWARD_SLUG="${REWARD//_/}"

# Murcko scaffold bucket (REINVENT-style); off unless SCAFFOLD_DIVERSITY is truthy.
SCAFFOLD_DIVERSITY="${SCAFFOLD_DIVERSITY:-1}"
SCAFFOLD_BUCKET_SIZE="${SCAFFOLD_BUCKET_SIZE:-10}"
SCAFFOLD_PENALTY="${SCAFFOLD_PENALTY:-0.5}"
SCAFFOLD_WINDOW_BATCHES="${SCAFFOLD_WINDOW_BATCHES:-50}"
SCAFFOLD_LABELED="${SCAFFOLD_LABELED:-0}"
# murcko (default) | canonical_smiles — passed as --scaffold_diversity_key
SCAFFOLD_DIVERSITY_KEY="${SCAFFOLD_DIVERSITY_KEY:-canonical_smiles}"

SCAFFOLD_FLAG=()
SCAFFOLD_SUFFIX=""
if [[ "$SCAFFOLD_DIVERSITY" =~ ^(1|true|yes|on)$ ]]; then
  PEN_TAG="${SCAFFOLD_PENALTY//./p}"
  WIN_TAG="${SCAFFOLD_WINDOW_BATCHES//-/m}"
  SCAFFOLD_SUFFIX="_scaff_b${SCAFFOLD_BUCKET_SIZE}_p${PEN_TAG}_w${WIN_TAG}"
  SCAFFOLD_FLAG=(
    --scaffold_diversity
    --scaffold_diversity_key "$SCAFFOLD_DIVERSITY_KEY"
    --scaffold_bucket_size "$SCAFFOLD_BUCKET_SIZE"
    --scaffold_penalty "$SCAFFOLD_PENALTY"
    --scaffold_window_batches "$SCAFFOLD_WINDOW_BATCHES"
  )
  if [[ "$SCAFFOLD_DIVERSITY_KEY" == "canonical_smiles" ]]; then
    SCAFFOLD_SUFFIX+="_canonsmi"
  fi
  if [[ "$SCAFFOLD_LABELED" =~ ^(1|true|yes|on)$ ]]; then
    SCAFFOLD_FLAG+=(--scaffold_labeled)
    SCAFFOLD_SUFFIX+="_labeled"
  fi
fi

KL_OMIT_POS="${KL_OMIT_POS:-1}"
KL_OMIT_FLAG=()
KL_OMIT_SUFFIX=""
if [[ "$KL_OMIT_POS" =~ ^(1|true|yes|on)$ ]]; then
  KL_OMIT_FLAG=(--kl_omit_pos)
  KL_OMIT_SUFFIX="omitposkl"
else
  KL_OMIT_SUFFIX="fullposkl"
fi

ELEM_SUFFIX=""
PER_ELEM_FLAG=()
if [[ "$PER_ELEMENT_LOGP_MEAN" =~ ^(1|true|yes|on)$ ]]; then
  ELEM_SUFFIX="_elementmean"
  PER_ELEM_FLAG=(--per_element_logp_mean)
fi

SIG_TAG="${SIGMA_EXPLORE//./p}"
RUN_TAG="${REWARD_SLUG}-seed${SEED}_sig${SIG_TAG}_g${GROUP_SIZE}_mu${UPDATE_PASSES}_kl${KL_COEF}_lr${LR}${ELEM_SUFFIX}_continue_maxa${MAX_ATOMS}_${KL_OMIT_SUFFIX}${SCAFFOLD_SUFFIX}"
CKPT_TAG="${REWARD_SLUG}_seed${SEED}_sig${SIG_TAG}_g${GROUP_SIZE}_mu${UPDATE_PASSES}_kl${KL_COEF}_lr${LR}${ELEM_SUFFIX}_continue_maxa${MAX_ATOMS}-${KL_OMIT_SUFFIX}${SCAFFOLD_SUFFIX}"

GRPO_WANDB_PROJECT="${GRPO_WANDB_PROJECT:-chemflow-grpo-tanimoto}"
GRPO_WANDB_GROUP="${GRPO_WANDB_GROUP:-}"

echo "host=$(hostname)  gpus=${CUDA_VISIBLE_DEVICES:-unset}  reward=${REWARD}  scaffold_diversity=${SCAFFOLD_DIVERSITY}  scaffold_diversity_key=${SCAFFOLD_DIVERSITY_KEY}  scaffold_bucket_size=${SCAFFOLD_BUCKET_SIZE}  scaffold_penalty=${SCAFFOLD_PENALTY}  scaffold_window_batches=${SCAFFOLD_WINDOW_BATCHES}  scaffold_labeled=${SCAFFOLD_LABELED}  sigma_explore=${SIGMA_EXPLORE}  seed=${SEED}  group_size=${GROUP_SIZE}  update_passes=${UPDATE_PASSES}  kl_coef=${KL_COEF}  kl_omit_pos=${KL_OMIT_POS}  lr=${LR}  n_updates=${N_UPDATES}  max_atoms=${MAX_ATOMS}  per_element_logp_mean=${PER_ELEMENT_LOGP_MEAN}  run=${RUN_TAG}  wandb_project=${GRPO_WANDB_PROJECT}  wandb_group=${GRPO_WANDB_GROUP:-<none>}"
nvidia-smi -L || true

WANDB_EXTR=()
if [[ -n "$GRPO_WANDB_GROUP" ]]; then
  WANDB_EXTR=(--wandb_group "$GRPO_WANDB_GROUP")
fi

uv run --env-file .env python -m rl.run_grpo \
    --ckpt .rl_ckpts/grpo_tanimoto_seed0_sig0p05_g8_mu2_kl0.02_lr1e-4_maxa100-omitposkl_scaff_b10_p0p5_w50_canonsmi.pt \
    --n_updates "$N_UPDATES" --num_steps 100 --max_atoms "$MAX_ATOMS" --sigma_explore "$SIGMA_EXPLORE" --lr "$LR" \
    --max_grad_norm 1.0 \
    --seed "$SEED" --group_size "$GROUP_SIZE" --update_passes "$UPDATE_PASSES" --kl_coef "$KL_COEF" \
    "${PER_ELEM_FLAG[@]}" \
    --reward "$REWARD" \
    "${SCAFFOLD_FLAG[@]}" \
    "${KL_OMIT_FLAG[@]}" \
    --wandb --wandb_project "$GRPO_WANDB_PROJECT" --wandb_name "$RUN_TAG" \
    "${WANDB_EXTR[@]}" \
    --save ".rl_ckpts/grpo_${CKPT_TAG}.pt" \
    --save_best ".rl_ckpts/grpo_${CKPT_TAG}_best.pt" \
    data.datamodule.batch_size.test=128

echo "done: ${RUN_TAG}"
