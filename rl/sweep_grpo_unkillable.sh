#!/usr/bin/env bash
#
# WRONG:  sbatch rl/sweep_grpo_unkillable.sh
# RIGHT:  ./rl/sweep_grpo_unkillable.sh    (from repo root; submits jobs via rl/run_grpo_slurm.sh)
#
# Minimal GRPO group-size (G) × exploration noise (σ) sweep — scaffold diversity OFF so reward
# variance is not altered by scaffold penalties (which would confound reading G’s effect).
# Per-element log-prob mean is OFF unless β has been calibrated for it.
#
# Base (fixed): reward=n_atoms, β=KL_COEF=0.05, lr=1e-4, kl_omit_pos=ON (same recipe as your
#              stable runs — not swept; only G and SIGMA_EXPLORE vary), update_passes=2,
#              N_UPDATES=50.
#
# Primary diagnostic on W&B for G>1: compare reward_within_std vs reward_between_std. If
# reward_within_std is tiny (≪ reward_between_std, e.g. <10%), within-group signal is weak —
# G>1 mostly burns compute relative to usable gradient signal.
#
# Usage:
#   ./rl/sweep_grpo_unkillable.sh                              # default W&B project: chemflow-grpo-g-only-<stamp>
#   ./rl/sweep_grpo_unkillable.sh chemflow-grpo-g-only-manual001
#   SWEEP_STAMP=my001 DRY_RUN=1 ./rl/sweep_grpo_unkillable.sh
#
set -euo pipefail

if [[ -n "${SLURM_JOB_ID:-}" && "${SLURM_JOB_NAME:-}" == "sweep_grpo_unkillable.sh" ]]; then
  echo "error: do not sbatch this file — it is not a GPU batch script (no #SBATCH)." >&2
  echo "  run from repo root:  ./rl/sweep_grpo_unkillable.sh" >&2
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
mkdir -p slurm_logs

SLURM_SCRIPT="${REPO_ROOT}/rl/run_grpo_slurm.sh"
STAMP="${SWEEP_STAMP:-$(date +%Y%m%d_%H%M%S)}"
# Dedicated project keeps G-vs-σ runs separate from scaffold / KL sweeps.
GRPO_WANDB_PROJECT="${1:-${GRPO_WANDB_PROJECT:-chemflow-grpo-g-only-${STAMP}}}"
GRPO_WANDB_GROUP="${GRPO_WANDB_GROUP:-$STAMP}"
export GRPO_WANDB_PROJECT GRPO_WANDB_GROUP

# Columns: GROUP_SIZE  SIGMA_EXPLORE — minimal four corners (baseline + three G/noise contrasts).
CONFIGS=(
  "1 0.05"
  "4 0.05"
  "4 0.10"  
  "2 0.10"
)

SEED="${SEED:-0}"
KL_COEF="${KL_COEF:-0.05}"
LR="${LR:-1e-4}"
UPDATE_PASSES="${UPDATE_PASSES:-2}"
N_UPDATES="${N_UPDATES:-50}"
SCAFFOLD_DIVERSITY="${SCAFFOLD_DIVERSITY:-0}"
PER_ELEMENT_LOGP_MEAN="${PER_ELEMENT_LOGP_MEAN:-0}"
# Explicit: run_grpo_slurm.sh also defaults to 1; set here so the sweep is self-contained.
KL_OMIT_POS="${KL_OMIT_POS:-1}"
SWEEP_TIME="${SWEEP_TIME:-2:00:00}"
n_jobs="${#CONFIGS[@]}"

echo "Submitting ${n_jobs} jobs (G × σ sweep, scaffold OFF, no per-element)"
echo "  W&B project: ${GRPO_WANDB_PROJECT}"
echo "  W&B group:   ${GRPO_WANDB_GROUP}"
echo "  Fixed: kl=${KL_COEF} lr=${LR} mu=${UPDATE_PASSES} n_updates=${N_UPDATES} kl_omit_pos=${KL_OMIT_POS} seed=${SEED}"
echo

submit_one() {
  local j="g${GROUP_SIZE}_sig${SIGMA_EXPLORE}_kl${KL_COEF}_lr${LR}"
  j="${j//./p}"
  if [[ -n "${DRY_RUN:-}" ]]; then
    echo "DRY: sbatch -t ${SWEEP_TIME} -J grpo-${j} --export=ALL ${SLURM_SCRIPT}"
    return 0
  fi
  export SIGMA_EXPLORE GROUP_SIZE KL_COEF LR SEED UPDATE_PASSES N_UPDATES \
    SCAFFOLD_DIVERSITY PER_ELEMENT_LOGP_MEAN KL_OMIT_POS
  sbatch -t "${SWEEP_TIME}" -J "grpo-${j}" --export=ALL "${SLURM_SCRIPT}"
}

for cfg in "${CONFIGS[@]}"; do
  read -r GROUP_SIZE SIGMA_EXPLORE <<< "$cfg"
  submit_one
done

if [[ -n "${DRY_RUN:-}" ]]; then
  echo "DRY_RUN: no jobs were submitted."
else
  echo "Done. Compare reward_within_std vs reward_between_std on G>1 runs."
  echo "  squeue -u \"\$USER\"    https://wandb.ai/<entity>/${GRPO_WANDB_PROJECT}"
fi
