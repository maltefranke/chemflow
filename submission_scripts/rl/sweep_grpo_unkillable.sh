#!/usr/bin/env bash
#
# WRONG:  sbatch submission_scripts/rl/sweep_grpo_unkillable.sh
# RIGHT:  ./submission_scripts/rl/sweep_grpo_unkillable.sh    (from repo root; submits jobs via submission_scripts/rl/run_grpo_slurm.sh)
#
# Four-run batch: tighten around g8 vs g4 winners (canonical_smiles scaffold diversity on).
#
#   A — g8, kl=0.02, σ=0.05   official g8 comparison
#   B — g8, kl=0.03, σ=0.05   safer KL (regularization vs late validity crash)
#   C — g4, kl=0.03, σ=0.05   current-best analogue + safer KL
#   D — g4, kl=0.02, σ=0.03   lower exploration vs late invalidity spike
#
# Fixed across runs: LR=1e-4, UPDATE_PASSES=2, N_UPDATES=200,
# scaffold_b10_p0.5_w50 canonical_smiles, KL_OMIT_POS=1 unless overridden by env.
#
# Usage:
#   ./submission_scripts/rl/sweep_grpo_unkillable.sh
#   ./submission_scripts/rl/sweep_grpo_unkillable.sh chemflow-grpo-other-project   # override W&B project
#   SWEEP_STAMP=my001 DRY_RUN=1 ./submission_scripts/rl/sweep_grpo_unkillable.sh
#
# Default W&B project matches submission_scripts/rl/run_grpo_slurm.sh (change both if you retarget logs).
#
set -euo pipefail

if [[ -n "${SLURM_JOB_ID:-}" && "${SLURM_JOB_NAME:-}" == "sweep_grpo_unkillable.sh" ]]; then
  echo "error: do not sbatch this file — it is not a GPU batch script (no #SBATCH)." >&2
  echo "  run from repo root:  ./submission_scripts/rl/sweep_grpo_unkillable.sh" >&2
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
mkdir -p slurm_logs

SLURM_SCRIPT="${REPO_ROOT}/submission_scripts/rl/run_grpo_slurm.sh"
STAMP="${SWEEP_STAMP:-$(date +%Y%m%d_%H%M%S)}"
# Same fallback as submission_scripts/rl/run_grpo_slurm.sh GRPO_WANDB_PROJECT
GRPO_WANDB_PROJECT="${1:-${GRPO_WANDB_PROJECT:-chemflow-grpo-sweep-20260424_111439}}"
GRPO_WANDB_GROUP="${GRPO_WANDB_GROUP:-$STAMP}"
export GRPO_WANDB_PROJECT GRPO_WANDB_GROUP

# Columns: GROUP_SIZE  KL_COEF  SIGMA_EXPLORE (see header for A–D labels).
CONFIGS=(
  "# A official g8"
  "8 0.02 0.05"
  "# B safer g8"
  "8 0.03 0.05"
  "# C safer g4"
  "4 0.03 0.05"
  "# D lower-noise g4"
  "4 0.02 0.03"
)
 
SEED="${SEED:-0}"
LR="${LR:-1e-4}"
UPDATE_PASSES="${UPDATE_PASSES:-2}"
N_UPDATES="${N_UPDATES:-200}"
MAX_ATOMS="${MAX_ATOMS:-60}"
SCAFFOLD_DIVERSITY="${SCAFFOLD_DIVERSITY:-1}"
SCAFFOLD_DIVERSITY_KEY="${SCAFFOLD_DIVERSITY_KEY:-canonical_smiles}"
SCAFFOLD_BUCKET_SIZE="${SCAFFOLD_BUCKET_SIZE:-10}"
SCAFFOLD_PENALTY="${SCAFFOLD_PENALTY:-0.5}"
SCAFFOLD_WINDOW_BATCHES="${SCAFFOLD_WINDOW_BATCHES:-50}"
PER_ELEMENT_LOGP_MEAN="${PER_ELEMENT_LOGP_MEAN:-0}"
KL_OMIT_POS="${KL_OMIT_POS:-1}"
SWEEP_TIME="${SWEEP_TIME:-6:00:00}"

# Count non-comment configs.
n_jobs=0
for cfg in "${CONFIGS[@]}"; do
  [[ "$cfg" =~ ^# ]] && continue
  n_jobs=$((n_jobs + 1))
done

echo "Submitting ${n_jobs} jobs (g × kl × σ, scaffold canonical_smiles b${SCAFFOLD_BUCKET_SIZE})"
echo "  W&B project: ${GRPO_WANDB_PROJECT}"
echo "  W&B group:   ${GRPO_WANDB_GROUP}"
echo "  Fixed: lr=${LR} mu=${UPDATE_PASSES} n_updates=${N_UPDATES} max_atoms=${MAX_ATOMS} kl_omit_pos=${KL_OMIT_POS} seed=${SEED}"
echo "  Scaffold: penalty=${SCAFFOLD_PENALTY} window_batches=${SCAFFOLD_WINDOW_BATCHES}"
echo

submit_one() {
  local j="g${GROUP_SIZE}_sig${SIGMA_EXPLORE}_kl${KL_COEF}_lr${LR}"
  j="${j//./p}"
  if [[ -n "${DRY_RUN:-}" ]]; then
    echo "DRY: sbatch -t ${SWEEP_TIME} -J grpo-${j} --export=ALL ${SLURM_SCRIPT}"
    return 0
  fi
  export SIGMA_EXPLORE GROUP_SIZE KL_COEF LR SEED UPDATE_PASSES N_UPDATES \
    MAX_ATOMS \
    SCAFFOLD_DIVERSITY SCAFFOLD_DIVERSITY_KEY SCAFFOLD_BUCKET_SIZE SCAFFOLD_PENALTY \
    SCAFFOLD_WINDOW_BATCHES PER_ELEMENT_LOGP_MEAN KL_OMIT_POS
  sbatch -t "${SWEEP_TIME}" -J "grpo-${j}" --export=ALL "${SLURM_SCRIPT}"
}

for cfg in "${CONFIGS[@]}"; do
  if [[ "$cfg" =~ ^# ]]; then
    continue
  fi
  read -r GROUP_SIZE KL_COEF SIGMA_EXPLORE <<< "$cfg"
  submit_one
done

if [[ -n "${DRY_RUN:-}" ]]; then
  echo "DRY_RUN: no jobs were submitted."
else
  echo "Done. Compare runs A–D on W&B (${GRPO_WANDB_PROJECT})."
  echo "  squeue -u \"\$USER\"    https://wandb.ai/<entity>/${GRPO_WANDB_PROJECT}"
fi
