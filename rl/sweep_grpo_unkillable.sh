#!/usr/bin/env bash
#
# WRONG:  sbatch rl/sweep_grpo_unkillable.sh
#   This file has no #SBATCH lines. Slurm will use the default queue (e.g. long-cpu, GPU=0).
#
# RIGHT:  ./rl/sweep_grpo_unkillable.sh
#   Run that on a login/head node. It *submits* N separate jobs; each is rl/run_grpo_slurm.sh
#   (unkillable + GPU per run_grpo_slurm.sh).
#
# Sparse sweep (not a full cross-product): sigma_explore, lr, kl.
# Each run uses 100 policy updates and 100 integration steps (see run_grpo_slurm.sh, N_UPDATES).
# Expect ~80 min per run; default SWEEP_TIME 2:00:00 is plenty.
#
# W&B: each invocation gets its own *project* (a separate "folder" in the UI) so
# runs do not mix with ad-hoc chemflow-grpo jobs.  Pass a project name as the
# first argument, or set GRPO_WANDB_PROJECT / SWEEP_STAMP in the environment.
# All runs in the sweep also share a W&B *group* (same STAMP) for table filters.
#
# Usage (from the repository root, where `sbatch` and `.env` live):
#   ./rl/sweep_grpo_unkillable.sh
#   ./rl/sweep_grpo_unkillable.sh chemflow-grpo-ablate-2026-04-24
#   SWEEP_STAMP=manual001 ./rl/sweep_grpo_unkillable.sh
#
# One-off job (e.g. `--per_element_logp_mean`) in the *same* W&B project as an
# existing sweep stamp — see `rl/submit_grpo_elementmean_slurm.sh` or set
# `GRPO_WANDB_PROJECT` / `GRPO_WANDB_GROUP` and `PER_ELEMENT_LOGP_MEAN=1`
# before `sbatch rl/run_grpo_slurm.sh`.
#
# Dry run (print sbatch lines only, do not submit):
#   DRY_RUN=1 ./rl/sweep_grpo_unkillable.sh
#   SWEEP_TIME=1:30:00 ./rl/sweep_grpo_unkillable.sh   # per-job time limit (default 2:00:00)
#
set -euo pipefail

# Catches mistaken `sbatch rl/sweep_grpo_unkillable.sh` (default job name = script basename).
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
GRPO_WANDB_PROJECT="${1:-${GRPO_WANDB_PROJECT:-chemflow-grpo-sweep-${STAMP}}}"
GRPO_WANDB_GROUP="${GRPO_WANDB_GROUP:-$STAMP}"
export GRPO_WANDB_PROJECT GRPO_WANDB_GROUP

# Columns: SIGMA_EXPLORE  KL_COEF  LR
CONFIGS=(
  "0.01 0.05 1e-4"   # very low exploration
  "0.01 0.05 3e-4"
  "0.05 0.05 1e-4"
  "0.05 0.05 3e-4"
  "0.05 0.1  1e-4"
  "0.05 0.1  3e-4"
  "0.1  0.05 1e-4"
  "0.1  0.05 3e-4"
  "0.1  0.1  1e-4"
  "0.1  0.1  3e-4"
)

SEED="${SEED:-0}"
GROUP_SIZE="${GROUP_SIZE:-1}"
UPDATE_PASSES="${UPDATE_PASSES:-1}"
SWEEP_TIME="${SWEEP_TIME:-2:00:00}"
n_jobs="${#CONFIGS[@]}"

echo "Submitting ${n_jobs} jobs to partition unkillable (from ${SLURM_SCRIPT})"
echo "  W&B project: ${GRPO_WANDB_PROJECT}"
echo "  W&B group:   ${GRPO_WANDB_GROUP}"
echo

submit_one() {
  # Unique Slurm name (no dots: partition/job tools tolerate alnum+_-)
  local j="s${SIGMA_EXPLORE}_k${KL_COEF}_lr${LR}"
  j="${j//./p}"
  if [[ -n "${DRY_RUN:-}" ]]; then
    echo "DRY: sbatch -t ${SWEEP_TIME} -J grpo-${j} --export=ALL ${SLURM_SCRIPT}"
    return 0
  fi
  export SIGMA_EXPLORE KL_COEF LR SEED GROUP_SIZE UPDATE_PASSES
  # Pass full env to the job; keep GRPO_WANDB_* from this script.
  sbatch -t "${SWEEP_TIME}" -J "grpo-${j}" --export=ALL "${SLURM_SCRIPT}"
}

for cfg in "${CONFIGS[@]}"; do
  read -r SIGMA_EXPLORE KL_COEF LR <<< "$cfg"
  submit_one
done

if [[ -n "${DRY_RUN:-}" ]]; then
  echo "DRY_RUN: no jobs were submitted."
else
  echo "Done. Check: squeue -u \"\$USER\"  and  https://wandb.ai/<entity>/${GRPO_WANDB_PROJECT}"
fi
