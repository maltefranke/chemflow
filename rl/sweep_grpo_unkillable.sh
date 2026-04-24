#!/usr/bin/env bash
#
# WRONG:  sbatch rl/sweep_grpo_unkillable.sh
#   This file has no #SBATCH lines. Slurm will use the default queue (e.g. long-cpu, GPU=0).
#
# RIGHT:  ./rl/sweep_grpo_unkillable.sh
#   Run that on a login/head node. It *submits* N separate jobs; each is rl/run_grpo_slurm.sh
#   (unkillable + GPU per run_grpo_slurm.sh).
#
# Launch a full grid of GRPO jobs: one Slurm job per (a_sde × kl_coef × lr) with GROUP_SIZE=1.
# Assumes each run is ~40 min; run_grpo_slurm.sh time limit (4h) is plenty.
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

# Grid — edit to ablate (3 × 3 × 2 = 18 jobs with defaults below).
A_SDE_LIST=(0 0.05 0.1)
KL_COEF_LIST=(0.01 0.05 0.1)
LR_LIST=(3e-5 1e-4)
SEED="${SEED:-0}"
GROUP_SIZE="${GROUP_SIZE:-1}"
SWEEP_TIME="${SWEEP_TIME:-2:00:00}"

n_jobs=0
for A_SDE in "${A_SDE_LIST[@]}"; do
  for KL_COEF in "${KL_COEF_LIST[@]}"; do
    for LR in "${LR_LIST[@]}"; do
      n_jobs=$((n_jobs + 1))
    done
  done
done

echo "Submitting ${n_jobs} jobs to partition unkillable (from ${SLURM_SCRIPT})"
echo "  W&B project: ${GRPO_WANDB_PROJECT}"
echo "  W&B group:   ${GRPO_WANDB_GROUP}"
echo

submit_one() {
  # Unique Slurm name (no dots: partition/job tools tolerate alnum+_-)
  local j="a${A_SDE}_k${KL_COEF}_lr${LR}"
  j="${j//./p}"
  if [[ -n "${DRY_RUN:-}" ]]; then
    echo "DRY: sbatch -t ${SWEEP_TIME} -J grpo-${j} --export=ALL ${SLURM_SCRIPT}"
    return 0
  fi
  export A_SDE KL_COEF LR SEED GROUP_SIZE
  # Pass full env to the job; keep GRPO_WANDB_* from this script.
  sbatch -t "${SWEEP_TIME}" -J "grpo-${j}" --export=ALL "${SLURM_SCRIPT}"
}

for A_SDE in "${A_SDE_LIST[@]}"; do
  for KL_COEF in "${KL_COEF_LIST[@]}"; do
    for LR in "${LR_LIST[@]}"; do
      submit_one
    done
  done
done

if [[ -n "${DRY_RUN:-}" ]]; then
  echo "DRY_RUN: no jobs were submitted."
else
  echo "Done. Check: squeue -u \"\$USER\"  and  https://wandb.ai/<entity>/${GRPO_WANDB_PROJECT}"
fi
