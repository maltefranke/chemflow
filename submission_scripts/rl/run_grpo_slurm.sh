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

# Line-buffer stdout/stderr in Slurm (otherwise the first [grpo] step can appear late).
export PYTHONUNBUFFERED=1

# Run from the directory where `sbatch` was invoked.
cd "${SLURM_SUBMIT_DIR:-$PWD}"

export PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
mkdir -p slurm_logs

source .env
source .venv/bin/activate

# tmqm_gap reward runs xTB relaxations in a CPU process pool. Pin the worker
# count to the Slurm CPU allocation (leave one core for the main/GPU thread).
# os.cpu_count() would otherwise report the whole node and oversubscribe.
export CHEMFLOW_XTB_JOBS="${CHEMFLOW_XTB_JOBS:-$(( ${SLURM_CPUS_PER_TASK:-6} - 1 ))}"

# CRITICAL: pin OpenMP/BLAS to one thread per xTB worker. These MUST be set
# before Python starts: spawn workers re-import run_grpo (-> tblite) at
# bootstrap, before the pool initializer can run, so tblite's OpenMP would
# otherwise grab the whole node's cores per worker and thrash (~40x slowdown).
# Spawn children inherit this parent env, which is the only reliable fix.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_STACKSIZE=1G
export OMP_MAX_ACTIVE_LEVELS=1

echo "host=$(hostname)  gpus=${CUDA_VISIBLE_DEVICES:-unset}  xtb_jobs=${CHEMFLOW_XTB_JOBS}"
nvidia-smi -L || true

# tmQM HOMO-LUMO gap GRPO. Reward = tmqm_gap (validity-gated gap); all
# molecule/organic wrappers are disabled in configs/rl/grpo.yaml for pointclouds.
uv run --active --no-sync --env-file .env python -m chemflow.rl.run_grpo \
    data=tmqm \
    model=dit \
    cfg=uncond \
    representation=pointcloud \
    "$@"
