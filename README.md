# Chemflow

Training code for flow-matching molecular generation models on QM9, with Hydra config composition, PyTorch Lightning training, and Slurm helpers for local clusters and CSCS.

## Repository layout

- `run.py`: main training entry point.
- `configs/`: Hydra configs (`default.yaml`, `data/`, `model/`, `trainer/`, `schedule/`, `logging/`, `callbacks/`).
- `src/chemflow/`: dataset, interpolation, model, and training module implementations.
- `setup_env.sh`: local environment bootstrap script.
- `submit.sh`, `submit_job.sh`: basic Slurm submission scripts.
- `submit_cscs_job.sh`: CSCS containerized Slurm launcher.
- `submit_single_cscs_job.sh`: one-off CSCS job helper with env-var controls.
- `sweep.sh`: shell-based hyperparameter sweep launcher (one Slurm job per combination).
- `sweep.yaml`: W&B sweep template for constructing Slurm-wrapped commands.

## Environment setup (local)

```bash
source setup_env.sh
source .venv/bin/activate
```

`setup_env.sh` creates:
- `.env` with `PROJECT_ROOT`, `HYDRA_JOBS`, and `WANDB_DIR`
- `.venv` via `uv venv`
- editable install (`uv pip install -e .`)
- `torch-cluster` wheels matched to your installed PyTorch version

## Training locally

Default run (from repo root):

```bash
python run.py
```

Change model and selected options with Hydra overrides:

```bash
python run.py model=egnn data.n_atoms_strategy=flexible trainer.trainer.max_epochs=200
```

The default config stack is defined in `configs/default.yaml`:
- `data=qm9`
- `model=semla`
- `schedule=smoothstep`
- `trainer=trainer`
- `callbacks=callbacks`
- `logging=wandb_logger`

## Slurm usage

### Basic local-cluster submission

```bash
sbatch submit.sh
```

or with explicit overrides through `submit_job.sh` (edit script contents to match your cluster environment and desired overrides).

### CSCS containerized submission

`submit_cscs_job.sh` is the low-level launcher called by higher-level sweep helpers:

```bash
sbatch <resource_flags> submit_cscs_job.sh <run_tag> "<hydra_overrides>"
```

It runs inside the configured EDF environment (`CONTAINER_ENV`, default `~/.edf/chemflow/chemflow.toml`) and executes:
- editable install in `/app`
- `python run.py ...` with your overrides

### Single CSCS run helper

`submit_single_cscs_job.sh` builds a run tag and override string for one job:

```bash
bash submit_single_cscs_job.sh
```

Common controls:
- `SINGLE_MODEL=semla|egnn`
- `SINGLE_GLOBAL_BUDGET_WEIGHT=<float>`
- `SINGLE_SBATCH_ARGS="..."`
- `SINGLE_RUN_TAG=<name>`
- `SINGLE_EXTRA_OVERRIDES="key=value ..."`
- `SINGLE_DRY_RUN=1` (print command only)

## Hyperparameter sweeps

### Shell sweep (`sweep.sh`)

Submits one Slurm job per Cartesian product from the arrays defined in the script:

```bash
bash sweep.sh
```

Useful flags:
- `SWEEP_DRY_RUN=1` prints sbatch commands without submitting
- `SWEEP_ONE_JOB=1` submits only the first combination
- `SWEEP_RUN_CAP=<N>` stops after `N` submissions
- `SWEEP_SBATCH_ARGS="..."` overrides account/partition/resource flags

### W&B sweep template (`sweep.yaml`)

`sweep.yaml` is a template for parameter sampling and command construction. Update the command block and sbatch options for your site before use.

## Notes

- Paths in Slurm scripts are currently cluster-specific and should be adapted for your environment.
- Logs/output roots in CSCS scripts default to `/capstor/store/cscs/swissai/a131/frankem/chemflow`.