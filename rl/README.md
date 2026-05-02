# Reinforcement learning (GRPO) — Chemflow

This folder implements **Group Relative Policy Optimization (GRPO)** fine-tuning on the Morph flow-matching model: rollout over the same discrete + continuous actions as `integrator.integrate_step_gnn` (positions with Gaussian exploration, atom/edge/charge/insertion topology, etc.). Run everything from the **repository root** unless a script says otherwise.

## Layout

| Path | Role |
|------|------|
| [`grpo.py`](grpo.py) | `GRPOConfig`, rollout storage, `train(...)` — algorithm and conventions (time direction, log-prob channels, KL / clipping). Start here for behavior details. |
| [`run_grpo.py`](run_grpo.py) | CLI: load Hydra config + checkpoint, attach reward, call `train`. Supports W&B, KL to frozen ref, scaffold/SMILES diversity wrapper, best-checkpoint saving. |
| [`rewards.py`](rewards.py) | Reward functions and `REWARDS` registry. New objectives: implement `(module, trajectory) -> (Tensor(B,), dict)` and register the name. |
| [`eval_pretrained_validity.py`](eval_pretrained_validity.py) | Sample / validity checks for a **frozen** pretrained checkpoint (shared Hydra setup with `run_grpo`). |

### Launch scripts

- [`run_grpo.sh`](run_grpo.sh) — Example local launch (`uv run` from repo root; see script for flags).
- [`run_grpo_slurm.sh`](run_grpo_slurm.sh) — **Single GPU** Slurm job; hyperparameters via env vars (`SIGMA_EXPLORE`, `SEED`, `KL_COEF`, `REWARD`, scaffold diversity, W&B project/group, …). Comments in-file list options.
- [`sweep_grpo_unkillable.sh`](sweep_grpo_unkillable.sh) — Run **on a login node** with `./rl/sweep_grpo_unkillable.sh` — it **submits** multiple `run_grpo_slurm.sh` jobs. **Do not** `sbatch` this file (it has no `#SBATCH` GPU allocation).

## Quick start

Checkpoint paths in this README, in `--ckpt` defaults (`run_grpo.py`), and in the launch shells (e.g. `.pretrained_model/...`) are **placeholders aligned with how this repo was set up**. Point `--ckpt` (and any hard-coded paths in your copy of `run_grpo.sh` / Slurm wrappers) at **your** checkpoint file and folder layout.

```bash
python -m rl.run_grpo \
  --ckpt .pretrained_model/epoch=499-step=48500.ckpt \
  --n_updates 100 \
  --reward n_atoms \
  --lr 1e-4 \
  data.datamodule.batch_size.test=128
```

Trailing arguments are **Hydra overrides** (same pattern as `eval_pretrained_validity.py`): config dir defaults to `configs/`, name to `default`.

Useful flags: `--sigma_explore`, `--kl_coef`, `--kl_omit_pos`, `--group_size`, `--update_passes`, `--per_element_logp_mean`, `--save` / `--save_best`, `--wandb` + project/name/group, and `--scaffold_diversity` with `--scaffold_diversity_key` (`murcko` | `canonical_smiles`).

## Rewards

Registered names (see [`rewards.py`](rewards.py)):

- `validity`, `qed`, `n_atoms`, `shape`

Built-in rewards gate on RDKit validity (invalid samples get zero reward). `--scaffold_diversity` layers REINVENT-style occurrence bucketing on top of any registered reward.

## GRPO training CLI (`python -m rl.run_grpo`)

| Flag | Role |
|------|------|
| `--ckpt` | Pretrained Lightning checkpoint to initialize weights. |
| `--config_path`, `--config_name` | Hydra (`configs` / `default` by default). |
| `--n_updates` | Policy gradient steps. |
| `--num_steps` | Integration steps per rollout (`None` = module default). |
| `--max_atoms` | Maximum atoms allowed by the variable-atom integrator. |
| `--lr` | Optimizer learning rate. |
| `--sigma_explore` | Position kernel std σ for `N(x+v·dt, σ²I)`. |
| `--clip_eps` | Clip range on probability ratios. |
| `--max_grad_norm` | Global grad clip (`≤0` disables). |
| `--group_size` | Rollouts per shared prompt within-batch advantage groups. |
| `--update_passes` | PPO-style passes over one sampled trajectory. |
| `--seed` | Python / NumPy / Torch seed. |
| `--kl_coef` | Reverse-KL to frozen ref; `0` skips ref forward. |
| `--kl_omit_pos` | Drop position channel from KL term only. |
| `--per_element_logp_mean` | Mean within each RL channel before summing channels. |
| `--device` | `cuda` or `cpu`. |
| `--log_every` | Logging interval. |
| `--reward` | `validity`, `qed`, `n_atoms`, or `shape`. |
| `--scaffold_diversity` | Enable bucket gating on top of `--reward`. |
| `--scaffold_diversity_key` | `murcko` or `canonical_smiles`. |
| `--scaffold_bucket_size`, `--scaffold_penalty`, `--scaffold_window_batches` | Capacity, multiplier when full (`0` = hard zero), rolling memory (`-1` = full run). |
| `--scaffold_labeled` | Labeled Murcko (no effect when key is `canonical_smiles`). |
| `--wandb`, `--wandb_project`, `--wandb_name`, `--wandb_group` | Weights and Biases logging. |
| `--save`, `--save_best` | Final `state_dict`; best-by-smoothed-reward checkpoint. |
| `--best_ema_beta`, `--best_warmup_steps` | EMA and warmup for `--save_best`. |
| trailing `overrides` | Hydra (e.g. `data.datamodule.batch_size.test=128`). |

After load, `run_grpo` sets `module.integrator.max_atoms` from `--max_atoms` (default `60`).

### Slurm: `run_grpo_slurm.sh` (environment variables)

Export or prefix before `sbatch rl/run_grpo_slurm.sh` (see script for defaults):

| Variable | Role |
|----------|------|
| `SIGMA_EXPLORE`, `SEED`, `LR`, `N_UPDATES`, `MAX_ATOMS` | Noise, reproducibility, learning rate, number of GRPO updates, atom cap. |
| `GROUP_SIZE`, `UPDATE_PASSES`, `KL_COEF`, `KL_OMIT_POS` | Group advantage normalization, multi-pass optimization, KL strength, KL on positions or not. |
| `PER_ELEMENT_LOGP_MEAN` | Truthy → `--per_element_logp_mean`. |
| `REWARD` | Maps to `--reward`. |
| `SCAFFOLD_DIVERSITY` | Truthy → `--scaffold_diversity` plus bucket flags. |
| `SCAFFOLD_DIVERSITY_KEY` | `murcko` or `canonical_smiles`. |
| `SCAFFOLD_BUCKET_SIZE`, `SCAFFOLD_PENALTY`, `SCAFFOLD_WINDOW_BATCHES`, `SCAFFOLD_LABELED` | Bucket behavior and Murcko labeling. |
| `GRPO_WANDB_PROJECT`, `GRPO_WANDB_GROUP` | W&B project and optional group. |

The embedded `--ckpt` path in that script is a placeholder—edit it for your checkpoint layout.

## Experiments (layout)

- [`experiments/natoms/`](experiments/natoms/) — Atom-count comparison, trajectory dumps, analysis notebooks.
- [`experiments/shape/`](experiments/shape/) — Shape-reward sample bundles and notebook scoring.

## `notes/`

The [`notes/`](notes/) markdown files are **not kept in sync** with the code. Prefer this README, the docstring at the top of [`grpo.py`](grpo.py), and inline comments in [`run_grpo.py`](run_grpo.py) / [`rewards.py`](rewards.py) for current behavior.
