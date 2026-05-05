# Reinforcement learning (GRPO) — Chemflow

This folder implements **Group Relative Policy Optimization (GRPO)** fine-tuning on the Morph flow-matching model: rollout over the same discrete + continuous actions as `integrator.integrate_step_gnn` (positions with Gaussian exploration, atom/edge/charge/insertion topology, etc.). Run everything from the **repository root** unless a script says otherwise.

## Layout

| Path | Role |
|------|------|
| [`grpo.py`](grpo.py) | `GRPOConfig`, rollout storage, `train(...)` — algorithm and conventions (time direction, log-prob channels, KL / clipping). Start here for behavior details. |
| [`run_grpo.py`](run_grpo.py) | CLI and shared setup helpers: load Hydra config + checkpoint, attach reward, call `train`. Supports W&B, KL to frozen ref, scaffold/SMILES diversity wrapper, best-checkpoint saving. |
| [`rewards.py`](rewards.py) | Reward functions and `REWARDS` registry. New objectives: implement `(module, trajectory) -> (Tensor(B,), dict)` and register the name. |

### Launch scripts

- [`run_grpo.sh`](run_grpo.sh) — Example local launch (`uv run` from repo root; env vars become Hydra overrides).
- [`run_grpo_slurm.sh`](run_grpo_slurm.sh) — **Single GPU** Slurm job; hyperparameters via env vars (`SIGMA_EXPLORE`, `SEED`, `KL_COEF`, `REWARD`, scaffold diversity, W&B project/group, …). Comments in-file list options.
- [`sweep_grpo_unkillable.sh`](sweep_grpo_unkillable.sh) — Run **on a login node** with `./rl/sweep_grpo_unkillable.sh` — it **submits** multiple `run_grpo_slurm.sh` jobs. **Do not** `sbatch` this file (it has no `#SBATCH` GPU allocation).

## Quick start

Checkpoint paths in this README, in [`configs/rl/grpo.yaml`](../configs/rl/grpo.yaml), and in the launch shells (e.g. `.pretrained_model/...`) are **placeholders aligned with how this repo was set up**. Point `rl.ckpt` (and any hard-coded paths in your copy of `run_grpo.sh` / Slurm wrappers) at **your** checkpoint file and folder layout.

```bash
python -m rl.run_grpo \
  rl.ckpt=.pretrained_model/epoch=499-step=48500.ckpt \
  rl.n_updates=100 \
  rl.reward.name=n_atoms \
  rl.lr=1e-4 \
  data.datamodule.batch_size.test=128
```

Arguments are **Hydra overrides** on top of `configs/rl/grpo.yaml`, which itself composes the normal ChemFlow `configs/default.yaml` stack. Inspect the composed job with:

```bash
python -m rl.run_grpo --cfg job
```

Useful overrides: `rl.grpo.sigma_explore`, `rl.grpo.kl_coef`, `rl.grpo.kl_omit_pos`, `rl.grpo.group_size`, `rl.grpo.update_passes`, `rl.grpo.per_element_logp_mean`, `rl.save` / `rl.save_best`, `rl.wandb.*`, and `rl.reward.scaffold_diversity` with `rl.reward.scaffold_diversity_key` (`murcko` | `canonical_smiles`).

## Rewards

Registered names (see [`rewards.py`](rewards.py)):

- `validity`, `qed`, `n_atoms`, `shape`, `tanimoto`

Built-in rewards gate on RDKit validity (invalid samples get zero reward). `rl.reward.scaffold_diversity=true` layers REINVENT-style occurrence bucketing on top of any registered reward.

## GRPO training CLI (`python -m rl.run_grpo`)

| Override | Role |
|------|------|
| `rl.ckpt` | Pretrained Lightning checkpoint to initialize weights. |
| `rl.n_updates` | Policy gradient steps. |
| `rl.grpo.num_integration_steps` | Integration steps per rollout (`null` = module default). |
| `rl.max_atoms` | Maximum atoms allowed by the variable-atom integrator. |
| `rl.lr` | Optimizer learning rate. |
| `rl.grpo.sigma_explore` | Position kernel std σ for `N(x+v·dt, σ²I)`. |
| `rl.grpo.clip_eps` | Clip range on probability ratios. |
| `rl.grpo.max_grad_norm` | Global grad clip (`null` or `≤0` disables). |
| `rl.grpo.group_size` | Rollouts per shared prompt within-batch advantage groups. |
| `rl.grpo.update_passes` | PPO-style passes over one sampled trajectory. |
| `rl.seed` | Python / NumPy / Torch seed. |
| `rl.grpo.kl_coef` | Reverse-KL to frozen ref; `0` skips ref forward. |
| `rl.grpo.kl_omit_pos` | Drop position channel from KL term only. |
| `rl.grpo.per_element_logp_mean` | Mean within each RL channel before summing channels. |
| `rl.device` | `auto`, `cuda`, or `cpu`. |
| `rl.log_every` | Logging interval. |
| `rl.reward.name` | `validity`, `qed`, `n_atoms`, `shape`, or `tanimoto`. |
| `rl.reward.scaffold_diversity` | Enable bucket gating on top of `rl.reward.name`. |
| `rl.reward.scaffold_diversity_key` | `murcko` or `canonical_smiles`. |
| `rl.reward.scaffold_bucket_size`, `rl.reward.scaffold_penalty`, `rl.reward.scaffold_window_batches` | Capacity, multiplier when full (`0` = hard zero), rolling memory (`-1` = full run). |
| `rl.reward.scaffold_labeled` | Labeled Murcko (no effect when key is `canonical_smiles`). |
| `rl.wandb.enabled`, `rl.wandb.project`, `rl.wandb.name`, `rl.wandb.group` | Weights and Biases logging. |
| `rl.save`, `rl.save_best` | Final `state_dict`; best-by-smoothed-reward checkpoint. |
| `rl.best.ema_beta`, `rl.best.warmup_steps` | EMA and warmup for `rl.save_best`. |
| Any normal ChemFlow key | Hydra override for model/data/trainer config, e.g. `data.datamodule.batch_size.test=128`. |

After load, `run_grpo` sets `module.integrator.max_atoms` from `rl.max_atoms` (default `60`).

### Slurm: `run_grpo_slurm.sh` (environment variables)

Export or prefix before `sbatch rl/run_grpo_slurm.sh` (see script for defaults):

| Variable | Role |
|----------|------|
| `SIGMA_EXPLORE`, `SEED`, `LR`, `N_UPDATES`, `MAX_ATOMS` | Noise, reproducibility, learning rate, number of GRPO updates, atom cap. |
| `GROUP_SIZE`, `UPDATE_PASSES`, `KL_COEF`, `KL_OMIT_POS` | Group advantage normalization, multi-pass optimization, KL strength, KL on positions or not. |
| `PER_ELEMENT_LOGP_MEAN` | Truthy → `rl.grpo.per_element_logp_mean=true`. |
| `REWARD` | Maps to `rl.reward.name`. |
| `SCAFFOLD_DIVERSITY` | Truthy → `rl.reward.scaffold_diversity=true` plus bucket overrides. |
| `SCAFFOLD_DIVERSITY_KEY` | `murcko` or `canonical_smiles`. |
| `SCAFFOLD_BUCKET_SIZE`, `SCAFFOLD_PENALTY`, `SCAFFOLD_WINDOW_BATCHES`, `SCAFFOLD_LABELED` | Bucket behavior and Murcko labeling. |
| `GRPO_WANDB_PROJECT`, `GRPO_WANDB_GROUP` | W&B project and optional group. |

The embedded `rl.ckpt` path in that script is a placeholder—edit it for your checkpoint layout.

## Experiments (layout)

- [`experiments/natoms/`](experiments/natoms/) — Atom-count comparison, trajectory dumps, analysis notebooks.
- [`experiments/shape/`](experiments/shape/) — Shape-reward sample bundles and notebook scoring.
- [`experiments/tanimoto/`](experiments/tanimoto/) — Tanimoto-reward sample bundles and top-unique-molecule notebook for Prilocaine similarity.

## `notes/`

The [`notes/`](notes/) markdown files are **not kept in sync** with the code. Prefer this README, the docstring at the top of [`grpo.py`](grpo.py), and inline comments in [`run_grpo.py`](run_grpo.py) / [`rewards.py`](rewards.py) for current behavior.
