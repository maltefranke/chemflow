# Stability Fixes (Phase 2 GRPO)

Running log of engineering + algorithmic fixes applied to `rl/grpo.py` and
friends, in the order they land.  Each entry states:

- **Problem** — what failure mode was observed (cite the training run / wandb
  run when possible).
- **Fix** — the concrete code change (one-liner summary; full diff in git).
- **Knobs** — new CLI flags / `GRPOConfig` fields introduced.
- **How to tell it's working** — specific numbers or metrics to watch in the
  next training run.

See also `DEPARTURES.md` for deviations from the overleaf derivations.
This file complements that one: everything here is *training stability*,
not mathematical correctness.

---

## 1. Gradient clipping + best-checkpoint tracking + save-dir mkdir

Date: 2026-04-22
Run that motivated this: `phase2-natoms-seed0_alpha0` (wandb `n0qlqqe4`).

### Problem

Two orthogonal issues in one run:

1. **Reward-hacking collapse + unclipped grads.**  With `reward=n_atoms`,
   `lr=1e-4`, `a_sde=0`, the policy climbed monotonically for 6 steps
   (`reward_mean` 17.3 → 23.9, `p_valid=1.0`) then collapsed by step 19
   (`reward_mean=10.1`, `p_valid=0.44`).  `grad_norm` doubled from 2.7 to
   6.6 during the collapse onset and briefly spiked to 8.7.  No gradient
   clipping was in place.  Step 5 was the peak and we had no way to
   recover it.

2. **Final-save crash.**  `run_grpo.py` called
   `torch.save({...}, args.save)` with `args.save=.rl_ckpts/...` but the
   parent directory did not exist:
   ```
   RuntimeError: Parent directory .rl_ckpts does not exist.
   ```
   The entire 50-step run was lost after wandb had already synced.

### Fix

`rl/grpo.py`:

- `GRPOConfig.max_grad_norm: float | None = 1.0`.
- `grpo_step(...)` now calls `torch.nn.utils.clip_grad_norm_` between
  the per-step `backward()` accumulation and `optimizer.step()`.
  - The returned value is logged as `grad_norm` (this is the *pre-clip*
    total norm, so historical plots stay comparable).
  - A new `grad_norm_post_clip` key is logged too.
  - `max_grad_norm=None` (or `<= 0` via CLI) disables clipping and falls
    back to the manual sum that was there before.
- `train(...)` now accepts `best_save_path`, `best_ema_beta=0.9`,
  `best_warmup_steps=3`.  Tracks an EMA of `reward_mean` and writes
  `{"state_dict", "step", "reward_ema", "reward_mean"}` to
  `best_save_path` every time the EMA hits a new max (after warmup).
  Also ensures the parent directory exists.

`rl/run_grpo.py`:

- New CLI flags: `--max_grad_norm` (default `1.0`), `--save_best`,
  `--best_ema_beta`, `--best_warmup_steps`.
- `os.makedirs(...)` on the parent of `--save` before the final
  `torch.save` — fixes the crash above.

`rl/run_grpo.sh`:

- Adds `--max_grad_norm 1.0` and
  `--save_best .rl_ckpts/grpo_natoms_seed0_alpha0_best.pt`.
- Adds the missing trailing `\` so
  `data.datamodule.batch_size.test=128` is actually part of the command
  (it was being executed as a separate, silently-failing shell
  statement before).

### Knobs

| Flag / field           | Default | Meaning                                                      |
|------------------------|---------|--------------------------------------------------------------|
| `--max_grad_norm`      | `1.0`   | `clip_grad_norm_` threshold.  Pass `0` or negative to disable. |
| `--save_best`          | `None`  | Path for best-EMA-reward checkpoint.                         |
| `--best_ema_beta`      | `0.9`   | `ema = β·ema + (1−β)·reward`.  Effective window ≈ `1/(1−β)`. |
| `--best_warmup_steps`  | `3`     | Don't start saving until this many steps have elapsed.       |

### On the best-checkpoint metric

We use *smoothed* `reward_mean` — **not** `reward_mean × p_valid` —
because all built-in rewards in `rl/rewards.py` already multiply by
RDKit validity (invalid → 0).  Concretely:

- `validity_reward`: `reward_mean = p_valid`.
- `qed_reward`: `reward_mean = p_valid × qed_mean_valid`.
- `n_atoms_reward`: `reward_mean = p_valid × n_atoms_mean_valid`.

Verified on the collapse run: step 19 shows `reward_mean = 10.12`,
`p_valid = 0.44`, `n_atoms_mean_valid = 23.14`, and
`0.44 × 23.14 ≈ 10.2` ✓.  Multiplying by `p_valid` again would
double-penalise validity collapse.  Under the current metric, step 5
(`23.94 · 1.00`) correctly beats step 27 (`22.34 · 0.94`).

Smoothing matters: raw `reward_mean` swings ±5 on batch 128, so a
single lucky step shouldn't win the save.  β=0.9 gives ≈10-step
effective window, which roughly matches the observed oscillation
period of the collapse.

### How to tell it's working

Next training run with these defaults (same seed, same recipe) should:

1. **Never print `RuntimeError: Parent directory ... does not exist`**
   at the final `torch.save`.
2. Log `grad_norm_post_clip ≤ 1.0` for every step after clipping kicks
   in (roughly, when `grad_norm` exceeds 1.0 — in the reference run
   that was from step 1 onwards).
3. Print a line like
   `[grpo] best: step=0005 reward_ema=... -> .rl_ckpts/..._best.pt`
   whenever a new EMA peak is reached.  On the reference run we expect
   this to fire a few times in steps 3–7, then taper off as the policy
   plateaus or collapses.
4. Leave a `_best.pt` file on disk at the end of the run whose
   `reward_ema` should be ≥ the final-step `reward_ema` in the wandb
   log.  That is the recovery mechanism for the step-5-peak problem.

### Not yet addressed

Logged here for the next entries:

- **KL penalty to a frozen reference policy** (bucket 2, next).  The
  `grad_norm_post_clip = 1.0` ceiling prevents runaway updates but
  does not prevent the *direction* of drift; KL is the principled fix.
- **Multi-epoch inner loop** — with μ=1 the PPO ratio is always 1 at
  the first inner step, so `clip_eps` is never active.  Clipping
  grads is a band-aid until we actually use the PPO clip.
- **Per-prompt group-relative advantage** — `adv = (r − r.mean())/std`
  is currently over the whole batch; GRPO proper standardises within
  groups sharing the same starting latent.

---

<!-- Next entry template:

## 2. ...

Date: YYYY-MM-DD
Run that motivated this: `...`

### Problem
### Fix
### Knobs
### How to tell it's working
### Not yet addressed

-->
