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

### Post-run verdict (wandb run `b2ejeqok`)

**Save + best-ckpt: worked.**  No crash at final `torch.save`.
Best-ckpt fired on steps 3–6, peak EMA `19.45` at step 6, checkpoint
landed at `.rl_ckpts/grpo_natoms_seed0_alpha0_best.pt`.  The EMA
smoother correctly picked step 6 (robust plateau) over the later
single-step peak at step 13 (`reward_mean=24.84` but already in the
middle of the collapse).

**Grad clip: fired as intended, but insufficient.**  From step 3 onwards
`grad_norm_post_clip = 1.0` (i.e. pre-clip norm exceeded threshold).
Despite the clip, the run still collapsed — and actually collapsed
*harder* than the pre-clip baseline:

| Metric                | Pre-clip run (`n0qlqqe4`) | Clip run (`b2ejeqok`) |
|-----------------------|---------------------------|------------------------|
| Peak `reward_mean`    | 23.94 (step 5)            | 24.84 (step 13)        |
| Minimum `p_valid`     | 0.44 (step 19)            | **0.02** (step 16)     |
| Final `reward_mean`   | 20.09 (step 49)           | 0.59 (step 16, run aborted) |

So clipping smoothed the trajectory on both sides — higher peak,
deeper pit — without changing its *direction*.  Magnitude clip does
not prevent a consistent directional drift in the policy.

**Root cause: edge channel runaway.**  The per-graph-summed edge
log-prob drifts monotonically:

```
step: 0    3    5    6    7    13     14     15     16
lp/edge: -3.7 -9.5 -18  -41  -71  -142  -157  -166  -207
```

The edge categorical smears toward uniform / rare bond types.  Garbage
bonds → valence errors → `p_valid = 0.02`.  Signature: `signal/edge`
flips sign at step 7 (negative → positive) and stays positive through
the crash, i.e. "trajectories with rarer edges get higher advantage"
— exactly the reward-hacking pattern `n_atoms × validity` incentivises
when there's nothing pulling edge log-probs back toward the prior.

**Implication.**  Entry 2 (KL penalty to a frozen reference) is not
optional; it is the structural fix.  Grad clip bounded magnitude but
there is no restoring force on *direction* without KL.

### Not yet addressed

- **Multi-epoch inner loop** — with μ=1 the PPO ratio is always 1 at
  the first inner step, so `clip_eps` is never active.  Clipping
  grads is a band-aid until we actually use the PPO clip.
- (Done elsewhere) **Group-relative advantage** and **KL to ref** are
  implemented in `rl/grpo.py`; see the corresponding entries in this
  file.

---

## 2. KL penalty to a frozen reference policy (k3 per channel)

Date: 2026-04-23  
Implemented in: `rl/grpo.py`, `rl/run_grpo.py`, `rl/run_grpo.sh` / `rl/run_grpo_slurm.sh`.  
Motivated by: wandb run `b2ejeqok` (see §1 post-run verdict) — **edge
channel runaway** without a restoring force.

### Problem (unchanged)

Gradient clipping bounds magnitude, not direction.  A frozen reference
is the standard structural fix: anchor the finetuned policy to the
pretrained one.

### What we implemented

**Config / CLI:** `GRPOConfig.kl_coef: float` (default `0.0`); CLI
`--kl_coef` (same as β below).  `0` disables: **no** `deepcopy` of the
module, **no** extra forward pass, behaviour matches pre-KL code paths.

**Reference:** `ref_module = copy.deepcopy(module)` at `train()` start
**after** the checkpoint is loaded, then `eval()`, all parameters
`requires_grad_(False)`, same device.  Frozen for the full run.

**Per-step loss:**

$$
\mathcal{L}_k
  = \underbrace{-\min(r_k \hat A,\, \mathrm{clip}(r_k)\, \hat A)}_{\text{PPO/GRPO surrogate}}
  + \beta \cdot \frac{1}{B}\sum_{i=1}^B
      \sum_{c \in \text{channels}} \mathrm{k3}\!\left(
        t_{i,c};\;
        t_{i,c} = \Big[\log \pi_\mathrm{ref}^c(a) - \log \pi_\theta^c(a)\Big]_{\text{clamped}}
      \right)
$$

**Estimator:** **Schulman k3** per channel, *not* a hybrid analytic
formulation from the pre-implementation design notes.  Rationale
(documented in full when we revisited the plan): same clamp as
`log_ratio` avoids `exp` overflow; `B \cdot T` samples per update make
k3’s variance negligible; per-channel k3 is uniform across
`pos, node, edge, …` because every channel reuses the same
`_per_channel_logprob` (including GMM / ins edges); and it matches
common GRPO / RLHF practice.  **Sum over channels** is a practical
**factorised** penalty; it is *not* the exact joint
$\mathrm{KL}(\pi_\theta\,\|\,\pi_\mathrm{ref})$ on the one-step
product measure — if we need exact joint KL later, we can add a
**single** k3 on the **total** log-prob only.

`k3(t) = e^t - t - 1$ with $t = (lp^c_\mathrm{ref} - lp^c_\theta)$ clamped
to `[-log_ratio_clamp, log_ratio_clamp]`.

**Compute:** one policy forward (with per-channel `lp` in the
autograd path when `kl_coef>0`) + one `inference_mode` ref forward
per scoring step.  Rollout unchanged.  Memory: second full model when
`kl_coef>0`.

**Logging:** `loss` = PPO + KL; `loss_ppo`, `loss_kl`; `kl/total` =
unscaled mean k3 (i.e. `loss_kl / β` at fixed β); and `kl/{channel}`
unscaled.  (β is not baked into the per-channel log keys.)

**Default β in `run_grpo.sh`:** `0.05`.  `run_grpo_slurm.sh` defaults
`KL_COEF=0` so cluster jobs do not double VRAM until you opt in
(`KL_COEF=0.05 sbatch ...`).

### Deviations from the earlier (pre-code) “hybrid analytic + k1” spec

- Analytic per-channel + mixture bounds + exact GMM KL would be more
  precise in principle but is ~3× the code and duplicates every
  distributional path; the shipped **k3-per-channel** path reuses
  `_per_channel_logprob` for both models and is easier to keep
  correct.  We can add analytic `edge` / discrete KL later if
  `kl/edge` is noisy in wandb.
- The **position–var at $t\to 1$** issue (huge
  $\|\Delta\mu\|^2/2\mathrm{var}$ in closed-form Kullback–Leibler) is
  sidestepped: k3 is evaluated on the **observed** log-prob difference
  at sampled actions, not on $\Delta\mu/\sigma$.

### How to tell it's working (predictions for the next run)

On the `n_atoms` recipe with `--kl_coef 0.05`:

1. **`lp/edge` shouldn’t** reproduce `−3.7 → −207` in &lt;20 steps. If
   it does, **raise β** (or try `lr/2` first; LR and KL both affect drift).
2. **`p_valid`** should stay higher than a no-KL run with the same seed.
3. **`kl/edge` vs `kl/pos`:** the dominant channel in `kl/*` is the
   one the ref–θ gap is fighting; compare to `signal/*` for the reward.
4. **`kl/total`:** should rise early then level off, not run away
  linearly (if it does, β is too small).

### Open questions (still)

- If **ins–edge** `kl/ins_e_*` becomes dominant, revisit mixture-KL
  exacts or a dedicated term.
- **Multi-epoch** ($\mu>1$): k3 is biased; fix with importance
  sampling or the analytic per-step KL when that lands.

---

<!-- Next entry template:

## N. ...

Date: YYYY-MM-DD
Run that motivated this: `...`

### Problem
### Fix
### Knobs
### How to tell it's working
### Not yet addressed

-->
