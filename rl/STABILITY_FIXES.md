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

Logged here for the next entries:

- **KL penalty to a frozen reference policy** (entry 2, next — design
  notes below).  Required to prevent the edge-channel drift we just
  saw.  `grad_norm_post_clip = 1.0` caps update magnitude but not
  direction.
- **Multi-epoch inner loop** — with μ=1 the PPO ratio is always 1 at
  the first inner step, so `clip_eps` is never active.  Clipping
  grads is a band-aid until we actually use the PPO clip.
- **Per-prompt group-relative advantage** — `adv = (r − r.mean())/std`
  is currently over the whole batch; GRPO proper standardises within
  groups sharing the same starting latent.

---

## 2. [TODO] KL penalty to a frozen reference policy

Status: **design agreed, not yet implemented.**
Motivated by: wandb run `b2ejeqok` (see §1 post-run verdict).

### Problem

Gradient clipping bounds update magnitude, not direction.  The edge
channel drifts monotonically (`lp/edge: −3.7 → −207` in 16 steps) with
no restoring force, because advantage is consistently highest on
trajectories that place exotic edges to grow atom count.  A frozen
reference policy (= the pretrained model at `train()` start) provides
that restoring force via a KL-to-ref penalty added to the per-step
loss:

$$
\mathcal{L}_k = -\min(r_k \hat A,\; \mathrm{clip}(r_k) \hat A) \;+\; \beta \cdot \mathrm{KL}\bigl(\pi_\theta^{\mathrm{step}} \,\|\, \pi_\mathrm{ref}^{\mathrm{step}}\bigr).
$$

### Design: reverse KL, hybrid analytic + k1

- **Direction:** reverse KL $\mathrm{KL}(\pi_\theta \,\|\, \pi_\mathrm{ref})$
  (mode-seeking; explodes when $\pi_\theta$ puts mass where ref has
  near-zero — exactly the exotic-bond-type failure mode).  Standard
  RLHF choice.

- **Estimator: analytic per-channel where closed-form exists, k1
  (log-ratio) for GMM only.**  Unlike LLM-RLHF (|V|=50k categorical,
  analytic KL expensive, folklore uses k3 from Schulman's note), our
  action space is small and structured.  7 of 8 channels have
  cheap analytic KL:

  | Channel           | Dist.                                   | Analytic KL?                       |
  |-------------------|-----------------------------------------|-------------------------------------|
  | Positions         | $\mathcal{N}(\mu,\sigma^2_t dt \mathbf{I})$, same $\sigma$ both sides | Yes: $\|\mu_\theta-\mu_\mathrm{ref}\|^2/(2\sigma^2 dt)$ |
  | Node 3-way + sub  | Cat(3) × conditional Cat($\|A\|$)       | Yes: tree-structured categorical    |
  | Edge sub          | Bernoulli + conditional Cat($\|E\|$)    | Yes                                 |
  | Charge            | Cat($\|C\|$)                            | Yes                                 |
  | Insertion gate    | Bernoulli per node                      | Yes                                 |
  | **GMM**           | Mixture of $K$ components               | **No** (no closed form for GMM KL) |
  | Ins→existing edge | $(1-\kappa) p_\mathrm{prior} + \kappa\,\pi_\theta$ | Upper bound: $\kappa \cdot \mathrm{KL}(\pi_\theta\,\|\,\pi_\mathrm{ref})$ (prior cancels by convexity) |
  | Ins→ins edge      | Same mixture form                       | Same bound                          |

  For categorical: `(p_theta * (log p_theta - log p_ref)).sum(-1)`
  per token, then per-graph scatter-sum.  Exact, no sampling noise,
  no exp-overflow, same per-channel debuggability we already have
  in `_per_channel_logprob`.

  For GMM: k1 estimator on the stored `(ins_x, ins_a, ins_c)` samples,
  i.e. `logp_gmm_theta − logp_gmm_ref`.  Biased under multi-epoch,
  but GMM is a minor channel (current `lp/gmm ≈ −0.1` vs
  `lp/edge = −207`), so the bias lives far from where KL needs to be
  most accurate.

- **Why not pure k3** (Schulman's `r − 1 − log r` with $r = \pi_\mathrm{ref}/\pi_\theta$):
  1. Accuracy where it matters — on the channel that just collapsed
     (edge), analytic KL is the *exact* quantity in the loss, while
     k3 is a noisy estimator whose variance grows with policy drift
     (i.e. worst precisely when KL is largest).
  2. No exp-overflow.  k3 on edge log-ratios of O(200 nats) requires
     clamping, same problem we already saw with the PPO ratio.
  3. Per-token debuggability.  Analytic KL lets us see *which*
     edges/nodes drift; k3 is scalar per graph.
  4. Same compute cost — both need one forward pass through the ref.

- **Reference policy:** `copy.deepcopy(module)` at `train()` start,
  `eval()`, `requires_grad_(False)`.  Keep on GPU (second copy of the
  backbone — fits on A100-40G at batch 128 with comfortable margin).
  Frozen for the entire run.

- **Where to evaluate:** compute `preds_ref` at scoring time (one extra
  forward per `grpo_step`, rollout unchanged).  Cache `logp_ref`
  per-channel in `StepData` later when we add multi-epoch and the
  duplicated forward actually starts to matter.

- **Default β:** `0.05`.  Sanity estimate: a single edge drifting
  from one-hot to uniform-over-5 contributes ~log(5) ≈ 1.6 nats of
  KL; ~100 triu edges per graph ⇒ full drift ≈ 160 nats per graph
  per step; at β=0.05 that's −8 nats of loss pressure per step —
  same order as the `adv × ratio` magnitudes we see in training, so
  competitive without being dominant.  Expose as CLI flag
  `--kl_beta`; sweep later.

- **Logging:** add `kl/{pos,node,edge,charge,ins_gate,gmm,ins_e_ext,ins_e_ii}`
  and `kl/total` to the `info` dict, one entry per channel.

### Cost

- **VRAM:** second module copy.  Baseline run on A100-40G used a
  fraction of available; two copies still fit.
- **Time:** ~2× per `grpo_step` (two forwards instead of one during
  scoring).  Rollout unchanged.  A 50-step test run goes from
  ~1 min → ~2 min.
- **Code:** new function `_per_channel_kl(preds_theta, preds_ref, step, cfg, module)`
  alongside `_per_channel_logprob`.  ~100 lines, mostly mechanical
  (7 two-liner analytic formulas + one k1 term).

### Implementation checklist

- [ ] `GRPOConfig.kl_beta: float = 0.05`
- [ ] `train()` creates `ref_module = deepcopy(module).eval().requires_grad_(False)`
      **after** the checkpoint load, pass into `grpo_step`.
- [ ] `_forward_preds_ref(ref_module, mol_t, t, cfg)` — same adapter
      plumbing as `_forward_preds`, `@torch.no_grad()`.
- [ ] `_per_channel_kl(preds_theta, preds_ref, step, cfg, module)`:
      returns `(kl_total, kl_breakdown)` with same contract as
      `_per_channel_logprob`.  Channels per table above.
- [ ] `grpo_step` assembles loss as
      `step_loss = ppo_loss + cfg.kl_beta * kl_total.mean()`,
      divides by `n_steps`, backwards.
- [ ] CLI `--kl_beta` in `run_grpo.py`.
- [ ] Log `kl/{channel}` and `kl/total` in the info dict.
- [ ] `run_grpo.sh` gets `--kl_beta 0.05`.

### How to tell it's working (predictions for the next run)

On the same recipe (`reward=n_atoms`, `lr=1e-4`, `a_sde=0`, batch 128)
with `--kl_beta 0.05`, expect:

1. **`lp/edge` bounded.**  Instead of `−3.7 → −207`, expect drift
   capped at some steady-state value (guess: `|lp/edge| < 30`).  If
   `|lp/edge|` still crosses 50, increase β.
2. **`p_valid ≥ 0.8` throughout training.**  The policy can still
   reward-hack, but only within the KL ball around `ref` — and
   `ref` has `p_valid ≈ 1.0`.
3. **Monotone non-decreasing `reward_ema`** for the first ~20
   steps.  KL acts as a soft trust region; shouldn't observe the
   step-6 → step-16 crash.
4. **`kl/edge` is the dominant channel.**  If some other channel
   (e.g. positions) dominates, that's a clue the reward is pulling
   a direction we didn't anticipate.
5. **`kl/total` grows then plateaus.**  Initial growth = policy
   learning, plateau = KL penalty balancing reward signal.  If it
   grows linearly to ∞, β is too small.

### Open questions / things to revisit

- **Mixture-bound for ins-edge KL:** the convexity bound is tight
  when $\kappa \to 1$ (data end) but loose at $\kappa \to 0$ (noise
  end).  Most integration steps live in between.  Check empirically
  whether ins-edge KL is ever the dominant channel; if yes, replace
  with an exact mixture-KL derivation (tractable but tedious).
- **Position KL at $t \to 1$:** `var = σ²_t · dt` goes to `var_floor`
  (1e-6) at the data end.  Position KL is $\|\Delta\mu\|^2/(2\mathrm{var})$
  so small drifts blow up.  Same floor applied both sides means the
  KL is internally consistent but may dominate unnaturally; might
  need a channel-specific β or a separate clamp.
- **k1 for GMM** is unbiased only at µ=1.  Once multi-epoch lands
  (entry 3), either switch to MC from $\pi_\theta$ or accept the
  bias and log it.

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
