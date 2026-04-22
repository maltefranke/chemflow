# Departures from the Overleaf (Phase 1)

This document lists every place the implementation in `rl/grpo.py` deviates from
the derivations in the project's overleaf, and why.

## 1. Convention

Everything in `rl/grpo.py` is translated into **code convention**
($t{=}0$ noise, $t{=}1$ data, sampling $t: 0 \to 1$). The overleaf is written
in the opposite convention ($x_0$ data, $x_1$ noise).

The overleaf's boxed SDE

$$
dx = \left[v + \frac{\sigma_t^2}{2}\,\frac{x_t + (1-t)\,v}{t+\sigma_n^2}\right] dt + \sigma_t\,dw
$$

becomes, in code convention,

$$
dx = \left[v - \frac{\sigma_t^2}{2}\,\frac{x_t - t\,v}{(1-t)+\sigma_n^2}\right] dt + \sigma_t\,dw,
\qquad \sigma_t^2 = a^2\,(1-t)/t.
$$

The minus on the score correction is correct — it is the same object
expressed in forward time. This is documented in the module docstring of
`rl/grpo.py`.

## 2. Self-conditioning disabled

`prev_preds = None` is passed at every step during both rollout and log-prob
recomputation. Keeping the pretrained model's self-conditioning faithfully
would require either quadratic re-rollouts, or storing the entire `preds`
dictionary per step as a detached observation. Easy to add back once we see it
matters.

## 3. Position SDE $\sigma_t^2$ clamp at $t \approx 0$

$\sigma_t^2 = a^2(1-t)/t$ has a singularity at $t=0$ (the noise end, which is
the **start** of sampling in code convention). We clamp $t$ from below with
`eps_t = 1e-2` inside the Gaussian moments. Open question; alternatives are:

- (a) skip the SDE on the first step (deterministic Euler for step 0);
- (b) pick a bounded schedule.

## 4. Insertion probability clamp deferred

$\tilde p_\mathrm{ins}^i = \min(p_\mathrm{ins}^i,\,1-\epsilon)$ from §4.1 of the
overleaf is **not** implemented yet because Phase 1 disables insertions
entirely. It will be added together with the insertion channel.

## 5. Per-step PPO-style ratio + shared trajectory-level advantage

The overleaf derives trajectory-level log-probabilities. We still compute
them step by step, but clip and apply the advantage **per step**, then sum.
This is the standard GRPO / PPO form and lets us call `backward()` once per
step so memory is $O(1\text{ step})$ instead of $O(\text{all steps})$. The
advantage is trajectory-level (single reward at the end), so the
mathematical object being optimized is identical to trajectory-level GRPO up
to the clip-before-sum vs. sum-then-clip choice.

Concretely, for each step $k$ we compute

$$
r_k(\theta) = \exp\bigl(\log\pi_\theta^{\mathrm{step}}(t_k) - \log\pi_{\theta_\mathrm{old}}^{\mathrm{step}}(t_k)\bigr),
\qquad
\mathcal{L}_k = -\min\bigl(r_k\,\hat A,\;\mathrm{clip}(r_k,1-\epsilon,1+\epsilon)\,\hat A\bigr),
$$

and backprop $\mathcal{L}_k / K$ where $K$ is the number of integration steps.

---

# Known numerical / structural concerns and fixes

This section logs issues raised during review of `rl/grpo.py` that are not
conceptual departures from the overleaf but do affect correctness or stability.
Each entry states the problem, my assessment, and the minimal fix.

## A. Log-ratio overflow / underflow (severity: high)

**Problem.** `grpo_step` computes `ratio = exp(lp_new - lp_old)` where both
`lp_new` and `lp_old` are *graph-summed* log-probs over all nodes, triu edges,
and position dof at one integration step. Even a small per-term drift
($\sim 10^{-2}$ nats) summed over a few hundred contributions per step pushes the
step log-ratio into the tens, and $\exp(\cdot)$ overflows to `inf` in fp32.
Once `ratio = inf`, `torch.minimum(ratio*adv, clipped*adv)` can return a
finite value but the gradient path through the `inf` branch is `NaN`, silently
breaking training.

**Fix.** Clamp the *log-ratio* before exponentiating:

```python
log_ratio = (lp_new - step.logp_old).clamp(-20.0, 20.0)
ratio = torch.exp(log_ratio)
```

$\pm 20$ corresponds to ratios in $[2\cdot 10^{-9},\;5\cdot 10^{8}]$ — far outside
the PPO clip region, so this is a pure safety net and does not change the
effective objective (the gradient outside the clip region is already zero).

## B. Position-log-prob variance floor (severity: high)

**Problem.** In code convention $\sigma_t^2 = a^2(1-t)/t$ vanishes as $t \to 1$
(the data end, where the log-time schedule spends most of its steps). With
`a_sde = 0.1` and $t = 1 - 10^{-3}$, `var = sigma_t2 * dt ≈ 2.5\cdot 10^{-7}$.
The Gaussian log-density contains $-\tfrac{1}{2}\|x_{\text{next}}-\mu\|^2 /
\text{var}$; once $\theta$ drifts, $|\mu_{\text{new}} - \mu_{\text{old}}|$
can be a few $\sqrt{\text{var}}$, the squared residual scales as $1/\text{var}$,
and per-step log-ratios blow up. Compounds directly with A.

The existing `eps_t` clamp only guards the noise end ($t \to 0$); the data
end is what bites us in practice.

**Fix.** Floor the variance inside `gaussian_logprob_positions`:

```python
var = (sigma_t2 * dt).clamp_min(var_floor)  # var_floor = 1e-6 is a reasonable default
```

This is a *soft smoothing* of the policy: the true Gaussian becomes a delta at
$t = 1$, we train against a smoothed version with $\text{std} \geq 10^{-3}$.
Because the same floor is applied in both the stored `logp_old` and the
recomputed `logp_new`, the ratio is consistent — no policy mismatch.

## C. Canonical `q → p` transformation (severity: low)

**Problem.** The sampling path in `_rollout_step` uses
`p_sub = (q * rate * dt).clamp(0, 1)` before drawing the Bernoulli, while the
scoring path in `_per_channel_logprob` uses the unclamped product and relies on
`node_action_logprob` to floor at `[EPS, 1-EPS]`. If `q*rate*dt ≥ 1`, the
sampler treats the action as certain ($p = 1$) while the scorer scores it as
$p = 1 - \epsilon$. In practice `q*rate*dt \ll 1` and the disagreement branch is
never taken, so the real policy mismatch is negligible — but having two
different transformations is a latent bug.

**Fix.** Factor out one canonical helper used by both paths:

```python
def _q_to_p(q, rate_node, dt):
    return (q * rate_node * dt).clamp(EPS, 1.0 - EPS)
```

Use this in `_rollout_step` for the Bernoulli draw and in `_per_channel_logprob`
for the log-prob; drop the separate `.clamp` inside `node_action_logprob`.

## D. Triu-edge alignment assertion (severity: low)

**Problem.** `step.edge_triu_idx` is stored during rollout but `_per_channel_logprob`
calls `_extract_triu(mol_t.edge_index, ...)` fresh and relies on the result
matching the stored ordering. `EDGE_ALIGNER.align_edges` is deterministic on a
fixed `edge_index`, so this is correct today, but a future refactor could
silently break the pairing of `step.e_triu_sub` / `step.hat_e_triu` with the
newly-extracted edge-level probabilities.

**Fix.** Cheap defensive check inside `_per_channel_logprob`:

```python
triu_idx, (q_sub_e_triu, edge_probs_triu) = _extract_triu(
    mol_t.edge_index, [q_sub_e_full, edge_probs_full],
)
assert torch.equal(triu_idx, step.edge_triu_idx), (
    "edge aligner ordering drifted between rollout and scoring"
)
```

O(E) equality check, loud failure on regression, no change to logic.

## E. Config mutation (severity: low, correctness OK today)

**Problem.** `rollout_trajectory` mutates `cfg.cfg_inputs` and downstream calls
read from the same instance. Works under the current synchronous, single-process
loop; breaks the first time we run data-parallel GRPO or interleave multiple
rollouts sharing the same `GRPOConfig` object.

**Fix.** Drop `cfg_inputs` from `GRPOConfig` and thread it explicitly through
`_rollout_step`, `_step_logprob_with_grad`, and `_per_channel_logprob`. Defer
until we actually parallelise.

## Priority order

1. A (log-ratio clamp) — required for stability.
2. B (variance floor) — required for stability; compounds with A.
3. C (canonical `q → p`) — hygiene; forecloses the latent mismatch.
4. D (alignment assertion) — one-line insurance.
5. E (config mutation) — defer to the parallelisation PR.

