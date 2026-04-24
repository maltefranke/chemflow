# Departures from the Overleaf (Phase 2)

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

## 2. Self-conditioning is dead code in the base repo

`prev_preds = None` is passed at every step. This is not actually a
departure: `SelfConditioningResidualLayer` is defined in
`src/chemflow/model/self_conditioning.py` but **never imported**, and the
active backbone `DiTBackboneWithHeads.forward` accepts `prev_outs` /
`is_random_self_conditioning` as kwargs but does not read them. The base
model never consumed self-conditioning, so our rollout passing `None`
matches `sample()` exactly.

## 3. Position SDE $\sigma_t^2$ clamp at $t \approx 0$

$\sigma_t^2 = a^2(1-t)/t$ has a singularity at $t=0$ (the noise end, which is
the **start** of sampling in code convention). We clamp $t$ from below with
`eps_t = 1e-2` inside the Gaussian moments. Open question; alternatives are:

- (a) skip the SDE on the first step (deterministic Euler for step 0);
- (b) pick a bounded schedule.

## 4. Insertion probability clamp (implemented)

$\tilde p_\mathrm{ins}^i = \min(p_\mathrm{ins}^i,\,1-\epsilon)$ from §4.1 of the
overleaf is applied both when sampling (so the Bernoulli probability is
well-defined) and when evaluating the log-prob. Because the integrator's
sampling path `torch.rand() < p` is equivalent to clamping at 1 (a uniform
draw always falls below any $p \ge 1$), this changes the numerical value
of the log-prob but not the action distribution.

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

## A. Log-ratio overflow / underflow (severity: high, **applied**)

**Problem.** `grpo_step` computes `ratio = exp(lp_new - lp_old)` where both
`lp_new` and `lp_old` are *graph-summed* log-probs over all nodes, triu edges,
and position dof at one integration step. Even a small per-term drift
($\sim 10^{-2}$ nats) summed over a few hundred contributions per step pushes the
step log-ratio into the tens, and $\exp(\cdot)$ overflows to `inf` in fp32.
Once `ratio = inf`, `torch.minimum(ratio*adv, clipped*adv)` can return a
finite value but the gradient path through the `inf` branch is `NaN`, silently
breaking training.

**Fix (applied in `grpo_step`, configurable via `GRPOConfig.log_ratio_clamp`,
default $\pm 20$).** Clamp the log-ratio before exponentiating:

```python
log_ratio = (lp_new - step.logp_old).clamp(
    -cfg.log_ratio_clamp, cfg.log_ratio_clamp,
)
ratio = torch.exp(log_ratio)
```

$\pm 20$ corresponds to ratios in $[2\cdot 10^{-9},\;5\cdot 10^{8}]$ — far outside
the PPO clip region, so this is a pure safety net and does not change the
effective objective (the gradient outside the clip region is already zero).

## B. Position-log-prob variance floor (severity: high, **applied**)

**Problem.** In code convention $\sigma_t^2 = a^2(1-t)/t$ vanishes as $t \to 1$
(the data end, where the log-time schedule spends most of its steps). With
`a_sde = 0.1` and $t = 1 - 10^{-3}$, `var = sigma_t2 * dt ≈ 2.5\cdot 10^{-7}$.
The Gaussian log-density contains $-\tfrac{1}{2}\|x_{\text{next}}-\mu\|^2 /
\text{var}$; once $\theta$ drifts, $|\mu_{\text{new}} - \mu_{\text{old}}|$
can be a few $\sqrt{\text{var}}$, the squared residual scales as $1/\text{var}$,
and per-step log-ratios blow up. Compounds directly with A.

The existing `eps_t` clamp only guards the noise end ($t \to 0$); the data
end is what bites us in practice.

**Fix (applied on both paths, configurable via `GRPOConfig.var_floor`,
default $10^{-6}$).** Floor the variance identically in scoring and sampling so
the behaviour policy equals the scoring policy:

```python
# gaussian_logprob_positions (scoring):
var = (sigma_t2 * dt).clamp_min(var_floor)

# _rollout_step (sampling):
var_sample = (sigma_t2 * dt).clamp_min(cfg.var_floor)
x_next = mu + torch.sqrt(var_sample).unsqueeze(-1) * noise_pos
```

This is a *soft smoothing* of the policy: the true Gaussian becomes a delta at
$t = 1$, we train against a smoothed version with $\text{std} \geq 10^{-3}$.
Applying the floor on both paths makes the PPO score-function estimator
unbiased (the sample distribution and the scoring distribution now match);
applying it only on scoring would keep the ratio consistent but leave the
gradient mis-scaled at tail times.

## C. Canonical `q → p` transformation (severity: low, **applied**)

**Problem.** The sampling path in `_rollout_step` uses
`p_sub = (q * rate * dt).clamp(0, 1)` before drawing the Bernoulli, while the
scoring path in `_per_channel_logprob` uses the unclamped product and relies on
`node_action_logprob` to floor at `[EPS, 1-EPS]`. If `q*rate*dt ≥ 1`, the
sampler treats the action as certain ($p = 1$) while the scorer scores it as
$p = 1 - \epsilon$. In practice `q*rate*dt \ll 1` and the disagreement branch is
never taken, so the real policy mismatch is negligible — but having two
different transformations is a latent bug.

**Fix (applied).** A single helper `_q_to_p` is defined near the top of
`grpo.py` and used in both paths (all `sub_a`, `del`, `sub_e` channels, plus
`_compute_p_ins` for the insertion gate):

```python
def _q_to_p(q, rate_node, dt):
    return (q * rate_node * dt).clamp(EPS, 1.0 - EPS)
```

`node_action_logprob` still defensively clamps the *sum* `p_any = p_sub + p_del`
to `1 - EPS` (since two independently-clamped probabilities can still sum to
more than one); this is not a path disagreement — rollout clamps the sum the
same way before the conditional split.

## D. Triu-edge alignment assertion (severity: low, **applied**)

**Problem.** `step.edge_triu_idx` is stored during rollout but `_per_channel_logprob`
calls `_extract_triu(mol_t.edge_index, ...)` fresh and relies on the result
matching the stored ordering. `EDGE_ALIGNER.align_edges` is deterministic on a
fixed `edge_index`, so this is correct today, but a future refactor could
silently break the pairing of `step.e_triu_sub` / `step.hat_e_triu` with the
newly-extracted edge-level probabilities.

**Fix (applied inside `_per_channel_logprob`).**

```python
triu_idx, (q_sub_e_triu, edge_probs_triu) = _extract_triu(
    mol_t.edge_index, [q_sub_e_full, edge_probs_full],
)
assert torch.equal(triu_idx, step.edge_triu_idx), (
    "edge aligner ordering drifted between rollout and scoring"
)
```

An analogous assert guards the `predict_edges_for_insertion` ordering for
ins→existing edges (rollout vs. scoring).

O(E) equality check, loud failure on regression, no change to logic.

## E. Config mutation (severity: low, correctness OK today, **not yet fixed**)

**Problem.** `rollout_trajectory` mutates `cfg.cfg_inputs` (line `cfg.cfg_inputs
= module._build_inference_cfg_inputs(...)`) and downstream calls read from the
same instance. Works under the current synchronous, single-process loop; breaks
the first time we run data-parallel GRPO or interleave multiple rollouts
sharing the same `GRPOConfig` object.

**Fix (pending).** Drop `cfg_inputs` from `GRPOConfig` and thread it explicitly
through `_rollout_step`, `_step_logprob_with_grad`, and `_per_channel_logprob`.
Defer until we actually parallelise.

---

# Phase-2 specific design choices

The entries below are introduced by enabling the full variable-atom regime
(insertions + deletions + typed-GMM + ins-edge heads). None of them is a
departure from the overleaf in the "we simplified the math" sense; they
are choices about how to thread the integrator's action-sampling into a
policy-gradient framework where every random draw must have a log-prob.

## F. Integrator safeguards treated as environment (severity: medium)

**Problem.** `integrate_step_gnn` applies two safety nets that modify the
sampled action outside the policy:

1. **Insertion overflow** (integration.py lines 272–288): if a graph's
   post-ins atom count would exceed `max_atoms`, randomly drop excess
   `do_ins` flags.
2. **Deletion underflow** (lines 369–379): if a graph would retain fewer
   than 2 nodes, restore some `do_del` flags.

Both are valid environmental behaviour but introduce a mismatch between
the *nominal* policy (Bernoulli with rate $p$) and the *realised* action
distribution.

**Fix.** We apply the exact same safeguards in `_rollout_step` so the
trajectory distribution matches the integrator bit-for-bit, then evaluate
the log-prob under the unmodified Bernoulli probabilities. The safeguard
is treated as part of the environment $p(\text{next state} \mid \text{action})$
rather than the policy. Practically this is a mild log-prob miscalibration
on the ~1% of steps where a safeguard fires; the bias is the same under
$\theta_\mathrm{old}$ and $\theta_\mathrm{new}$, so the GRPO *ratio* is
unaffected on those steps (both sides score the same action under the
same Bernoulli).

## G. Two-stage ins-edge marginalisation

**Source.** Overleaf Eq. `ins_edge_marginal`.

The integrator samples insertion edges in two stages: a clean prediction
$e_1 \sim \mathrm{Cat}(\pi_\mathrm{edge})$, one-hot encoded, mixed with
the prior at the current noise level
$\kappa_t = \kappa_{\mathrm{sub},e}(t+\Delta t)$, then the final
$\hat e \sim \mathrm{Cat}((1-\kappa_t)\,p_\mathrm{prior} + \kappa_t\,\mathbf{1}[e_1])$.

We do not store the latent $e_1$. At scoring time we use the closed-form
marginal
$$P(\hat e = k) = (1-\kappa_t)\,p_\mathrm{prior}[k] + \kappa_t\,\pi_\mathrm{edge}[k],$$
re-computing $\kappa_t$ from `step.t + step.dt` via
`module.integrator.sub_e_schedule.kappa_t`. This applies to both
ins→existing and ins→ins edges.

**Note on rollout.** We *also* sample directly from this marginal in
`_rollout_step` rather than replicating the integrator's two-stage draw.
The two procedures produce the same distribution over $\hat e$ (by the
law of total probability), so no additional departure is introduced.
The practical benefit is that we never need to store $e_1$.

## H. GMM component marginalisation

**Source.** Overleaf Remark after Eq. `gmm_logprob_lse`.

`sample_from_typed_gmm` picks a component $k$ internally but we do not
store it. The log-prob is the logsumexp over components
$$\log\pi^\mathrm{GMM}(x,a,c) = \mathrm{logsumexp}_k\Big[\log\pi_k +
\log\mathcal{N}(x|\mu_k,\sigma_k^2 I) + \log p_{\mathrm{atom},k}[a] +
\log p_{\mathrm{charge},k}[c]\Big],$$
which is invariant to which component was drawn. This reduces variance
in the policy-gradient signal, at the cost of evaluating all $K$
components ($K \le 8$) once per inserted atom.

## I. `node_atom_types` to the insertion edge head

**Situation.** The integrator constructs the argument to
`ins_edge_head.predict_edges_for_insertion(..., node_atom_types=a_t, ...)`
*after* applying substitutions (integration.py line 320 sets
`a_t[do_sub_a] = a_1[do_sub_a]`, then line 444 passes the post-sub `a_t`).
This creates an asymmetry inside one step: the latent features
`h_latent` were produced from the *pre-sub* atom types, but the explicit
`node_atom_types` tensor used by the edge MLP sees *post-sub* types.

**Our choice.** We mirror the integrator exactly: rollout rebuilds
`a_next = mol_t.a; a_next[do_sub_a] = hat_a[do_sub_a]` and passes that,
and scoring rebuilds the same tensor from `step.a_choice == 1` and
`step.hat_a`. Any alternative (pass pre-sub, or recompute `h_latent`
from post-sub) would differ from the integrator and therefore from the
policy the pretrained model was trained under. First implementation
pass used pre-sub `mol_t.a` in both places and was silently wrong
(rollout didn't match the integrator; scoring and rollout agreed with
each other only because both used the wrong value). Caught during
review; see commit.

## J. Invalid ins→existing edges: predicted in full, zeroed in log-prob

**Situation.** `ins_edge_head.predict_edges_for_insertion` returns
pairs `(spawn_orig, existing_orig)` for *every* same-graph node of every
spawn, regardless of whether the `existing` endpoint survives deletion.
The integrator filters these to the valid subset
(`orig_to_postdel[existing] >= 0`) before placing them on the graph;
invalid edges are simply never placed.

**Our choice.** At rollout we sample `hat_e_ins_full` for **all**
returned pairs (valid and invalid), then record both the full edge set
and a `ins_edge_valid_mask`. At scoring time we re-run
`predict_edges_for_insertion` on the same `do_ins` mask (so the pair
ordering matches — asserted) and evaluate the marginal log-prob on the
full set, then zero out invalid-edge contributions before the
scatter-sum.

**Why not subset.** Either (a) include all pairs in the log-prob or
(b) subset to valid-only before sampling both the edge and the log-prob.
We picked (a) for a simpler ordering assertion; subsetting would require
stable re-ordering of the valid subset at scoring time. The gradient on
invalid edges is zero (we multiplied by `valid_mask`) so the effective
objective is identical under (a) and (b).

## K. Deleted spawn nodes may still insert atoms

**Situation.** Integrator line 397: `do_ins_valid = do_ins`. A node
that will be deleted this step can still spawn an insertion — the spawn
node only provides the GMM conditioning signal (its latent features and
position), not a structural parent in the post-step graph. The
post-deletion mapping `orig_to_postdel[spawn] == -1` is fine because
the *new atom* uses its own fresh index, not the spawn's.

**Our choice.** We match this exactly: `do_ins` is applied without
being intersected with `~do_del`. The GMM conditioning tensors in
`ins_gmm_preds` are indexed by the original `do_ins` mask, which
reaches deleted nodes just as well as survivors.

## L. Phantom sampling on deleted nodes

**Situation.** The integrator evaluates every per-node channel (position
update, atom-type sample via `a_pred`, charge sample via `c_pred`) on
**all** $N_\mathrm{orig}$ nodes before filtering (lines 259, 878–895
in lightning_module.py, line 391 in integration.py). There is no
short-circuit that skips these draws for nodes marked `do_del`.

**Our choice.** Rollout matches: we compute `x_next`, `hat_a`, `hat_c`
for all original nodes. Deleted rows are "phantom" — sampled but never
placed in `mol_t_next`. The corresponding log-prob terms multiply by
`survive_mask = step.a_choice != 2` before `_scatter_sum`, which zeros
them out per-graph. The phantom samples are independent of the reward
and of $\theta$, so their masked contribution is structurally zero and
does not affect the gradient.

Alternative (skip these draws on deleted nodes) would add branching
without changing the distribution and would make the scoring path
harder to keep in lock-step with the rollout path.

## M. Charge channel partitioning: existing vs inserted

**Invariant.** Every atom in the final state has exactly one charge
log-prob contribution.  Existing survivors get theirs from the
`charge_head` categorical (node-channel log-prob, masked by
`survive_mask`).  Inserted atoms get theirs from the typed GMM term
(which jointly models `(x, a, c)` for each insertion).  Double-counting
either side would bias the gradient.

**Where this shows up in code.** `charge_logprob(...)` is called with
`survive_mask` derived from `step.a_choice != 2`, scattering only over
mol_t's *original* nodes.  `gmm_marginal_logprob(...)` reads `ins_c`
from `step.ins_c` and uses the GMM's `c_probs` head — these atoms are
newly inserted, not in mol_t, so the two sets are disjoint by
construction.

## N. Integrator parity (documentation only)

These points match the integrator and `LightningModuleRates.sample()`
exactly; they are listed only because they are easy to get wrong.

- **COM removal**: `remove_com()` is called once before the loop and
  once after every integrator step (`sample()` lightning_module.py
  lines 829 and 957). We match.
- **Topology edit order**: within a step, the integrator applies
  substitutions → deletions → insertions on `mol_post_sub`. Insertions
  receive the post-deletion index mapping. We match.
- **`n_atoms_strategy == "fixed"`**: zero out `q_del` and
  `num_ins_pred` *before* any safeguard or clamp, exactly as
  `sample()` does.
- **`p_sub + p_del > 1` edge case**: neither rollout nor scoring
  re-normalise the two rates. Both paths clamp
  `p_any = (p_sub + p_del).clamp(max=1-EPS)` consistently, so the
  effective distribution is always a valid categorical over
  $\{\text{noop}, \text{sub}, \text{del}\}$ and rollout / scoring
  agree.
- **`cfg_inputs`**: built once per rollout from the `(mol_t, mol_1)`
  pair at $t=0$ and reused for every step and every scoring pass —
  the CFG conditioning is constant over the trajectory by design.

## Status summary

Applied in the current `grpo.py`:

- §A (log-ratio clamp) — `grpo_step`, configurable via `GRPOConfig.log_ratio_clamp`.
- §B (variance floor) — `gaussian_logprob_positions`, configurable via `GRPOConfig.var_floor`.
- §C (canonical `q → p`) — `_q_to_p` helper used by all sigmoid-gated channels.
- §D (triu alignment assert + ins-edge ordering assert) — `_per_channel_logprob`.
- §F (integrator safeguards as environment) — `_apply_overflow_safeguard`,
  `_apply_underflow_safeguard` in `_rollout_step`.
- §G (two-stage ins-edge marginalisation) — closed-form marginal sampling in
  rollout, `ins_edge_marginal_logprob` in scoring.
- §H (GMM component marginalisation) — `gmm_marginal_logprob` via logsumexp.
- §I (post-sub `node_atom_types` to the ins-edge head) — rebuilt consistently
  in both rollout (`a_next`) and scoring (`a_next_score`).
- §J (invalid ins→existing edges zeroed in log-prob) — via `ins_edge_valid_mask`.
- §K/§L/§M — matched to integrator semantics.

Still pending:

- §E (config mutation of `cfg.cfg_inputs`) — waiting on parallelisation.
- §3 (position SDE $\sigma_t^2$ lower clamp) — `eps_t = 1e-2` is the current
  pragmatic choice; the `(a)` / `(b)` alternatives in §3 are still open.

