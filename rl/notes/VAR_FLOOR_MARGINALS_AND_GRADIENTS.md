# Variance floor: marginals, gradients, and RL bookkeeping

This note ties together three things that are easy to conflate:

1. **Marginal preservation** (continuous-time SDE / flow-matching story).
2. **What actually runs** in `rl/grpo.py` (sampler vs scored log-probability).
3. **How that affects policy gradients** and why **advantages stay “real”** while **likelihood gradients can be scaled or biased**.

Code anchors: `position_policy_moments`, `gaussian_logprob_positions` in `rl/grpo.py`; rollout sampling in `_rollout_step`.

*Rendering:* math uses `$…$` (inline) and `$$…$$` (display) so it shows in GitHub, VS Code/Cursor preview, and other engines that enable KaTeX/MathJax on `$` delimiters.

---

## 1. Two different objects

**Sampling kernel (rollout)**  
Positions update as

$$
x_{t+\Delta t} = \mu_\theta(x_t, t) + \sqrt{\sigma_t^2\,\Delta t}\,\xi,
\quad
\xi \sim \mathcal{N}(0, I),
$$

with **no** variance floor on $\sigma_t^2 \Delta t$. So when `a_sde = 0`, $\sigma_t^2 = 0$ and the position step is **deterministic**: $x_{t+\Delta t} = \mu_\theta$.

**Scored kernel (log-probability)**  
`gaussian_logprob_positions` evaluates a Gaussian with the **same** $\mu_\theta$ but variance

$$
\mathrm{var}_{\text{score}} = \max\bigl(\sigma_t^2 \Delta t,\; v_{\text{floor}}\bigr),
$$

where $v_{\text{floor}}$ is the config `var_floor`.

So the density $\log \pi_\theta(x \mid x_t, \ldots)$ is **not** necessarily the density of the kernel that generated $x$ when $\sigma_t^2 \Delta t < v_{\text{floor}}$, or when `a_sde = 0`.

**Takeaway:** the **Markov transition** implied by sampling and the **transition** implied by `log π` can differ as soon as the floor binds (or for positions when `a_sde = 0`).

---

## 2. “Marginal preservation” — what we lose

In ideal SDE / flow matching, one designs noise and drift so that **forward simulation** and **the probability path** stay compatible (up to time discretization and model error). A **floor only in the log-density** does not add noise in simulation; it **redefines** the probabilistic model used in the loss. Then:

- **Simulated trajectories** follow the **sampler’s** noise ($\sigma_t^2 \Delta t$ at each step, possibly zero).
- **Theorem-style marginals** of the **written-down** SDE are not what you get from rolling out if you do **not** inject matching noise in `x_next = …`.

So the fuss is not about $\mu$ or velocity $v$: those match between rollout and scoring via `position_policy_moments`. The break is **variance-only**: **same mean, different agreement on how noisy the step was**.

---

## 3. Gradients w.r.t. the mean (positions)

For an isotropic Gaussian on the mean (schematically),

$$
\nabla_{\mu} \log \mathcal{N}(x \mid \mu, \mathrm{var}\, I)
\propto \frac{x - \mu}{\mathrm{var}}.
$$

**When `var_score` is dominated by `var_floor`:**

- **Magnitude** $\propto 1/\mathrm{var}$ is **smaller** than the “honest” gradient using $\mathrm{var} = \sigma_t^2 \Delta t$ (for the same residual $x - \mu$).
- **Direction** is still “increase $\log \pi$ by moving $\mu$ toward $x$” along each coordinate — the floor **damps** updates; it does not flip the sign of the mean gradient.

So clamping the **variance** in the log-prob acts like a **gradient gain limiter** on position heads on floored steps.

**Extreme case `a_sde = 0`:** rollout gives $x = \mu_{\text{old}}$ at sampling time. On the **first** PPO pass with $\theta \approx \theta_{\text{old}}$, the residual is **zero**, so $\nabla_\mu \log \pi$ is **zero** for positions even though $\log \pi$ is finite (finite `var_floor`). Later multi-pass steps can give non-zero mean gradients because $\mu_\theta$ moves while $x$ stays fixed — that is not fresh exploration signal, it is **reuse + policy drift**.

---

## 4. Advantages vs log-probs (why rewards are “fine” but likelihood is warped)

**Advantages** are built from **rollout rewards** (here, after a full trajectory). That signal reflects **whatever stochastic process actually ran** — real SDE noise, discrete draws, insertions, etc. It does **not** go through `var_floor`.

**Policy-gradient factors** for the PPO surrogate use $\nabla_\theta \log \pi_\theta(a \mid s)$ where $\log \pi$ is computed with **`var_score`**, not necessarily the variance that produced $x$.

So you get a **split**:

| Quantity | Tied to |
|----------|--------|
| **Which** $(x_t, x_{t+1})$, discrete actions, graphs | **Sampler** (true dynamics + schedules). |
| **How good** each trajectory is (**advantage**) | **Rewards** from that sampler. |
| **How strongly** $\theta$ is nudged along $\nabla \log \pi$ for positions | **Scored** Gaussian, with $\mathrm{var}_{\text{score}} \ge v_{\text{floor}}$. |

Intuition: the algorithm still **prefers** high-advantage trajectories, but the **sensitivity** of the update to changing $\mu_\theta$ is scaled by **$1/\mathrm{var}_{\text{score}}$**. When the floor binds, position updates are **gentler** than they would be for the true sampling variance — a **stability vs fidelity** trade.

**Log-prob level bias:** if $\mathrm{var}_{\text{score}} > \mathrm{var}_{\text{sample}}$ for a given step, the same residual $x - \mu$ gets a **higher** $\log \pi$ (flatter Gaussian) than under the true kernel — **optimistic** likelihood vs truth. For PPO **ratios** $\log \pi_{\text{new}} - \log \pi_{\text{old}}$, $\mathrm{var}$ for positions does **not** depend on $\theta$ in this codebase (only $\mu$ does), so old vs new uses the **same** $\mathrm{var}_{\text{score}}$: **differences** are less biased than absolute $\log \pi$, but **gradient scale** is still set by **$1/\mathrm{var}_{\text{score}}$**.

---

## 5. Other clamps (different role)

- **`log_ratio_clamp`**, PPO **`clip_eps`**: constrain **updates across passes** and **importance ratios**; they do not change the continuous-time marginal story — they bound **optimization**, not $\sigma_t^2$.
- **KL to ref (`k3`)**: adds a term whose gradients pull $\theta$ toward a frozen ref; consistent with using the **same** floored per-channel log-probs as PPO, but still not the same object as the **sampler** when the floor binds.

---

## 6. Summary

| Topic | Effect of `var_floor` **in log-prob only** |
|-------|-------------------------------------------|
| **Marginals of the SDE on paper** | **Not** matched by “score with extra variance” if simulation uses smaller or zero noise. |
| **Gradients w.r.t. $\mu$** | **Scaled down** when $\max(\sigma_t^2 \Delta t, v_{\text{floor}}) > \sigma_t^2 \Delta t$ (floor binds). |
| **Advantages** | Still from **real** rollouts — **un**affected by the floor. |
| **What training optimizes** | A **surrogate** likelihood for PPO/KL, stable but not identical to the generating kernel when the floor binds. |

For fixes that aim to **align** sampler and scorer (e.g. floor in both, or floor only when $\sigma_t^2 \Delta t > 0$), treat that as an **ablation**: you trade **exact** kernel identity for the current **stability** behavior.
