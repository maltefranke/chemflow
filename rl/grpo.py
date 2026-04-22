"""Minimal GRPO on the Morph architecture.

Scope (Phase 1)
---------------
Fixed-atom-count regime (no insertions, no deletions).  The per-step policy
covers four channels:
    * positions (SDE with score correction)
    * atom-type substitution (2-way Bernoulli + categorical target)
    * edge-type substitution (2-way Bernoulli + categorical target, upper-tri)
    * charge (categorical, applied to all surviving nodes)

The outer rollout / log-prob / clipped-ratio machinery does not change when
insertions / deletions are later enabled; only `StepData` and the per-channel
log-prob functions grow.

Convention
----------
We use the *code* convention throughout this file:
    t in [0, 1],   t = 0 is noise,   t = 1 is data,   sampling is t: 0 -> 1.
This is the convention of `chemflow.flow_matching.integration` and of `sample()`.
The math in the project's overleaf is written in the opposite convention
(x_0 = data, x_1 = noise).  Translation:
    t_user         <-> 1 - t_code
    (1 - t_user)   <-> t_code
    v_theta (both) == hat x_data - hat x_noise  (same sign here)
    sigma_t^2      (user) = a^2 t/(1-t)   ->  (code) a^2 (1-t)/t
After translating, the forward-time SDE in code convention is
    dx = [v - (sigma_t^2/2) * (x_t - t v) / ((1-t) + sigma_noise^2)] dt + sigma_t dw.
So the sign on the score correction is a minus here, unlike the plus in the
overleaf box -- this is the same object, expressed in the other direction of time.

Explicit simplifications vs. `sample()`
---------------------------------------
    * `prev_preds = None` always (no self-conditioning).
    * Positions evolve as an SDE (Gaussian transition), not a deterministic
      ODE; this is what makes the continuous channel trainable.
    * Insertions and deletions are disabled: we never call `filter_nodes`,
      never sample from the GMM, never call `ins_edge_head`.

Open knobs / questions left as TODO
-----------------------------------
    * `eps_t` floor on t inside sigma_t^2: hack to avoid 1/0 at t = 0.
    * No KL term to a frozen reference policy.
    * Per-step ratio (as in LLM-GRPO); change to per-trajectory ratio by
      summing then clipping if preferred.
    * Reward is RDKit validity (binary); swap by replacing `validity_reward`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional

import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from chemflow.dataset.molecule_data import MoleculeBatch
from chemflow.utils.utils import EDGE_ALIGNER

# Rewards live in rl/rewards.py (registry + shared RDKit loop).  `validity_reward`
# is re-exported here so the existing `reward_fn=validity_reward` default keeps
# working for callers that import it from `rl.grpo` directly.
from rl.rewards import validity_reward  # noqa: F401


EPS = 1e-8
EPS_T = 1e-2
DEFAULT_CLIP_EPS = 0.2
DEFAULT_VAR_FLOOR = 1e-6
DEFAULT_LOG_RATIO_CLAMP = 20.0


# ─────────────────────────────────────────────────────────────────────────────
# Config + rollout storage
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class GRPOConfig:
    sigma_noise: float = 0.2       # must match Interpolator.move_noise_scale used at training
    a_sde: float = 0.1             # SDE noise coefficient; sigma_t^2 = a^2 (1-t)/t
    clip_eps: float = DEFAULT_CLIP_EPS
    eps_t: float = EPS_T
    var_floor: float = DEFAULT_VAR_FLOOR           # floor on position Gaussian variance (see DEPARTURES.md §B)
    log_ratio_clamp: float = DEFAULT_LOG_RATIO_CLAMP  # clamp |lp_new - lp_old| before exp (see §A)
    num_integration_steps: int | None = None  # None -> module's default
    # CFG inputs to feed guided_predict.  Built at rollout time from a batch's
    # (mol_t, mol_1) pair and stashed here so every step + log-prob call shares
    # the exact same conditioning.
    cfg_inputs: dict | None = None


@dataclass
class StepData:
    """Per-step rollout record.  All tensors detached from theta_old's graph."""

    mol_t: object           # MoleculeBatch snapshot at step start (COM-removed)
    t: torch.Tensor         # (B,) graph-level time
    dt: float

    # Continuous channel
    x_next: torch.Tensor    # (N, 3) realised x_{t+dt} (pre-COM-removal of next step)

    # Discrete node channel (0 = NOOP, 1 = SUB, 2 = DEL)
    # Phase 1 never emits 2; kept for compat with Phase 2.
    a_choice: torch.Tensor  # (N,) int8
    hat_a: torch.Tensor     # (N,) sampled atom-type target
    hat_c: torch.Tensor     # (N,) sampled charge

    # Edge channel (upper triangular of mol_t.edge_index)
    edge_triu_idx: torch.Tensor  # (2, E_triu) in mol_t node space
    e_triu_sub: torch.Tensor     # (E_triu,) bool
    hat_e_triu: torch.Tensor     # (E_triu,) sampled edge-type target

    # Cached per-graph log-prob under theta_old (computed at rollout time)
    logp_old: torch.Tensor  # (B,)


@dataclass
class Trajectory:
    steps: List[StepData]
    mol_final: object
    reward: Optional[torch.Tensor] = None  # (B,)


# ─────────────────────────────────────────────────────────────────────────────
# Low-level helpers
# ─────────────────────────────────────────────────────────────────────────────


def _scatter_sum(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Per-graph sum of per-element values (CPU/GPU agnostic, autograd-safe)."""
    if src.numel() == 0:
        return src.new_zeros(dim_size)
    out = src.new_zeros(dim_size)
    out.index_add_(0, index, src)
    return out


def _gather_logp(probs: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """log probs[torch.arange(N), idx] with numerical floor."""
    return torch.log(probs.clamp(min=EPS)).gather(-1, idx.view(-1, 1)).squeeze(-1)


def _q_to_p(q: torch.Tensor, rate_node: torch.Tensor, dt: float) -> torch.Tensor:
    """Canonical conversion raw-sigmoid -> per-step probability (see DEPARTURES.md §C).

    Used identically by sampling and scoring so the two paths cannot diverge.
    """
    return (q * rate_node * dt).clamp(EPS, 1.0 - EPS)


# ─────────────────────────────────────────────────────────────────────────────
# Position SDE (code convention)
# ─────────────────────────────────────────────────────────────────────────────


def position_policy_moments(
    x_t: torch.Tensor,      # (N, 3)
    x1_pred: torch.Tensor,  # (N, 3)
    t_node: torch.Tensor,   # (N,)
    dt: float,
    sigma_noise: float,
    a_sde: float,
    eps_t: float = EPS_T,
):
    """Return (mu, sigma_t^2) for the Euler-Maruyama Gaussian step at time t.

    mu_i    = x_{i,t} + dt * [v_i - (sigma_t^2 / 2) * (x_{i,t} - t v_i) / ((1-t)+sigma_noise^2)]
    var_i   = sigma_t^2 * dt  (isotropic on R^3)

    v_i     = (x1_pred_i - x_{i,t}) / (1 - t)      (code convention)
    """
    t = t_node.clamp(min=eps_t, max=1.0 - 1e-6).unsqueeze(-1)  # (N, 1)
    one_minus_t = 1.0 - t
    sigma_n2 = sigma_noise ** 2

    v = (x1_pred - x_t) / one_minus_t                              # (N, 3)
    score = -(x_t - t * v) / (one_minus_t + sigma_n2)              # (N, 3)
    sigma_t2 = (a_sde ** 2) * one_minus_t / t                      # (N, 1)
    mu = x_t + dt * (v + 0.5 * sigma_t2 * score)                   # (N, 3)
    return mu, sigma_t2.squeeze(-1)                                # mu: (N, 3), var: (N,)


def gaussian_logprob_positions(
    x_next: torch.Tensor,    # (N, 3)
    mu: torch.Tensor,        # (N, 3)
    sigma_t2: torch.Tensor,  # (N,)
    dt: float,
    batch_id: torch.Tensor,  # (N,)
    num_graphs: int,
    var_floor: float = DEFAULT_VAR_FLOOR,
) -> torch.Tensor:
    """Per-graph sum of Gaussian log-densities.

    `var_floor` smooths the policy near t=1, where `sigma_t^2 = a^2 (1-t)/t -> 0`
    would otherwise explode the squared-residual term.  Applied identically in
    rollout and scoring so the ratio is unaffected.  See DEPARTURES.md §B.
    """
    var = (sigma_t2 * dt).clamp_min(var_floor)                      # (N,)
    sq = ((x_next - mu) ** 2).sum(-1)                               # (N,)
    logp_node = -0.5 * sq / var - 1.5 * torch.log(2 * math.pi * var)
    return _scatter_sum(logp_node, batch_id, num_graphs)


# ─────────────────────────────────────────────────────────────────────────────
# Discrete per-channel log-probs
# ─────────────────────────────────────────────────────────────────────────────


def node_action_logprob(
    p_sub: torch.Tensor,        # (N,) probability of SUB this step
    p_del: torch.Tensor,        # (N,) probability of DEL this step
    atom_probs: torch.Tensor,   # (N, |A|)
    a_choice: torch.Tensor,     # (N,) 0/1/2
    hat_a: torch.Tensor,        # (N,)
    batch_id: torch.Tensor,
    num_graphs: int,
) -> torch.Tensor:
    """log pi^{node} per graph (overleaf Eq. 22).  `p_del` may be all zeros (Phase 1).

    Assumes `p_sub`, `p_del` already passed through `_q_to_p` (so in (EPS, 1-EPS)).
    We still clamp `p_any` / `p_noop` because their sum can still exceed 1.
    """
    p_any = (p_sub + p_del).clamp(max=1.0 - EPS)
    p_noop = (1.0 - p_any).clamp(min=EPS)

    log_noop = torch.log(p_noop)
    log_sub = torch.log(p_sub.clamp(min=EPS))
    log_del = torch.log(p_del.clamp(min=EPS))
    log_pi_atom = _gather_logp(atom_probs, hat_a)

    logp = torch.where(a_choice == 0, log_noop, torch.zeros_like(log_noop))
    logp = torch.where(a_choice == 1, log_sub + log_pi_atom, logp)
    logp = torch.where(a_choice == 2, log_del, logp)
    return _scatter_sum(logp, batch_id, num_graphs)


def edge_sub_logprob(
    p_sub_e: torch.Tensor,       # (E_triu,)
    edge_probs: torch.Tensor,    # (E_triu, |E|)
    e_triu_sub: torch.Tensor,    # (E_triu,) bool
    hat_e: torch.Tensor,         # (E_triu,)
    edge_batch_id: torch.Tensor,
    num_graphs: int,
) -> torch.Tensor:
    """Assumes `p_sub_e` is already in (EPS, 1-EPS) via `_q_to_p`."""
    log_noop = torch.log(1.0 - p_sub_e)
    log_sub = torch.log(p_sub_e)
    log_pi_edge = _gather_logp(edge_probs, hat_e)
    logp = torch.where(e_triu_sub, log_sub + log_pi_edge, log_noop)
    return _scatter_sum(logp, edge_batch_id, num_graphs)


def charge_logprob(
    charge_probs: torch.Tensor,  # (N, |C|)
    hat_c: torch.Tensor,         # (N,)
    survive_mask: torch.Tensor,  # (N,) bool
    batch_id: torch.Tensor,
    num_graphs: int,
) -> torch.Tensor:
    log_pi = _gather_logp(charge_probs, hat_c)
    log_pi = torch.where(survive_mask, log_pi, torch.zeros_like(log_pi))
    return _scatter_sum(log_pi, batch_id, num_graphs)


# ─────────────────────────────────────────────────────────────────────────────
# One-step building blocks (shared by rollout and log-prob recomputation)
# ─────────────────────────────────────────────────────────────────────────────


def _extract_triu(
    edge_index: torch.Tensor,
    attrs: list[torch.Tensor],
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Extract upper-triangular edges + corresponding attrs using the project's aligner."""
    infos = EDGE_ALIGNER.align_edges(source_group=(edge_index, attrs))
    return infos["edge_index"], list(infos["edge_attr"])


def _per_channel_logprob(
    preds: dict,
    mol_t,
    step: StepData,
    t: torch.Tensor,
    dt: float,
    cfg: GRPOConfig,
    *,
    sub_rate_schedule,
    sub_e_rate_schedule,
) -> torch.Tensor:
    """Compute per-graph log-prob from `preds` (under ANY theta) and stored actions."""
    batch_id = mol_t.batch
    num_graphs = t.shape[0]

    # --- Positions ---
    x1_pred = preds["pos_head"]
    t_node = t[batch_id]
    mu, sigma_t2 = position_policy_moments(
        mol_t.x, x1_pred, t_node, dt, cfg.sigma_noise, cfg.a_sde, cfg.eps_t,
    )
    lp_pos = gaussian_logprob_positions(
        step.x_next, mu, sigma_t2, dt, batch_id, num_graphs, cfg.var_floor,
    )

    # --- Atom sub/del ---
    q_sub = torch.sigmoid(preds["do_sub_a_head"].view(-1))
    atom_probs = F.softmax(preds["atom_type_head"], dim=-1)
    sub_rate_node = sub_rate_schedule.rate(t)[batch_id]
    p_sub = _q_to_p(q_sub, sub_rate_node, dt)
    # Phase 1: del disabled.
    p_del = torch.zeros_like(p_sub)
    lp_node = node_action_logprob(
        p_sub, p_del, atom_probs,
        step.a_choice, step.hat_a,
        batch_id, num_graphs,
    )

    # --- Edge sub (upper-triangular of mol_t) ---
    q_sub_e_full = torch.sigmoid(preds["do_sub_e_head"].view(-1))
    edge_probs_full = F.softmax(preds["edge_type_head"], dim=-1)
    triu_idx, (q_sub_e_triu, edge_probs_triu) = _extract_triu(
        mol_t.edge_index, [q_sub_e_full, edge_probs_full],
    )
    # DEPARTURES.md §D: defensive check; aligner is deterministic but this
    # catches any future ordering drift immediately.
    assert torch.equal(triu_idx, step.edge_triu_idx), (
        "edge aligner ordering drifted between rollout and scoring"
    )
    edge_batch_id = batch_id[triu_idx[0]]
    sub_e_rate_triu = sub_e_rate_schedule.rate(t)[edge_batch_id]
    p_sub_e_triu = _q_to_p(q_sub_e_triu, sub_e_rate_triu, dt)
    lp_edge = edge_sub_logprob(
        p_sub_e_triu, edge_probs_triu,
        step.e_triu_sub, step.hat_e_triu,
        edge_batch_id, num_graphs,
    )

    # --- Charge ---
    charge_probs = F.softmax(preds["charge_head"], dim=-1)
    survive_mask = step.a_choice != 2  # all True in Phase 1
    lp_charge = charge_logprob(
        charge_probs, step.hat_c, survive_mask, batch_id, num_graphs,
    )

    return lp_pos + lp_node + lp_edge + lp_charge


# ─────────────────────────────────────────────────────────────────────────────
# Rollout (no grad)
# ─────────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def _rollout_step(module, mol_t, t: torch.Tensor, dt: float, cfg: GRPOConfig):
    """Sample one SDE + discrete step from theta_old, recording actions."""
    model = module._get_model()
    model.set_inference()

    preds = module.cfg_adapter.guided_predict(
        model, mol_t, t, None, cfg.cfg_inputs,
    )

    batch_id = mol_t.batch
    num_graphs = t.shape[0]

    # --- Positions (SDE) ---
    x1_pred = preds["pos_head"]
    t_node = t[batch_id]
    mu, sigma_t2 = position_policy_moments(
        mol_t.x, x1_pred, t_node, dt, cfg.sigma_noise, cfg.a_sde, cfg.eps_t,
    )
    noise = torch.randn_like(mol_t.x)
    x_next = mu + torch.sqrt(sigma_t2.unsqueeze(-1) * dt) * noise

    # --- Sample categorical targets ---
    atom_probs = F.softmax(preds["atom_type_head"], dim=-1)
    charge_probs = F.softmax(preds["charge_head"], dim=-1)
    edge_probs_full = F.softmax(preds["edge_type_head"], dim=-1)
    hat_a = Categorical(probs=atom_probs).sample()
    hat_c = Categorical(probs=charge_probs).sample()
    hat_e_full = Categorical(probs=edge_probs_full).sample()

    # --- Node sub decision (Phase 1: no del).  Canonical q->p so sampling and
    #     scoring use identical probabilities (DEPARTURES.md §C). ---
    q_sub = torch.sigmoid(preds["do_sub_a_head"].view(-1))
    sub_rate_node = module.integrator.sub_schedule.rate(t)[batch_id]
    p_sub = _q_to_p(q_sub, sub_rate_node, dt)
    do_sub_a = torch.rand_like(p_sub) < p_sub
    a_choice = do_sub_a.to(torch.int8)  # 0 or 1 in Phase 1

    # --- Edge sub decision (triu) ---
    q_sub_e_full = torch.sigmoid(preds["do_sub_e_head"].view(-1))
    triu_idx, (q_sub_e_triu, edge_probs_triu, hat_e_triu, e_triu_current) = _extract_triu(
        mol_t.edge_index, [q_sub_e_full, edge_probs_full, hat_e_full, mol_t.e],
    )
    edge_batch_id = batch_id[triu_idx[0]]
    sub_e_rate_triu = module.integrator.sub_e_schedule.rate(t)[edge_batch_id]
    p_sub_e_triu = _q_to_p(q_sub_e_triu, sub_e_rate_triu, dt)
    e_triu_sub = torch.rand_like(p_sub_e_triu) < p_sub_e_triu

    # --- Build next molecule state ---
    a_next = mol_t.a.clone()
    a_next[do_sub_a] = hat_a[do_sub_a]
    c_next = hat_c
    e_triu_next = e_triu_current.clone()
    e_triu_next[e_triu_sub] = hat_e_triu[e_triu_sub]
    edge_index_full, (e_next,) = EDGE_ALIGNER.symmetrize_edges(
        triu_idx, [e_triu_next],
    )
    mol_next = MoleculeBatch(
        x=x_next.clone(),
        a=a_next,
        c=c_next,
        e=e_next,
        edge_index=edge_index_full,
        batch=batch_id.clone(),
    )

    # --- Record step data + log-prob under theta_old ---
    step = StepData(
        mol_t=mol_t.clone(),
        t=t.clone(),
        dt=dt,
        x_next=x_next.detach().clone(),
        a_choice=a_choice,
        hat_a=hat_a,
        hat_c=hat_c,
        edge_triu_idx=triu_idx.clone(),
        e_triu_sub=e_triu_sub,
        hat_e_triu=hat_e_triu,
        logp_old=torch.zeros(num_graphs, device=mol_t.x.device),  # filled below
    )
    # Fill logp_old by re-using the same preds + freshly sampled actions
    step.logp_old = _per_channel_logprob(
        preds, step.mol_t, step, t, dt, cfg,
        sub_rate_schedule=module.integrator.sub_schedule,
        sub_e_rate_schedule=module.integrator.sub_e_schedule,
    ).detach()

    return mol_next, step


@torch.no_grad()
def rollout_trajectory(module, batch, cfg: GRPOConfig) -> Trajectory:
    mol_t, mol_1 = batch
    mol_t = mol_t.clone()
    _ = mol_t.remove_com()

    batch_size = mol_t.batch_size
    device = mol_t.x.device
    t = torch.zeros(batch_size, device=device)
    n_steps = cfg.num_integration_steps or module.integrator.num_integration_steps
    step_sizes = module.integrator.get_time_steps(num_steps=n_steps)

    cfg.cfg_inputs = module._build_inference_cfg_inputs(mol_t, mol_1, batch_size)

    steps: List[StepData] = []
    for dt in step_sizes:
        mol_t, step = _rollout_step(module, mol_t, t, dt, cfg)
        steps.append(step)
        _ = mol_t.remove_com()
        t = t + dt

    return Trajectory(steps=steps, mol_final=mol_t)


# ─────────────────────────────────────────────────────────────────────────────
# GRPO update
# ─────────────────────────────────────────────────────────────────────────────


def _step_logprob_with_grad(module, step: StepData, cfg: GRPOConfig) -> torch.Tensor:
    model = module._get_model()
    model.set_inference()
    preds = module.cfg_adapter.guided_predict(
        model, step.mol_t, step.t, None, cfg.cfg_inputs,
    )
    return _per_channel_logprob(
        preds, step.mol_t, step, step.t, step.dt, cfg,
        sub_rate_schedule=module.integrator.sub_schedule,
        sub_e_rate_schedule=module.integrator.sub_e_schedule,
    )


def grpo_step(
    module,
    batch,
    cfg: GRPOConfig,
    optimizer: torch.optim.Optimizer,
    reward_fn: Callable = validity_reward,
) -> dict:
    """One GRPO update: rollout -> reward -> per-step clipped loss -> step."""
    # 1. Rollout under theta_old; cached logp_old in each StepData.
    trajectory = rollout_trajectory(module, batch, cfg)
    reward_tensor, reward_aux = reward_fn(module, trajectory)  # (B,), diagnostics
    trajectory.reward = reward_tensor

    # 2. Advantage (group = current batch)
    r = trajectory.reward
    adv = (r - r.mean()) / (r.std() + 1e-6)            # (B,)

    # 3. Per-step clipped loss, backward as we go to bound memory.
    optimizer.zero_grad(set_to_none=True)
    loss_sum = 0.0
    ratio_log: list[torch.Tensor] = []
    n_steps = len(trajectory.steps)

    for step in trajectory.steps:
        lp_new = _step_logprob_with_grad(module, step, cfg)
        # DEPARTURES.md §A: clamp log-ratio before exp to avoid inf/NaN from
        # graph-summed log-probs.  Far outside the PPO clip region, so this is
        # pure safety and does not change the effective objective.
        log_ratio = (lp_new - step.logp_old).clamp(
            -cfg.log_ratio_clamp, cfg.log_ratio_clamp,
        )
        ratio = torch.exp(log_ratio)                               # (B,)
        clipped = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps)
        step_loss = -torch.minimum(ratio * adv, clipped * adv).mean()
        (step_loss / n_steps).backward()
        loss_sum += step_loss.item()
        ratio_log.append(ratio.detach())

    optimizer.step()

    ratios = torch.stack(ratio_log)
    return {
        "loss": loss_sum / n_steps,
        "reward_mean": float(r.mean()),
        "reward_std": float(r.std()),
        "ratio_mean": float(ratios.mean()),
        "ratio_max": float(ratios.max()),
        "ratio_min": float(ratios.min()),
        **reward_aux,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────


def train(
    module,
    dataloader: Iterable,
    cfg: GRPOConfig,
    *,
    n_updates: int = 100,
    lr: float = 1e-5,
    device: str | torch.device = "cuda",
    log_every: int = 1,
    reward_fn: Callable = validity_reward,
) -> None:
    module = module.to(device)
    # Train mode on parameters; backbone internals that care about "inference vs train"
    # are handled by model.set_inference() inside the forward pass.
    for p in module.parameters():
        p.requires_grad_(True)

    optimizer = torch.optim.Adam(
        [p for p in module.parameters() if p.requires_grad], lr=lr,
    )

    # Piggy-back on an externally-initialised wandb run if the caller started one.
    wandb_run = None
    try:
        import wandb  # noqa: F401
        wandb_run = wandb.run
    except ImportError:
        pass

    it = iter(dataloader)
    for step in range(n_updates):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(dataloader)
            batch = next(it)

        batch = _batch_to_device(batch, device)
        info = grpo_step(module, batch, cfg, optimizer, reward_fn=reward_fn)

        if wandb_run is not None:
            wandb_run.log(info, step=step)
        if step % log_every == 0:
            info_str = " ".join(f"{k}={v:.4f}" for k, v in info.items())
            print(f"[grpo] step={step:04d} {info_str}", flush=True)


def _batch_to_device(batch, device):
    if isinstance(batch, (list, tuple)):
        return type(batch)(_batch_to_device(b, device) for b in batch)
    if hasattr(batch, "to"):
        return batch.to(device, non_blocking=True)
    return batch
