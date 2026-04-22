"""GRPO on the Morph architecture (Phase 2: full variable-atom regime).

Scope
-----
Per-step policy covers every channel in `integrator.integrate_step_gnn`:
    * positions                    - SDE; log-prob restricted to survivors
    * atom-type 3-way action       - noop / sub (+target) / del
    * edge-type substitution       - upper-tri Bernoulli + categorical target,
                                     log-prob restricted to surviving edges
    * charge                       - categorical over surviving existing nodes
    * insertion gate               - per-node Bernoulli(p_ins)
    * typed GMM                    - joint (x, a, c) for each inserted atom,
                                     log-prob marginalised over K components
    * ins->existing / ins->ins     - two-stage edge types, log-prob
                                     marginalised over the clean prediction

One step of the sampler consumes O(20) random draws and rewires the graph
topology.  Rollout performs the full set of draws; scoring recomputes all
log-probs from the stored captured actions + a fresh forward pass.

Convention
----------
Code convention throughout this file:
    t in [0, 1],   t = 0 is noise,   t = 1 is data,   sampling is t: 0 -> 1.
Matches `chemflow.flow_matching.integration` and `LightningModuleRates.sample`.
The overleaf is written in the opposite convention (x_0 data, x_1 noise).
Translation:
    t_user         <-> 1 - t_code
    (1 - t_user)   <-> t_code
    v_theta (both) == hat x_data - hat x_noise          (same sign here)
    sigma_t^2      (user) = a^2 t/(1-t)   ->  (code) a^2 (1-t)/t
After translating, the forward-time SDE is
    dx = [v - (sigma_t^2/2) * (x_t - t v) / ((1-t) + sigma_noise^2)] dt + sigma_t dw
so the score-correction sign is minus (same object, forward time).

Explicit differences vs. `sample()`
-----------------------------------
    * `prev_preds = None` always.  The model accepts a `prev_outs` kwarg in
      its forward signatures but the active backbone
      (`DiTBackboneWithHeads`) never reads it, and
      `SelfConditioningResidualLayer` is never imported.  So
      self-conditioning is effectively dead code in the base repo; passing
      `prev_preds` is a no-op.  Documented in DEPARTURES.md §2.
    * Positions evolve as an SDE, not a deterministic ODE: with `a_sde > 0`
      the Gaussian transition gives a well-defined policy for positions.
      Setting `a_sde = 0` collapses the SDE to the same Euler step
      `integrate_step_gnn` takes (still well-defined through `var_floor`).

Safeguards (replicated from `integrate_step_gnn`)
-------------------------------------------------
    * Insertion overflow: if `n_atoms + n_ins > max_atoms` for a graph,
      randomly drop excess `do_ins` flags.
    * Deletion underflow: if a graph would retain fewer than 2 nodes,
      restore some `do_del` flags.
Both are non-policy random modifications to the sampled action.  We apply
them identically to the integrator and then compute log-prob under the
*unmodified* Bernoulli probabilities, treating the safeguard as part of
the environment.  See DEPARTURES.md §E.

Open knobs / TODOs
------------------
    * `eps_t` floor on t inside sigma_t^2.
    * No KL term to a frozen reference policy.
    * Per-step clipped ratio (LLM-GRPO style); trajectory-level is a sum.
    * Reward defaults to RDKit validity; swap via `rl.rewards.REWARDS`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Optional

import torch
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

from chemflow.dataset.molecule_data import (
    MoleculeBatch,
    PointCloud,
    filter_nodes,
    join_molecules_with_atoms,
    join_molecules_with_predicted_edges,
)
from chemflow.utils.utils import EDGE_ALIGNER

from rl.rewards import validity_reward  # noqa: F401 (re-exported for callers)


EPS = 1e-8
EPS_T = 1e-2
DEFAULT_CLIP_EPS = 0.2
DEFAULT_VAR_FLOOR = 1e-6
DEFAULT_LOG_RATIO_CLAMP = 20.0
DEFAULT_P_INS_CLAMP = 1.0 - 1e-3


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
    p_ins_clamp: float = DEFAULT_P_INS_CLAMP       # overleaf §4.1: `min(p_ins, 1-eps)` (see §4)
    num_integration_steps: int | None = None       # None -> module's default
    # CFG inputs to feed guided_predict.  Built at rollout time from a batch's
    # (mol_t, mol_1) pair and stashed here so every step + log-prob call shares
    # the exact same conditioning.
    cfg_inputs: dict | None = None


@dataclass
class StepData:
    """Per-step rollout record.  All tensors detached from theta_old's graph.

    Captured actions are the minimum set needed to re-derive every
    log-prob under a fresh parameterisation `theta_new`: actual sampled
    values, not intermediate component indices.
    """

    mol_t: object              # MoleculeBatch snapshot at step start (COM-removed)
    t: torch.Tensor            # (B,) graph-level time
    dt: float

    # Continuous channel --------------------------------------------------
    x_next: torch.Tensor       # (N_orig, 3) realised x_{t+dt} on ALL original nodes
                               # (deleted nodes' rows are phantom; masked at log-prob time)

    # Discrete node channel (0 = NOOP, 1 = SUB, 2 = DEL) -----------------
    a_choice: torch.Tensor     # (N_orig,) int8, ENVIRONMENT-adjusted via the
                               # deletion underflow safeguard
    hat_a: torch.Tensor        # (N_orig,) sampled atom-type target (only used if a_choice==1)
    hat_c: torch.Tensor        # (N_orig,) sampled charge (all survivors)

    # Upper-tri edge channel ---------------------------------------------
    edge_triu_idx: torch.Tensor   # (2, E_triu) in mol_t's node space
    e_triu_sub: torch.Tensor      # (E_triu,) bool
    hat_e_triu: torch.Tensor      # (E_triu,) sampled edge-type target

    # Insertion channel ---------------------------------------------------
    do_ins: torch.Tensor       # (N_orig,) bool; ENVIRONMENT-adjusted (overflow clamp)
    ins_x: torch.Tensor        # (N_ins, 3) sampled GMM positions
    ins_a: torch.Tensor        # (N_ins,) sampled GMM atom types
    ins_c: torch.Tensor        # (N_ins,) sampled GMM charges

    # Ins -> existing edges (filtered to valid targets + valid spawns) ----
    ins_edge_spawn_orig: torch.Tensor  # (E_ins,) original-node indices of spawns
    ins_edge_existing_orig: torch.Tensor  # (E_ins,) original-node indices of targets
    hat_e_ins: torch.Tensor    # (E_ins,) sampled final edge types

    # Ins -> ins edges (all upper-tri pairs within a graph, among survivors) --
    ins_ii_spawn_src_orig: torch.Tensor  # (E_ii,) original spawn index for src side
    ins_ii_src_local: torch.Tensor       # (E_ii,) new-atom local index (src)
    ins_ii_dst_local: torch.Tensor       # (E_ii,) new-atom local index (dst)
    hat_e_ins_ii: torch.Tensor           # (E_ii,) sampled final ins->ins edge types

    # Cached per-graph log-prob under theta_old (computed at rollout time)
    logp_old: torch.Tensor     # (B,)


@dataclass
class Trajectory:
    steps: List[StepData] = field(default_factory=list)
    mol_final: object = None
    reward: Optional[torch.Tensor] = None  # (B,)


# ─────────────────────────────────────────────────────────────────────────────
# Low-level helpers
# ─────────────────────────────────────────────────────────────────────────────


def _scatter_sum(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Per-graph sum (CPU/GPU agnostic, autograd-safe)."""
    if src.numel() == 0:
        return src.new_zeros(dim_size) if src.dtype.is_floating_point else torch.zeros(
            dim_size, device=src.device, dtype=torch.float32,
        )
    out = src.new_zeros(dim_size)
    out.index_add_(0, index, src)
    return out


def _gather_logp(probs: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """log probs[..., idx] with a numerical floor."""
    return torch.log(probs.clamp(min=EPS)).gather(-1, idx.view(-1, 1)).squeeze(-1)


def _q_to_p(q: torch.Tensor, rate_node: torch.Tensor, dt: float) -> torch.Tensor:
    """Canonical raw-sigmoid -> per-step probability (see DEPARTURES.md §C)."""
    return (q * rate_node * dt).clamp(EPS, 1.0 - EPS)


def _extract_triu(
    edge_index: torch.Tensor, attrs: list[torch.Tensor],
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    infos = EDGE_ALIGNER.align_edges(source_group=(edge_index, attrs))
    return infos["edge_index"], list(infos["edge_attr"])


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

    v = (x1_pred - x_t) / one_minus_t
    score = -(x_t - t * v) / (one_minus_t + sigma_n2)
    sigma_t2 = (a_sde ** 2) * one_minus_t / t
    mu = x_t + dt * (v + 0.5 * sigma_t2 * score)
    return mu, sigma_t2.squeeze(-1)


def gaussian_logprob_positions(
    x_next: torch.Tensor,    # (N, 3)
    mu: torch.Tensor,        # (N, 3)
    sigma_t2: torch.Tensor,  # (N,)
    dt: float,
    batch_id: torch.Tensor,  # (N,)
    num_graphs: int,
    survive_mask: torch.Tensor | None = None,  # (N,) bool
    var_floor: float = DEFAULT_VAR_FLOOR,
) -> torch.Tensor:
    """Per-graph sum of Gaussian log-densities, restricted to `survive_mask`."""
    var = (sigma_t2 * dt).clamp_min(var_floor)
    sq = ((x_next - mu) ** 2).sum(-1)
    logp_node = -0.5 * sq / var - 1.5 * torch.log(2 * math.pi * var)
    if survive_mask is not None:
        logp_node = torch.where(survive_mask, logp_node, torch.zeros_like(logp_node))
    return _scatter_sum(logp_node, batch_id, num_graphs)


# ─────────────────────────────────────────────────────────────────────────────
# Discrete per-channel log-probs
# ─────────────────────────────────────────────────────────────────────────────


def node_action_logprob(
    p_sub: torch.Tensor,        # (N,)
    p_del: torch.Tensor,        # (N,) -- may be all zeros when n_atoms_strategy == "fixed"
    atom_probs: torch.Tensor,   # (N, |A|)
    a_choice: torch.Tensor,     # (N,) 0/1/2
    hat_a: torch.Tensor,        # (N,)
    batch_id: torch.Tensor,
    num_graphs: int,
) -> torch.Tensor:
    """log pi^{node} per graph (overleaf Eq. 22).

    `p_sub`, `p_del` already passed through `_q_to_p` (in (EPS, 1-EPS)).  We
    re-clamp `p_any` / `p_noop` defensively because the sum can still exceed 1.
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
    survive_mask: torch.Tensor | None = None,  # (E_triu,) bool - both endpoints survive
) -> torch.Tensor:
    log_noop = torch.log(1.0 - p_sub_e)
    log_sub = torch.log(p_sub_e)
    log_pi_edge = _gather_logp(edge_probs, hat_e)
    logp = torch.where(e_triu_sub, log_sub + log_pi_edge, log_noop)
    if survive_mask is not None:
        logp = torch.where(survive_mask, logp, torch.zeros_like(logp))
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


def ins_gate_logprob(
    p_ins: torch.Tensor,         # (N,) already-clamped Bernoulli probability
    do_ins: torch.Tensor,        # (N,) bool
    batch_id: torch.Tensor,
    num_graphs: int,
) -> torch.Tensor:
    """Sum of per-node Bernoulli log-probs (overleaf Eq. logprob_ins_gate)."""
    log_pos = torch.log(p_ins.clamp(min=EPS))
    log_neg = torch.log((1.0 - p_ins).clamp(min=EPS))
    logp = torch.where(do_ins, log_pos, log_neg)
    return _scatter_sum(logp, batch_id, num_graphs)


def gmm_marginal_logprob(
    gmm_dict: dict,              # keys: mu (n,K,D), sigma (n,K), pi (n,K), a_probs (n,K,|A|), c_probs (n,K,|C|)
    ins_x: torch.Tensor,         # (n, D)
    ins_a: torch.Tensor,         # (n,)
    ins_c: torch.Tensor,         # (n,)
) -> torch.Tensor:
    """log pi^GMM per inserted atom; logsumexp over K components (overleaf Eq. gmm_logprob_lse)."""
    if ins_x.numel() == 0:
        return ins_x.new_zeros(0)

    mu = gmm_dict["mu"]                                # (n, K, D)
    sigma = gmm_dict["sigma"]                          # (n, K)
    pi = gmm_dict["pi"]                                # (n, K)
    a_probs = gmm_dict["a_probs"]                      # (n, K, |A|)
    c_probs = gmm_dict["c_probs"]                      # (n, K, |C|)

    # Spatial: Normal with isotropic sigma.  Sum log-density over D, broadcast (n,1,D) - (n,K,D).
    loc = mu
    scale = sigma.unsqueeze(-1).expand_as(loc).clamp(min=EPS)
    log_N = Normal(loc, scale).log_prob(ins_x.unsqueeze(1)).sum(-1)  # (n, K)

    # Types: gather at ins_a / ins_c, then log.
    log_pa = torch.log(a_probs.clamp(min=EPS)).gather(-1, ins_a.view(-1, 1, 1).expand(-1, a_probs.size(1), 1)).squeeze(-1)
    log_pc = torch.log(c_probs.clamp(min=EPS)).gather(-1, ins_c.view(-1, 1, 1).expand(-1, c_probs.size(1), 1)).squeeze(-1)

    log_pi = torch.log(pi.clamp(min=EPS))
    logits = log_pi + log_N + log_pa + log_pc               # (n, K)
    return torch.logsumexp(logits, dim=-1)                  # (n,)


def ins_edge_marginal_logprob(
    edge_probs: torch.Tensor,    # (E, |E|) pi_edge
    prior_probs: torch.Tensor,   # (|E|,) prior
    kappa_t: torch.Tensor,       # (E, 1) in [0, 1]
    hat_e: torch.Tensor,         # (E,)
) -> torch.Tensor:
    """Overleaf Eq. ins_edge_marginal.

    P(hat_e = k) = (1 - kappa) * p_prior[k] + kappa * pi_edge[k]
    Closed-form marginal over the latent `e1` that the integrator samples.
    """
    if hat_e.numel() == 0:
        return edge_probs.new_zeros(0)
    prior_mix = prior_probs.view(1, -1).expand_as(edge_probs)
    mixed = (1.0 - kappa_t) * prior_mix + kappa_t * edge_probs          # (E, |E|)
    mixed = mixed.clamp(min=EPS)
    return torch.log(mixed).gather(-1, hat_e.view(-1, 1)).squeeze(-1)   # (E,)


# ─────────────────────────────────────────────────────────────────────────────
# Shared per-step forward + log-prob (used by rollout and scoring)
# ─────────────────────────────────────────────────────────────────────────────


def _forward_preds(module, mol_t, t: torch.Tensor, cfg: GRPOConfig):
    """Single forward pass that returns the full preds dict (with h_latent)."""
    model = module._get_model()
    model.set_inference()
    return module.cfg_adapter.guided_predict(model, mol_t, t, None, cfg.cfg_inputs)


def _compute_p_ins(
    num_ins_pred: torch.Tensor,   # (N,) raw rate-head output
    ins_rate_node: torch.Tensor,  # (N,) schedule rate * dt multiplier
    dt: float,
    p_ins_clamp: float,
) -> torch.Tensor:
    """Overleaf §4.1: expected_ins * ins_rate * dt, clamped below 1 - eps."""
    return (num_ins_pred * ins_rate_node * dt).clamp(EPS, p_ins_clamp)


def _compute_p_sub_del(
    q_sub: torch.Tensor, q_del: torch.Tensor,
    sub_rate_node: torch.Tensor, del_rate_node: torch.Tensor, dt: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Raw `_q_to_p` on each channel.  Both rollout and `node_action_logprob`
    clamp `p_any = p_sub + p_del` to `1 - EPS`, so no rescaling here is
    needed for the two paths to stay consistent.
    """
    return _q_to_p(q_sub, sub_rate_node, dt), _q_to_p(q_del, del_rate_node, dt)


def _per_channel_logprob(
    preds: dict,
    step: StepData,
    cfg: GRPOConfig,
    module,
) -> torch.Tensor:
    """Compute per-graph log-prob from `preds` (under ANY theta) and `step`'s captured actions.

    Mirrors the rollout channel-by-channel.  Every randomness draw in
    `_rollout_step` has a term here; `step.logp_old` is just this function
    evaluated under theta_old's preds.
    """
    integrator = module.integrator
    mol_t = step.mol_t
    t, dt = step.t, step.dt
    batch_id = mol_t.batch
    num_graphs = t.shape[0]
    device = mol_t.x.device

    # ─── Schedules ──────────────────────────────────────────────────────
    sub_rate_node = integrator.sub_schedule.rate(t)[batch_id]
    del_rate_node = integrator.del_schedule.rate(t)[batch_id]
    ins_rate_node = integrator.ins_schedule.rate(t)[batch_id]
    sub_e_rate = integrator.sub_e_schedule.rate(t)
    prior_edge_probs = integrator._cat_edge.probs.to(device)  # (|E|,)

    survive_mask = step.a_choice != 2                # (N_orig,) bool

    # ─── Positions ──────────────────────────────────────────────────────
    x1_pred = preds["pos_head"]
    t_node = t[batch_id]
    mu, sigma_t2 = position_policy_moments(
        mol_t.x, x1_pred, t_node, dt, cfg.sigma_noise, cfg.a_sde, cfg.eps_t,
    )
    lp_pos = gaussian_logprob_positions(
        step.x_next, mu, sigma_t2, dt, batch_id, num_graphs,
        survive_mask=survive_mask, var_floor=cfg.var_floor,
    )

    # ─── Atom sub / del 3-way ───────────────────────────────────────────
    q_sub = torch.sigmoid(preds["do_sub_a_head"].view(-1))
    q_del = torch.sigmoid(preds["do_del_head"].view(-1))
    atom_probs = F.softmax(preds["atom_type_head"], dim=-1)
    if getattr(module, "n_atoms_strategy", "fixed") == "fixed":
        q_del = torch.zeros_like(q_del)
    p_sub, p_del = _compute_p_sub_del(q_sub, q_del, sub_rate_node, del_rate_node, dt)
    lp_node = node_action_logprob(
        p_sub, p_del, atom_probs,
        step.a_choice.to(torch.long), step.hat_a,
        batch_id, num_graphs,
    )

    # ─── Edge sub (upper-tri of mol_t) ──────────────────────────────────
    q_sub_e_full = torch.sigmoid(preds["do_sub_e_head"].view(-1))
    edge_probs_full = F.softmax(preds["edge_type_head"], dim=-1)
    triu_idx, (q_sub_e_triu, edge_probs_triu) = _extract_triu(
        mol_t.edge_index, [q_sub_e_full, edge_probs_full],
    )
    # DEPARTURES.md §D
    assert torch.equal(triu_idx, step.edge_triu_idx), (
        "edge aligner ordering drifted between rollout and scoring"
    )
    edge_batch_id = batch_id[triu_idx[0]]
    sub_e_rate_triu = sub_e_rate[edge_batch_id]
    p_sub_e_triu = _q_to_p(q_sub_e_triu, sub_e_rate_triu, dt)
    # Surviving edges: both endpoints survive.
    edge_survive_mask = survive_mask[triu_idx[0]] & survive_mask[triu_idx[1]]
    lp_edge = edge_sub_logprob(
        p_sub_e_triu, edge_probs_triu,
        step.e_triu_sub, step.hat_e_triu,
        edge_batch_id, num_graphs,
        survive_mask=edge_survive_mask,
    )

    # ─── Charge (surviving existing nodes only) ─────────────────────────
    charge_probs = F.softmax(preds["charge_head"], dim=-1)
    lp_charge = charge_logprob(
        charge_probs, step.hat_c, survive_mask, batch_id, num_graphs,
    )

    # ─── Insertion gate (all nodes) ─────────────────────────────────────
    num_ins_pred = preds["ins_rate_head"].view(-1)
    if getattr(module, "n_atoms_strategy", "fixed") == "fixed":
        num_ins_pred = torch.zeros_like(num_ins_pred)
    p_ins = _compute_p_ins(num_ins_pred, ins_rate_node, dt, cfg.p_ins_clamp)
    lp_ins_gate = ins_gate_logprob(p_ins, step.do_ins, batch_id, num_graphs)

    # ─── GMM (inserted atoms) ───────────────────────────────────────────
    lp_gmm = torch.zeros(num_graphs, device=device)
    if step.do_ins.any():
        gmm_full = preds["gmm_head"]                    # dict with per-node tensors
        gmm_sub = {k: v[step.do_ins] for k, v in gmm_full.items()}
        gmm_per_ins = gmm_marginal_logprob(
            gmm_sub, step.ins_x, step.ins_a, step.ins_c,
        )                                                # (N_ins,)
        ins_batch_id = batch_id[step.do_ins]
        lp_gmm = _scatter_sum(gmm_per_ins, ins_batch_id, num_graphs)

    # ─── Ins -> existing edges ──────────────────────────────────────────
    lp_ins_edge_ext = torch.zeros(num_graphs, device=device)
    if step.do_ins.any() and step.ins_edge_spawn_orig_full.numel() > 0:
        ins_edge_head = getattr(module._get_model(), "ins_edge_head", None)
        if ins_edge_head is not None:
            # Rebuild the post-sub atom types exactly as rollout did, so the
            # edge head sees the same `node_atom_types` signal on both paths.
            a_next_score = mol_t.a.clone()
            sub_mask_score = step.a_choice == 1
            a_next_score[sub_mask_score] = step.hat_a[sub_mask_score]

            spawn_idx, existing_idx, ins_logits = ins_edge_head.predict_edges_for_insertion(
                h=preds["h_latent"],
                x=mol_t.x,
                node_atom_types=a_next_score,
                batch=batch_id,
                insertion_mask=step.do_ins,
                ins_x=step.ins_x,
                ins_a=step.ins_a,
                ins_c=step.ins_c,
            )
            # Match the stored edges (both sides keyed by original node indices).
            # `predict_edges_for_insertion` builds pairs in a deterministic order:
            # for each spawn in `torch.where(do_ins)[0]`, iterate over same-graph
            # nodes. We assert the returned (spawn_idx, existing_idx) matches the
            # stored pairs by equality; if order ever drifts, we fall back to
            # matching via a hash.
            assert torch.equal(spawn_idx, step.ins_edge_spawn_orig_full) and \
                torch.equal(existing_idx, step.ins_edge_existing_orig_full), (
                    "insertion edge ordering drifted between rollout and scoring"
                )
            ins_probs = F.softmax(ins_logits, dim=-1)          # (E_full, |E|)
            t_ins = t[batch_id[spawn_idx]] + dt
            kappa_t_e = integrator.sub_e_schedule.kappa_t(t_ins).unsqueeze(1)

            # hat_e_ins_full is stored aligned with (spawn_idx, existing_idx);
            # invalid edges (deleted endpoints) were NEVER placed, so their log-prob
            # is dominated by the Bernoulli-through-softmax stage we skipped --
            # but scoring only over the *valid* set keeps us consistent with rollout:
            # at rollout, invalid edges are dropped before being placed in the graph
            # and thus contribute nothing to the trajectory log-prob.
            lp_per_edge = ins_edge_marginal_logprob(
                ins_probs, prior_edge_probs, kappa_t_e, step.hat_e_ins_full,
            )
            valid = step.ins_edge_valid_mask
            lp_per_edge = torch.where(valid, lp_per_edge, torch.zeros_like(lp_per_edge))
            edge_batch_id_ext = batch_id[spawn_idx]
            lp_ins_edge_ext = _scatter_sum(lp_per_edge, edge_batch_id_ext, num_graphs)

    # ─── Ins -> ins edges ───────────────────────────────────────────────
    lp_ins_edge_ii = torch.zeros(num_graphs, device=device)
    if step.hat_e_ins_ii.numel() > 0:
        ins_edge_head = getattr(module._get_model(), "ins_edge_head", None)
        if ins_edge_head is not None:
            new_atoms_a = step.ins_a
            new_atoms_x = step.ins_x
            new_atoms_c = step.ins_c
            ii_logits = ins_edge_head.forward_ins_to_ins(
                h=preds["h_latent"],
                x=mol_t.x,
                spawn_src_idx=step.ins_ii_spawn_src_orig,
                ins_x_src=new_atoms_x[step.ins_ii_src_local],
                ins_a_src=new_atoms_a[step.ins_ii_src_local],
                ins_c_src=new_atoms_c[step.ins_ii_src_local],
                ins_a_dst=new_atoms_a[step.ins_ii_dst_local],
                ins_x_dst=new_atoms_x[step.ins_ii_dst_local],
            )
            ii_probs = F.softmax(ii_logits, dim=-1)
            t_ii = t[batch_id[step.ins_ii_spawn_src_orig]] + dt
            kappa_t_ii = integrator.sub_e_schedule.kappa_t(t_ii).unsqueeze(1)
            lp_per_ii = ins_edge_marginal_logprob(
                ii_probs, prior_edge_probs, kappa_t_ii, step.hat_e_ins_ii,
            )
            edge_batch_id_ii = batch_id[step.ins_ii_spawn_src_orig]
            lp_ins_edge_ii = _scatter_sum(lp_per_ii, edge_batch_id_ii, num_graphs)

    return lp_pos + lp_node + lp_edge + lp_charge + lp_ins_gate + lp_gmm + lp_ins_edge_ext + lp_ins_edge_ii


# ─────────────────────────────────────────────────────────────────────────────
# Rollout (no grad)
# ─────────────────────────────────────────────────────────────────────────────


def _apply_overflow_safeguard(
    do_ins: torch.Tensor, batch_id: torch.Tensor, num_graphs: int, max_atoms: int,
) -> torch.Tensor:
    """Insertion overflow safeguard (matches integrator lines 272-288)."""
    if not do_ins.any():
        return do_ins
    n_atoms = torch.bincount(batch_id, minlength=num_graphs)
    n_ins = torch.bincount(batch_id[do_ins], minlength=num_graphs)
    overflow = (n_atoms + n_ins - max_atoms).clamp(min=0)
    if not overflow.any():
        return do_ins
    do_ins = do_ins.clone()
    for g in torch.nonzero(overflow, as_tuple=False).flatten().tolist():
        n_to_remove = int(overflow[g].item())
        removed = torch.where((batch_id == g) & do_ins)[0]
        removed = removed[torch.randperm(removed.shape[0], device=do_ins.device)]
        do_ins[removed[:n_to_remove]] = False
    return do_ins


def _apply_underflow_safeguard(
    do_del: torch.Tensor, batch_id: torch.Tensor, num_graphs: int,
) -> torch.Tensor:
    """Deletion underflow safeguard (matches integrator lines 369-379)."""
    keep = ~do_del
    n_kept = torch.bincount(batch_id[keep], minlength=num_graphs)
    under = (2 - n_kept).clamp(min=0)
    if not under.any():
        return do_del
    do_del = do_del.clone()
    for g in torch.nonzero(under, as_tuple=False).flatten().tolist():
        n_restore = int(under[g].item())
        deleted_in_g = torch.where((batch_id == g) & do_del)[0]
        do_del[deleted_in_g[:n_restore]] = False
    return do_del


@torch.no_grad()
def _rollout_step(module, mol_t, t: torch.Tensor, dt: float, cfg: GRPOConfig):
    """Sample one full SDE + discrete + insertion step from theta_old."""
    integrator = module.integrator
    preds = _forward_preds(module, mol_t, t, cfg)

    batch_id = mol_t.batch
    num_graphs = t.shape[0]
    device = mol_t.x.device
    N_orig = mol_t.x.size(0)
    n_atoms_strategy = getattr(module, "n_atoms_strategy", "fixed")
    prior_edge_probs = integrator._cat_edge.probs.to(device)

    # ─── Positions (SDE) ────────────────────────────────────────────────
    x1_pred = preds["pos_head"]
    t_node = t[batch_id]
    mu, sigma_t2 = position_policy_moments(
        mol_t.x, x1_pred, t_node, dt, cfg.sigma_noise, cfg.a_sde, cfg.eps_t,
    )
    noise_pos = torch.randn_like(mol_t.x)
    x_next = mu + torch.sqrt(sigma_t2.unsqueeze(-1) * dt) * noise_pos

    # ─── Categorical predictions (shared across rollout + scoring) ─────
    atom_probs = F.softmax(preds["atom_type_head"], dim=-1)
    charge_probs = F.softmax(preds["charge_head"], dim=-1)
    edge_probs_full = F.softmax(preds["edge_type_head"], dim=-1)
    hat_a = Categorical(probs=atom_probs).sample()
    hat_c = Categorical(probs=charge_probs).sample()
    hat_e_full = Categorical(probs=edge_probs_full).sample()

    # ─── 3-way node action (noop / sub / del) ───────────────────────────
    q_sub = torch.sigmoid(preds["do_sub_a_head"].view(-1))
    q_del = torch.sigmoid(preds["do_del_head"].view(-1))
    if n_atoms_strategy == "fixed":
        q_del = torch.zeros_like(q_del)
    sub_rate_node = integrator.sub_schedule.rate(t)[batch_id]
    del_rate_node = integrator.del_schedule.rate(t)[batch_id]
    p_sub, p_del = _compute_p_sub_del(q_sub, q_del, sub_rate_node, del_rate_node, dt)

    # Sampling scheme matches integrator: Bernoulli(p_any), then conditional
    # split into del vs sub.  Keeps the marginal distribution exact.
    p_any = (p_sub + p_del).clamp(EPS, 1.0 - EPS)
    do_edit = torch.rand_like(p_any) < p_any
    prob_cond_del = p_del / (p_any + EPS)
    is_del = torch.rand_like(prob_cond_del) < prob_cond_del
    do_del = do_edit & is_del
    do_sub_a = do_edit & (~is_del)

    # Deletion underflow safeguard (environment, not policy).
    if n_atoms_strategy != "fixed":
        do_del = _apply_underflow_safeguard(do_del, batch_id, num_graphs)
    a_choice = torch.zeros(N_orig, dtype=torch.int8, device=device)
    a_choice[do_sub_a] = 1
    a_choice[do_del] = 2

    # ─── Edge sub on triu ───────────────────────────────────────────────
    q_sub_e_full = torch.sigmoid(preds["do_sub_e_head"].view(-1))
    triu_idx, (q_sub_e_triu, edge_probs_triu, hat_e_triu, e_triu_current) = _extract_triu(
        mol_t.edge_index, [q_sub_e_full, edge_probs_full, hat_e_full, mol_t.e],
    )
    edge_batch_id = batch_id[triu_idx[0]]
    sub_e_rate_triu = integrator.sub_e_schedule.rate(t)[edge_batch_id]
    p_sub_e_triu = _q_to_p(q_sub_e_triu, sub_e_rate_triu, dt)
    e_triu_sub = torch.rand_like(p_sub_e_triu) < p_sub_e_triu

    # ─── Insertion gate ─────────────────────────────────────────────────
    num_ins_pred = preds["ins_rate_head"].view(-1)
    if n_atoms_strategy == "fixed":
        num_ins_pred = torch.zeros_like(num_ins_pred)
    ins_rate_node = integrator.ins_schedule.rate(t)[batch_id]
    p_ins = _compute_p_ins(num_ins_pred, ins_rate_node, dt, cfg.p_ins_clamp)
    do_ins = torch.rand_like(p_ins) < p_ins

    # Insertion overflow safeguard (environment, not policy).
    if n_atoms_strategy != "fixed":
        do_ins = _apply_overflow_safeguard(
            do_ins, batch_id, num_graphs, integrator.max_atoms,
        )

    # ─── GMM sample for inserted atoms ──────────────────────────────────
    ins_x = x_next.new_zeros(0, 3)
    ins_a = torch.zeros(0, dtype=torch.long, device=device)
    ins_c = torch.zeros(0, dtype=torch.long, device=device)
    if do_ins.any():
        gmm_full = preds["gmm_head"]
        gmm_sub = {k: v[do_ins] for k, v in gmm_full.items()}
        t_ins = t[batch_id[do_ins]] + dt
        new_atoms = integrator.sample_insertions(gmm_sub, t_ins)
        ins_x, ins_a, ins_c = new_atoms.x, new_atoms.a, new_atoms.c

    # ─── Build next molecule state before topology edits ────────────────
    a_next = mol_t.a.clone()
    a_next[do_sub_a] = hat_a[do_sub_a]
    e_triu_next = e_triu_current.clone()
    e_triu_next[e_triu_sub] = hat_e_triu[e_triu_sub]
    edge_index_full, (e_next,) = EDGE_ALIGNER.symmetrize_edges(triu_idx, [e_triu_next])
    mol_post_sub = MoleculeBatch(
        x=x_next.clone(),
        a=a_next,
        c=hat_c,
        e=e_next,
        edge_index=edge_index_full,
        batch=batch_id.clone(),
    )

    # ─── Ins -> existing edges: sample via two-stage marginalisation ────
    ins_edge_head = getattr(module._get_model(), "ins_edge_head", None)
    hat_e_ins_full = torch.zeros(0, dtype=torch.long, device=device)
    ins_edge_spawn_full = torch.zeros(0, dtype=torch.long, device=device)
    ins_edge_existing_full = torch.zeros(0, dtype=torch.long, device=device)
    ins_edge_valid_mask = torch.zeros(0, dtype=torch.bool, device=device)

    hat_e_ins_valid = torch.zeros(0, dtype=torch.long, device=device)
    ins_edge_spawn_orig_valid = torch.zeros(0, dtype=torch.long, device=device)
    ins_edge_existing_orig_valid = torch.zeros(0, dtype=torch.long, device=device)

    hat_e_ins_ii = torch.zeros(0, dtype=torch.long, device=device)
    ins_ii_spawn_src_orig = torch.zeros(0, dtype=torch.long, device=device)
    ins_ii_src_local = torch.zeros(0, dtype=torch.long, device=device)
    ins_ii_dst_local = torch.zeros(0, dtype=torch.long, device=device)

    if do_ins.any() and ins_edge_head is not None:
        # Match integrator: edge head sees POST-SUB atom types (but pre-del).
        spawn_idx, existing_idx, ins_logits = ins_edge_head.predict_edges_for_insertion(
            h=preds["h_latent"],
            x=mol_t.x,
            node_atom_types=a_next,
            batch=batch_id,
            insertion_mask=do_ins,
            ins_x=ins_x, ins_a=ins_a, ins_c=ins_c,
        )
        ins_edge_spawn_full = spawn_idx.clone()
        ins_edge_existing_full = existing_idx.clone()

        if ins_logits.numel() > 0:
            ins_probs = F.softmax(ins_logits, dim=-1)
            t_ins_graph = t[batch_id[spawn_idx]] + dt
            kappa_t_e = integrator.sub_e_schedule.kappa_t(t_ins_graph).unsqueeze(1)
            # Closed-form marginal sampling: P(hat_e) = (1 - kappa) * prior + kappa * pi.
            # Matches integrator's two-stage draw in distribution (DEPARTURES.md §F).
            mixed_probs = (
                (1.0 - kappa_t_e) * prior_edge_probs.view(1, -1) + kappa_t_e * ins_probs
            )
            hat_e_ins_full = Categorical(probs=mixed_probs.clamp(min=EPS)).sample()

            # Edges are valid if endpoint survives deletion *and* spawn is in do_ins
            # (always true here since spawn_idx comes from do_ins).
            target_survives = (~do_del)[existing_idx]
            ins_edge_valid_mask = target_survives

            hat_e_ins_valid = hat_e_ins_full[ins_edge_valid_mask]
            ins_edge_spawn_orig_valid = spawn_idx[ins_edge_valid_mask]
            ins_edge_existing_orig_valid = existing_idx[ins_edge_valid_mask]

    # ─── Ins -> ins edges: build upper-tri pairs within each graph ──────
    if do_ins.any() and ins_edge_head is not None and ins_x.size(0) >= 2:
        new_atoms_batch = batch_id[do_ins]
        spawn_orig_indices = torch.where(do_ins)[0]
        ii_src_list, ii_dst_list, ii_spawn_src_list = [], [], []
        for g in new_atoms_batch.unique():
            mask_g = new_atoms_batch == g
            local_idx_g = torch.where(mask_g)[0]
            if local_idx_g.numel() >= 2:
                pairs = torch.combinations(local_idx_g, r=2)
                ii_src_list.append(pairs[:, 0])
                ii_dst_list.append(pairs[:, 1])
                ii_spawn_src_list.append(spawn_orig_indices[pairs[:, 0]])

        if ii_src_list:
            ii_src = torch.cat(ii_src_list)
            ii_dst = torch.cat(ii_dst_list)
            ii_spawn_src = torch.cat(ii_spawn_src_list)

            ii_logits = ins_edge_head.forward_ins_to_ins(
                h=preds["h_latent"],
                x=mol_t.x,
                spawn_src_idx=ii_spawn_src,
                ins_x_src=ins_x[ii_src],
                ins_a_src=ins_a[ii_src],
                ins_c_src=ins_c[ii_src],
                ins_a_dst=ins_a[ii_dst],
                ins_x_dst=ins_x[ii_dst],
            )
            ii_probs = F.softmax(ii_logits, dim=-1)
            t_ii = t[batch_id[ii_spawn_src]] + dt
            kappa_t_ii = integrator.sub_e_schedule.kappa_t(t_ii).unsqueeze(1)
            mixed_ii = (
                (1.0 - kappa_t_ii) * prior_edge_probs.view(1, -1) + kappa_t_ii * ii_probs
            )
            hat_e_ins_ii = Categorical(probs=mixed_ii.clamp(min=EPS)).sample()
            ins_ii_spawn_src_orig = ii_spawn_src
            ins_ii_src_local = ii_src
            ins_ii_dst_local = ii_dst

    # ─── Rewire topology: delete, then insert ───────────────────────────
    if n_atoms_strategy != "fixed":
        keep_mask = ~do_del
        if do_del.any():
            mol_post_del = filter_nodes(mol_post_sub, keep_mask)
        else:
            mol_post_del = mol_post_sub

        if do_ins.any():
            # Original -> post-deletion node mapping.
            orig_to_postdel = torch.full((N_orig,), -1, dtype=torch.long, device=device)
            kept_indices = torch.where(keep_mask)[0]
            orig_to_postdel[kept_indices] = torch.arange(kept_indices.numel(), device=device)

            new_atoms_pc = PointCloud(x=ins_x, a=ins_a, c=ins_c)
            new_atoms_pc.batch = batch_id[do_ins]
            edge_dist = integrator.distributions.edge_type_distribution.to(device)

            # Map spawn-originals to new-atom indices (0..N_ins-1).
            orig_to_new_atom = torch.full((N_orig,), -1, dtype=torch.long, device=device)
            spawn_orig_all = torch.where(do_ins)[0]
            orig_to_new_atom[spawn_orig_all] = torch.arange(spawn_orig_all.numel(), device=device)

            if ins_edge_spawn_full.numel() > 0 and ins_edge_valid_mask.any():
                # Subset valid edges and remap into post-deletion space.
                valid = ins_edge_valid_mask
                spawn_new = orig_to_new_atom[ins_edge_spawn_full[valid]]
                target_post = orig_to_postdel[ins_edge_existing_full[valid]]
                e_t_ins_placed = hat_e_ins_full[valid]

                mol_t_next = join_molecules_with_predicted_edges(
                    mol=mol_post_del,
                    new_atoms=new_atoms_pc,
                    e_ins=e_t_ins_placed,
                    spawn_node_idx=spawn_new,
                    target_node_idx=target_post,
                    fallback_edge_dist=edge_dist,
                    e_ins_to_ins=hat_e_ins_ii if hat_e_ins_ii.numel() > 0 else None,
                    ins_to_ins_src_idx=ins_ii_src_local if ins_ii_src_local.numel() > 0 else None,
                    ins_to_ins_dst_idx=ins_ii_dst_local if ins_ii_dst_local.numel() > 0 else None,
                )
            else:
                mol_t_next = join_molecules_with_atoms(mol_post_del, new_atoms_pc, edge_dist)
        else:
            mol_t_next = mol_post_del
    else:
        mol_t_next = mol_post_sub

    # ─── Pack StepData ──────────────────────────────────────────────────
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
        do_ins=do_ins,
        ins_x=ins_x.detach().clone(),
        ins_a=ins_a.clone(),
        ins_c=ins_c.clone(),
        ins_edge_spawn_orig=ins_edge_spawn_orig_valid,
        ins_edge_existing_orig=ins_edge_existing_orig_valid,
        hat_e_ins=hat_e_ins_valid,
        ins_ii_spawn_src_orig=ins_ii_spawn_src_orig,
        ins_ii_src_local=ins_ii_src_local,
        ins_ii_dst_local=ins_ii_dst_local,
        hat_e_ins_ii=hat_e_ins_ii,
        logp_old=torch.zeros(num_graphs, device=device),  # filled below
    )
    # Scoring needs the full (pre-validity-filter) spawn/target pairing so that
    # the fresh forward's `predict_edges_for_insertion` produces tensors in the
    # same order.  Attach as extra fields to avoid a 30-arg dataclass.
    step.ins_edge_spawn_orig_full = ins_edge_spawn_full
    step.ins_edge_existing_orig_full = ins_edge_existing_full
    step.ins_edge_valid_mask = ins_edge_valid_mask
    step.hat_e_ins_full = hat_e_ins_full

    # Fill logp_old using the fresh preds from this rollout pass.
    step.logp_old = _per_channel_logprob(preds, step, cfg, module).detach()
    return mol_t_next, step


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
    preds = _forward_preds(module, step.mol_t, step.t, cfg)
    return _per_channel_logprob(preds, step, cfg, module)


def grpo_step(
    module,
    batch,
    cfg: GRPOConfig,
    optimizer: torch.optim.Optimizer,
    reward_fn: Callable = validity_reward,
) -> dict:
    """One GRPO update: rollout -> reward -> per-step clipped loss -> step."""
    trajectory = rollout_trajectory(module, batch, cfg)
    reward_tensor, reward_aux = reward_fn(module, trajectory)
    trajectory.reward = reward_tensor

    r = trajectory.reward
    adv = (r - r.mean()) / (r.std() + 1e-6)

    optimizer.zero_grad(set_to_none=True)
    loss_sum = 0.0
    ratio_log: list[torch.Tensor] = []
    n_steps = len(trajectory.steps)

    for step in trajectory.steps:
        lp_new = _step_logprob_with_grad(module, step, cfg)
        # DEPARTURES.md §A: clamp log-ratio before exp.
        log_ratio = (lp_new - step.logp_old).clamp(
            -cfg.log_ratio_clamp, cfg.log_ratio_clamp,
        )
        ratio = torch.exp(log_ratio)
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
    for p in module.parameters():
        p.requires_grad_(True)

    optimizer = torch.optim.Adam(
        [p for p in module.parameters() if p.requires_grad], lr=lr,
    )

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
