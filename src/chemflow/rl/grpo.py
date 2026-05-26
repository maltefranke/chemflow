"""GRPO on the Morph architecture."""

from __future__ import annotations

import copy
import math
import os
from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Optional

import torch
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

# Import rewards before any chemflow module that imports RDKit so WrapLogs runs first.
from chemflow.rl.rewards import validity_reward  # noqa: F401 (re-exported for callers)

from chemflow.dataset.molecule_data import (
    MoleculeBatch,
    PointCloud,
    filter_nodes,
    join_molecules_with_atoms,
    join_molecules_with_predicted_edges,
)
from chemflow.dataset.representation import neutral_charge_index
from chemflow.utils.utils import EDGE_ALIGNER


EPS = 1e-8
EPS_T = 1e-2
DEFAULT_CLIP_EPS = 0.2
DEFAULT_LOG_RATIO_CLAMP = 20.0
DEFAULT_P_INS_CLAMP = 1.0 - 1e-3


# ─────────────────────────────────────────────────────────────────────────────
# Config + rollout storage
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class GRPOConfig:
    sigma_explore: float = 0.05    # per-coordinate std for N(x_t + v_θ dt, σ² I) position kernel
    clip_eps: float = DEFAULT_CLIP_EPS
    eps_t: float = EPS_T
    log_ratio_clamp: float = DEFAULT_LOG_RATIO_CLAMP  # pre-exp bound: ratio / k3, avoid exp Inf
    p_ins_clamp: float = DEFAULT_P_INS_CLAMP       # cap p_ins below 1 (insert Bernoulli floor headroom)
    num_integration_steps: int | None = None       # None -> module's default
    max_grad_norm: float | None = 1.0              # clip_grad_norm_ threshold; None disables
    # GRPO group size: number of rollouts sharing the same prompt. G=1 keeps
    # the legacy batch-relative baseline; G>1 replicates each prompt G times
    # in the batch and normalises advantages within each group (DeepSeek-GRPO
    # style).  Effective unique-prompt count per update = batch_size // G.
    group_size: int = 1
    # β · (sum of per-channel k3) added to the step loss, k3 = exp(t) - t - 1
    # with t = lp_ref - lp_θ (clamped). 0 disables: no ref model, no extra forward.
    kl_coef: float = 0.0
    # If True, the `pos` channel is skipped in the k3 sum (GRPO still uses full
    # joint log-prob for the clipped surrogate; only the KL anchor omits positions).
    kl_omit_pos: bool = False
    # PPO-style number of optimization passes over one sampled trajectory.
    # 1 = legacy behavior (single pass). >1 reuses rollout data via clipped
    # importance ratios exp(log π_θ(a) - log π_old(a)).
    update_passes: int = 1
    # If True, each channel's log-prob is the mean over its natural units,
    # then channels are summed into `total` (Flow-GRPO-style resolution scaling).
    # Details: positions use mean over **3 × surviving atoms** (one factor per
    # Cartesian coordinate); charge/node/ins_gate/edge/GMM/ins edges use the
    # counts documented in `_per_channel_logprob`.  Default False keeps sums
    # (exact factorised joint within each channel).
    #
    # KL: `_k3_kl_per_channel` uses the same per-channel breakdown.  With
    # per-element means, typical |lp_ref − lp_θ| per channel shrinks by about
    # the number of averaged units (e.g. position ~3 N_surv atoms).  Increase
    # `kl_coef` when you rely on the ref anchor (often an order of magnitude
    # or more — tune on validation).
    per_element_logp_mean: bool = False
    # CFG signal overrides to feed guided_predict. Built at rollout time from a
    # batch's (mol_t, mol_1) pair and stashed here so every step + log-prob call
    # shares the exact same conditioning.
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


def _to_per_element_mean(per_graph_sum: torch.Tensor, denom: torch.Tensor) -> torch.Tensor:
    """`per_graph_sum / max(denom,1)` — safe when denom==0 (numerator is zero there)."""
    return per_graph_sum / denom.clamp(min=1.0)


@torch.no_grad()
def _rl_diagnostics(module, mol_final) -> dict[str, float]:
    """Run the module's batch + distribution metric collections on the final batch.

    Coords are stored normalized during the rollout; pretraining's metrics expect
    Angstroms, so scale a clone before updating. Reset before and after to keep
    metric state local to this call.
    """
    out: dict[str, float] = {}
    mol = mol_final.clone()
    coord_std = getattr(module.distributions, "coordinate_std", None)
    if coord_std is not None:
        mol.x = mol.x * coord_std

    bm = getattr(module, "batch_metrics", None)
    if bm is not None and len(bm) > 0:
        bm.reset()
        bm.update(mol)
        out.update({f"diag/batch/{k}": float(v.detach().cpu()) for k, v in bm.compute().items()})
        bm.reset()

    dm = getattr(module, "distribution_metrics", None)
    if dm is not None and len(dm) > 0:
        dm.reset()
        dm.update(mol)
        out.update({f"diag/{k}": float(v.detach().cpu()) for k, v in dm.compute().items()})
        dm.reset()
    return out


def _sync_model_ema_to_live(module) -> None:
    """Copy live `model` weights into `model_ema` before checkpointing.

    Prevents a stale EMA copy from overwriting the trained policy on reload
    (see ``load_ckpt_into_module`` — it copies EMA -> model when the loaded
    module has ``use_ema_for_eval=True``).
    """
    if getattr(module, "model_ema", None) is not None:
        module.model_ema.load_state_dict(module.model.state_dict())


def _gather_logp(probs: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """log probs[..., idx] with a numerical floor."""
    return torch.log(probs.clamp(min=EPS)).gather(-1, idx.view(-1, 1)).squeeze(-1)


def _q_to_p(q: torch.Tensor, rate_node: torch.Tensor, dt: float) -> torch.Tensor:
    """Canonical raw-sigmoid -> per-step probability : 
    sigmoid(q) * rate * dt; clamp interior (0,1) for logs."""
    return (q * rate_node * dt).clamp(EPS, 1.0 - EPS)


def _extract_triu(
    edge_index: torch.Tensor, attrs: list[torch.Tensor],
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    infos = EDGE_ALIGNER.align_edges(source_group=(edge_index, attrs))
    return infos["edge_index"], list(infos["edge_attr"])


# ─────────────────────────────────────────────────────────────────────────────
# Position kernel (fixed-Gaussian exploration)
# ─────────────────────────────────────────────────────────────────────────────


def position_explore_mean(
    x_t: torch.Tensor,      # (N, 3)
    x1_pred: torch.Tensor,  # (N, 3)
    t_node: torch.Tensor,   # (N,)
    dt: float,
    eps_t: float = EPS_T,
) -> torch.Tensor:
    """Mean of π(x_{t+dt}|x_t) = N(x_t + v_θ dt, σ² I); v_θ matches FM velocity."""
    t = t_node.clamp(min=eps_t, max=1.0 - 1e-6).unsqueeze(-1)
    one_minus_t = 1.0 - t
    v_theta = (x1_pred - x_t) / one_minus_t
    return x_t + v_theta * dt


def gaussian_logprob_positions(
    x_next: torch.Tensor,    # (N, 3)
    mu: torch.Tensor,        # (N, 3)
    var: float,              # per-coordinate variance (σ²); isotropic on R^3
    batch_id: torch.Tensor,  # (N,)
    num_graphs: int,
    survive_mask: torch.Tensor | None = None,  # (N,) bool
) -> torch.Tensor:
    """Per-graph sum of Gaussian log-densities, restricted to `survive_mask`."""
    sq = ((x_next - mu) ** 2).sum(-1)
    log_2pi_var = math.log(2 * math.pi * var)
    logp_node = -0.5 * sq / var - 1.5 * log_2pi_var
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
    """log pi^{node} per-graph: 3-way noop/sub/del + atom cat (factorised).

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
    """Sum of per-node Bernoulli log-probs."""
    log_pos = torch.log(p_ins.clamp(min=EPS))
    log_neg = torch.log((1.0 - p_ins).clamp(min=EPS))
    logp = torch.where(do_ins, log_pos, log_neg)
    return _scatter_sum(logp, batch_id, num_graphs)


def gmm_marginal_logprob(
    gmm_dict: dict,              # keys: mu (n,K,D), sigma (n,K), pi (n,K), a_probs (n,K,|A|), c_probs (n,K,|C|)
    ins_x: torch.Tensor,         # (n, D)
    ins_a: torch.Tensor,         # (n,)
    ins_c: torch.Tensor,         # (n,)
    include_c: bool = True,
) -> torch.Tensor:
    """log pi^GMM per inserted atom; logsumexp over K components."""
    if ins_x.numel() == 0:
        return ins_x.new_zeros(0)

    mu = gmm_dict["mu"]                                # (n, K, D)
    sigma = gmm_dict["sigma"]                          # (n, K)
    pi = gmm_dict["pi"]                                # (n, K)
    a_probs = gmm_dict["a_probs"]                      # (n, K, |A|)

    # Spatial: Normal with isotropic sigma.  Sum log-density over D, broadcast (n,1,D) - (n,K,D).
    loc = mu
    scale = sigma.unsqueeze(-1).expand_as(loc).clamp(min=EPS)
    log_N = Normal(loc, scale).log_prob(ins_x.unsqueeze(1)).sum(-1)  # (n, K)

    # Types: gather at ins_a / ins_c, then log.
    log_pa = torch.log(a_probs.clamp(min=EPS)).gather(-1, ins_a.view(-1, 1, 1).expand(-1, a_probs.size(1), 1)).squeeze(-1)

    log_pi = torch.log(pi.clamp(min=EPS))
    logits = log_pi + log_N + log_pa                        # (n, K)
    if include_c:
        c_probs = gmm_dict["c_probs"]                      # (n, K, |C|)
        log_pc = torch.log(c_probs.clamp(min=EPS)).gather(-1, ins_c.view(-1, 1, 1).expand(-1, c_probs.size(1), 1)).squeeze(-1)
        logits = logits + log_pc
    return torch.logsumexp(logits, dim=-1)                  # (n,)


def ins_edge_marginal_logprob(
    edge_probs: torch.Tensor,    # (E, |E|) pi_edge
    prior_probs: torch.Tensor,   # (|E|,) prior
    kappa_t: torch.Tensor,       # (E, 1) in [0, 1]
    hat_e: torch.Tensor,         # (E,)
) -> torch.Tensor:
    """
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


def _ensure_max_seqlen(mol: MoleculeBatch) -> MoleculeBatch:
    mol.max_seqlen = int(torch.bincount(mol.batch).max().item())
    return mol


def _forward_preds(module, mol_t, t: torch.Tensor, cfg: GRPOConfig):
    """Single forward pass that returns the full preds dict (with h_latent)."""
    mol_t = _ensure_max_seqlen(mol_t)
    model = module._get_model()
    model.set_inference()
    return module.cfg_guidance.guided_predict(model, mol_t, t, None, cfg.cfg_inputs)


def _compute_p_ins(
    num_ins_pred: torch.Tensor,   # (N,) raw rate-head output
    ins_rate_node: torch.Tensor,  # (N,) schedule rate * dt multiplier
    dt: float,
    p_ins_clamp: float,
) -> torch.Tensor:
    """EditFlow : expected_ins * ins_rate * dt, clamped below 1 - eps."""
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
    *,
    detach_breakdown: bool = True,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
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
    requires_charges = module.representation.requires_charges
    requires_topology = module.representation.requires_topology

    # ─── Positions ──────────────────────────────────────────────────────
    x1_pred = preds["pos_head"]
    t_node = t[batch_id]
    mu = position_explore_mean(mol_t.x, x1_pred, t_node, dt, cfg.eps_t)
    var = cfg.sigma_explore**2
    lp_pos = gaussian_logprob_positions(
        step.x_next, mu, var, batch_id, num_graphs,
        survive_mask=survive_mask,
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
    if requires_topology:
        q_sub_e_full = torch.sigmoid(preds["do_sub_e_head"].view(-1))
        edge_probs_full = F.softmax(preds["edge_type_head"], dim=-1)
        triu_idx, (q_sub_e_triu, edge_probs_triu) = _extract_triu(
            mol_t.edge_index, [q_sub_e_full, edge_probs_full],
        )
        # triu order: rollout vs logp must share EDGE_ALIGNER permutation
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
    else:
        lp_edge = torch.zeros(num_graphs, device=device)

    # ─── Charge (surviving existing nodes only) ─────────────────────────
    if requires_charges:
        charge_probs = F.softmax(preds["charge_head"], dim=-1)
        lp_charge = charge_logprob(
            charge_probs, step.hat_c, survive_mask, batch_id, num_graphs,
        )
    else:
        lp_charge = torch.zeros(num_graphs, device=device)

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
            include_c=requires_charges,
        )                                                # (N_ins,)
        ins_batch_id = batch_id[step.do_ins]
        lp_gmm = _scatter_sum(gmm_per_ins, ins_batch_id, num_graphs)

    # ─── Ins -> existing edges ──────────────────────────────────────────
    lp_ins_edge_ext = torch.zeros(num_graphs, device=device)
    if requires_topology and step.do_ins.any() and step.ins_edge_spawn_orig_full.numel() > 0:
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
    if requires_topology and step.hat_e_ins_ii.numel() > 0:
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

    if cfg.per_element_logp_mean:
        dtype = lp_pos.dtype
        n_nodes = torch.bincount(batch_id, minlength=num_graphs).to(device=device, dtype=dtype)
        n_surv = _scatter_sum(survive_mask.float(), batch_id, num_graphs).to(dtype=dtype)
        # `gaussian_logprob_positions` is a sum of one 3D isotropic factor per atom
        # (−0.5‖x−μ‖²/var − 1.5 log(2πvar)); match Flow-GRPO's per-pixel mean by
        # normalising with **3 coordinates per surviving atom**.
        lp_pos = _to_per_element_mean(lp_pos, n_surv * 3.0)
        lp_node = _to_per_element_mean(lp_node, n_nodes)
        lp_charge = _to_per_element_mean(lp_charge, n_surv)
        lp_ins_gate = _to_per_element_mean(lp_ins_gate, n_nodes)
        if requires_topology:
            assert edge_survive_mask.shape == edge_batch_id.shape, (
                "edge_survive_mask and edge_batch_id must both be (E_triu,)"
            )
            edge_denom = _scatter_sum(edge_survive_mask.float(), edge_batch_id, num_graphs).to(dtype=dtype)
            lp_edge = _to_per_element_mean(lp_edge, edge_denom)
        if step.do_ins.any():
            ins_batch_id = batch_id[step.do_ins]
            n_ins = torch.bincount(ins_batch_id, minlength=num_graphs).to(device=device, dtype=dtype)
            lp_gmm = _to_per_element_mean(lp_gmm, n_ins)
        den_ins_ext = torch.zeros(num_graphs, device=device, dtype=dtype)
        if step.do_ins.any() and step.ins_edge_spawn_orig_full.numel() > 0:
            den_ins_ext = _scatter_sum(
                step.ins_edge_valid_mask.float(),
                batch_id[step.ins_edge_spawn_orig_full],
                num_graphs,
            ).to(dtype=dtype)
        lp_ins_edge_ext = _to_per_element_mean(lp_ins_edge_ext, den_ins_ext)
        den_ii = torch.zeros(num_graphs, device=device, dtype=dtype)
        if step.hat_e_ins_ii.numel() > 0:
            den_ii = torch.bincount(
                batch_id[step.ins_ii_spawn_src_orig],
                minlength=num_graphs,
            ).to(dtype=dtype)
        lp_ins_edge_ii = _to_per_element_mean(lp_ins_edge_ii, den_ii)

    total = lp_pos + lp_node + lp_edge + lp_charge + lp_ins_gate + lp_gmm + lp_ins_edge_ext + lp_ins_edge_ii
    per_ch = {
        "pos": lp_pos, "node": lp_node, "edge": lp_edge, "charge": lp_charge,
        "ins_gate": lp_ins_gate, "gmm": lp_gmm, "ins_e_ext": lp_ins_edge_ext,
        "ins_e_ii": lp_ins_edge_ii,
    }
    if detach_breakdown:
        per_ch = {k: v.detach() for k, v in per_ch.items()}
    return total, per_ch


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
    """Sample one full position-explore + discrete + insertion step from theta_old."""
    integrator = module.integrator
    preds = _forward_preds(module, mol_t, t, cfg)

    batch_id = mol_t.batch
    num_graphs = t.shape[0]
    device = mol_t.x.device
    N_orig = mol_t.x.size(0)
    n_atoms_strategy = getattr(module, "n_atoms_strategy", "fixed")
    prior_edge_probs = integrator._cat_edge.probs.to(device)
    requires_charges = module.representation.requires_charges
    requires_topology = module.representation.requires_topology
    neutral_c = None if requires_charges else neutral_charge_index(module.vocab)

    # ─── Positions (fixed-Gaussian exploration) ──────────────────────────
    x1_pred = preds["pos_head"]
    t_node = t[batch_id]
    mu = position_explore_mean(mol_t.x, x1_pred, t_node, dt, cfg.eps_t)
    noise_pos = torch.randn_like(mol_t.x)
    x_next = mu + cfg.sigma_explore * noise_pos

    # ─── Categorical predictions (shared across rollout + scoring) ─────
    atom_probs = F.softmax(preds["atom_type_head"], dim=-1)
    charge_probs = F.softmax(preds["charge_head"], dim=-1)
    edge_probs_full = F.softmax(preds["edge_type_head"], dim=-1)
    hat_a = Categorical(probs=atom_probs).sample()
    hat_c = Categorical(probs=charge_probs).sample()
    hat_e_full = Categorical(probs=edge_probs_full).sample()
    # Mirror main sample(): force dummies for channels the representation doesn't carry.
    if not requires_charges:
        hat_c.fill_(neutral_c)
    if not requires_topology:
        hat_e_full.fill_(0)  # <NO_BOND> at canonical edge-vocab index 0

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
    if not requires_topology:
        q_sub_e_full = torch.zeros_like(q_sub_e_full)  # never substitute dummy edges
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
        if not requires_charges:
            ins_c = torch.full_like(ins_c, neutral_c)

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
    ins_edge_head = (
        getattr(module._get_model(), "ins_edge_head", None) if requires_topology else None
    )
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
            # ins edges: same (1−κ)·prior + κ·π mix as integrator
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
    lp_total_old, _ = _per_channel_logprob(preds, step, cfg, module)
    step.logp_old = lp_total_old.detach()
    return mol_t_next, step


def _replicate_batch_for_groups(batch, group_size: int):
    """Replicate each graph in a `(mol_t, mol_1)` batch `group_size` times.

    Layout is prompt-major: output indices `i*G .. i*G + G-1` are independent
    clones of the i-th input graph.  This lets downstream code reshape a
    per-graph reward tensor as `(K, G)` to compute group-relative advantages
    directly, without any permutation.

    If `batch_size` is not a multiple of `group_size`, the batch is trimmed to
    the largest divisible prefix (so K = batch_size // G).  G=1 is a no-op.
    """
    if group_size == 1:
        return batch
    mol_t, mol_1 = batch
    batch_size = mol_t.num_graphs
    K = batch_size // group_size
    if K == 0:
        raise ValueError(
            f"batch_size ({batch_size}) must be >= group_size ({group_size})"
        )
    t_list = mol_t.to_data_list()[:K]
    o_list = mol_1.to_data_list()[:K]
    t_reps, o_reps = [], []
    for t_d, o_d in zip(t_list, o_list):
        for _ in range(group_size):
            t_reps.append(t_d.clone())
            o_reps.append(o_d.clone())
    return type(mol_t).from_data_list(t_reps), type(mol_1).from_data_list(o_reps)


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

    ctx = module.cfg_guidance.build_ctx()
    cfg.cfg_inputs = module.cfg_guidance.build_overrides(mol_1, ctx)

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


def _step_logprob_with_grad(
    module, step: StepData, cfg: GRPOConfig, *, detach_breakdown: bool = True,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Returns `(total_lp, channel_breakdown)`.

    By default, breakdown values are `.detach()`'d (cheap diagnostics, no big graph).
    Set `detach_breakdown=False` for KL, which needs the same per-channel
    `log p_θ` in the autograd path.
    """
    preds = _forward_preds(module, step.mol_t, step.t, cfg)
    return _per_channel_logprob(
        preds, step, cfg, module, detach_breakdown=detach_breakdown,
    )


def _k3_kl_per_channel(
    breakdown_theta: dict[str, torch.Tensor],
    breakdown_ref: dict[str, torch.Tensor],
    cfg: GRPOConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Schulman k3 = exp(t) - t - 1 with t = log π_ref - log π_θ, per channel and graph.
    Sum over channels (factorised per-step policy).  Clamp `t` before `exp` (overflow).
    """
    kl_by_ch: dict[str, torch.Tensor] = {}
    for name, lp_t in breakdown_theta.items():
        lp_r = breakdown_ref.get(name)
        if lp_r is None:
            continue
        if cfg.kl_omit_pos and name == "pos":
            continue
        t = (lp_r - lp_t).clamp(
            -cfg.log_ratio_clamp, cfg.log_ratio_clamp,
        )
        kl_by_ch[name] = torch.exp(t) - t - 1.0
    if not kl_by_ch:
        z = next(iter(breakdown_theta.values()))
        return torch.zeros_like(z), {}
    return sum(kl_by_ch.values()), kl_by_ch


def grpo_step(
    module,
    batch,
    cfg: GRPOConfig,
    optimizer: torch.optim.Optimizer,
    reward_fn: Callable = validity_reward,
    ref_module: Optional[torch.nn.Module] = None,
    do_diagnostics: bool = False,
) -> dict:
    """One GRPO update: rollout -> reward -> per-step clipped loss -> step.

    Advantage computation
    ---------------------
    With `cfg.group_size == 1` (legacy behaviour), advantages are computed
    batch-relative: `adv_i = (r_i - mean(r)) / std(r)`.

    With `cfg.group_size == G > 1` (DeepSeek-style GRPO), the batch is
    pre-replicated so each prompt appears G times consecutively.  Advantages
    are then normalised *within* each group of G:
        `adv_{k,g} = (r_{k,g} - mean_g r_{k,g}) / std_g r_{k,g}`.
    This cancels between-prompt reward heterogeneity from the advantage,
    leaving only the contribution of the stochastic rollout choices.

    Note on diagnostics:
        * With `cfg.update_passes == 1` (legacy), `ratio ≡ 1` within this call
          because `lp_new` and `lp_old` are both evaluated under theta_old, so
          ratio metrics carry little information.
        * With `cfg.update_passes > 1`, later passes use the same `lp_old` but
          updated θ, so ratio drift reflects policy movement.
        * `signal/{channel}` = mean over steps of `mean_batch(adv * lp_channel)`.
          This is the policy-gradient "learning signal" per channel: its sign
          and magnitude tell us whether that channel's log-prob correlates
          with reward across the batch.
        * `grad_norm` is the L2 norm of the accumulated gradient before
          `optimizer.step()`.  Non-zero => the policy is actually moving.
        * `reward_within_std` / `reward_between_std` (group mode only) show
          how much of the reward variance is attributable to action noise
          (within) vs prompt heterogeneity (between).
        * When `cfg.kl_coef > 0` and `ref_module` is set, each step adds
          `β · mean(sum_c k3_c)` to the loss (Schulman k3 on per-channel
          log-prob ratios).  Logged as `kl/total` and `kl/{channel}`.
    """
    if cfg.update_passes < 1:
        raise ValueError(f"cfg.update_passes must be >= 1, got {cfg.update_passes}")

    batch = _replicate_batch_for_groups(batch, cfg.group_size)
    trajectory = rollout_trajectory(module, batch, cfg)
    reward_tensor, reward_aux = reward_fn(module, trajectory)
    trajectory.reward = reward_tensor

    r = trajectory.reward
    G = cfg.group_size
    group_aux: dict[str, float] = {}
    if G > 1:
        assert r.numel() % G == 0, (
            f"reward size {r.numel()} not divisible by group_size {G}; "
            "did `_replicate_batch_for_groups` run?"
        )
        r_groups = r.view(-1, G)
        mean = r_groups.mean(dim=1, keepdim=True)
        std = r_groups.std(dim=1, keepdim=True)
        adv = ((r_groups - mean) / (std + 1e-6)).reshape(-1)
        group_aux["reward_within_std"] = float(r_groups.std(dim=1).mean())
        group_aux["reward_between_std"] = float(r_groups.mean(dim=1).std())
    else:
        adv = (r - r.mean()) / (r.std() + 1e-6)

    n_passes = cfg.update_passes
    loss_ppo_sum = 0.0
    loss_kl_sum = 0.0
    n_steps = len(trajectory.steps)
    use_kl = cfg.kl_coef > 0.0 and ref_module is not None

    # Per-channel accumulators (averaged over steps and passes at the end).
    lp_mean: dict[str, float] = {}
    adv_signal: dict[str, float] = {}
    kl_mean: dict[str, float] = {}
    grad_norm_pre_sum = 0.0
    grad_norm_post_sum = 0.0

    ref_breakdowns: list[dict[str, torch.Tensor]] | None = None
    if use_kl:
        ref_breakdowns = []
        with torch.inference_mode():
            for step in trajectory.steps:
                _, breakdown_ref = _step_logprob_with_grad(
                    ref_module, step, cfg, detach_breakdown=True,
                )
                ref_breakdowns.append(breakdown_ref)

    for _ in range(n_passes):
        optimizer.zero_grad(set_to_none=True)
        for step_idx, step in enumerate(trajectory.steps):
            lp_new, breakdown = _step_logprob_with_grad(
                module, step, cfg, detach_breakdown=not use_kl,
            )
            # clamp Δlogp before exp (importance weight / grad stability)
            log_ratio = (lp_new - step.logp_old).clamp(
                -cfg.log_ratio_clamp, cfg.log_ratio_clamp,
            )
            ratio = torch.exp(log_ratio)
            clipped = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps)
            ppo_term = -torch.minimum(ratio * adv, clipped * adv).mean()
            step_loss = ppo_term

            if use_kl:
                breakdown_ref = ref_breakdowns[step_idx]
                kl_total, kl_by_ch = _k3_kl_per_channel(
                    breakdown, breakdown_ref, cfg,
                )
                step_loss = step_loss + cfg.kl_coef * kl_total.mean()
                loss_kl_sum += float((cfg.kl_coef * kl_total.detach().mean()).item())
                for name, kch in kl_by_ch.items():
                    kl_mean[name] = kl_mean.get(name, 0.0) + float(kch.detach().mean()) / (n_steps * n_passes)

            (step_loss / n_steps).backward()
            loss_ppo_sum += float(ppo_term.item())

            for name, lp_ch in breakdown.items():
                lp_mean[name] = lp_mean.get(name, 0.0) + float(lp_ch.detach().mean()) / (n_steps * n_passes)
                adv_signal[name] = adv_signal.get(name, 0.0) + float(
                    (adv * lp_ch.detach()).mean(),
                ) / (n_steps * n_passes)

        # Gradient norm BEFORE clipping: the raw magnitude of one optimizer update.
        # With multi-pass updates, we average these norms across passes.
        if cfg.max_grad_norm is not None:
            grad_norm_pre = float(
                torch.nn.utils.clip_grad_norm_(
                    [p for p in module.parameters() if p.requires_grad],
                    max_norm=cfg.max_grad_norm,
                )
            )
            grad_norm_post = min(grad_norm_pre, float(cfg.max_grad_norm))
        else:
            grad_sq = 0.0
            for p in module.parameters():
                if p.grad is not None:
                    grad_sq += float(p.grad.pow(2).sum())
            grad_norm_pre = grad_sq ** 0.5
            grad_norm_post = grad_norm_pre

        grad_norm_pre_sum += grad_norm_pre
        grad_norm_post_sum += grad_norm_post
        optimizer.step()

    loss_total = (loss_ppo_sum + loss_kl_sum) / (n_steps * n_passes)
    out: dict = {
        "loss": loss_total,
        "loss_ppo": loss_ppo_sum / (n_steps * n_passes),
        "grad_norm": grad_norm_pre_sum / n_passes,
        "grad_norm_post_clip": grad_norm_post_sum / n_passes,
        "update_passes": n_passes,
        "reward_mean": float(r.mean()),
        "reward_std": float(r.std()),
        "reward_min": float(r.min()),
        "reward_max": float(r.max()),
        "adv_abs_mean": float(adv.abs().mean()),
        **group_aux,
        **{f"lp/{k}": v for k, v in lp_mean.items()},
        **{f"signal/{k}": v for k, v in adv_signal.items()},
        **reward_aux,
    }
    if use_kl:
        out["loss_kl"] = loss_kl_sum / (n_steps * n_passes)
        # Unscaled k3: same as (step loss KL term) / β, i.e. mean_batch(sum_c k3_c) averaged over time.
        out["kl/total"] = (loss_kl_sum / (n_steps * n_passes)) / cfg.kl_coef
        out.update({f"kl/{k}": v for k, v in kl_mean.items()})
    if do_diagnostics:
        out.update(_rl_diagnostics(module, trajectory.mol_final))
    return out


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
    best_save_path: str | None = None,
    best_ema_beta: float = 0.9,
    best_warmup_steps: int = 3,
    diagnostics_every: int = 0,
    ) -> None:
    """GRPO training loop.

    When `cfg.kl_coef > 0`, a **frozen** deepcopy of the policy at `train()`
    start (post-checkpoint) is the ref in `grpo_step` (k3 on per-channel logp).

    Best-checkpoint tracking
    ------------------------
    If `best_save_path` is set, we track an EMA of `reward_mean` with
    coefficient `best_ema_beta` (default 0.9, ~10-step effective window) and
    save the module state_dict whenever the EMA hits a new maximum, after a
    short warmup of `best_warmup_steps` steps so the very first noisy update
    doesn't win by default.  All built-in rewards already gate on RDKit
    validity (invalid -> 0), so `reward_mean` is a validity-aware signal --
    no extra `p_valid` multiplier needed.
    """
    module = module.to(device)
    for p in module.parameters():
        p.requires_grad_(True)
    # GRPO trains the live `model`. Keep `model_ema` frozen so it doesn't enter
    # the optimizer's param list (no Adam state allocated) and stays inert if
    # something later puts it back in the forward graph.
    if getattr(module, "model_ema", None) is not None:
        for p in module.model_ema.parameters():
            p.requires_grad_(False)

    ref_module: torch.nn.Module | None = None
    if cfg.kl_coef > 0.0:
        ref_module = copy.deepcopy(module)
        ref_module.eval()
        for p in ref_module.parameters():
            p.requires_grad_(False)
        ref_module = ref_module.to(device)

    optimizer = torch.optim.Adam(
        [p for p in module.parameters() if p.requires_grad], lr=lr,
    )

    wandb_run = None
    try:
        import wandb  # noqa: F401
        wandb_run = wandb.run
    except ImportError:
        pass

    if best_save_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(best_save_path)) or ".", exist_ok=True)
    reward_ema: float | None = None
    best_ema: float = -float("inf")

    it = iter(dataloader)
    for step in range(n_updates):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(dataloader)
            batch = next(it)

        batch = _batch_to_device(batch, device)
        do_diag = diagnostics_every > 0 and step % diagnostics_every == 0
        info = grpo_step(
            module, batch, cfg, optimizer, reward_fn=reward_fn, ref_module=ref_module,
            do_diagnostics=do_diag,
        )

        r_mean = info["reward_mean"]
        if reward_ema is None:
            reward_ema = r_mean
        else:
            reward_ema = best_ema_beta * reward_ema + (1.0 - best_ema_beta) * r_mean
        info["reward_ema"] = reward_ema

        if best_save_path is not None and step >= best_warmup_steps and reward_ema > best_ema:
            best_ema = reward_ema
            rep = getattr(module, "representation", None)
            hp: dict = {"rl_checkpoint": True, "use_ema_for_eval": False}
            if rep is not None:
                hp["representation"] = rep.value
            _sync_model_ema_to_live(module)
            torch.save(
                {
                    "state_dict": module.state_dict(),
                    "hyper_parameters": hp,
                    "step": step,
                    "reward_ema": reward_ema,
                    "reward_mean": r_mean,
                },
                best_save_path,
            )
            info["best_ema"] = best_ema
            info["best_saved_at_step"] = step
            print(f"[grpo] best: step={step:04d} reward_ema={reward_ema:.4f} -> {best_save_path}", flush=True)

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
