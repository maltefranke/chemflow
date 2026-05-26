"""Simple non-reference GRPO rewards."""

from __future__ import annotations

from typing import Callable

import torch

from chemflow.dataset.representation import Representation

from .common import _as_tensor, _iter_valid_mols
from .spec import RewardSpec, WrapperSpec


def validity_reward(module, trajectory) -> tuple[torch.Tensor, dict[str, float]]:
    """Binary RDKit validity per graph."""
    device = trajectory.mol_final.x.device
    vals = [1.0 if ok else 0.0 for _, ok in _iter_valid_mols(module, trajectory)]
    r = _as_tensor(vals, device)
    return r, {"p_valid": float(r.mean())}


def n_atoms_reward(module, trajectory) -> tuple[torch.Tensor, dict[str, float]]:
    """Per-graph atom count. Higher = better. Pure geometry, no RDKit.

    Uses ``mol.num_graphs`` (set by the integrator / batching layer) as the
    canonical batch size, not ``mol.batch.max() + 1``. The max+1 form silently
    truncates the reward vector when the trailing graphs lose all their atoms
    to deletion, which then misaligns with GRPO's sampled groups.
    """
    mol = trajectory.mol_final
    num_graphs = int(getattr(mol, "num_graphs", None) or mol.batch_size)
    counts = torch.bincount(mol.batch, minlength=num_graphs).float()
    return counts, {
        "n_atoms_mean": float(counts.mean()),
        "n_atoms_max": float(counts.max()) if counts.numel() else 0.0,
    }


def validity_gate_wrapper(base_reward_fn: Callable) -> Callable:
    """Multiply base reward by 0/1 RDKit validity per graph.

    Adds ``p_valid``, ``{base}_mean_valid``, ``{base}_max_valid`` diagnostics
    so users can see the gated and ungated reward signal side-by-side.
    """

    def wrapped(module, trajectory) -> tuple[torch.Tensor, dict[str, float]]:
        r, diag = base_reward_fn(module, trajectory)
        diag_out = dict(diag)

        device = r.device
        valid_mask = _as_tensor(
            [1.0 if ok else 0.0 for _, ok in _iter_valid_mols(module, trajectory)],
            device,
        )
        r_gated = r * valid_mask

        n_valid = int(valid_mask.sum().item())
        valid_vals = r_gated[valid_mask.bool()]
        diag_out["p_valid"] = float(valid_mask.mean())
        diag_out["reward_mean_pre_validity"] = float(r.mean())
        diag_out["reward_mean_post_validity"] = float(r_gated.mean())
        if n_valid > 0:
            diag_out["reward_mean_valid"] = float(valid_vals.mean())
            diag_out["reward_max_valid"] = float(valid_vals.max())
        return r_gated, diag_out

    return wrapped


# Binary validity is itself an RDKit measurement, so MOLECULE-only.
VALIDITY_SPEC = RewardSpec(
    fn=validity_reward,
    supported_representations=frozenset({Representation.MOLECULE}),
)
# Atom counting needs only positions+batch — works in every representation.
N_ATOMS_SPEC = RewardSpec(
    fn=n_atoms_reward,
    supported_representations=frozenset(Representation),
)
# Validity gating requires RDKit-parseable molecules, so MOLECULE-only.
VALIDITY_GATE_SPEC = WrapperSpec(
    make=validity_gate_wrapper,
    supported_representations=frozenset({Representation.MOLECULE}),
)
