"""Simple non-reference GRPO rewards."""

from __future__ import annotations

import torch

from .common import _as_tensor, _iter_valid_mols


def validity_reward(module, trajectory) -> tuple[torch.Tensor, dict[str, float]]:
    """Binary RDKit validity per graph."""
    device = trajectory.mol_final.x.device
    vals = [1.0 if ok else 0.0 for _, ok in _iter_valid_mols(module, trajectory)]
    r = _as_tensor(vals, device)
    return r, {"p_valid": float(r.mean())}


def n_atoms_reward(module, trajectory) -> tuple[torch.Tensor, dict[str, float]]:
    """Total atom count if valid, else 0. Higher = better."""
    device = trajectory.mol_final.x.device
    counts_valid_gated: list[float] = []
    counts_all: list[float] = []
    for rd, ok in _iter_valid_mols(module, trajectory):
        n_atoms = float(rd.GetNumAtoms()) if rd is not None else 0.0
        counts_all.append(n_atoms)
        counts_valid_gated.append(n_atoms if ok else 0.0)
    r = _as_tensor(counts_valid_gated, device)
    n_valid = sum(1 for c in counts_valid_gated if c > 0)
    sum_valid = sum(c for c in counts_valid_gated if c > 0)
    sum_all = sum(counts_all)
    return r, {
        "p_valid": n_valid / max(len(counts_valid_gated), 1),
        "n_atoms_mean_valid": (sum_valid / n_valid) if n_valid > 0 else 0.0,
        "n_atoms_max_valid": max(counts_valid_gated) if counts_valid_gated else 0.0,
        "n_atoms_mean_all": (sum_all / len(counts_all)) if counts_all else 0.0,
        "n_atoms_max_all": max(counts_all) if counts_all else 0.0,
    }
