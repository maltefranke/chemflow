"""GRPO reward functions and registry.

A reward function has signature
    ``(module, trajectory) -> (Tensor(B,), dict[str, float])``
where the tensor is the per-graph reward used by GRPO and the dict is merged
into training diagnostics.
"""

from __future__ import annotations

from typing import Callable

from .common import _as_tensor, _iter_valid_mols
from .diversity import ScaffoldBucketMemory, scaffold_diversity_wrapper
from .prilocaine_topology import (
    _SHAPE_REF_SMILES,
    _get_tanimoto_ref,
    _heavy_atom_mol,
    _score_topology_motif_single,
    _score_topology_single,
    tanimoto_reward,
    topology_motif_reward,
    topology_reward,
)
from .atom_number import n_atoms_reward, validity_reward

REWARDS: dict[str, Callable] = {
    "validity": validity_reward,
    "n_atoms": n_atoms_reward,
    "tanimoto": tanimoto_reward,
    "topology": topology_reward,
    "topology_motif": topology_motif_reward,
}

__all__ = [
    "REWARDS",
    "ScaffoldBucketMemory",
    "_SHAPE_REF_SMILES",
    "_as_tensor",
    "_get_tanimoto_ref",
    "_heavy_atom_mol",
    "_iter_valid_mols",
    "_score_topology_motif_single",
    "_score_topology_single",
    "n_atoms_reward",
    "scaffold_diversity_wrapper",
    "tanimoto_reward",
    "topology_motif_reward",
    "topology_reward",
    "validity_reward",
]
