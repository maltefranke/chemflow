"""GRPO reward functions and registry.

A reward function has signature
    ``(module, trajectory) -> (Tensor(B,), dict[str, float])``
where the tensor is the per-graph reward used by GRPO and the dict is merged
into training diagnostics.

Each reward is registered as a :class:`RewardSpec` declaring the dataset
representations it can run against. Wrappers (e.g. scaffold diversity) are
registered as :class:`WrapperSpec`. ``build_reward`` intersects the base
reward's representation set with every active wrapper's set before training
starts.
"""

from __future__ import annotations

from typing import Callable

from chemflow.dataset.representation import Representation

from .atom_number import (
    N_ATOMS_SPEC,
    VALIDITY_GATE_SPEC,
    VALIDITY_SPEC,
    n_atoms_reward,
    validity_gate_wrapper,
    validity_reward,
)
from .common import _as_tensor, _iter_valid_mols
from .diversity import (
    SCAFFOLD_DIVERSITY_SPEC,
    ScaffoldBucketMemory,
    scaffold_diversity_wrapper,
)
from .organic_xyz_validity import (
    ORGANIC_XYZ_VALIDITY_GATE_SPEC,
    ORGANIC_XYZ_VALIDITY_SPEC,
    organic_xyz_validity_gate_wrapper,
    organic_xyz_validity_reward,
)
from .tmqm_xtb import (
    TMQM_GAP_SPEC,
    TMQM_VALIDITY_GATE_SPEC,
    TMQM_VALIDITY_SPEC,
    tmqm_gap_reward,
    tmqm_validity_gate_wrapper,
    tmqm_validity_reward,
)
from .prilocaine_topology import (
    _SHAPE_REF_SMILES,
    TANIMOTO_SPEC,
    TOPOLOGY_MOTIF_SPEC,
    TOPOLOGY_SPEC,
    _get_tanimoto_ref,
    _heavy_atom_mol,
    _score_topology_motif_single,
    _score_topology_single,
    tanimoto_reward,
    topology_motif_reward,
    topology_reward,
)
from .spec import RewardSpec, WrapperSpec

REWARDS: dict[str, RewardSpec] = {
    "validity": VALIDITY_SPEC,
    "organic_xyz_validity": ORGANIC_XYZ_VALIDITY_SPEC,
    "tmqm_validity": TMQM_VALIDITY_SPEC,
    "tmqm_gap": TMQM_GAP_SPEC,
    "n_atoms": N_ATOMS_SPEC,
    "tanimoto": TANIMOTO_SPEC,
    "topology": TOPOLOGY_SPEC,
    "topology_motif": TOPOLOGY_MOTIF_SPEC,
}

WRAPPERS: dict[str, WrapperSpec] = {
    "scaffold_diversity": SCAFFOLD_DIVERSITY_SPEC,
    "validity_gate": VALIDITY_GATE_SPEC,
    "organic_xyz_validity_gate": ORGANIC_XYZ_VALIDITY_GATE_SPEC,
    "tmqm_validity_gate": TMQM_VALIDITY_GATE_SPEC,
}


def build_reward(
    name: str,
    *,
    representation: Representation,
    wrappers: list[tuple[str, dict]] | None = None,
) -> Callable:
    """Resolve a reward (+ optional wrappers) and validate representation compat.

    ``wrappers`` is an ordered list of ``(wrapper_name, kwargs)`` pairs. The
    base reward's supported set is intersected with every wrapper's supported
    set; ``representation`` must lie in the intersection.
    """
    if name not in REWARDS:
        raise ValueError(
            f"Unknown RL reward {name!r}. Available rewards: {sorted(REWARDS)}"
        )
    spec = REWARDS[name]
    wrappers = wrappers or []

    effective = spec.supported_representations
    for w_name, _ in wrappers:
        if w_name not in WRAPPERS:
            raise ValueError(
                f"Unknown reward wrapper {w_name!r}. Available: {sorted(WRAPPERS)}"
            )
        effective = effective & WRAPPERS[w_name].supported_representations

    if representation not in effective:
        active = [w for w, _ in wrappers] or ["(none)"]
        raise ValueError(
            f"reward={name!r} with wrappers={active} supports "
            f"representations={sorted(r.value for r in effective)}, "
            f"but RL is running with representation={representation.value!r}."
        )

    fn = spec.fn
    for w_name, kwargs in wrappers:
        fn = WRAPPERS[w_name].make(fn, **kwargs)
    return fn


__all__ = [
    "REWARDS",
    "RewardSpec",
    "ScaffoldBucketMemory",
    "WRAPPERS",
    "WrapperSpec",
    "_SHAPE_REF_SMILES",
    "_as_tensor",
    "_get_tanimoto_ref",
    "_heavy_atom_mol",
    "_iter_valid_mols",
    "_score_topology_motif_single",
    "_score_topology_single",
    "build_reward",
    "n_atoms_reward",
    "organic_xyz_validity_gate_wrapper",
    "organic_xyz_validity_reward",
    "scaffold_diversity_wrapper",
    "validity_gate_wrapper",
    "tanimoto_reward",
    "tmqm_gap_reward",
    "tmqm_validity_gate_wrapper",
    "tmqm_validity_reward",
    "topology_motif_reward",
    "topology_reward",
    "validity_reward",
]
