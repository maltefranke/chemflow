"""Representation mode for chemflow training runs.

Declares what subset of (x, a, c, edge_index, e) a run should train on:
- POINTCLOUD: x, a only (c and edges projected to dummies)
- CHARGED_POINTCLOUD: x, a, c (edges projected to dummies)
- GEOMETRIC_GRAPH: full chemistry (no projection)

A dataset's Capabilities declares what fields it provides. ``validate_representation``
checks the requested mode is achievable by the dataset.

The cache and vocab are **canonical** in all modes (one preprocessing per dataset).
The two projection helpers transform canonical artifacts into the model-facing view:
- ``project_molecule_to_representation``: canonical MoleculeData → model-facing.
- ``project_distributions_to_representation``: canonical priors → projected priors
  (used by sample_prior_graph / interpolator / integrator so sampled tokens match
  projected targets — see refactor doc §2).
"""

from dataclasses import dataclass, replace
from enum import Enum

import torch
import torch.nn.functional as F

from chemflow.dataset.molecule_data import MoleculeData
from chemflow.dataset.vocab import Distributions, Vocab


class Representation(str, Enum):
    POINTCLOUD = "pointcloud"
    CHARGED_POINTCLOUD = "charged_pointcloud"
    GEOMETRIC_GRAPH = "geometric_graph"

    @property
    def requires_charges(self) -> bool:
        return self != Representation.POINTCLOUD

    @property
    def requires_topology(self) -> bool:
        return self == Representation.GEOMETRIC_GRAPH


@dataclass(frozen=True)
class Capabilities:
    provides_charges: bool
    provides_topology: bool


def validate_representation(caps: Capabilities, mode: Representation) -> None:
    """Raise if the dataset cannot satisfy the requested representation."""
    if mode.requires_charges and not caps.provides_charges:
        raise ValueError(
            f"Representation {mode} requires charges but dataset does not provide them"
        )
    if mode.requires_topology and not caps.provides_topology:
        raise ValueError(
            f"Representation {mode} requires topology but dataset does not provide it"
        )


def build_fully_connected_edge_index(n: int, device=None) -> torch.Tensor:
    """Directed, no-self-loop, symmetric edge_index of shape (2, N*(N-1))."""
    if n <= 1:
        return torch.empty((2, 0), dtype=torch.long, device=device)
    rng = torch.arange(n, device=device)
    src, dst = torch.meshgrid(rng, rng, indexing="ij")
    mask = src != dst
    return torch.stack([src[mask], dst[mask]], dim=0)


def neutral_charge_index(vocab: Vocab) -> int:
    """Index of the neutral ``"0"`` charge token in the canonical vocab; falls back
    to literal 0 when ``"0"`` isn't a discovered token (e.g. TMQM without charges).
    The fallback is just a constant the embedding absorbs into bias.
    """
    try:
        return vocab.charge_tokens.index("0")
    except (ValueError, AttributeError):
        return 0


def project_molecule_to_representation(
    mol: MoleculeData, vocab: Vocab, mode: Representation
) -> MoleculeData:
    """Project a (possibly partial) canonical MoleculeData onto the run's representation.

    Mutates and returns ``mol`` so any extra attributes (e.g. ``y``) are preserved.

    - non-topology modes: ``edge_index = fully_connected(N)``, ``e = zeros(E)``
      (token 0 of canonical edge vocab is ``<NO_BOND>`` by construction).
    - non-charge modes: ``c = neutral_idx(N)`` (canonical index of ``"0"``).
    - geometric_graph: unchanged.
    """
    if mode == Representation.GEOMETRIC_GRAPH:
        return mol

    n = mol.x.shape[0]
    device = mol.x.device

    if not mode.requires_topology:
        mol.edge_index = build_fully_connected_edge_index(n, device=device)
        mol.e = torch.zeros(mol.edge_index.shape[1], dtype=torch.long, device=device)

    if not mode.requires_charges:
        mol.c = torch.full((n,), neutral_charge_index(vocab), dtype=torch.long, device=device)

    return mol


def project_distributions_to_representation(
    distributions: Distributions, vocab: Vocab, mode: Representation
) -> Distributions:
    """Project canonical (or uniform) distributions onto the run's representation.

    Mirrors ``project_molecule_to_representation`` for the priors that
    ``sample_prior_graph``, the interpolator's internal Categoricals, and the
    integrator's insertion sampler draw from. Pins ``edge_type_distribution`` to
    one-hot at the canonical ``<NO_BOND>`` token (index 0) in non-topology modes,
    and ``charge_type_distribution`` to one-hot at the neutral charge index in
    non-charge modes. Other fields pass through untouched.
    """
    if mode == Representation.GEOMETRIC_GRAPH:
        return distributions

    new = replace(distributions)
    if not mode.requires_topology:
        new.edge_type_distribution = F.one_hot(
            torch.tensor(0), len(vocab.edge_tokens)
        ).float()
    if not mode.requires_charges:
        new.charge_type_distribution = F.one_hot(
            torch.tensor(neutral_charge_index(vocab)), len(vocab.charge_tokens)
        ).float()
    return new
