"""Organic XYZ validity reward.

Infers bonds from 3D coordinates and atom types with RDKit's ``DetermineBonds``
and applies the same core criteria as graph ``validity``: sanitizable, one
fragment, and total formal charge zero.

This is deliberately QM9-focused for now: total charge is fixed to zero and no
charged-state search is attempted. It can also score neutral organic GEOM-style
pointclouds, but it is not a transition-metal-complex validity check.
"""

from __future__ import annotations

from typing import Callable, Iterator

import torch
from rdkit import Chem
from rdkit.Chem.rdDetermineBonds import DetermineBonds

from chemflow.dataset.representation import Representation

from .common import _as_tensor
from .spec import RewardSpec, WrapperSpec

# Conservative organic element set. This covers QM9 and neutral organic GEOM
# molecules, while causing metal-containing vocabularies to fail fast.
_ORGANIC_Z = frozenset({1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53})
_PT = Chem.GetPeriodicTable()
_DEFAULT_COV_FACTOR = 1.3
_DEFAULT_MAX_ITERATIONS = 200
_Z_LOOKUP_CACHE: dict[tuple[str, ...], list[int]] = {}


def _coord_scale(module) -> float:
    coord_std = getattr(module.distributions, "coordinate_std", None)
    if coord_std is None:
        return 1.0
    return float(coord_std.item() if hasattr(coord_std, "item") else coord_std)


def _organic_z_lookup(atom_tokens: list[str]) -> list[int]:
    out: list[int] = []
    invalid: list[str] = []
    for token in atom_tokens:
        z = int(_PT.GetAtomicNumber(token))
        if z not in _ORGANIC_Z:
            invalid.append(token)
        out.append(z)
    if invalid:
        bad = ", ".join(sorted(set(invalid)))
        raise ValueError(
            "organic_xyz_validity is only for neutral organic XYZ pointclouds; "
            f"vocab contains unsupported atom token(s): {bad}"
        )
    return out


def _cached_organic_z_lookup(atom_tokens: list[str]) -> list[int]:
    key = tuple(atom_tokens)
    lookup = _Z_LOOKUP_CACHE.get(key)
    if lookup is None:
        lookup = _organic_z_lookup(atom_tokens)
        _Z_LOOKUP_CACHE[key] = lookup
    return lookup


def _xyz_is_organic_valid(
    z: list[int],
    pos: list[list[float]],
    *,
    cov_factor: float,
    max_iterations: int,
) -> bool:
    """True if bonds infer and sanitize into one neutral organic molecule."""
    if not z:
        return False

    rw = Chem.RWMol()
    for zi in z:
        rw.AddAtom(Chem.Atom(int(zi)))
    mol = rw.GetMol()

    conf = Chem.Conformer(len(z))
    for i, xyz in enumerate(pos):
        conf.SetAtomPosition(i, (float(xyz[0]), float(xyz[1]), float(xyz[2])))
    mol.AddConformer(conf, assignId=True)

    try:
        DetermineBonds(
            mol,
            charge=0,
            covFactor=float(cov_factor),
            allowChargedFragments=True,
            useHueckel=False,
            useVdw=False,
            embedChiral=False,
            maxIterations=int(max_iterations),
        )
    except Exception:
        return False

    if Chem.SanitizeMol(mol, catchErrors=True) != Chem.SanitizeFlags.SANITIZE_NONE:
        return False
    if len(Chem.GetMolFrags(mol)) > 1:
        return False
    if Chem.GetFormalCharge(mol) != 0:
        return False
    return True


def _iter_organic_xyz_valid(
    module,
    trajectory,
    *,
    cov_factor: float,
    max_iterations: int,
) -> Iterator[bool]:
    """Yield neutral organic XYZ validity per graph in ``mol_final``."""
    z_lookup = _cached_organic_z_lookup(module.vocab.atom_tokens)
    scale = _coord_scale(module)

    for mol_i in trajectory.mol_final.to_data_list():
        a = mol_i.a
        if a.ndim > 1:
            a = a.argmax(dim=-1)
        z = [z_lookup[int(i)] for i in a]
        pos = (mol_i.x.detach().cpu() * scale).tolist()
        yield _xyz_is_organic_valid(
            z,
            pos,
            cov_factor=cov_factor,
            max_iterations=max_iterations,
        )


def organic_xyz_validity_reward(
    module,
    trajectory,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Binary neutral organic validity per graph, inferred from XYZ."""
    device = trajectory.mol_final.x.device
    vals = [
        1.0 if ok else 0.0
        for ok in _iter_organic_xyz_valid(
            module,
            trajectory,
            cov_factor=_DEFAULT_COV_FACTOR,
            max_iterations=_DEFAULT_MAX_ITERATIONS,
        )
    ]
    r = _as_tensor(vals, device)
    return r, {"p_organic_xyz_valid": float(r.mean())}


def organic_xyz_validity_gate_wrapper(
    base_reward_fn: Callable,
    *,
    cov_factor: float = _DEFAULT_COV_FACTOR,
    max_iterations: int = _DEFAULT_MAX_ITERATIONS,
) -> Callable:
    """Gate a reward by neutral organic validity inferred from XYZ."""

    def wrapped(module, trajectory) -> tuple[torch.Tensor, dict[str, float]]:
        r, diag = base_reward_fn(module, trajectory)
        diag_out = dict(diag)

        mask = _as_tensor(
            [
                1.0 if ok else 0.0
                for ok in _iter_organic_xyz_valid(
                    module,
                    trajectory,
                    cov_factor=cov_factor,
                    max_iterations=max_iterations,
                )
            ],
            r.device,
        )
        r_gated = r * mask

        n_valid = int(mask.sum().item())
        diag_out["p_organic_xyz_valid"] = float(mask.mean())
        diag_out["reward_mean_pre_organic_xyz_validity"] = float(r.mean())
        diag_out["reward_mean_post_organic_xyz_validity"] = float(r_gated.mean())
        if n_valid > 0:
            valid_vals = r_gated[mask.bool()]
            diag_out["reward_mean_organic_xyz_valid"] = float(valid_vals.mean())
            diag_out["reward_max_organic_xyz_valid"] = float(valid_vals.max())
        return r_gated, diag_out

    return wrapped


_ALL_REPR = frozenset(Representation)
ORGANIC_XYZ_VALIDITY_SPEC = RewardSpec(
    fn=organic_xyz_validity_reward,
    supported_representations=_ALL_REPR,
)
ORGANIC_XYZ_VALIDITY_GATE_SPEC = WrapperSpec(
    make=organic_xyz_validity_gate_wrapper,
    supported_representations=_ALL_REPR,
)
