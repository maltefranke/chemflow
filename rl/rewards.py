"""GRPO reward functions.

Contract
--------
A reward function has signature
    `(module, trajectory) -> (Tensor(B,), dict[str, float])`
where the tensor is the per-graph reward used by GRPO and the dict is
diagnostics merged into the training logs (e.g. `p_valid`, `qed_mean_valid`).
Add a new reward by writing a function and registering it in `REWARDS`.

All built-in rewards gate on RDKit validity: invalid molecules receive 0, so
validity pressure is preserved regardless of the property being maximized.
"""

from __future__ import annotations

from typing import Callable, Iterator, Optional

import torch
from rdkit import Chem


# ─────────────────────────────────────────────────────────────────────────────
# Shared RDKit conversion / validity loop (one place so rewards can't drift)
# ─────────────────────────────────────────────────────────────────────────────


def _iter_valid_mols(
    module, trajectory,
) -> Iterator[tuple[Optional[Chem.Mol], bool]]:
    """Yield (rdkit_mol_or_None, is_valid) for each graph in `mol_final`."""
    from chemflow.utils import rdkit as chemflowRD  # noqa: N812

    v = module.vocab
    for mol_i in trajectory.mol_final.to_data_list():
        rd = mol_i.to_rdkit_mol(v.atom_tokens, v.edge_tokens, v.charge_tokens)
        if rd is None:
            yield None, False
            continue
        try:
            ok = chemflowRD.mol_is_valid(rd)
        except Exception:
            ok = False
        yield rd, ok


def _as_tensor(vals: list[float], device) -> torch.Tensor:
    return torch.tensor(vals, device=device, dtype=torch.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Built-in rewards
# ─────────────────────────────────────────────────────────────────────────────


def validity_reward(module, trajectory) -> tuple[torch.Tensor, dict[str, float]]:
    """Binary RDKit validity per graph."""
    device = trajectory.mol_final.x.device
    vals = [1.0 if ok else 0.0 for _, ok in _iter_valid_mols(module, trajectory)]
    r = _as_tensor(vals, device)
    return r, {"p_valid": float(r.mean())}


def qed_reward(module, trajectory) -> tuple[torch.Tensor, dict[str, float]]:
    """QED drug-likeness in [0, 1] if valid, else 0.

    RDKit's QED is Bickerton et al. (2012): a weighted aggregate of MW, logP,
    HBA, HBD, PSA, rotatable bonds, aromatic rings, and alert count.  Bounded,
    mildly prefers medium-sized drug-like molecules, so ins/del gains show up
    as a size-distribution shift.

    Diagnostics: `p_valid` (fraction of valid mols) and `qed_mean_valid` (mean
    QED restricted to valid mols, 0 if none).  `reward_mean == p_valid * qed_mean_valid`.
    """
    from rdkit.Chem import QED

    device = trajectory.mol_final.x.device
    qed_vals: list[float] = []
    valid_mask: list[bool] = []
    for rd, ok in _iter_valid_mols(module, trajectory):
        valid_mask.append(ok)
        if not ok:
            qed_vals.append(0.0)
            continue
        try:
            qed_vals.append(float(QED.qed(rd)))
        except Exception:
            qed_vals.append(0.0)
    r = _as_tensor(qed_vals, device)
    n_valid = sum(valid_mask)
    qed_sum_valid = sum(q for q, v in zip(qed_vals, valid_mask) if v)
    return r, {
        "p_valid": n_valid / max(len(valid_mask), 1),
        "qed_mean_valid": (qed_sum_valid / n_valid) if n_valid > 0 else 0.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Registry (add new rewards here)
# ─────────────────────────────────────────────────────────────────────────────


REWARDS: dict[str, Callable] = {
    "validity": validity_reward,
    "qed": qed_reward,
}
