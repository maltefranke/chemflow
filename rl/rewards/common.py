"""Shared helpers for GRPO reward functions."""

from __future__ import annotations

from typing import Iterator, Optional

import torch
from rdkit import Chem, RDLogger

try:
    Chem.WrapLogs()
except AttributeError:
    pass
RDLogger.DisableLog("rdApp.*")


def _iter_valid_mols(
    module, trajectory,
) -> Iterator[tuple[Optional[Chem.Mol], bool]]:
    """Yield (rdkit_mol_or_None, is_valid) for each graph in `mol_final`."""
    from chemflow.utils import rdkit_utils as chemflowRD  # noqa: N812

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
