"""Pure functions for computing conditioning targets from molecules.

These run on CPU/GPU tensors and are independent of any CFG plumbing —
they're called from `signals.ConditioningSignal.extract()` and from the
dataset (Phase 5 caching).
"""

from __future__ import annotations

import torch
from rdkit import Chem
from rdkit.Chem import Crippen, GetPeriodicTable
from rdkit.Chem.QED import qed as _rdkit_qed

_PERIODIC_TABLE = GetPeriodicTable()


def compute_molecular_weight(
    atom_indices: torch.Tensor,
    atom_tokens: list[str],
    batch: torch.Tensor | None = None,
    num_graphs: int | None = None,
) -> torch.Tensor:
    """Per-graph molecular weight (Daltons) via RDKit's periodic table."""
    weights = torch.tensor(
        [_PERIODIC_TABLE.GetAtomicWeight(tok) for tok in atom_tokens],
        dtype=torch.float,
        device=atom_indices.device,
    )
    per_atom = weights[atom_indices.long()]
    if batch is not None and num_graphs is not None:
        mw = torch.zeros(num_graphs, device=atom_indices.device)
        mw.scatter_add_(0, batch, per_atom)
        return mw
    return per_atom.sum().unsqueeze(0)


def compute_logp(
    smiles: list[str] | str,
    device: torch.device | None = None,
) -> torch.Tensor:
    """RDKit Crippen MolLogP per SMILES; unparsable → 0.0 (null prior)."""
    if isinstance(smiles, str):
        smiles = [smiles]
    out: list[float] = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s) if s else None
        out.append(float(Crippen.MolLogP(mol)) if mol is not None else 0.0)
    return torch.tensor(out, dtype=torch.float, device=device)


def compute_qed(
    smiles: list[str] | str,
    device: torch.device | None = None,
) -> torch.Tensor:
    """RDKit QED per SMILES; unparsable → 0.0 (lower bound of [0, 1])."""
    if isinstance(smiles, str):
        smiles = [smiles]
    out: list[float] = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s) if s else None
        try:
            out.append(float(_rdkit_qed(mol)) if mol is not None else 0.0)
        except Exception:
            out.append(0.0)
    return torch.tensor(out, dtype=torch.float, device=device)
