"""RDKit-mol metrics + the helpers they need (process pool, SEMLA valence
tables, PoseBusters). Everything in this file pulls RDKit; the tensor /
distribution side stays RDKit-free.
"""

import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor

import faulthandler
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from torchmetrics import Metric

from posebusters import PoseBusters

from chemflow.utils import rdkit_utils as chemflowRD


faulthandler.enable()
# Silence RDKit warnings (in addition to chemflow.utils.rdkit_utils) because
# this module may be imported first from some code paths.
RDLogger.DisableLog("rdApp.*")


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class GenerativeMetric(Metric):
    """torchmetrics base for metrics consuming a list of RDKit mols."""

    def update(self, mols: list[Chem.rdchem.Mol]) -> None:
        raise NotImplementedError

    def compute(self) -> torch.Tensor:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Shared process pool for CPU-heavy RDKit work (optimise_mol / calc_energy).
#
# Why this exists:
#   * MMFF optimisation + energy calc is Python/C serial per-molecule.
#   * Running it on the main thread inside validation_step starves the
#     TorchElastic rendezvous keep-alive thread and triggers
#     RendezvousTimeoutError on long val cycles.
#   * We use a "spawn" context so the children do not inherit the parent's
#     CUDA context (fork + CUDA = undefined behavior).
#   * The pool is lazy so simply importing this module does not pay the
#     spawn cost; it starts on the first validation step.
# ---------------------------------------------------------------------------

_RD_POOL: ProcessPoolExecutor | None = None
_RD_POOL_MAX_WORKERS = int(os.environ.get("CHEMFLOW_RD_POOL_WORKERS", "4"))


def _get_rdkit_pool() -> ProcessPoolExecutor:
    global _RD_POOL
    if _RD_POOL is None:
        affinity = (
            len(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else os.cpu_count() or 1
        )
        # Leave at least half the cores for DataLoader workers + rendezvous.
        n = max(1, min(_RD_POOL_MAX_WORKERS, affinity // 2))
        _RD_POOL = ProcessPoolExecutor(
            max_workers=n,
            mp_context=mp.get_context("spawn"),
        )
    return _RD_POOL


def _pool_map(fn, items: list) -> list:
    """Run ``fn`` over ``items`` in the shared RDKit pool. Falls back to a
    plain list comprehension if the pool cannot be constructed (e.g. spawn is
    unavailable) so metrics never break training.
    """
    if not items:
        return []
    try:
        pool = _get_rdkit_pool()
        n_workers = pool._max_workers  # type: ignore[attr-defined]
        chunk = max(1, len(items) // (n_workers * 4))
        return list(pool.map(fn, items, chunksize=chunk))
    except Exception:
        return [fn(x) for x in items]


def _rd_optimise_mol(mol):
    return chemflowRD.optimise_mol(mol)


def _rd_calc_energy(mol):
    return chemflowRD.calc_energy(mol)


def _rd_calc_energy_per_atom(mol):
    return chemflowRD.calc_energy(mol, per_atom=True)


def _is_valid_float(num):
    return num not in [None, float("inf"), float("-inf"), float("nan")]


def _canonical_smiles(smi: str) -> str | None:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


# ---------------------------------------------------------------------------
# Stability helpers
# ---------------------------------------------------------------------------


def calc_atom_stabilities(mol):
    """Per-atom RDKit-sanity flags. ``True`` means *stable* (no problem
    detected), ``False`` means the atom is involved in a
    ``DetectChemistryProblems`` finding. Aligned so that downstream metrics
    named ``*_stability`` report a *higher = better* number.
    """
    stabilities = [True] * mol.GetNumAtoms()
    for p in Chem.DetectChemistryProblems(mol):
        if hasattr(p, "GetAtomIdx"):
            stabilities[p.GetAtomIdx()] = False
    return stabilities


SEMLA_ALLOWED_VALENCIES = {
    "H": {0: 1, 1: 0, -1: 0},
    "C": {0: [3, 4], 1: 3, -1: 3},
    "N": {0: [2, 3], 1: [2, 3, 4], -1: 2},  # In QM9, N+ seems to be present in the form NH+ and NH2+
    "O": {0: 2, 1: 3, -1: 1},
    "F": {0: 1, -1: 0},
    "B": 3,
    "Al": 3,
    "Si": 4,
    "P": {0: [3, 5], 1: 4},
    "S": {0: [2, 6], 1: [2, 3], 2: 4, 3: 5, -1: 3},
    "Cl": 1,
    "As": 3,
    "Br": {0: 1, 1: 2},
    "I": 1,
    "Hg": [1, 2],
    "Bi": [3, 5],
    "Se": [2, 4, 6],
}


def _SEMLA_is_valid_valence(valence, allowed, charge):
    if isinstance(allowed, int):
        return allowed == valence
    if isinstance(allowed, list):
        return valence in allowed
    if isinstance(allowed, dict):
        sub = allowed.get(charge)
        if sub is None:
            return False
        return _SEMLA_is_valid_valence(valence, sub, charge)
    return False


def SEMLA_mol_is_valid(
    mol: Chem.rdchem.Mol, with_hs: bool = True, connected: bool = True
) -> bool:
    """Whether the mol can be sanitised and, optionally, is fully connected."""
    if mol is None:
        return False
    mol_copy = Chem.Mol(mol)
    if not with_hs:
        mol_copy = Chem.RemoveAllHs(mol_copy)
    try:
        AllChem.SanitizeMol(mol_copy)
    except Exception:
        return False
    if connected and len(AllChem.GetMolFrags(mol_copy)) != 1:
        return False
    return True


def SEMLA_calc_atom_stabilities(mol):
    stabilities = []
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        if sym not in SEMLA_ALLOWED_VALENCIES:
            stabilities.append(False)
            continue
        stabilities.append(
            _SEMLA_is_valid_valence(
                atom.GetExplicitValence(),
                SEMLA_ALLOWED_VALENCIES[sym],
                atom.GetFormalCharge(),
            )
        )
    return stabilities


# ---------------------------------------------------------------------------
# PoseBusters
# ---------------------------------------------------------------------------


def calc_posebusters_metrics(
    rdkit_mols: list[Chem.rdchem.Mol | None],
) -> dict[str, float]:
    """Run PoseBusters molecular plausibility checks and return averaged pass
    rates. Uses the "mol" config (standalone, no protein context).
    """
    valid_mols = [mol for mol in rdkit_mols if mol is not None]
    if not valid_mols:
        return {}

    df = PoseBusters(config="mol", max_workers=-1).bust(valid_mols, None, None)

    results: dict[str, float] = {}
    for col in df.columns:
        if col in ("file", "molecule"):
            continue
        try:
            numeric = pd.to_numeric(df[col], errors="coerce")
            if not numeric.isna().all():
                results[col] = float(numeric.mean())
        except Exception:
            print(f"Warning: could not process PoseBusters column '{col}'.")
    return results


# ---------------------------------------------------------------------------
# RDKit-stability metrics (fed precomputed bools)
# ---------------------------------------------------------------------------


class AtomStability(Metric):
    """Fraction of atoms passing RDKit ``DetectChemistryProblems``. Fed
    precomputed per-atom bools by ``calc_atom_stabilities`` (True = stable)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("atom_stable", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, stabilities: list[list[bool]]) -> None:
        flat = [b for mol in stabilities for b in mol]
        self.atom_stable += sum(flat)
        self.total += len(flat)

    def compute(self) -> torch.Tensor:
        return self.atom_stable.float() / self.total


class MoleculeStability(Metric):
    """Fraction of molecules where every atom is stable."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("mol_stable", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, stabilities: list[list[bool]]) -> None:
        mol_stables = [len(mol) > 0 and all(mol) for mol in stabilities]
        self.mol_stable += sum(mol_stables)
        self.total += len(mol_stables)

    def compute(self) -> torch.Tensor:
        return self.mol_stable.float() / self.total


# ---------------------------------------------------------------------------
# SEMLA stability + validity (per-element valence table, RDKit-mol input)
# ---------------------------------------------------------------------------


class SEMLAValidity(GenerativeMetric):
    """Validity using SEMLA's sanitisation + connectedness rule (no charge / radical filter)."""

    def __init__(self, with_hs: bool = True, connected: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.with_hs = with_hs
        self.connected = connected
        self.add_state("valid", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, mols: list[Chem.rdchem.Mol]) -> None:
        is_valid = [
            SEMLA_mol_is_valid(mol, with_hs=self.with_hs, connected=self.connected)
            for mol in mols
        ]
        self.valid += sum(is_valid)
        self.total += len(mols)

    def compute(self) -> torch.Tensor:
        return self.valid.float() / self.total


class SEMLAAtomStability(GenerativeMetric):
    """Atom-stability using SEMLA's per-element valence table."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("atom_stable", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, mols: list[Chem.rdchem.Mol]) -> None:
        for mol in mols:
            if mol is None:
                continue
            try:
                stabilities = SEMLA_calc_atom_stabilities(mol)
            except Exception:
                continue
            self.atom_stable += sum(stabilities)
            self.total += len(stabilities)

    def compute(self) -> torch.Tensor:
        return self.atom_stable.float() / self.total


class SEMLAMoleculeStability(GenerativeMetric):
    """Fraction of mols where every atom passes SEMLA's valence check."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("mol_stable", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, mols: list[Chem.rdchem.Mol]) -> None:
        for mol in mols:
            if mol is None:
                self.total += 1
                continue
            try:
                stabilities = SEMLA_calc_atom_stabilities(mol)
            except Exception:
                self.total += 1
                continue
            if len(stabilities) > 0 and all(stabilities):
                self.mol_stable += 1
            self.total += 1

    def compute(self) -> torch.Tensor:
        return self.mol_stable.float() / self.total


# ---------------------------------------------------------------------------
# Validity / uniqueness / novelty
# ---------------------------------------------------------------------------


class Validity(GenerativeMetric):
    def __init__(self, allow_charged: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.allow_charged = allow_charged
        self.add_state("valid", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, mols: list[Chem.rdchem.Mol]) -> None:
        is_valid = [
            chemflowRD.mol_is_valid(mol, allow_charged=self.allow_charged)
            for mol in mols
            if mol is not None
        ]
        self.valid += sum(is_valid)
        self.total += len(mols)

    def compute(self) -> torch.Tensor:
        return self.valid.float() / self.total


# TODO I don't think this will work with DDP
class Uniqueness(GenerativeMetric):
    """Note: only tracks uniqueness of molecules which can be converted into SMILES"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.valid_smiles = []

    def reset(self):
        super().reset()
        self.valid_smiles = []

    def update(self, mols: list[Chem.rdchem.Mol]) -> None:
        smiles = [
            Chem.MolToSmiles(mol, canonical=True) for mol in mols if mol is not None
        ]
        self.valid_smiles.extend(smi for smi in smiles if smi is not None)

    def compute(self) -> torch.Tensor:
        return torch.tensor(len(set(self.valid_smiles))) / len(self.valid_smiles)


class Novelty(GenerativeMetric):
    def __init__(self, train_smiles: list[str], **kwargs):
        super().__init__(**kwargs)

        n_workers = min(8, len(os.sched_getaffinity(0)))
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            canonical = list(executor.map(_canonical_smiles, train_smiles))

        self.smiles = set(smi for smi in canonical if smi is not None)

        self.add_state("novel", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, mols: list[Chem.rdchem.Mol]) -> None:
        smiles = [
            chemflowRD.smiles_from_mol(mol, canonical=True)
            for mol in mols
            if mol is not None
        ]
        valid_smiles = [smi for smi in smiles if smi is not None]
        novel = [smi not in self.smiles for smi in valid_smiles]

        self.novel += sum(novel)
        self.total += len(novel)

    def compute(self) -> torch.Tensor:
        return self.novel.float() / self.total


# ---------------------------------------------------------------------------
# Energy / strain / RMSD
# ---------------------------------------------------------------------------


class EnergyValidity(GenerativeMetric):
    def __init__(self, optimise=False, **kwargs):
        super().__init__(**kwargs)
        self.optimise = optimise
        self.add_state("n_valid", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, mols: list[Chem.rdchem.Mol]) -> None:
        num_mols = len(mols)
        non_none = [m for m in mols if m is not None]
        if self.optimise:
            non_none = [m for m in _pool_map(_rd_optimise_mol, non_none) if m is not None]
        energies = _pool_map(_rd_calc_energy, non_none)
        valid_energies = [e for e in energies if _is_valid_float(e)]
        self.n_valid += len(valid_energies)
        self.total += num_mols

    def compute(self) -> torch.Tensor:
        return self.n_valid.float() / self.total


class AverageEnergy(GenerativeMetric):
    """Average energy for molecules for which energy can be calculated.

    Energy can't be calculated for invalid mols and pose optimisation isn't
    guaranteed to succeed; failures don't count towards the metric. Does not
    require sanitisation, but sanitising first is usually a good idea.
    """

    def __init__(self, optimise=False, per_atom=False, **kwargs):
        super().__init__(**kwargs)
        self.optimise = optimise
        self.per_atom = per_atom
        self.add_state("energy", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state(
            "n_valid_energies", default=torch.tensor(0), dist_reduce_fx="sum"
        )

    def update(self, mols: list[Chem.rdchem.Mol]) -> None:
        non_none = [m for m in mols if m is not None]
        if self.optimise:
            non_none = [m for m in _pool_map(_rd_optimise_mol, non_none) if m is not None]
        energy_fn = _rd_calc_energy_per_atom if self.per_atom else _rd_calc_energy
        energies = _pool_map(energy_fn, non_none)
        valid_energies = [e for e in energies if _is_valid_float(e)]
        self.energy += sum(valid_energies)
        self.n_valid_energies += len(valid_energies)

    def compute(self) -> torch.Tensor:
        return self.energy / self.n_valid_energies


# TODO: Add xTB as level of theory option and add forces as a metric


class AverageStrainEnergy(GenerativeMetric):
    """Energy difference between a molecule's pose and its optimised pose.

    Only counted when the molecule has an energy, pose optimisation succeeds,
    and the optimised-pose energy is calculable. Combine with
    ``EnergyValidity(optimise=True)`` to track the fraction of molecules this
    metric can be calculated on.
    """

    def __init__(self, per_atom=False, **kwargs):
        super().__init__(**kwargs)
        self.per_atom = per_atom
        self.add_state(
            "total_energy_diff", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("n_valid", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, mols: list[Chem.rdchem.Mol]) -> None:
        indexed = [(idx, m) for idx, m in enumerate(mols) if m is not None]
        if not indexed:
            return

        indices, non_none = zip(*indexed)
        optimised = _pool_map(_rd_optimise_mol, list(non_none))

        energy_fn = _rd_calc_energy_per_atom if self.per_atom else _rd_calc_energy

        opt_pairs = [(i, m) for i, m in zip(indices, optimised) if m is not None]
        if not opt_pairs:
            return
        opt_indices, opt_mols_list = zip(*opt_pairs)
        opt_energies = _pool_map(energy_fn, list(opt_mols_list))

        pairs = [(i, e) for i, e in zip(opt_indices, opt_energies) if e is not None]
        if not pairs:
            return

        valid_indices, valid_energies = zip(*pairs)
        original_energies = _pool_map(energy_fn, [mols[i] for i in valid_indices])

        energy_diffs = [
            orig - opt for orig, opt in zip(original_energies, valid_energies)
        ]
        self.total_energy_diff += sum(energy_diffs)
        self.n_valid += len(energy_diffs)

    def compute(self) -> torch.Tensor:
        return self.total_energy_diff / self.n_valid


class AverageOptRmsd(GenerativeMetric):
    """Average RMSD between a molecule and its optimised pose.

    Only counted when the molecule is valid and pose optimisation succeeds.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("total_rmsd", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_valid", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, mols: list[Chem.rdchem.Mol]) -> None:
        indexed = [(idx, m) for idx, m in enumerate(mols) if m is not None]
        if not indexed:
            return

        indices, non_none = zip(*indexed)
        optimised = _pool_map(_rd_optimise_mol, list(non_none))

        pairs = [(i, m) for i, m in zip(indices, optimised) if m is not None]
        if not pairs:
            return

        valid_indices, opt_mols = zip(*pairs)
        original_mols = [mols[i] for i in valid_indices]
        rmsds = [
            chemflowRD.conf_distance(m1, m2)
            for m1, m2 in zip(original_mols, opt_mols)
        ]
        self.total_rmsd += sum(rmsds)
        self.n_valid += len(rmsds)

    def compute(self) -> torch.Tensor:
        return self.total_rmsd / self.n_valid
