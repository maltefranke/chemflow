"""Code adapted from SemlaFlow"""

import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import torch
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from torchmetrics import Metric
from torchmetrics import MetricCollection

from chemflow.utils import rdkit_utils as chemflowRD
from chemflow.utils.utils import token_to_index

from posebusters import PoseBusters

import faulthandler

faulthandler.enable()

# Silence RDKit warnings here too (in addition to chemflow.utils.rdkit_utils) because
# this module may be imported first from some code paths.
RDLogger.DisableLog("rdApp.*")


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
#   * The pool is lazy so simply importing metrics.py does not pay the
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
    """Run `fn` over `items` in the shared RDKit pool.

    Falls back to a plain list comprehension if the pool cannot be
    constructed (e.g. spawn is unavailable) so metrics never break
    training.
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


_BOND_TYPE_TO_EDGE_TOKEN = {
    Chem.BondType.SINGLE: "1",
    Chem.BondType.DOUBLE: "2",
    Chem.BondType.TRIPLE: "3",
    Chem.BondType.AROMATIC: "4",
}


def _rdkit_mols_atom_type_indices(
    mols: list,
    atom_tokens: list[str],
    device: torch.device,
) -> torch.Tensor:
    """Map each atom in each non-None RDKit mol to an atom token index; skip mols with unknown symbols."""
    vals: list[int] = []
    for mol in mols:
        if mol is None:
            continue
        try:
            for atom in mol.GetAtoms():
                vals.append(token_to_index(atom_tokens, atom.GetSymbol()))
        except ValueError:
            continue
    if not vals:
        return torch.tensor([], dtype=torch.long, device=device)
    return torch.tensor(vals, dtype=torch.long, device=device)


def _rdkit_mols_edge_type_indices(
    mols: list,
    edge_tokens: list[str],
    device: torch.device,
) -> torch.Tensor:
    """Upper-triangular pair types (incl. NO_BOND at index 0) aligned with training preprocessing."""
    vals: list[int] = []
    for mol in mols:
        if mol is None:
            continue
        n = mol.GetNumAtoms()
        try:
            for i in range(n):
                for j in range(i + 1, n):
                    bond = mol.GetBondBetweenAtoms(i, j)
                    if bond is None:
                        vals.append(0)
                    else:
                        bt = bond.GetBondType()
                        if bt not in _BOND_TYPE_TO_EDGE_TOKEN:
                            raise ValueError("unsupported bond type")
                        tok = _BOND_TYPE_TO_EDGE_TOKEN[bt]
                        vals.append(token_to_index(edge_tokens, tok))
        except ValueError:
            continue
    if not vals:
        return torch.tensor([], dtype=torch.long, device=device)
    return torch.tensor(vals, dtype=torch.long, device=device)


def _rdkit_mols_charge_type_indices(
    mols: list,
    charge_tokens: list[str],
    device: torch.device,
) -> torch.Tensor:
    """Map each atom's formal charge to a charge token index; skip mols with unknown charges."""
    vals: list[int] = []
    for mol in mols:
        if mol is None:
            continue
        try:
            for atom in mol.GetAtoms():
                vals.append(token_to_index(charge_tokens, str(atom.GetFormalCharge())))
        except ValueError:
            continue
    if not vals:
        return torch.tensor([], dtype=torch.long, device=device)
    return torch.tensor(vals, dtype=torch.long, device=device)


def _kl_divergence(gen: torch.Tensor, target: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.sum(
        gen * (torch.log(gen.clamp(min=eps)) - torch.log(target.clamp(min=eps)))
    )


def atom_count_distribution_metrics(
    mols: list,
    target_distribution: torch.Tensor,
    device: torch.device | str,
    eps: float = 1e-8,
) -> dict[str, torch.Tensor]:
    """Compare generated atom-count distribution to a target atom-count distribution (KL only)."""
    if len(mols) == 0:
        zero = torch.tensor(0.0, device=device)
        return {"atom_count_dist_kl": zero}

    n_atoms_gen = torch.tensor(
        [mol.num_nodes for mol in mols],
        device=device,
        dtype=torch.long,
    )
    max_gen = int(n_atoms_gen.max().item())

    target = target_distribution.to(device=device, dtype=torch.float32)
    min_len = max(max_gen + 1, target.numel())
    gen_hist = torch.bincount(n_atoms_gen, minlength=min_len)
    gen = gen_hist.to(dtype=torch.float32)
    gen = gen / gen.sum().clamp(min=1.0)

    if target.numel() < gen.numel():
        target = torch.cat(
            [target, torch.zeros(gen.numel() - target.numel(), device=device)],
            dim=0,
        )
    elif target.numel() > gen.numel():
        gen = torch.cat(
            [gen, torch.zeros(target.numel() - gen.numel(), device=device)],
            dim=0,
        )

    target = target / target.sum().clamp(min=eps)

    return {"atom_count_dist_kl": _kl_divergence(gen, target, eps)}


def calc_atom_stabilities(mol):
    problems = Chem.DetectChemistryProblems(mol)

    # Number of atoms involved in problems
    stabilities = [False] * mol.GetNumAtoms()

    for p in problems:
        if hasattr(p, "GetAtomIdx"):
            stabilities[p.GetAtomIdx()] = True

    return stabilities


def _is_valid_float(num):
    return num not in [None, float("inf"), float("-inf"), float("nan")]


### old semla functions
def SEMLA_mol_is_valid(mol: Chem.rdchem.Mol, with_hs: bool = True, connected: bool = True) -> bool:
    """Whether the mol can be sanitised and, optionally, whether it's fully connected

    Args:
        mol (Chem.Mol): RDKit molecule to check
        with_hs (bool): Whether to check validity including hydrogens (if they are in the input mol), default True
        connected (bool): Whether to also assert that the mol must not have disconnected atoms, default True

    Returns:
        bool: Whether the mol is valid
    """

    if mol is None:
        return False

    mol_copy = Chem.Mol(mol)
    if not with_hs:
        mol_copy = Chem.RemoveAllHs(mol_copy)

    try:
        AllChem.SanitizeMol(mol_copy)
    except Exception:
        return False

    n_frags = len(AllChem.GetMolFrags(mol_copy))
    if connected and n_frags != 1:
        return False

    return True

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
        valid = allowed == valence

    elif isinstance(allowed, list):
        valid = valence in allowed

    elif isinstance(allowed, dict):
        allowed = allowed.get(charge)
        if allowed is None:
            return False

        valid = _SEMLA_is_valid_valence(valence, allowed, charge)

    return valid

def SEMLA_calc_atom_stabilities(mol):
    stabilities = []

    for atom in mol.GetAtoms():
        atom_type = atom.GetSymbol()
        valence = atom.GetExplicitValence()
        charge = atom.GetFormalCharge()

        if atom_type not in SEMLA_ALLOWED_VALENCIES:
            stabilities.append(False)
            continue

        allowed = SEMLA_ALLOWED_VALENCIES[atom_type]
        atom_stable = _SEMLA_is_valid_valence(valence, allowed, charge)
        stabilities.append(atom_stable)

    return stabilities 

class GenerativeMetric(Metric):
    # TODO add metric attributes - see torchmetrics doc

    def __init__(self, **kwargs):
        # Pass extra kwargs (defined in Metric class) to parent
        super().__init__(**kwargs)

    def update(self, mols: list[Chem.rdchem.Mol]) -> None:
        raise NotImplementedError()

    def compute(self) -> torch.Tensor:
        raise NotImplementedError()


class PairMetric(Metric):
    def __init__(self, **kwargs):
        # Pass extra kwargs (defined in Metric class) to parent
        super().__init__(**kwargs)

    def update(
        self, predicted: list[Chem.rdchem.Mol], actual: list[Chem.rdchem.Mol]
    ) -> None:
        raise NotImplementedError()

    def compute(self) -> torch.Tensor:
        raise NotImplementedError()


class AtomStability(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.add_state("atom_stable", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, stabilities: list[list[bool]]) -> None:
        all_atom_stables = [
            atom_stable for atom_stbs in stabilities for atom_stable in atom_stbs
        ]
        self.atom_stable += sum(all_atom_stables)
        self.total += len(all_atom_stables)

    def compute(self) -> torch.Tensor:
        return self.atom_stable.float() / self.total


class MoleculeStability(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.add_state("mol_stable", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, stabilities: list[list[bool]]) -> None:
        mol_stables = [sum(atom_stbs) == len(atom_stbs) for atom_stbs in stabilities]
        self.mol_stable += sum(mol_stables)
        self.total += len(mol_stables)

    def compute(self) -> torch.Tensor:
        return self.mol_stable.float() / self.total


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
    """Atom-stability using SEMLA's per-element valence table. Takes mols directly."""

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
    """Fraction of mols where every atom passes SEMLA's valence check. Takes mols directly."""

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


class AtomCountDistributionMetric(GenerativeMetric):
    """KL(gen || target) for generated atom counts vs training atom-count distribution."""

    def __init__(
        self,
        target_distribution: torch.Tensor,
        eps: float = 1e-8,
        **kwargs,
    ):
        super().__init__(**kwargs)

        target = target_distribution.detach().to(dtype=torch.float32)
        target = target / target.sum().clamp(min=eps)

        self.eps = eps
        self.target_len = int(target.numel())

        self.register_buffer("target_distribution", target)
        self.add_state(
            "gen_hist",
            default=torch.zeros(self.target_len, dtype=torch.float32),
            dist_reduce_fx="sum",
        )
        self.add_state("n_total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, mols: list[Chem.rdchem.Mol]) -> None:
        counts = [mol.GetNumAtoms() for mol in mols if mol is not None]
        if len(counts) == 0:
            return

        counts_t = torch.tensor(
            counts,
            device=self.gen_hist.device,
            dtype=torch.long,
        )
        counts_t = counts_t.clamp(min=0, max=self.target_len - 1)

        hist = torch.bincount(counts_t, minlength=self.target_len).to(
            dtype=torch.float32
        )
        self.gen_hist += hist
        self.n_total += float(len(counts))

    def compute(self) -> torch.Tensor:
        if self.n_total <= 0:
            return torch.tensor(0.0, device=self.gen_hist.device)

        gen = self.gen_hist / self.gen_hist.sum().clamp(min=1.0)
        target = self.target_distribution
        target = target / target.sum().clamp(min=self.eps)

        return _kl_divergence(gen, target, self.eps)


class AtomTypeDistributionMetric(GenerativeMetric):
    """KL(gen || target) for atom-type histogram vs training distribution."""

    def __init__(
        self,
        target_distribution: torch.Tensor,
        atom_tokens: list[str],
        eps: float = 1e-8,
        **kwargs,
    ):
        super().__init__(**kwargs)

        target = target_distribution.detach().to(dtype=torch.float32)
        target = target / target.sum().clamp(min=eps)

        self.eps = eps
        self.target_len = int(target.numel())
        self.atom_tokens = list(atom_tokens)

        if self.target_len != len(self.atom_tokens):
            raise ValueError(
                "atom_type_distribution length must match len(atom_tokens)"
            )

        self.register_buffer("target_distribution", target)
        self.add_state(
            "gen_hist",
            default=torch.zeros(self.target_len, dtype=torch.float32),
            dist_reduce_fx="sum",
        )
        self.add_state("n_total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, mols: list[Chem.rdchem.Mol]) -> None:
        idx = _rdkit_mols_atom_type_indices(
            mols, self.atom_tokens, self.gen_hist.device
        )
        if idx.numel() == 0:
            return
        idx = idx.clamp(min=0, max=self.target_len - 1)
        hist = torch.bincount(idx, minlength=self.target_len).to(dtype=torch.float32)
        self.gen_hist += hist
        self.n_total += float(idx.numel())

    def compute(self) -> torch.Tensor:
        if self.n_total <= 0:
            return torch.tensor(0.0, device=self.gen_hist.device)

        gen = self.gen_hist / self.gen_hist.sum().clamp(min=1.0)
        target = self.target_distribution
        target = target / target.sum().clamp(min=self.eps)

        return _kl_divergence(gen, target, self.eps)


class EdgeTypeDistributionMetric(GenerativeMetric):
    """KL(gen || target) for edge-type pairs (upper triangle, incl. NO_BOND) vs training."""

    def __init__(
        self,
        target_distribution: torch.Tensor,
        edge_tokens: list[str],
        eps: float = 1e-8,
        **kwargs,
    ):
        super().__init__(**kwargs)

        target = target_distribution.detach().to(dtype=torch.float32)
        target = target / target.sum().clamp(min=eps)

        self.eps = eps
        self.target_len = int(target.numel())
        self.edge_tokens = list(edge_tokens)

        if self.target_len != len(self.edge_tokens):
            raise ValueError(
                "edge_type_distribution length must match len(edge_tokens)"
            )

        self.register_buffer("target_distribution", target)
        self.add_state(
            "gen_hist",
            default=torch.zeros(self.target_len, dtype=torch.float32),
            dist_reduce_fx="sum",
        )
        self.add_state("n_total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, mols: list[Chem.rdchem.Mol]) -> None:
        idx = _rdkit_mols_edge_type_indices(
            mols, self.edge_tokens, self.gen_hist.device
        )
        if idx.numel() == 0:
            return
        idx = idx.clamp(min=0, max=self.target_len - 1)
        hist = torch.bincount(idx, minlength=self.target_len).to(dtype=torch.float32)
        self.gen_hist += hist
        self.n_total += float(idx.numel())

    def compute(self) -> torch.Tensor:
        if self.n_total <= 0:
            return torch.tensor(0.0, device=self.gen_hist.device)

        gen = self.gen_hist / self.gen_hist.sum().clamp(min=1.0)
        target = self.target_distribution
        target = target / target.sum().clamp(min=self.eps)

        return _kl_divergence(gen, target, self.eps)


class ChargeTypeDistributionMetric(GenerativeMetric):
    """KL(gen || target) for atom formal-charge histogram vs training distribution."""

    def __init__(
        self,
        target_distribution: torch.Tensor,
        charge_tokens: list[str],
        eps: float = 1e-8,
        **kwargs,
    ):
        super().__init__(**kwargs)

        target = target_distribution.detach().to(dtype=torch.float32)
        target = target / target.sum().clamp(min=eps)

        self.eps = eps
        self.target_len = int(target.numel())
        self.charge_tokens = list(charge_tokens)

        if self.target_len != len(self.charge_tokens):
            raise ValueError(
                "charge_type_distribution length must match len(charge_tokens)"
            )

        self.register_buffer("target_distribution", target)
        self.add_state(
            "gen_hist",
            default=torch.zeros(self.target_len, dtype=torch.float32),
            dist_reduce_fx="sum",
        )
        self.add_state("n_total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, mols: list[Chem.rdchem.Mol]) -> None:
        idx = _rdkit_mols_charge_type_indices(
            mols, self.charge_tokens, self.gen_hist.device
        )
        if idx.numel() == 0:
            return
        idx = idx.clamp(min=0, max=self.target_len - 1)
        hist = torch.bincount(idx, minlength=self.target_len).to(dtype=torch.float32)
        self.gen_hist += hist
        self.n_total += float(idx.numel())

    def compute(self) -> torch.Tensor:
        if self.n_total <= 0:
            return torch.tensor(0.0, device=self.gen_hist.device)

        gen = self.gen_hist / self.gen_hist.sum().clamp(min=1.0)
        target = self.target_distribution
        target = target / target.sum().clamp(min=self.eps)

        return _kl_divergence(gen, target, self.eps)


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
        # Chain to the base Metric.reset() so that torchmetrics' internal
        super().reset()
        self.valid_smiles = []

    def update(self, mols: list[Chem.rdchem.Mol]) -> None:
        smiles = [
            Chem.MolToSmiles(mol, canonical=True) for mol in mols if mol is not None
        ]
        valid_smiles = [smi for smi in smiles if smi is not None]
        self.valid_smiles.extend(valid_smiles)

    def compute(self) -> torch.Tensor:
        num_unique = len(set(self.valid_smiles))
        uniqueness = torch.tensor(num_unique) / len(self.valid_smiles)
        return uniqueness


def _canonical_smiles(smi: str) -> str | None:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


class Novelty(GenerativeMetric):
    def __init__(self, train_smiles: list[str], **kwargs):
        super().__init__(**kwargs)

        n_workers = min(8, len(os.sched_getaffinity(0)))
        executor = ProcessPoolExecutor(max_workers=n_workers)

        futures = [executor.submit(_canonical_smiles, smi) for smi in train_smiles]
        canonical = [future.result() for future in futures]

        executor.shutdown()

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
            non_none = [
                m for m in _pool_map(_rd_optimise_mol, non_none) if m is not None
            ]

        energies = _pool_map(_rd_calc_energy, non_none)
        valid_energies = [e for e in energies if _is_valid_float(e)]

        self.n_valid += len(valid_energies)
        self.total += num_mols

    def compute(self) -> torch.Tensor:
        return self.n_valid.float() / self.total


class AverageEnergy(GenerativeMetric):
    """Average energy for molecules for which energy can be calculated

    Note that the energy cannot be calculated for some molecules (specifically invalid ones) and the pose optimisation
    is not guaranteed to succeed. Molecules for which the energy cannot be calculated do not count towards the metric.

    This metric doesn't require that input molecules have been sanitised by RDKit, however, it is usually a good idea
    to do this anyway to ensure that all of the required molecular and atom properties are calculated and stored.
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
            non_none = [
                m for m in _pool_map(_rd_optimise_mol, non_none) if m is not None
            ]

        energy_fn = _rd_calc_energy_per_atom if self.per_atom else _rd_calc_energy
        energies = _pool_map(energy_fn, non_none)
        valid_energies = [e for e in energies if _is_valid_float(e)]

        self.energy += sum(valid_energies)
        self.n_valid_energies += len(valid_energies)

    def compute(self) -> torch.Tensor:
        return self.energy / self.n_valid_energies


# TODO: Add xTB as level of theory option and add forces as a metric


class AverageStrainEnergy(GenerativeMetric):
    """
    The strain energy is the energy difference between a molecule's pose and its optimised pose. Estimated using RDKit.
    Only calculated when all of the following are true:
    1. The molecule is valid and an energy can be calculated
    2. The pose optimisation succeeds
    3. The energy can be calculated for the optimised pose

    Note that molecules which do not meet these criteria will not count towards the metric and can therefore give
    unexpected results. Use the EnergyValidity metric with the optimise flag set to True to track the proportion of
    molecules for which this metric can be calculated.

    This metric doesn't require that input molecules have been sanitised by RDKit, however, it is usually a good idea
    to do this anyway to ensure that all of the required molecular and atom properties are calculated and stored.
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
    """
    Average RMSD between a molecule and its optimised pose. Only calculated when all of the following are true:
    1. The molecule is valid
    2. The pose optimisation succeeds

    Note that molecules which do not meet these criteria will not count towards the metric and can therefore give
    unexpected results.
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
            chemflowRD.conf_distance(mol1, mol2)
            for mol1, mol2 in zip(original_mols, opt_mols)
        ]

        self.total_rmsd += sum(rmsds)
        self.n_valid += len(rmsds)

    def compute(self) -> torch.Tensor:
        return self.total_rmsd / self.n_valid


def init_metrics(
    train_smiles: list[str] | None = None,
    target_n_atoms_distribution: torch.Tensor | None = None,
    atom_type_distribution: torch.Tensor | None = None,
    edge_type_distribution: torch.Tensor | None = None,
    charge_type_distribution: torch.Tensor | None = None,
    atom_tokens: list[str] | None = None,
    edge_tokens: list[str] | None = None,
    charge_tokens: list[str] | None = None,
    allow_charged: bool = False,
    distributions=None,
):

    metrics = {
        "validity": Validity(allow_charged=allow_charged),
        "semla-validity": SEMLAValidity(),
        "semla-atom-stability": SEMLAAtomStability(),
        "semla-molecule-stability": SEMLAMoleculeStability(),
        "uniqueness": Uniqueness(),
        **({"novelty": Novelty(train_smiles)} if train_smiles is not None else {}),
        "energy-validity": EnergyValidity(),
        "opt-energy-validity": EnergyValidity(optimise=True),
        "energy": AverageEnergy(),
        "energy-per-atom": AverageEnergy(per_atom=True),
        "strain": AverageStrainEnergy(),
        "strain-per-atom": AverageStrainEnergy(per_atom=True),
        "opt-rmsd": AverageOptRmsd(),
    }

    # Distribution metrics are kept in their own collection so they can accumulate
    # across an entire validation epoch (rather than being reset per batch) and
    # so the lightning module can additionally render ground-truth-vs-generated
    # marginal plots from their internal histograms.
    distribution_metrics: dict = {}
    if target_n_atoms_distribution is not None:
        distribution_metrics["atom_count_dist_kl"] = AtomCountDistributionMetric(
            target_distribution=target_n_atoms_distribution,
        )
    if atom_type_distribution is not None and atom_tokens is not None:
        distribution_metrics["atom_type_dist_kl"] = AtomTypeDistributionMetric(
            target_distribution=atom_type_distribution,
            atom_tokens=atom_tokens,
        )
    if edge_type_distribution is not None and edge_tokens is not None:
        distribution_metrics["edge_type_dist_kl"] = EdgeTypeDistributionMetric(
            target_distribution=edge_type_distribution,
            edge_tokens=edge_tokens,
        )
    if charge_type_distribution is not None and charge_tokens is not None:
        distribution_metrics["charge_type_dist_kl"] = ChargeTypeDistributionMetric(
            target_distribution=charge_type_distribution,
            charge_tokens=charge_tokens,
        )

    stability_metrics = {
        "atom-stability": AtomStability(),
        "molecule-stability": MoleculeStability(),
    }

    metrics = MetricCollection(metrics, compute_groups=False)
    stability_metrics = MetricCollection(stability_metrics, compute_groups=False)
    distribution_metrics = MetricCollection(distribution_metrics, compute_groups=False)

    # Pointcloud-mode metrics — built from training-set target stats stored in
    # Distributions. The collection may be empty if Distributions lacks them.
    from chemflow.utils.pointcloud_metrics import build_pointcloud_metrics

    pc_metrics: dict = {}
    if distributions is not None and atom_tokens is not None:
        pc_metrics = build_pointcloud_metrics(distributions, len(atom_tokens))
    pointcloud_metrics = MetricCollection(pc_metrics, compute_groups=False)

    return metrics, stability_metrics, distribution_metrics, pointcloud_metrics


# ---------------------------------------------------------------------------
# Marginal distribution plotting for wandb logging.
# ---------------------------------------------------------------------------


def _labels_for_n_atoms(target_len: int) -> list[str]:
    return [str(i) for i in range(target_len)]


def plot_marginal_comparison(
    gen_hist: torch.Tensor,
    target_hist: torch.Tensor,
    labels: list[str] | None,
    title: str,
    xlabel: str,
    eps: float = 1e-8,
):
    """Return a matplotlib Figure comparing ground-truth vs generated marginal densities.

    Both inputs are unnormalized histograms; they are normalized to densities here.
    """
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    import numpy as np

    gen = gen_hist.detach().to("cpu", dtype=torch.float32)
    target = target_hist.detach().to("cpu", dtype=torch.float32)

    n = max(gen.numel(), target.numel())
    if gen.numel() < n:
        gen = torch.cat([gen, torch.zeros(n - gen.numel())])
    if target.numel() < n:
        target = torch.cat([target, torch.zeros(n - target.numel())])

    gen_sum = float(gen.sum())
    target_sum = float(target.sum())
    gen = gen / max(gen_sum, eps)
    target = target / max(target_sum, eps)

    if labels is None or len(labels) != n:
        labels = [str(i) for i in range(n)]

    x = np.arange(n)
    width = 0.4

    fig, ax = plt.subplots(figsize=(max(6.0, n * 0.35), 4.0))
    ax.bar(
        x - width / 2.0,
        target.numpy(),
        width,
        label="ground truth",
        color="steelblue",
        alpha=0.85,
    )
    ax.bar(
        x + width / 2.0,
        gen.numpy(),
        width,
        label="generated",
        color="darkorange",
        alpha=0.85,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45 if n > 10 else 0, ha="right")
    ax.set_ylabel("probability")
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def build_marginal_plots(distribution_metrics: MetricCollection) -> dict:
    """Build one matplotlib Figure per distribution metric present in the collection.

    Returns a dict keyed by a short plot name (e.g. ``"n_atoms"``) mapping to a Figure.
    Metrics that have not yet seen any samples (``n_total == 0``) are skipped.
    Caller is responsible for closing the figures after logging.
    """
    plots: dict = {}

    def _has_samples(metric) -> bool:
        n_total = getattr(metric, "n_total", None)
        if n_total is None:
            return False
        try:
            return (
                float(n_total.item() if isinstance(n_total, torch.Tensor) else n_total)
                > 0.0
            )
        except Exception:
            return False

    if "atom_count_dist_kl" in distribution_metrics:
        m = distribution_metrics["atom_count_dist_kl"]
        if _has_samples(m):
            plots["n_atoms"] = plot_marginal_comparison(
                gen_hist=m.gen_hist,
                target_hist=m.target_distribution,
                labels=_labels_for_n_atoms(m.target_len),
                title="Number of atoms: ground truth vs generated",
                xlabel="n_atoms",
            )

    if "atom_type_dist_kl" in distribution_metrics:
        m = distribution_metrics["atom_type_dist_kl"]
        if _has_samples(m):
            plots["atom_types"] = plot_marginal_comparison(
                gen_hist=m.gen_hist,
                target_hist=m.target_distribution,
                labels=list(m.atom_tokens),
                title="Atom types: ground truth vs generated",
                xlabel="atom type",
            )

    if "edge_type_dist_kl" in distribution_metrics:
        m = distribution_metrics["edge_type_dist_kl"]
        if _has_samples(m):
            plots["edge_types"] = plot_marginal_comparison(
                gen_hist=m.gen_hist,
                target_hist=m.target_distribution,
                labels=list(m.edge_tokens),
                title="Edge types (upper-tri pairs incl. NO_BOND): "
                "ground truth vs generated",
                xlabel="edge type",
            )

    if "charge_type_dist_kl" in distribution_metrics:
        m = distribution_metrics["charge_type_dist_kl"]
        if _has_samples(m):
            plots["charges"] = plot_marginal_comparison(
                gen_hist=m.gen_hist,
                target_hist=m.target_distribution,
                labels=list(m.charge_tokens),
                title="Formal charges: ground truth vs generated",
                xlabel="formal charge",
            )

    return plots


def calc_posebusters_metrics(
    rdkit_mols: list[Chem.rdchem.Mol | None],
) -> dict[str, float]:
    """Run PoseBusters molecular plausibility checks and return averaged pass rates.

    Uses the "mol" config (standalone molecule checks, no protein context).
    Each check produces a boolean per molecule; we return the mean pass rate.
    """

    valid_mols = [mol for mol in rdkit_mols if mol is not None]
    if len(valid_mols) == 0:
        return {}

    buster = PoseBusters(config="mol", max_workers=-1)

    df = buster.bust(valid_mols, None, None)

    results = {}
    for col in df.columns:
        # Skip string index columns if they are in the dataframe
        if col in ["file", "molecule"]:
            continue

        try:
            # Force the column to numeric (True=1.0, False=0.0, NaN=NaN)
            # errors='coerce' turns anything it can't convert into a NaN
            numeric_series = pd.to_numeric(df[col], errors="coerce")

            # If the column isn't entirely NaNs after conversion, get the mean
            if not numeric_series.isna().all():
                # Note: Adding the "posebusters/" prefix to match your logs!
                results[col] = float(numeric_series.mean())
        except Exception:
            print(
                f"Warning: Could not process column '{col}' in PoseBusters results. Skipping this metric."
            )
            pass

    return results


# {'modules': [{'name': 'Loading', 'function': 'loading', 'chosen_binary_test_output': ['mol_pred_loaded'], 'rename_outputs': {'mol_pred_loaded': 'MOL_PRED loaded'}}, {'name': 'Chemistry', 'function': 'rdkit_sanity', 'chosen_binary_test_output': ['passes_rdkit_sanity_checks'], 'rename_outputs': {'passes_rdkit_sanity_checks': 'Sanitization'}}, {'name': 'Chemistry', 'function': 'inchi_convertible', 'chosen_binary_test_output': ['inchi_convertible'], 'rename_outputs': {'inchi_convertible': 'InChI convertible'}}, {'name': 'Chemistry', 'function': 'atoms_connected', 'chosen_binary_test_output': ['all_atoms_connected'], 'rename_outputs': {'all_atoms_connected': 'All atoms connected'}}, {'name': 'Chemistry', 'function': 'check_radicals', 'chosen_binary_test_output': ['no_radicals'], 'rename_outputs': {'no_radicals': 'No radicals'}}, {'name': 'Geometry', 'function': 'distance_geometry', 'parameters': {'bound_matrix_params': {'set15bounds': True, 'scaleVDW': True, 'doTriangleSmoothing': True, 'useMacrocycle14config': False}, 'threshold_bad_bond_length': 0.25, 'threshold_bad_angle': 0.25, 'threshold_clash': 0.3, 'ignore_hydrogens': True, 'sanitize': True}, 'chosen_binary_test_output': ['bond_lengths_within_bounds', 'bond_angles_within_bounds', 'no_internal_clash'], 'rename_outputs': {'bond_lengths_within_bounds': 'Bond lengths', 'bond_angles_within_bounds': 'Bond angles', 'no_internal_clash': 'Internal steric clash'}}, {'name': 'Ring flatness', 'function': 'flatness', 'parameters': {'flat_systems': {'aromatic_5_membered_rings_sp2': '[ar5^2]1[ar5^2][ar5^2][ar5^2][ar5^2]1', 'aromatic_6_membered_rings_sp2': '[ar6^2]1[ar6^2][ar6^2][ar6^2][ar6^2][ar6^2]1'}, 'threshold_flatness': 0.25}, 'chosen_binary_test_output': ['flatness_passes'], 'rename_outputs': {'num_systems_checked': 'number_aromatic_rings_checked', 'num_systems_passed': 'number_aromatic_rings_pass', 'max_distance': 'aromatic_ring_maximum_distance_from_plane', 'flatness_passes': 'Aromatic ring flatness'}}, {'name': 'Ring non-flatness', 'function': 'flatness', 'parameters': {'check_nonflat': True, 'flat_systems': {'non-aromatic_6_membered_rings': '[C,O,S,N;R1]~1[C,O,S,N;R1][C,O,S,N;R1][C,O,S,N;R1][C,O,S,N;R1][C,O,S,N;R1]1', 'non-aromatic_6_membered_rings_db03_0': '[C;R1]~1[C;R1][C,O,S,N;R1]~[C,O,S,N;R1][C;R1][C;R1]1', 'non-aromatic_6_membered_rings_db03_1': '[C;R1]~1[C;R1][C;R1]~[C;R1][C,O,S,N;R1][C;R1]1', 'non-aromatic_6_membered_rings_db02_0': '[C;R1]~1[C;R1][C;R1][C,O,S,N;R1]~[C,O,S,N;R1][C;R1]1', 'non-aromatic_6_membered_rings_db02_1': '[C;R1]~1[C;R1][C,O,S,N;R1][C;R1]~[C;R1][C;R1]1'}, 'threshold_flatness': 0.05}, 'chosen_binary_test_output': ['flatness_passes'], 'rename_outputs': {'num_systems_checked': 'number_non-aromatic_rings_checked', 'num_systems_passed': 'number_non-aromatic_rings_pass', 'max_distance': 'non-aromatic_ring_maximum_distance_from_plane', 'flatness_passes': 'Non-aromatic ring non-flatness'}}, {'name': 'Double bond flatness', 'function': 'flatness', 'parameters': {'flat_systems': {'trigonal_planar_double_bonds': '[C;X3;^2](*)(*)=[C;X3;^2](*)(*)'}, 'threshold_flatness': 0.25}, 'chosen_binary_test_output': ['flatness_passes'], 'rename_outputs': {'num_systems_checked': 'number_double_bonds_checked', 'num_systems_passed': 'number_double_bonds_pass', 'max_distance': 'double_bond_maximum_distance_from_plane', 'flatness_passes': 'Double bond flatness'}}, {'name': 'Energy ratio', 'function': 'energy_ratio', 'parameters': {'threshold_energy_ratio': 100.0, 'ensemble_number_conformations': 50}}, {'name': 'Energy ratio', 'function': 'energy_ratio', 'parameters': {'threshold_energy_ratio': 100.0, 'ensemble_number_conformations': 50}, 'chosen_binary_test_output': ['energy_ratio_passes'], 'rename_outputs': {'energy_ratio_passes': 'Internal energy'}}], 'loading': {'mol_pred': {'cleanup': False, 'sanitize': False, 'add_hs': False, 'assign_stereo': False, 'load_all': True}, 'mol_true': {'cleanup': False, 'sanitize': False, 'add_hs': False, 'assign_stereo': False, 'load_all': True}, 'mol_cond': {'cleanup': False, 'sanitize': False, 'add_hs': False, 'assign_stereo': False, 'proximityBonding': False}}, 'top_n': None, 'max_workers': 0, 'chunk_size': 100}


def calc_metrics_(
    rdkit_mols,
    metrics,
    stab_metrics=None,
    mol_stabs=None,
    mols=None,
    target_n_atoms_distribution: torch.Tensor | None = None,
    device: torch.device | str | None = None,
):
    metrics.reset()
    metrics.update(rdkit_mols)
    raw = metrics.compute()
    results = {
        k: v.item() if isinstance(v, torch.Tensor) else v for k, v in raw.items()
    }

    if mols is not None and target_n_atoms_distribution is not None:
        if device is None:
            if isinstance(target_n_atoms_distribution, torch.Tensor):
                device = target_n_atoms_distribution.device
            else:
                device = "cpu"
        atom_count_results = atom_count_distribution_metrics(
            mols=mols,
            target_distribution=target_n_atoms_distribution,
            device=device,
        )
        results = {
            **results,
            **{
                k: v.item() if isinstance(v, torch.Tensor) else v
                for k, v in atom_count_results.items()
            },
        }

    if stab_metrics is None:
        return results

    stab_metrics.reset()
    stab_metrics.update(mol_stabs)
    stab_raw = stab_metrics.compute()
    stab_results = {
        k: v.item() if isinstance(v, torch.Tensor) else v for k, v in stab_raw.items()
    }

    results = {**results, **stab_results}
    return results
