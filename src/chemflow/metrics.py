"""Code adapted from SemlaFlow"""

import os
from concurrent.futures import ProcessPoolExecutor
from typing import Literal

import torch
from rdkit import Chem
from torchmetrics import Metric
from torchmetrics import MetricCollection

from chemflow import rdkit as chemflowRD


def atom_count_distribution_metrics(
    mols: list,
    target_distribution: torch.Tensor,
    device: torch.device | str,
    eps: float = 1e-8,
) -> dict[str, torch.Tensor]:
    """Compare generated atom-count distribution to a target atom-count distribution."""
    if len(mols) == 0:
        zero = torch.tensor(0.0, device=device)
        return {
            "atom_count_dist_tvd": zero,
            "atom_count_dist_l1": zero,
            "atom_count_dist_kl": zero,
            "atom_count_mean_abs_diff": zero,
        }

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

    l1 = torch.sum(torch.abs(gen - target))
    tvd = 0.5 * l1
    kl = torch.sum(
        gen * (torch.log(gen.clamp(min=eps)) - torch.log(target.clamp(min=eps)))
    )

    support = torch.arange(gen.numel(), device=device, dtype=torch.float32)
    mean_gen = torch.sum(gen * support)
    mean_target = torch.sum(target * support)
    mean_abs_diff = torch.abs(mean_gen - mean_target)

    return {
        "atom_count_dist_tvd": tvd,
        "atom_count_dist_l1": l1,
        "atom_count_dist_kl": kl,
        "atom_count_mean_abs_diff": mean_abs_diff,
    }

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


class AtomCountDistributionMetric(GenerativeMetric):
    """Distribution metric comparing generated atom counts to target distribution."""

    def __init__(
        self,
        target_distribution: torch.Tensor,
        metric_type: Literal["tvd", "l1", "kl", "mean_abs_diff"],
        eps: float = 1e-8,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if metric_type not in {"tvd", "l1", "kl", "mean_abs_diff"}:
            raise ValueError(f"Unknown metric_type: {metric_type}")

        target = target_distribution.detach().to(dtype=torch.float32)
        target = target / target.sum().clamp(min=eps)

        self.metric_type = metric_type
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

        hist = torch.bincount(counts_t, minlength=self.target_len).to(dtype=torch.float32)
        self.gen_hist += hist
        self.n_total += float(len(counts))

    def compute(self) -> torch.Tensor:
        if self.n_total <= 0:
            return torch.tensor(0.0, device=self.gen_hist.device)

        gen = self.gen_hist / self.gen_hist.sum().clamp(min=1.0)
        target = self.target_distribution
        target = target / target.sum().clamp(min=self.eps)

        l1 = torch.sum(torch.abs(gen - target))
        if self.metric_type == "l1":
            return l1

        if self.metric_type == "tvd":
            return 0.5 * l1

        if self.metric_type == "kl":
            return torch.sum(
                gen
                * (
                    torch.log(gen.clamp(min=self.eps))
                    - torch.log(target.clamp(min=self.eps))
                )
            )

        support = torch.arange(self.target_len, device=gen.device, dtype=torch.float32)
        mean_gen = torch.sum(gen * support)
        mean_target = torch.sum(target * support)
        return torch.abs(mean_gen - mean_target)


class Validity(GenerativeMetric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("valid", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, mols: list[Chem.rdchem.Mol]) -> None:
        is_valid = [
            chemflowRD.mol_is_valid(mol)
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
        self.valid_smiles = []

    def update(self, mols: list[Chem.rdchem.Mol]) -> None:
        smiles = [
            Chem.MolToSmiles(mol, canonical=True)
            for mol in mols
            if mol is not None
        ]
        valid_smiles = [smi for smi in smiles if smi is not None]
        self.valid_smiles.extend(valid_smiles)

    def compute(self) -> torch.Tensor:
        num_unique = len(set(self.valid_smiles))
        uniqueness = torch.tensor(num_unique) / len(self.valid_smiles)
        return uniqueness


class Novelty(GenerativeMetric):
    def __init__(self, existing_mols: list[Chem.rdchem.Mol], **kwargs):
        super().__init__(**kwargs)

        n_workers = min(8, len(os.sched_getaffinity(0)))
        executor = ProcessPoolExecutor(max_workers=n_workers)

        futures = [
            executor.submit(chemflowRD.smiles_from_mol, mol, canonical=True)
            for mol in existing_mols
        ]
        smiles = [future.result() for future in futures]
        smiles = [smi for smi in smiles if smi is not None]

        executor.shutdown()

        self.smiles = set(smiles)

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

        if self.optimise:
            mols = [chemflowRD.optimise_mol(mol) for mol in mols if mol is not None]

        energies = [chemflowRD.calc_energy(mol) for mol in mols if mol is not None]
        valid_energies = [energy for energy in energies if _is_valid_float(energy)]

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
        if self.optimise:
            mols = [chemflowRD.optimise_mol(mol) for mol in mols if mol is not None]

        energies = [
            chemflowRD.calc_energy(mol, per_atom=self.per_atom)
            for mol in mols
            if mol is not None
        ]
        valid_energies = [energy for energy in energies if _is_valid_float(energy)]

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
        opt_mols = [
            (idx, chemflowRD.optimise_mol(mol))
            for idx, mol in list(enumerate(mols))
            if mol is not None
        ]
        energies = [
            (idx, chemflowRD.calc_energy(mol, per_atom=self.per_atom))
            for idx, mol in opt_mols
            if mol is not None
        ]
        valids = [(idx, energy) for idx, energy in energies if energy is not None]

        if len(valids) == 0:
            return

        valid_indices, valid_energies = tuple(zip(*valids))
        original_energies = [
            chemflowRD.calc_energy(mols[idx], per_atom=self.per_atom)
            for idx in valid_indices
        ]
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
        valids = [
            (idx, chemflowRD.optimise_mol(mol))
            for idx, mol in list(enumerate(mols))
            if mol is not None
        ]
        valids = [(idx, mol) for idx, mol in valids if mol is not None]

        if len(valids) == 0:
            return

        valid_indices, opt_mols = tuple(zip(*valids))
        original_mols = [mols[idx] for idx in valid_indices]
        rmsds = [
            chemflowRD.conf_distance(mol1, mol2)
            for mol1, mol2 in zip(original_mols, opt_mols)
        ]

        self.total_rmsd += sum(rmsds)
        self.n_valid += len(rmsds)

    def compute(self) -> torch.Tensor:
        return self.total_rmsd / self.n_valid


def init_metrics(train_mols=None, target_n_atoms_distribution: torch.Tensor | None = None):

    metrics = {
        "validity": Validity(),
        "uniqueness": Uniqueness(),
    #     "novelty": Novelty(train_mols),
        "energy-validity": EnergyValidity(),
        "opt-energy-validity": EnergyValidity(optimise=True),
        "energy": AverageEnergy(),
        "energy-per-atom": AverageEnergy(per_atom=True),
        "strain": AverageStrainEnergy(),
        "strain-per-atom": AverageStrainEnergy(per_atom=True),
        "opt-rmsd": AverageOptRmsd(),
    }
    if target_n_atoms_distribution is not None:
        metrics["atom_count_dist_tvd"] = AtomCountDistributionMetric(
            target_distribution=target_n_atoms_distribution,
            metric_type="tvd",
        )
        metrics["atom_count_dist_l1"] = AtomCountDistributionMetric(
            target_distribution=target_n_atoms_distribution,
            metric_type="l1",
        )
        metrics["atom_count_dist_kl"] = AtomCountDistributionMetric(
            target_distribution=target_n_atoms_distribution,
            metric_type="kl",
        )
        metrics["atom_count_mean_abs_diff"] = AtomCountDistributionMetric(
            target_distribution=target_n_atoms_distribution,
            metric_type="mean_abs_diff",
        )
    stability_metrics = {
        "atom-stability": AtomStability(),
        "molecule-stability": MoleculeStability(),
    }

    metrics = MetricCollection(metrics, compute_groups=False)
    stability_metrics = MetricCollection(stability_metrics, compute_groups=False)

    return metrics, stability_metrics


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
    results = metrics.compute()

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
        results = {**results, **atom_count_results}

    if stab_metrics is None:
        return results

    stab_metrics.reset()
    stab_metrics.update(mol_stabs)
    stab_results = stab_metrics.compute()

    results = {**results, **stab_results}
    return results
