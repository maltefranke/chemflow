"""Prilocaine-dependent GRPO rewards."""

from __future__ import annotations

import math

import torch
from rdkit import Chem

from chemflow.dataset.representation import Representation

from .common import _as_tensor, _iter_valid_mols
from .spec import RewardSpec

_PRILOCAINE_REF_SMILES = "CCCNC(C)C(=O)Nc1ccccc1C"
_SHAPE_REF_SMILES = _PRILOCAINE_REF_SMILES  # Back-compat for analysis notebooks.

_TANIMOTO_REF_CACHE: dict | None = None
_TOPOLOGY_REF_CACHE: dict | None = None
_TOPOLOGY_MOTIF_REF_CACHE: dict | None = None


def _heavy_atom_mol(rd: Chem.Mol) -> Chem.Mol:
    """Return a heavy-atom copy for graph fingerprints."""
    try:
        heavy = Chem.RemoveHs(Chem.Mol(rd), sanitize=True)
        Chem.SanitizeMol(heavy)
        return heavy
    except Exception:
        return rd


def _get_tanimoto_ref() -> dict:
    global _TANIMOTO_REF_CACHE
    if _TANIMOTO_REF_CACHE is None:
        from rdkit.Chem import rdFingerprintGenerator

        mol = Chem.MolFromSmiles(_PRILOCAINE_REF_SMILES)
        mol = _heavy_atom_mol(mol)
        morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        _TANIMOTO_REF_CACHE = {
            "fp": morgan_gen.GetFingerprint(mol),
            "morgan_gen": morgan_gen,
        }
    return _TANIMOTO_REF_CACHE


def _get_topology_ref() -> dict:
    global _TOPOLOGY_REF_CACHE
    if _TOPOLOGY_REF_CACHE is None:
        from rdkit.Chem import rdFingerprintGenerator
        from rdkit.Chem.Scaffolds import MurckoScaffold

        mol = Chem.MolFromSmiles(_PRILOCAINE_REF_SMILES)
        mol = _heavy_atom_mol(mol)
        morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        _TOPOLOGY_REF_CACHE = {
            "mol": mol,
            "scaffold": MurckoScaffold.GetScaffoldForMol(mol),
            "rdk_fp": Chem.RDKFingerprint(mol),
            "morgan_fp": morgan_gen.GetFingerprint(mol),
            "morgan_gen": morgan_gen,
            "heavy_atoms": mol.GetNumHeavyAtoms(),
        }
    return _TOPOLOGY_REF_CACHE


def _mcs_sim(mol: Chem.Mol, target: Chem.Mol, timeout: int = 1) -> float:
    from rdkit.Chem import rdFMCS

    try:
        res = rdFMCS.FindMCS(
            [mol, target],
            atomCompare=rdFMCS.AtomCompare.CompareElements,
            bondCompare=rdFMCS.BondCompare.CompareOrderExact,
            ringMatchesRingOnly=True,
            timeout=timeout,
        )
    except Exception:
        return 0.0
    if res.canceled or res.numAtoms == 0:
        return 0.0
    denom = max(target.GetNumHeavyAtoms(), mol.GetNumHeavyAtoms(), 1)
    return float(res.numAtoms) / float(denom)


def _scaffold_sim(mol: Chem.Mol, target_scaffold: Chem.Mol) -> float:
    from rdkit.Chem import DataStructs
    from rdkit.Chem.Scaffolds import MurckoScaffold

    try:
        mol_scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    except Exception:
        return 0.0

    mol_atoms = mol_scaffold.GetNumHeavyAtoms()
    target_atoms = target_scaffold.GetNumHeavyAtoms()
    if target_atoms == 0:
        return 1.0 if mol_atoms == 0 else 0.0
    if mol_atoms == 0:
        return 0.0

    try:
        fp1 = Chem.RDKFingerprint(mol_scaffold)
        fp2 = Chem.RDKFingerprint(target_scaffold)
        return float(DataStructs.TanimotoSimilarity(fp1, fp2))
    except Exception:
        return 0.0


def _fp_blend_sim(mol: Chem.Mol, ref: dict) -> float:
    from rdkit.Chem import DataStructs

    try:
        rdk_sim = DataStructs.TanimotoSimilarity(
            Chem.RDKFingerprint(mol),
            ref["rdk_fp"],
        )
    except Exception:
        rdk_sim = 0.0
    try:
        morgan_sim = DataStructs.TanimotoSimilarity(
            ref["morgan_gen"].GetFingerprint(mol),
            ref["morgan_fp"],
        )
    except Exception:
        morgan_sim = 0.0
    return float(0.5 * rdk_sim + 0.5 * morgan_sim)


def _size_penalty(mol: Chem.Mol, target_heavy_atoms: int) -> float:
    if target_heavy_atoms <= 0:
        return 0.0
    ratio = mol.GetNumHeavyAtoms() / target_heavy_atoms
    return float(math.exp(-0.5 * ((ratio - 1.0) / 0.3) ** 2))


def _score_topology_single(
    rd: Chem.Mol,
) -> tuple[float, float, float, float, float, float]:
    """MCS/scaffold/fingerprint/size composite toward the topology reference."""
    ref = _get_topology_ref()
    mol = _heavy_atom_mol(rd)
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    mcs = _mcs_sim(mol, ref["mol"])
    scaffold = _scaffold_sim(mol, ref["scaffold"])
    fp = _fp_blend_sim(mol, ref)
    size = _size_penalty(mol, ref["heavy_atoms"])
    mcs_t = 1.0 / (1.0 + math.exp(-12.0 * (mcs - 0.4)))
    raw = 0.45 * mcs_t + 0.25 * scaffold + 0.20 * fp + 0.10 * size
    return raw, mcs, mcs_t, scaffold, fp, size


def _is_nonring_functional_fragment(mol: Chem.Mol) -> bool:
    """Keep reference path motifs that add chemistry beyond carbon ring paths."""
    try:
        has_hetero = any(atom.GetAtomicNum() not in (1, 6) for atom in mol.GetAtoms())
        has_non_aromatic_multiple = any(
            bond.GetBondType() not in (Chem.BondType.SINGLE, Chem.BondType.AROMATIC)
            for bond in mol.GetBonds()
        )
        return bool(has_hetero or has_non_aromatic_multiple)
    except Exception:
        return False


def _extract_topology_motifs(
    mol: Chem.Mol,
    min_bonds: int = 1,
    max_bonds: int = 4,
) -> dict[str, Chem.Mol]:
    """Auto-extract non-ring-only path motifs from the reference molecule."""
    motifs: dict[str, Chem.Mol] = {}
    for length in range(min_bonds, max_bonds + 1):
        try:
            paths = Chem.FindAllPathsOfLengthN(mol, length, useBonds=True)
        except Exception:
            continue
        for path in paths:
            bond_ids = set(int(bond_id) for bond_id in path)
            atom_ids: set[int] = set()
            for bond_id in bond_ids:
                bond = mol.GetBondWithIdx(bond_id)
                atom_ids.add(bond.GetBeginAtomIdx())
                atom_ids.add(bond.GetEndAtomIdx())
            try:
                smi = Chem.MolFragmentToSmiles(
                    mol,
                    atomsToUse=sorted(atom_ids),
                    bondsToUse=sorted(bond_ids),
                    canonical=True,
                    isomericSmiles=False,
                )
            except Exception:
                continue
            if not smi or smi in motifs:
                continue
            query = Chem.MolFromSmiles(smi)
            if query is None or not _is_nonring_functional_fragment(query):
                continue
            motifs[smi] = query
    return motifs


def _get_topology_motif_ref() -> dict:
    global _TOPOLOGY_MOTIF_REF_CACHE
    if _TOPOLOGY_MOTIF_REF_CACHE is None:
        mol = Chem.MolFromSmiles(_PRILOCAINE_REF_SMILES)
        mol = _heavy_atom_mol(mol)
        motifs = _extract_topology_motifs(mol)
        _TOPOLOGY_MOTIF_REF_CACHE = {
            "motifs": motifs,
            "motif_count": len(motifs),
        }
    return _TOPOLOGY_MOTIF_REF_CACHE


def _topology_motif_recall(mol: Chem.Mol, motifs: dict[str, Chem.Mol]) -> float:
    if not motifs:
        return 1.0
    matched = 0
    for query in motifs.values():
        try:
            if mol.HasSubstructMatch(query):
                matched += 1
        except Exception:
            pass
    return float(matched) / float(len(motifs))


def _score_topology_motif_single(
    rd: Chem.Mol,
) -> tuple[float, float, float, float, float, float, float]:
    """Topology reward plus reference-derived functional path motif recall."""
    ref = _get_topology_ref()
    motif_ref = _get_topology_motif_ref()
    mol = _heavy_atom_mol(rd)
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    mcs = _mcs_sim(mol, ref["mol"])
    scaffold = _scaffold_sim(mol, ref["scaffold"])
    fp = _fp_blend_sim(mol, ref)
    size = _size_penalty(mol, ref["heavy_atoms"])
    motif = _topology_motif_recall(mol, motif_ref["motifs"])
    mcs_t = 1.0 / (1.0 + math.exp(-12.0 * (mcs - 0.4)))
    raw = (
        0.32 * mcs_t
        + 0.18 * scaffold
        + 0.15 * fp
        + 0.10 * size
        + 0.25 * motif
    )
    return raw, mcs, mcs_t, scaffold, fp, size, motif


def tanimoto_reward(module, trajectory) -> tuple[torch.Tensor, dict[str, float]]:
    """Morgan Tanimoto to Prilocaine if valid, else 0."""
    from rdkit.Chem import DataStructs

    device = trajectory.mol_final.x.device
    ref = _get_tanimoto_ref()
    rewards: list[float] = []
    vals: list[float] = []
    n_valid = 0

    for rd, ok in _iter_valid_mols(module, trajectory):
        if not ok or rd is None:
            rewards.append(0.0)
            continue
        n_valid += 1
        try:
            rd_heavy = _heavy_atom_mol(rd)
            sim = DataStructs.TanimotoSimilarity(
                ref["morgan_gen"].GetFingerprint(rd_heavy),
                ref["fp"],
            )
        except Exception:
            sim = 0.0
        rewards.append(sim)
        vals.append(sim)

    return _as_tensor(rewards, device), {
        "p_valid": n_valid / max(len(rewards), 1),
        "tanimoto_mean_valid": (sum(vals) / len(vals)) if vals else 0.0,
        "tanimoto_max_valid": max(vals) if vals else 0.0,
    }


def topology_reward(module, trajectory) -> tuple[torch.Tensor, dict[str, float]]:
    """Composite topology reward to Prilocaine if valid, else 0."""
    device = trajectory.mol_final.x.device
    rewards: list[float] = []
    raw_vals: list[float] = []
    mcs_vals: list[float] = []
    mcs_t_vals: list[float] = []
    scaffold_vals: list[float] = []
    fp_vals: list[float] = []
    size_vals: list[float] = []
    n_valid = 0

    for rd, ok in _iter_valid_mols(module, trajectory):
        if not ok or rd is None:
            rewards.append(0.0)
            continue
        n_valid += 1
        try:
            raw, mcs, mcs_t, scaffold, fp, size = _score_topology_single(rd)
        except Exception:
            raw, mcs, mcs_t, scaffold, fp, size = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        rewards.append(raw)
        raw_vals.append(raw)
        mcs_vals.append(mcs)
        mcs_t_vals.append(mcs_t)
        scaffold_vals.append(scaffold)
        fp_vals.append(fp)
        size_vals.append(size)

    return _as_tensor(rewards, device), {
        "p_valid": n_valid / max(len(rewards), 1),
        "topology_mean_valid": (sum(raw_vals) / len(raw_vals)) if raw_vals else 0.0,
        "topology_max_valid": max(raw_vals) if raw_vals else 0.0,
        "topology_mcs_mean_valid": (sum(mcs_vals) / len(mcs_vals)) if mcs_vals else 0.0,
        "topology_mcs_t_mean_valid": (
            sum(mcs_t_vals) / len(mcs_t_vals)
        ) if mcs_t_vals else 0.0,
        "topology_scaffold_mean_valid": (
            sum(scaffold_vals) / len(scaffold_vals)
        ) if scaffold_vals else 0.0,
        "topology_fp_mean_valid": (sum(fp_vals) / len(fp_vals)) if fp_vals else 0.0,
        "topology_size_mean_valid": (
            sum(size_vals) / len(size_vals)
        ) if size_vals else 0.0,
    }


def topology_motif_reward(module, trajectory) -> tuple[torch.Tensor, dict[str, float]]:
    """Topology reward with a functional path-motif recall term."""
    device = trajectory.mol_final.x.device
    motif_ref = _get_topology_motif_ref()
    rewards: list[float] = []
    raw_vals: list[float] = []
    mcs_vals: list[float] = []
    mcs_t_vals: list[float] = []
    scaffold_vals: list[float] = []
    fp_vals: list[float] = []
    size_vals: list[float] = []
    motif_vals: list[float] = []
    n_valid = 0

    for rd, ok in _iter_valid_mols(module, trajectory):
        if not ok or rd is None:
            rewards.append(0.0)
            continue
        n_valid += 1
        try:
            raw, mcs, mcs_t, scaffold, fp, size, motif = (
                _score_topology_motif_single(rd)
            )
        except Exception:
            raw, mcs, mcs_t, scaffold, fp, size, motif = (
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            )
        rewards.append(raw)
        raw_vals.append(raw)
        mcs_vals.append(mcs)
        mcs_t_vals.append(mcs_t)
        scaffold_vals.append(scaffold)
        fp_vals.append(fp)
        size_vals.append(size)
        motif_vals.append(motif)

    return _as_tensor(rewards, device), {
        "p_valid": n_valid / max(len(rewards), 1),
        "topology_motif_mean_valid": (
            sum(raw_vals) / len(raw_vals)
        ) if raw_vals else 0.0,
        "topology_motif_max_valid": max(raw_vals) if raw_vals else 0.0,
        "topology_motif_mcs_mean_valid": (
            sum(mcs_vals) / len(mcs_vals)
        ) if mcs_vals else 0.0,
        "topology_motif_mcs_t_mean_valid": (
            sum(mcs_t_vals) / len(mcs_t_vals)
        ) if mcs_t_vals else 0.0,
        "topology_motif_scaffold_mean_valid": (
            sum(scaffold_vals) / len(scaffold_vals)
        ) if scaffold_vals else 0.0,
        "topology_motif_fp_mean_valid": (
            sum(fp_vals) / len(fp_vals)
        ) if fp_vals else 0.0,
        "topology_motif_size_mean_valid": (
            sum(size_vals) / len(size_vals)
        ) if size_vals else 0.0,
        "topology_motif_recall_mean_valid": (
            sum(motif_vals) / len(motif_vals)
        ) if motif_vals else 0.0,
        "topology_motif_ref_count": float(motif_ref["motif_count"]),
    }


# All three rewards build RDKit molecules from the final batch and score
# against a reference scaffold, so they require full chemistry.
_MOLECULE_ONLY = frozenset({Representation.MOLECULE})
TANIMOTO_SPEC = RewardSpec(fn=tanimoto_reward, supported_representations=_MOLECULE_ONLY)
TOPOLOGY_SPEC = RewardSpec(fn=topology_reward, supported_representations=_MOLECULE_ONLY)
TOPOLOGY_MOTIF_SPEC = RewardSpec(
    fn=topology_motif_reward, supported_representations=_MOLECULE_ONLY
)
