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
from rdkit import Chem, RDLogger

# Silence RDKit's "Explicit valence for atom..." spam emitted during GRPO
# rollouts where intermediate (and some final) molecules are invalid by
# construction.  The validity flag returned by `mol_is_valid` is the signal
# we care about; the log lines just obscure `print`-based training metrics.
try:  # RDKit >= 2020: route C++ logging through Python (then DisableLog works)
    Chem.WrapLogs()  # type: ignore[attr-defined]
except AttributeError:
    pass
RDLogger.DisableLog("rdApp.*")


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


def n_atoms_reward(module, trajectory) -> tuple[torch.Tensor, dict[str, float]]:
    """Total atom count if valid, else 0.  Higher = better.

    Returned as raw integer count (not normalized): GRPO standardizes the
    advantage `(r - r.mean()) / (r.std() + 1e-6)` anyway, so reward scale is
    irrelevant for the gradient -- only ordering matters.  Keeping the raw
    count makes `reward_mean` directly readable as "average atoms this
    batch", and `reward_max` as "largest valid mol seen".

    Diagnostics:
        - `p_valid`
        - `n_atoms_mean_valid` / `n_atoms_max_valid` over valid mols only
        - `n_atoms_mean_all` / `n_atoms_max_all` over all RDKit-convertible mols
    """
    device = trajectory.mol_final.x.device
    counts_valid_gated: list[float] = []
    counts_all: list[float] = []
    for rd, ok in _iter_valid_mols(module, trajectory):
        n_atoms = float(rd.GetNumAtoms()) if rd is not None else 0.0
        counts_all.append(n_atoms)
        counts_valid_gated.append(n_atoms if ok else 0.0)
    r = _as_tensor(counts_valid_gated, device)
    n_valid = sum(1 for c in counts_valid_gated if c > 0)
    sum_valid = sum(c for c in counts_valid_gated if c > 0)
    sum_all = sum(counts_all)
    return r, {
        "p_valid": n_valid / max(len(counts_valid_gated), 1),
        "n_atoms_mean_valid": (sum_valid / n_valid) if n_valid > 0 else 0.0,
        "n_atoms_max_valid": max(counts_valid_gated) if counts_valid_gated else 0.0,
        "n_atoms_mean_all": (sum_all / len(counts_all)) if counts_all else 0.0,
        "n_atoms_max_all": max(counts_all) if counts_all else 0.0,
    }

# ─────────────────────────────────────────────────────────────────────────────
# Shape reward: USRCAT × Tanimoto toward a reference molecule (Prilocaine)
# ─────────────────────────────────────────────────────────────────────────────

_SHAPE_REF_SMILES = "CCCNC(C)C(=O)Nc1ccccc1C"  # Prilocaine
_SHAPE_REF_CACHE: dict | None = None


def _get_shape_ref() -> dict:
    """Build reference molecule once. Includes explicit Hs to match QM9."""
    global _SHAPE_REF_CACHE
    if _SHAPE_REF_CACHE is not None:
        return _SHAPE_REF_CACHE

    from rdkit.Chem import AllChem, rdMolDescriptors, rdFingerprintGenerator

    mol = Chem.MolFromSmiles(_SHAPE_REF_SMILES)
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    cids = AllChem.EmbedMultipleConfs(mol, numConfs=50, params=params)
    best_e, best_cid = float("inf"), -1
    for cid in cids:
        AllChem.MMFFOptimizeMolecule(mol, confId=cid, maxIters=2000)
        mp = AllChem.MMFFGetMoleculeProperties(mol)
        ff = AllChem.MMFFGetMoleculeForceField(mol, mp, confId=cid)
        if ff is not None and ff.CalcEnergy() < best_e:
            best_e, best_cid = ff.CalcEnergy(), cid
    for cid in [c.GetId() for c in mol.GetConformers() if c.GetId() != best_cid]:
        mol.RemoveConformer(cid)

    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    _SHAPE_REF_CACHE = {
        "usrcat": rdMolDescriptors.GetUSRCAT(mol),
        "fp": morgan_gen.GetFingerprint(mol),
        "morgan_gen": morgan_gen,
    }
    return _SHAPE_REF_CACHE


def _score_shape_single(rd: Chem.Mol) -> tuple[float, float, float]:
    """Score one mol. No AddHs, no fallback embedding — model coords or nothing."""
    from rdkit.Chem import rdMolDescriptors, DataStructs

    if rd.GetNumConformers() == 0:
        return 0.0, 0.0, 0.0

    ref = _get_shape_ref()
    try:
        usrcat = rdMolDescriptors.GetUSRCAT(rd)
        shape_sim = rdMolDescriptors.GetUSRScore(usrcat, ref["usrcat"])
    except Exception:
        shape_sim = 0.0
    try:
        fp = ref["morgan_gen"].GetFingerprint(rd)
        graph_sim = DataStructs.TanimotoSimilarity(fp, ref["fp"])
    except Exception:
        graph_sim = 0.0
    return shape_sim, graph_sim, shape_sim * graph_sim


def shape_reward(module, trajectory) -> tuple[torch.Tensor, dict[str, float]]:
    """USRCAT × Tanimoto to Prilocaine if valid, else 0."""
    device = trajectory.mol_final.x.device
    rewards: list[float] = []
    usrcat_vals: list[float] = []
    tanimoto_vals: list[float] = []
    product_vals: list[float] = []
    n_valid = 0

    for rd, ok in _iter_valid_mols(module, trajectory):
        if not ok or rd is None:
            rewards.append(0.0)
            continue
        n_valid += 1
        try:
            s, g, p = _score_shape_single(rd)
        except Exception:
            s, g, p = 0.0, 0.0, 0.0
        rewards.append(p)
        usrcat_vals.append(s)
        tanimoto_vals.append(g)
        product_vals.append(p)

    r = _as_tensor(rewards, device)
    return r, {
        "p_valid": n_valid / max(len(rewards), 1),
        "shape_usrcat_mean_valid": (sum(usrcat_vals) / len(usrcat_vals)) if usrcat_vals else 0.0,
        "shape_tanimoto_mean_valid": (sum(tanimoto_vals) / len(tanimoto_vals)) if tanimoto_vals else 0.0,
        "shape_product_mean_valid": (sum(product_vals) / len(product_vals)) if product_vals else 0.0,
        "shape_product_max_valid": max(product_vals) if product_vals else 0.0,
    }

# ─────────────────────────────────────────────────────────────────────────────
# Registry (add new rewards here)
# ─────────────────────────────────────────────────────────────────────────────


REWARDS: dict[str, Callable] = {
    "validity": validity_reward,
    "qed": qed_reward,
    "n_atoms": n_atoms_reward,
    "shape": shape_reward,
}
