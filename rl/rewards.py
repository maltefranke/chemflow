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

Use ``scaffold_diversity_wrapper`` (see ``run_grpo --scaffold_diversity``) to layer
REINVENT-style occurrence bucketing on top of any registered reward (Murcko scaffold
or full-molecule canonical SMILES).
"""

from __future__ import annotations

from collections import deque
from typing import Callable, Iterator, Optional

import torch
from rdkit import Chem, RDLogger

try: 
    Chem.WrapLogs() 
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


# ─────────────────────────────────────────────────────────────────────────────
# REINVENT-style scaffold bucket memory (Murcko counts; optional rolling window)
# ─────────────────────────────────────────────────────────────────────────────


class ScaffoldBucketMemory:
    """Counts Murcko scaffold occurrences for diversity gating.

    If ``window_batches`` is None, counts never expire (full-run memory like REINVENT).
    If set to ``N > 0``, only occurrences from the last ``N`` committed batches
    contribute — older batches are forgotten when new ones arrive.
    """

    def __init__(self, window_batches: int | None = None) -> None:
        if window_batches is not None and window_batches < 1:
            raise ValueError("window_batches must be None or >= 1")
        self._window_batches = window_batches
        self._counts: dict[str, int] = {}
        self._history: deque[dict[str, int]] | None = (
            deque(maxlen=window_batches) if window_batches is not None else None
        )

    def counts_snapshot(self) -> dict[str, int]:
        """Copy of current scaffold -> occurrence counts (for debugging)."""
        return dict(self._counts)

    def commit_batch(self, scaffold_counts: dict[str, int]) -> None:
        """Add scaffold occurrence counts from a finished batch (accepted slots only)."""
        increment = {k: v for k, v in scaffold_counts.items() if v > 0}
        if self._history is None:
            for smi, c in increment.items():
                self._counts[smi] = self._counts.get(smi, 0) + c
            return
        if len(self._history) == self._history.maxlen:
            dropped = self._history.popleft()
            for smi, c in dropped.items():
                prev = self._counts.get(smi, 0) - c
                if prev <= 0:
                    self._counts.pop(smi, None)
                else:
                    self._counts[smi] = prev
        self._history.append(increment)
        for smi, c in increment.items():
            self._counts[smi] = self._counts.get(smi, 0) + c

    def total_seen(self, scaffold_smi: str) -> int:
        return self._counts.get(scaffold_smi, 0)


def _murcko_scaffold_smiles(rd: Chem.Mol, *, generic: bool) -> Optional[str]:
    """Canonical SMILES for Murcko scaffold; generic=True uses carbon skeleton only."""
    try:
        from rdkit.Chem.Scaffolds import MurckoScaffold

        core = MurckoScaffold.GetScaffoldForMol(rd)
        if core is None:
            return None
        if generic:
            try:
                MurckoScaffold.MakeScaffoldGeneric(core)
            except Exception:
                pass
        return Chem.MolToSmiles(core, canonical=True)
    except Exception:
        return None


def _canonical_mol_smiles(rd: Chem.Mol) -> Optional[str]:
    """Canonical SMILES for the full molecule (same graph as ``rd``)."""
    try:
        return Chem.MolToSmiles(rd, canonical=True)
    except Exception:
        return None


def _diversity_bucket_id(
    rd: Chem.Mol,
    *,
    diversity_bucket: str,
    generic_scaffold: bool,
) -> Optional[str]:
    if diversity_bucket == "murcko":
        return _murcko_scaffold_smiles(rd, generic=generic_scaffold)
    if diversity_bucket == "canonical_smiles":
        return _canonical_mol_smiles(rd)
    raise ValueError(
        f"diversity_bucket must be 'murcko' or 'canonical_smiles', got {diversity_bucket!r}"
    )


def scaffold_diversity_wrapper(
    base_reward_fn: Callable,
    *,
    bucket_size: int = 10,
    penalty: float = 0.0,
    generic_scaffold: bool = True,
    diversity_bucket: str = "murcko",
    window_batches: int | None = None,
    memory: ScaffoldBucketMemory | None = None,
) -> Callable:
    """Wrap any reward with REINVENT-style occurrence bucketing.

    ``diversity_bucket``:
      - ``murcko``: Bemis–Murcko scaffold (see ``generic_scaffold`` /
        ``--scaffold_labeled``).
      - ``canonical_smiles``: full molecule canonical SMILES; labeled/generic
        scaffold flags are ignored.

    Computes the base ``(tensor, diagnostics)`` first, then multiplies per-graph
    rewards by ``penalty`` (default 0 = hard zero) when that key's bucket is
    full. Invalid molecules already have reward 0 from typical bases and stay 0.

    Diagnostics from the base function are copied; ``scaffold_*`` keys are added.
    A second pass over ``_iter_valid_mols`` extracts the bucket id (Murcko or SMILES).
    """
    if bucket_size < 1:
        raise ValueError(f"bucket_size must be >= 1, got {bucket_size}")
    if diversity_bucket not in ("murcko", "canonical_smiles"):
        raise ValueError(
            "diversity_bucket must be 'murcko' or 'canonical_smiles', "
            f"got {diversity_bucket!r}",
        )
    mem = memory if memory is not None else ScaffoldBucketMemory(window_batches=window_batches)

    def wrapped(module, trajectory) -> tuple[torch.Tensor, dict[str, float]]:
        r, diag = base_reward_fn(module, trajectory)
        diag_out = dict(diag)

        batch_scaffolds: dict[str, int] = {}
        n_valid = 0
        n_penalized = 0
        n_scaffold_fail = 0
        mask = torch.ones_like(r, dtype=r.dtype, device=r.device)

        for idx, (rd, ok) in enumerate(_iter_valid_mols(module, trajectory)):
            if not ok or rd is None:
                continue
            n_valid += 1

            bucket_id = _diversity_bucket_id(
                rd,
                diversity_bucket=diversity_bucket,
                generic_scaffold=generic_scaffold,
            )
            if bucket_id is None:
                n_scaffold_fail += 1
                continue

            total_seen = mem.total_seen(bucket_id)
            batch_seen = batch_scaffolds.get(bucket_id, 0)
            if total_seen + batch_seen >= bucket_size:
                mask[idx] = penalty
                n_penalized += 1
            else:
                batch_scaffolds[bucket_id] = batch_seen + 1

        mem.commit_batch(batch_scaffolds)

        r_gated = r * mask

        diag_out["scaffold_penalty_frac"] = (
            (n_penalized / n_valid) if n_valid > 0 else 0.0
        )
        diag_out["scaffold_unique_in_batch"] = float(len(batch_scaffolds))
        diag_out["scaffold_extract_fail_frac"] = (
            (n_scaffold_fail / n_valid) if n_valid > 0 else 0.0
        )
        # 0 = murcko, 1 = canonical_smiles (constant per run; for filtering W&B runs).
        diag_out["scaffold_diversity_key_id"] = (
            0.0 if diversity_bucket == "murcko" else 1.0
        )
        diag_out["scaffold_reward_mean_pre"] = float(r.mean())
        diag_out["scaffold_reward_mean_post"] = float(r_gated.mean())

        return r_gated, diag_out

    return wrapped


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
