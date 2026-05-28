"""Diversity wrappers for GRPO reward functions."""

from __future__ import annotations

from collections import deque
from typing import Callable, Optional

import torch
from rdkit import Chem

from chemflow.dataset.representation import Representation

from .common import _iter_valid_mols
from .spec import WrapperSpec


class ScaffoldBucketMemory:
    """Counts Murcko scaffold or canonical-SMILES occurrences for diversity gating.

    If ``window_batches`` is None, counts never expire (full-run memory like REINVENT).
    If set to ``N > 0``, only occurrences from the last ``N`` committed batches
    contribute; older batches are forgotten when new ones arrive.
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
        """Copy of current bucket id -> occurrence counts (for debugging)."""
        return dict(self._counts)

    def commit_batch(self, bucket_counts: dict[str, int]) -> None:
        """Add bucket occurrence counts from a finished batch (accepted slots only)."""
        increment = {k: v for k, v in bucket_counts.items() if v > 0}
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

    def total_seen(self, bucket_id: str) -> int:
        return self._counts.get(bucket_id, 0)


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
      - ``murcko``: Bemis-Murcko scaffold (see ``generic_scaffold`` /
        ``rl.reward.scaffold_labeled``).
      - ``canonical_smiles``: full molecule canonical SMILES; labeled/generic
        scaffold flags are ignored.

    Computes the base ``(tensor, diagnostics)`` first, then multiplies per-graph
    rewards by ``penalty`` (default 0 = hard zero) when that key's bucket is
    full. Invalid molecules already have reward 0 from typical bases and stay 0.
    """
    if bucket_size < 1:
        raise ValueError(f"bucket_size must be >= 1, got {bucket_size}")
    if diversity_bucket not in ("murcko", "canonical_smiles"):
        raise ValueError(
            "diversity_bucket must be 'murcko' or 'canonical_smiles', "
            f"got {diversity_bucket!r}",
        )
    # Negative window = full-run memory (mirrors the old config flag's contract).
    win = None if (window_batches is not None and window_batches < 0) else window_batches
    mem = memory if memory is not None else ScaffoldBucketMemory(window_batches=win)

    def wrapped(module, trajectory) -> tuple[torch.Tensor, dict[str, float]]:
        r, diag = base_reward_fn(module, trajectory)
        diag_out = dict(diag)

        n_valid = 0
        n_penalized = 0
        n_scaffold_fail = 0
        mask = torch.ones_like(r, dtype=r.dtype, device=r.device)
        bucket_to_indices: dict[str, list[int]] = {}

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
            bucket_to_indices.setdefault(bucket_id, []).append(idx)

        batch_buckets: dict[str, int] = {}
        for bucket_id, indices in bucket_to_indices.items():
            available = bucket_size - mem.total_seen(bucket_id)
            if available <= 0:
                accepted: set[int] = set()
            elif available >= len(indices):
                accepted = set(indices)
            else:
                perm = torch.randperm(len(indices), device=r.device)[:available]
                accepted = {indices[int(i)] for i in perm.detach().cpu().tolist()}

            accepted_count = 0
            for idx in indices:
                if idx in accepted:
                    accepted_count += 1
                else:
                    mask[idx] = penalty
                    n_penalized += 1
            if accepted_count > 0:
                batch_buckets[bucket_id] = accepted_count

        mem.commit_batch(batch_buckets)

        r_gated = r * mask

        diag_out["scaffold_penalty_frac"] = (
            (n_penalized / n_valid) if n_valid > 0 else 0.0
        )
        diag_out["scaffold_unique_in_batch"] = float(len(batch_buckets))
        diag_out["scaffold_extract_fail_frac"] = (
            (n_scaffold_fail / n_valid) if n_valid > 0 else 0.0
        )
        diag_out["scaffold_diversity_key_id"] = (
            0.0 if diversity_bucket == "murcko" else 1.0
        )
        diag_out["scaffold_reward_mean_pre"] = float(r.mean())
        diag_out["scaffold_reward_mean_post"] = float(r_gated.mean())

        return r_gated, diag_out

    return wrapped


# Scaffold bucketing reads Murcko scaffolds / canonical SMILES off RDKit
# molecules, so it only applies in full-chemistry mode.
SCAFFOLD_DIVERSITY_SPEC = WrapperSpec(
    make=scaffold_diversity_wrapper,
    supported_representations=frozenset({Representation.MOLECULE}),
)
