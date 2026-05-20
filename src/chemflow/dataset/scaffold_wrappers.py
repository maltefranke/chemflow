"""Scaffold-aware flow-matching dataset wrapper.

A single class :class:`FlowMatchingDatasetWrapperScaffold` covers two
distinct tasks, selected via the ``mode`` argument:

- ``mode="molecule_optimization"`` — Source AND target are *complete*
  molecules that happen to share the same Bemis-Murcko scaffold. The model
  learns to edit one molecule into another by changing its substituents
  (atom substitutions, insertions, deletions on the decoration atoms only).
  The scaffold is preserved verbatim. (In Stefan's branch this was called
  ``FlowMatchingDatasetWrapperScaffoldDecoration``, even though the source
  already carries decorations — hence the rename here for clarity.)

- ``mode="scaffold_decoration"`` — Source is the *bare scaffold* of the
  target (just the scaffold heavy atoms, no decorations); target is the
  full molecule. The model learns to *grow* the molecule by inserting
  decoration atoms around the fixed scaffold. (In Stefan's branch this
  was called ``FlowMatchingDatasetWrapperScaffoldGrowth``.)

Both modes attach a ``scaffold_mask`` LongTensor of shape ``[N]`` to
``mol_t`` (training) or ``source`` (eval), with ones at scaffold atoms
and zeros at decoration atoms. The scaffold-aware interpolator places
scaffold atoms at the front of the node list, so
``scaffold_mask[:n_scaffold] == 1``.

Select via Hydra ``data.wrapper_kind: "molecule_optimization"`` or
``"scaffold_decoration"``; the field is threaded through to ``mode``.
"""

import os
import random

import torch

from chemflow.dataset.flow_matching_wrapper import FlowMatchingDatasetWrapper


def _qed_delta(source, target) -> torch.Tensor | None:
    """Return ``QED(target) - QED(source)`` as a ``[1, 1]`` float tensor.

    Reads the per-molecule ``qed`` scalar cached at dataset-load time
    (see ``FlowMatchingQM9Dataset``). Returns ``None`` if either side is
    missing the cache.
    """
    src_q = getattr(source, "qed", None)
    tgt_q = getattr(target, "qed", None)
    if src_q is None or tgt_q is None:
        return None
    return (tgt_q.float() - src_q.float()).view(1, 1)
from chemflow.dataset.scaffold_utils import (
    _substituents_need_rebuild,
    align_scaffold_bond_types,
    compute_scaffold_atom_indices,
    compute_scaffold_decoration_counts,
    compute_scaffold_groups,
    compute_scaffold_substituents,
    select_scaffold_pairs_spatially,
)
from chemflow.utils.utils import build_fully_connected_edge_index


def _load_or_build_scaffold_cache(
    base_dataset,
    scaffold_groups_dir: str,
    stage: str,
    *,
    need_decoration_counts: bool,
    need_substituents: bool,
) -> dict:
    """Load the persisted scaffold cache or build it lazily.

    Cached under ``scaffold_groups_dir/scaffold_groups_<stage>.pt``. Each
    required sub-cache is built on demand and the file is re-saved so a single
    run pays the (expensive) RDKit cost once.
    """
    if scaffold_groups_dir is None:
        raise ValueError(
            "scaffold_groups_dir must be provided for scaffold-aware wrappers"
        )
    path = os.path.join(scaffold_groups_dir, f"scaffold_groups_{stage}.pt")

    if os.path.exists(path):
        scaffold_groups = torch.load(path, weights_only=False)
    else:
        mol_to_group, groups = compute_scaffold_groups(base_dataset)
        scaffold_groups = {"mol_to_group": mol_to_group, "groups": groups}
        os.makedirs(scaffold_groups_dir, exist_ok=True)
        torch.save(scaffold_groups, path)

    if "scaffold_atom_indices" not in scaffold_groups or not isinstance(
        scaffold_groups["scaffold_atom_indices"][0], list
    ):
        scaffold_groups["scaffold_atom_indices"] = compute_scaffold_atom_indices(
            base_dataset
        )
        torch.save(scaffold_groups, path)

    if need_decoration_counts and "scaffold_decoration_counts" not in scaffold_groups:
        scaffold_groups["scaffold_decoration_counts"] = (
            compute_scaffold_decoration_counts(
                base_dataset, scaffold_groups["scaffold_atom_indices"]
            )
        )
        torch.save(scaffold_groups, path)

    if need_substituents and _substituents_need_rebuild(
        scaffold_groups.get("scaffold_substituents")
    ):
        scaffold_groups["scaffold_substituents"] = compute_scaffold_substituents(
            base_dataset, scaffold_groups["scaffold_atom_indices"]
        )
        torch.save(scaffold_groups, path)

    return scaffold_groups


_MODES = ("molecule_optimization", "scaffold_decoration")


class FlowMatchingDatasetWrapperScaffold(FlowMatchingDatasetWrapper):
    """Unified scaffold-aware wrapper, switchable between two tasks via ``mode``.

    Mode ``"molecule_optimization"`` (full → full):
        Source is sampled from the pool of same-scaffold molecules and
        carries its own decorations. The model learns to edit one
        molecule's substituents into another's while preserving the
        shared scaffold. Requires ≥2 valid molecules per scaffold group.

    Mode ``"scaffold_decoration"`` (bare → full):
        Source is the target's own bare scaffold (only the scaffold heavy
        atoms, no decorations). The model learns to *grow* the full
        molecule by inserting decoration atoms around the fixed scaffold.

    Args:
        mode: ``"molecule_optimization"`` or ``"scaffold_decoration"``.
        assignment_method: Used only by ``molecule_optimization`` — either
            ``"mcs_constrained"`` (default) or ``"substituent"`` for
            branch-aware OT on the decoration atoms.
        compute_qed_delta: Only meaningful for ``molecule_optimization``;
            silently disabled in ``scaffold_decoration`` since a bare
            scaffold has no QED. When ``True``, attaches
            ``target_props = [[QED(target) - QED(source)]]`` to ``mol_t``
            (train) / ``source`` (eval).
    """

    def __init__(
        self,
        base_dataset,
        distributions,
        interpolator,
        mode: str = "molecule_optimization",
        n_atoms_strategy="flexible",
        time_dist=None,
        stage="train",
        rotate: bool = False,
        n_augmentations: int = 1,
        scaffold_groups_dir: str | None = None,
        assignment_method: str = "mcs_constrained",
        compute_qed_delta: bool = False,
    ):
        super().__init__(
            base_dataset,
            distributions,
            interpolator,
            n_atoms_strategy=n_atoms_strategy,
            time_dist=time_dist,
            stage=stage,
            rotate=rotate,
            n_augmentations=n_augmentations,
        )
        if mode not in _MODES:
            raise ValueError(f"mode must be one of {_MODES}, got {mode!r}.")
        self._mode = mode
        self._assignment_method = assignment_method
        # QED delta is only meaningful when source is a full molecule.
        self._compute_qed_delta = compute_qed_delta and mode == "molecule_optimization"

        # Cache loading: molecule_optimization needs decoration counts for
        # spatial pair selection; scaffold_decoration does not.
        scaffold_groups = _load_or_build_scaffold_cache(
            base_dataset,
            scaffold_groups_dir,
            stage,
            need_decoration_counts=(mode == "molecule_optimization"),
            need_substituents=True,
        )
        self._scaffold_atom_indices = scaffold_groups["scaffold_atom_indices"]
        self._scaffold_substituents = scaffold_groups["scaffold_substituents"]

        if mode == "molecule_optimization":
            mol_to_group = scaffold_groups["mol_to_group"]
            groups = scaffold_groups["groups"]
            self._scaffold_decoration_counts = scaffold_groups[
                "scaffold_decoration_counts"
            ]
            filtered_groups = [
                [i for i in grp if self._scaffold_atom_indices[i]] for grp in groups
            ]
            # Need ≥2 valid members in the group (one target, one source).
            self._filtered_indices = [
                i
                for i in range(len(self.base_dataset))
                if mol_to_group[i] >= 0
                and self._scaffold_atom_indices[i]
                and len(filtered_groups[mol_to_group[i]]) >= 2
            ]
            self._mol_to_group = mol_to_group
            self._groups = filtered_groups
        else:  # scaffold_decoration
            self._filtered_indices = [
                i
                for i in range(len(self.base_dataset))
                if self._scaffold_atom_indices[i]
            ]
            self._mol_to_group = None
            self._groups = None

    def __len__(self):
        return len(self._filtered_indices)

    # ------------------------------------------------------------------
    # Mode-specific source selection
    # ------------------------------------------------------------------

    def _select_source(self, base_idx, target):
        """Build ``(source, scaffold_pairs, scaffold_substituents)`` for one target.

        ``scaffold_pairs`` is a list of ``(src_atom_idx, tgt_atom_idx)`` pairs
        passed straight to the scaffold-aware interpolator.
        ``scaffold_substituents`` is ``(src_subs, tgt_subs)`` or ``None``.
        """
        if self._mode == "molecule_optimization":
            gid = int(self._mol_to_group[base_idx])
            candidates = [i for i in self._groups[gid] if i != base_idx]
            source_idx = random.choice(candidates)
            source = self.base_dataset[source_idx]

            src_matches = self._scaffold_atom_indices[source_idx]
            tgt_matches = self._scaffold_atom_indices[base_idx]
            if src_matches and tgt_matches:
                pairs = select_scaffold_pairs_spatially(
                    src_matches,
                    tgt_matches,
                    source.x.detach().cpu().numpy(),
                    target.x.detach().cpu().numpy(),
                    self._scaffold_substituents[source_idx],
                    self._scaffold_substituents[base_idx],
                )
                source = align_scaffold_bond_types(source, target, pairs)
            else:
                pairs = []

            if (
                self._assignment_method == "substituent"
                and pairs
                and self._scaffold_substituents is not None
            ):
                subs = (
                    self._scaffold_substituents[source_idx],
                    self._scaffold_substituents[base_idx],
                )
            else:
                subs = None
            return source, pairs, subs

        # scaffold_decoration: source = target's bare scaffold.
        scaffold_indices = list(random.choice(self._scaffold_atom_indices[base_idx]))
        idx_tensor = torch.tensor(scaffold_indices, dtype=torch.long)
        source = target.get_permuted_subgraph(idx_tensor)
        # Identity-like mapping: source atom i ↔ target atom scaffold_indices[i].
        pairs = [(i, scaffold_indices[i]) for i in range(len(scaffold_indices))]
        return source, pairs, None

    def __getitem__(self, index):
        base_idx = self._filtered_indices[index]
        target = self.base_dataset[base_idx]

        if self.stage == "train":
            source, scaffold_pairs, scaffold_substituents = self._select_source(
                base_idx, target
            )

            t = self.time_dist.sample((1,)).squeeze(0)
            t = torch.clamp(t, min=0.0, max=1 - 1e-8)

            mol_t, mol_1, ins_targets = self.interpolator.interpolate_single(
                source,
                target,
                t.item(),
                scaffold_pairs=scaffold_pairs,
                scaffold_substituents=scaffold_substituents,
            )

            n_scaffold = len(scaffold_pairs) if scaffold_pairs else 0
            scaffold_mask = torch.zeros(mol_t.num_nodes, dtype=torch.long)
            scaffold_mask[:n_scaffold] = 1
            mol_t.scaffold_mask = scaffold_mask

            if hasattr(target, "y") and target.y is not None:
                mol_t.y = target.y
            if hasattr(target, "smiles"):
                mol_1.smiles = target.smiles
                mol_t.smiles = target.smiles

            if self._compute_qed_delta:
                delta = _qed_delta(source, target)
                if delta is not None:
                    mol_t.target_props = delta

            if self.n_augmentations <= 1:
                if self.rotate:
                    return self._augment(mol_t, mol_1, ins_targets, t)
                return mol_t, mol_1, ins_targets, t
            return [
                self._augment(mol_t, mol_1, ins_targets, t)
                for _ in range(self.n_augmentations)
            ]

        # --- eval / test --------------------------------------------------
        source, scaffold_pairs, _ = self._select_source(base_idx, target)

        # scaffold_mask in source-coordinate space.
        if self._mode == "molecule_optimization":
            # Source is a full molecule; mark its scaffold atoms (the source
            # side of the spatial-OT pairs).
            scaffold_indices = [int(s) for s, _ in scaffold_pairs] if scaffold_pairs else []
            if scaffold_indices:
                scaffold_mask = torch.zeros(source.num_nodes, dtype=torch.long)
                scaffold_mask[scaffold_indices] = 1
                source.scaffold_mask = scaffold_mask
        else:
            # Source is the bare scaffold, so all of its atoms are scaffold.
            source.scaffold_mask = torch.ones(source.num_nodes, dtype=torch.long)

        if self._compute_qed_delta:
            delta = _qed_delta(source, target)
            if delta is not None:
                source.target_props = delta

        # Inflate to fully-connected edge layout (mirrors base wrapper's eval contract).
        N = source.num_nodes
        full_ei = build_fully_connected_edge_index(N)
        dense = torch.zeros(N, N, dtype=source.e.dtype)
        dense[source.edge_index[0], source.edge_index[1]] = source.e
        source.e = dense[full_ei[0], full_ei[1]]
        source.edge_index = full_ei

        return source, target


