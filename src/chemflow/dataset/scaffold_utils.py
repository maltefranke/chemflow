"""Scaffold-decoration / molecule-optimization data utilities.

Bemis-Murcko scaffold grouping, atom-index matching, and pair-selection helpers
used by the scaffold-aware flow-matching wrappers. Kept in its own module so
the standard unconditional / property-conditioned flow stays free of scaffold
concerns.
"""

from collections import deque

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import RWMol
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.optimize import linear_sum_assignment

from chemflow.utils.rdkit_utils import IDX_BOND_MAP
from chemflow.utils.utils import rigid_alignment


def compute_scaffold_groups(dataset):
    """Group dataset indices by Bemis-Murcko scaffold SMILES.

    Molecules with empty (acyclic) or disconnected scaffolds are dropped:
    ``mol_to_group[i] == -1`` for those.

    Returns:
        mol_to_group: LongTensor[N], -1 for dropped molecules.
        groups: list[list[int]], group_id -> list of dataset indices.
    """
    scaffold_to_group: dict[str, int] = {}
    mol_to_group = torch.full((len(dataset),), -1, dtype=torch.long)
    groups: list[list[int]] = []

    for idx in range(len(dataset)):
        data = dataset.get(idx)
        mol = Chem.MolFromSmiles(data.smiles)
        if mol is None:
            continue
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        smi = Chem.MolToSmiles(scaffold)
        if not smi or "." in smi:
            continue

        if smi not in scaffold_to_group:
            scaffold_to_group[smi] = len(groups)
            groups.append([])

        gid = scaffold_to_group[smi]
        mol_to_group[idx] = gid
        groups[gid].append(idx)

    return mol_to_group, groups


def compute_scaffold_atom_indices(dataset) -> list[list[tuple[int, ...]]]:
    """For each molecule, return ALL scaffold atom-index matches.

    Uses ``GetSubstructMatches(uniquify=False)`` to capture every automorphism
    of the scaffold inside the molecule. The returned indices are in the same
    space as the SDF atom order (i.e. as ``MoleculeData.x``), achieved by
    reconstructing an RWMol from ``data.z`` / ``data.edge_index`` /
    ``data.edge_attr`` instead of going through canonical SMILES.

    Returns an empty list for molecules with no valid scaffold or RDKit
    sanitization failures.
    """
    result: list[list[tuple[int, ...]]] = []
    for idx in range(len(dataset)):
        data = dataset.get(idx)
        rwmol = RWMol()
        for z_val in data.z.tolist():
            rwmol.AddAtom(Chem.Atom(int(z_val)))
        for k in range(data.edge_index.shape[1]):
            i, j = int(data.edge_index[0, k]), int(data.edge_index[1, k])
            if i < j:
                rwmol.AddBond(i, j, IDX_BOND_MAP[int(data.edge_attr[k])])
        try:
            mol = rwmol.GetMol()
            Chem.SanitizeMol(mol)
        except Exception:
            result.append([])
            continue
        scaffold_smi = Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))
        if not scaffold_smi or "." in scaffold_smi:
            result.append([])
            continue
        canonical_scaffold = Chem.MolFromSmiles(scaffold_smi)
        if canonical_scaffold is None:
            result.append([])
            continue
        matches = mol.GetSubstructMatches(canonical_scaffold, uniquify=False)
        result.append(list(matches))
    return result


def compute_scaffold_decoration_counts(dataset, scaffold_atom_indices):
    """Per-atom non-H non-scaffold neighbour count for each molecule.

    Returns list[Tensor | None]: a 1D LongTensor of shape ``[N_atoms]`` per
    molecule. Entry ``i`` is the number of heavy, non-scaffold neighbours of
    atom ``i`` (0 for non-scaffold atoms and scaffold atoms with no
    decoration). ``None`` for molecules with no scaffold match.
    """
    result = []
    for idx in range(len(dataset)):
        matches = scaffold_atom_indices[idx]
        if not matches:
            result.append(None)
            continue
        data = dataset.get(idx)
        scaffold_set = set(matches[0])  # atom *set* is invariant across automorphisms
        N = data.z.shape[0]
        is_heavy = data.z != 1
        is_scaffold = torch.zeros(N, dtype=torch.bool)
        is_scaffold[list(scaffold_set)] = True
        if data.edge_index.numel() == 0:
            result.append(torch.zeros(N, dtype=torch.long))
            continue
        src_nodes = data.edge_index[0]
        dst_nodes = data.edge_index[1]
        contrib = (is_heavy[dst_nodes] & ~is_scaffold[dst_nodes]).long()
        counts = torch.zeros(N, dtype=torch.long)
        counts.scatter_add_(0, src_nodes, contrib)
        result.append(counts)
    return result


def compute_scaffold_substituents(dataset, scaffold_atom_indices):
    """For each molecule: ``{scaffold_atom_idx -> list of residue branches}``.

    Each branch is a list of atom indices rooted at one direct non-scaffold
    neighbour of the scaffold atom. BFS ownership is deterministic: each
    non-scaffold atom is owned by the lowest-indexed scaffold atom whose BFS
    reaches it first.

    Returns ``None`` for molecules with no valid scaffold match.
    """
    result = []
    for idx in range(len(dataset)):
        matches = scaffold_atom_indices[idx]
        if not matches:
            result.append(None)
            continue
        data = dataset.get(idx)
        N = data.z.shape[0]
        scaffold_set = set(matches[0])

        adj: dict[int, list[int]] = {i: [] for i in range(N)}
        for k in range(data.edge_index.shape[1]):
            src = int(data.edge_index[0, k])
            dst = int(data.edge_index[1, k])
            adj[src].append(dst)

        subs: dict[int, list[list[int]]] = {s: [] for s in scaffold_set}
        owned: dict[int, int] = {}

        for s in sorted(scaffold_set):
            for seed in sorted(adj[s]):
                if seed in scaffold_set or seed in owned:
                    continue
                branch: list[int] = [seed]
                owned[seed] = s
                queue = deque([seed])
                while queue:
                    node = queue.popleft()
                    for nbr in adj[node]:
                        if nbr not in scaffold_set and nbr not in owned:
                            owned[nbr] = s
                            branch.append(nbr)
                            queue.append(nbr)
                subs[s].append(branch)

        result.append(subs)
    return result


def select_scaffold_pairs_by_neighbor_count(
    src_dec: torch.Tensor,
    src_matches: list,
    tgt_dec: torch.Tensor,
    tgt_matches: list,
) -> list:
    """Pick the (src_auto, tgt_auto) pair minimising L1 of decoration-count signatures."""
    if not src_matches or not tgt_matches:
        return []
    src_idx_t = torch.tensor([list(sm) for sm in src_matches], dtype=torch.long)
    tgt_idx_t = torch.tensor([list(tm) for tm in tgt_matches], dtype=torch.long)
    src_sig_mat = src_dec[src_idx_t]
    tgt_sig_mat = tgt_dec[tgt_idx_t]
    scores = (src_sig_mat[:, None, :] - tgt_sig_mat[None, :, :]).abs().sum(dim=-1)
    best = scores.argmin()
    best_src = src_matches[best // len(tgt_matches)]
    best_tgt = tgt_matches[best % len(tgt_matches)]
    return list(zip(best_src, best_tgt, strict=True))


def select_scaffold_pairs_spatially(
    src_matches: list,
    tgt_matches: list,
    x0,
    x1,
    src_subs,
    tgt_subs,
    c_ins: float = 500.0,
) -> list:
    """Pick (src_auto, tgt_auto) by minimising Kabsch-aligned branch-root cost.

    For each candidate pair, Kabsch-aligns ``x0[src_auto]`` to ``x1[tgt_auto]``
    and sums Hungarian-matched branch-root distances across scaffold positions.
    Falls back gracefully when substituents are absent.
    """
    if not src_matches or not tgt_matches:
        return []

    best_cost = float("inf")
    best_src = src_matches[0]
    best_tgt = tgt_matches[0]

    for src_auto in src_matches:
        for tgt_auto in tgt_matches:
            pts_s = torch.tensor(x0[list(src_auto)], dtype=torch.float32)
            pts_t = torch.tensor(x1[list(tgt_auto)], dtype=torch.float32)
            if pts_s.shape[0] >= 3:
                R, t = rigid_alignment(pts_s, pts_t)
                R_np = R.detach().cpu().numpy()
                t_np = t.detach().cpu().numpy()
            else:
                R_np = np.eye(3, dtype=np.float32)
                t_np = np.zeros(3, dtype=np.float32)

            cost = 0.0
            for src_atom, tgt_atom in zip(src_auto, tgt_auto):
                sb = (src_subs or {}).get(src_atom, [])
                tb = (tgt_subs or {}).get(tgt_atom, [])
                if not sb or not tb:
                    continue
                s_roots = np.array([x0[b[0]] for b in sb], dtype=np.float32)
                s_roots = s_roots @ R_np.T + t_np
                t_roots = np.array([x1[b[0]] for b in tb], dtype=np.float32)
                s_sizes = np.array([len(b) for b in sb], dtype=np.float32)
                t_sizes = np.array([len(b) for b in tb], dtype=np.float32)
                cost_mat = (
                    np.linalg.norm(s_roots[:, None, :] - t_roots[None, :, :], axis=-1)
                    + c_ins * np.abs(s_sizes[:, None] - t_sizes[None, :])
                )
                ri, ci = linear_sum_assignment(cost_mat)
                cost += cost_mat[ri, ci].sum()

            if cost < best_cost:
                best_cost = cost
                best_src = src_auto
                best_tgt = tgt_auto

    return list(zip(best_src, best_tgt, strict=True))


def align_scaffold_bond_types(source, target, scaffold_pairs):
    """Rewrite ``source.e`` on scaffold-internal edges to match ``target.e``.

    Source and target share a Bemis-Murcko scaffold, but the dataset stores
    bonds in kekulized form. Two molecules can pick different Kekulé tautomers
    for the same aromatic ring (e.g. benzene as S-D-S-D-S-D vs D-S-D-S-D-S),
    which makes the scaffold edges disagree even after atoms are matched. The
    interpolator then schedules bogus bond substitutions across "preserved"
    scaffold bonds, polluting the loss and the eval comparison.

    Given ``scaffold_pairs`` of ``(src_atom, tgt_atom)`` tuples, this returns
    a clone of ``source`` whose scaffold-internal edge types are copied from
    the matching edges in ``target`` via the atom mapping. Non-scaffold edges
    are left untouched. ``target`` is returned unchanged.
    """
    if not scaffold_pairs:
        return source

    src_to_tgt = {int(s): int(t) for s, t in scaffold_pairs}
    src_scaffold = set(src_to_tgt.keys())
    tgt_scaffold = set(src_to_tgt.values())

    tgt_bond_at: dict[tuple[int, int], int] = {}
    for k in range(target.edge_index.shape[1]):
        i = int(target.edge_index[0, k])
        j = int(target.edge_index[1, k])
        if i in tgt_scaffold and j in tgt_scaffold:
            tgt_bond_at[(i, j)] = int(target.e[k])

    if not tgt_bond_at:
        return source

    new_e = source.e.clone()
    for k in range(source.edge_index.shape[1]):
        s_i = int(source.edge_index[0, k])
        s_j = int(source.edge_index[1, k])
        if s_i in src_scaffold and s_j in src_scaffold:
            bond = tgt_bond_at.get((src_to_tgt[s_i], src_to_tgt[s_j]))
            if bond is not None:
                new_e[k] = bond

    source = source.clone()
    source.e = new_e
    return source


def _substituents_need_rebuild(subs) -> bool:
    """True if scaffold_substituents key is absent or in the old flat list[int] format."""
    if subs is None:
        return True
    for mol_subs in subs:
        if mol_subs is None:
            continue
        for branches in mol_subs.values():
            return bool(branches) and not isinstance(branches[0], list)
    return False
