"""A dataset for editing molecules."""

import torch

<<<<<<< HEAD
from collections import deque
=======
>>>>>>> 34234e8 (MCS OT)
from rdkit import Chem
from rdkit.Chem import RWMol
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import rdFMCS

from chemflow.utils.rdkit import IDX_BOND_MAP


def compute_scaffold_groups(dataset):
    """Compute Bemis-Murcko scaffold groups for a dataset.

    Molecules with empty scaffolds (acyclic) or disconnected scaffolds
    (multiple ring systems, indicated by "." in SMILES) are dropped.

    Returns:
        mol_to_group: np.ndarray[N] of int64, -1 for dropped molecules
        groups: list[list[int]], group_id -> list of dataset indices
    """
    scaffold_to_group = {}
    mol_to_group = torch.full((len(dataset),), -1, dtype=torch.long)
    groups = []

    for idx in range(len(dataset)):
        data = dataset.get(idx)
        mol = Chem.MolFromSmiles(data.smiles)
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        smi = Chem.MolToSmiles(scaffold)

        # Drop empty scaffold (acyclic) and disconnected scaffolds
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
    """For each molecule, return ALL scaffold atom index matches (handles symmetry).

    Uses GetSubstructMatches(uniquify=False) to capture all automorphisms of the
    scaffold within the molecule. Returns a list of matches per molecule; each
    match is a tuple of atom indices. Returns an empty list for molecules with
    no valid scaffold.
    """
    result = []
    for idx in range(len(dataset)):
        data = dataset.get(idx)
        # Reconstruct the SDF molecule (with explicit Hs, SDF atom ordering) so
        # that the returned atom indices are in the same space as MoleculeData.x.
        # Using Chem.MolFromSmiles(data.smiles) would give canonical-SMILES
        # heavy-atom ordering — a different permutation — causing wrong scaffold pairs.
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
    """For each molecule, compute per-atom non-H non-scaffold neighbor counts.

    Returns list[torch.Tensor | None]: for each molecule, a 1D LongTensor of
    shape [N_atoms] where entry i = number of non-H, non-scaffold neighbors of
    atom i (0 for non-scaffold atoms and atoms with no decoration).
    Returns None for molecules with no scaffold matches.
    """
    result = []
    for idx in range(len(dataset)):
        matches = scaffold_atom_indices[idx]
        if not matches:
            result.append(None)
            continue
        data = dataset.get(idx)
        scaffold_set = set(matches[0])  # same atom set for all automorphisms
        N = data.z.shape[0]
        is_heavy = (data.z != 1)
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
    """For each molecule: {scaffold_atom_mol_idx -> [substituent atom mol indices]}.

    BFS from each scaffold atom's non-scaffold neighbors through the non-scaffold
    subgraph. Each non-scaffold atom is owned by the lowest-indexed scaffold atom
    whose BFS reaches it first (deterministic, automorphism-invariant).
    Returns None for molecules with no valid scaffold match.
    """
    result = []
    for idx in range(len(dataset)):
        matches = scaffold_atom_indices[idx]
        if not matches:
            result.append(None)
            continue
        data = dataset.get(idx)
        N = data.z.shape[0]
        scaffold_set = set(matches[0])  # same atom set for all automorphisms

        # Build undirected adjacency list
        adj = {i: [] for i in range(N)}
        for k in range(data.edge_index.shape[1]):
            src = int(data.edge_index[0, k])
            dst = int(data.edge_index[1, k])
            adj[src].append(dst)

        subs = {s: [] for s in scaffold_set}
        owned = {}  # non_scaffold_atom -> owning scaffold atom

        for s in sorted(scaffold_set):  # sorted for determinism
            queue = deque()
            for nbr in adj[s]:
                if nbr not in scaffold_set and nbr not in owned:
                    owned[nbr] = s
                    subs[s].append(nbr)
                    queue.append(nbr)
            while queue:
                node = queue.popleft()
                for nbr in adj[node]:
                    if nbr not in scaffold_set and nbr not in owned:
                        owned[nbr] = s
                        subs[s].append(nbr)
                        queue.append(nbr)

        result.append(subs)
    return result


def select_scaffold_pairs_by_neighbor_count(
    src_dec: torch.Tensor,
    src_matches: list,
    tgt_dec: torch.Tensor,
    tgt_matches: list,
) -> list:
    """Pick scaffold automorphism pair minimising L1 of decoration-count signatures.

    Each scaffold atom's signature = number of non-H, non-scaffold neighbors
    (decoration attachment count). Selects the (src_automorphism, tgt_automorphism)
    pair with minimum sum of |src_sig[k] - tgt_sig[k]| over scaffold positions k.

    Args:
        src_dec: 1D LongTensor [N_src_atoms], precomputed decoration counts.
        src_matches: list of automorphism tuples for source scaffold.
        tgt_dec: 1D LongTensor [N_tgt_atoms], precomputed decoration counts.
        tgt_matches: list of automorphism tuples for target scaffold.

    Returns list of (src_atom_idx, tgt_atom_idx) pairs, or [] if inputs empty.
    """
    if not src_matches or not tgt_matches:
        return []
    src_idx_t = torch.tensor(
        [list(sm) for sm in src_matches], dtype=torch.long
    )  # (n_src, K)
    tgt_idx_t = torch.tensor(
        [list(tm) for tm in tgt_matches], dtype=torch.long
    )  # (n_tgt, K)
    src_sig_mat = src_dec[src_idx_t]   # (n_src, K)
    tgt_sig_mat = tgt_dec[tgt_idx_t]   # (n_tgt, K)
    # scores[i, j] = L1 distance between sig vectors for match pair (i, j)
    scores = (src_sig_mat[:, None, :] - tgt_sig_mat[None, :, :]).abs().sum(dim=-1)
    print("Score matrix:\n", scores)
    best = scores.argmin()
    best_src = src_matches[best // len(tgt_matches)]
    best_tgt = tgt_matches[best  % len(tgt_matches)]
    return list(zip(best_src, best_tgt, strict=True))


def compute_scaffold_groups(dataset):
    """Compute Bemis-Murcko scaffold groups for a dataset.

    Molecules with empty scaffolds (acyclic) or disconnected scaffolds
    (multiple ring systems, indicated by "." in SMILES) are dropped.

    Returns:
        mol_to_group: np.ndarray[N] of int64, -1 for dropped molecules
        groups: list[list[int]], group_id -> list of dataset indices
    """
    scaffold_to_group = {}
    mol_to_group = torch.full((len(dataset),), -1, dtype=torch.long)
    groups = []

    for idx in range(len(dataset)):
        data = dataset.get(idx)
        mol = Chem.MolFromSmiles(data.smiles)
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        smi = Chem.MolToSmiles(scaffold)

        # Drop empty scaffold (acyclic) and disconnected scaffolds
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
    """For each molecule, return ALL scaffold atom index matches (handles symmetry).

    Uses GetSubstructMatches(uniquify=False) to capture all automorphisms of the
    scaffold within the molecule. Returns a list of matches per molecule; each
    match is a tuple of atom indices. Returns an empty list for molecules with
    no valid scaffold.
    """
    result = []
    for idx in range(len(dataset)):
        data = dataset.get(idx)
        mol = Chem.MolFromSmiles(data.smiles)
        if mol is None:
            result.append([])
            continue
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        matches = mol.GetSubstructMatches(scaffold, uniquify=False)
        result.append(list(matches))
    return result


def sort_by_scaffold(data_list):
    scaffold_groups = {}

    for idx, data in enumerate(data_list):
        smiles = data.smiles

        scaffold = Chem.MolFromSmiles(smiles)

        # get Bemis-Murcko scaffold
        scaffold = MurckoScaffold.GetScaffoldForMol(scaffold)

        if scaffold not in scaffold_groups:
            scaffold_groups[scaffold] = []

        scaffold_groups[scaffold].append(idx)

    return scaffold_groups


def get_mcs_atom_mapping(smiles1, smiles2):
    """
    Finds the Maximum Common Substructure between two SMILES strings
    and returns the atom mapping as a list of index pairs.
    """
    # 1. Convert SMILES to RDKit molecule objects
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)

    mol1 = Chem.AddHs(mol1)
    mol2 = Chem.AddHs(mol2)

    if not mol1 or not mol2:
        raise ValueError("One or both SMILES strings are invalid.")

    # 2. Find the Maximum Common Substructure (MCS)
    # Default parameters match exact atom types and bond types.
    mcs_result = rdFMCS.FindMCS([mol1, mol2])

    if not mcs_result.smartsString:
        return []  # No common structure found

    # 3. Create a query molecule from the MCS SMARTS string
    mcs_query = Chem.MolFromSmarts(mcs_result.smartsString)

    # 4. Find the matching atom indices in both molecules
    # GetSubstructMatch returns a tuple of atom indices.
    # The i-th index in the tuple corresponds to the i-th atom in the mcs_query.
    match1 = mol1.GetSubstructMatch(mcs_query)
    match2 = mol2.GetSubstructMatch(mcs_query)

    # 5. Zip the matches together to pair the corresponding atoms
    # This automatically excludes any atoms not present in the MCS.
    atom_mapping = list(zip(match1, match2))

    return atom_mapping
