import numpy as np
import torch
from rdkit import Chem
from scipy.optimize import linear_sum_assignment
import hydra

from chemflow.dataset.data_utils import get_mcs_atom_mapping
from chemflow.dataset.molecule_data import (
    AugmentedMoleculeData,
    MoleculeBatch,
    MoleculeData,
    filter_nodes,
)
from chemflow.utils.utils import rigid_alignment


def distance_based_assignment(x0, x1):
    """
    Assign targets from x1 to x0 using the Hungarian algorithm.

    Args:
        x0 (np.ndarray): Shape (N_x0, D)
        x1 (np.ndarray): Shape (N_x1, D)

    Returns:
        tuple: A tuple of two arrays:
        - row_ind (np.ndarray): Shape (N_x0,)
        - col_ind (np.ndarray): Shape (N_x1,)
    """
    # Calculate the cost matrix *only* for valid items
    # (N_valid, 1, D) - (1, M_valid, D) => (N_valid, M_valid, D)
    cost_matrix_b = np.linalg.norm(x0[:, np.newaxis] - x1[np.newaxis, :], axis=2)
    # cost_matrix_b shape is (N_x0, N_x1)

    # Perform the assignment
    # row_ind and col_ind are indices *into* x0 and x1
    row_ind, col_ind = linear_sum_assignment(cost_matrix_b)
    # K_b = len(row_ind)

    return row_ind, col_ind


def pre_align_positions(x0, x1, num_iterations=5):
    """
    Pre-align the positions of x0 and x1 iterating linear sum assignment and the Kabsch algorithm.
    """
    for _ in range(num_iterations):
        row_ind, col_ind = distance_based_assignment(x0, x1)
        R, t = rigid_alignment(torch.tensor(x0[row_ind]), torch.tensor(x1[col_ind]))
        x0 = (torch.tensor(x0) @ R.T) + t
        x0 = x0.numpy()

    return x0, x1


def distance_and_class_based_assignment(
    x0,
    x1,
    a0,
    a1,
    c_move=1.0,
    c_sub=10.0,
    c_ins=5.0,
    c_del=10.0,
):
    """
    Assign targets from x1 to x0 using the Hungarian algorithm,
    handling batches with flexible graph sizes using batch_id.

    Special cases:
    1) Distance-based assignment
    c_move=const, c_sub=0, c_ins>>c_move

    2) Class-based assignment.
    Note: This will not be useful for our application!
    c_move=0, c_sub=const, c_ins>>c_sub

    3) Pure insertion-deletion assignment - no move.
    c_move>>c_ins, c_sub=const, c_ins=const

    Args:
        x0 (np.ndarray): Shape (N_x0, D)
        x1 (np.ndarray): Shape (N_x1, D)
        a0 (np.ndarray): Shape (N_x0, C)
        a1 (np.ndarray): Shape (N_x1, C)
        c_move (float): Distance cost weight
        c_sub (float): Class cost weight
        c_ins (float): Insertion cost weight

    Returns:
        tuple: A tuple of two arrays:
        - row_ind (np.ndarray): Shape (N_x0,)
        - col_ind (np.ndarray): Shape (N_x1,)
    """
    N = x0.shape[0]
    M = x1.shape[0]

    if N == 0 or M == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    # 1. Calculate Real Costs
    dist_matrix = np.linalg.norm(x0[:, np.newaxis] - x1[np.newaxis, :], axis=2)
    class_diff_matrix = (a0[:, np.newaxis] != a1[np.newaxis, :]).astype(float)
    cost_real = (c_move * dist_matrix) + (c_sub * class_diff_matrix)

    # 2. Construct Augmented Cost Matrix (N+M, N+M)
    total_size = N + M
    aug_cost = np.zeros((total_size, total_size))

    # [Top-Left] Real matches (N x M)
    aug_cost[:N, :M] = cost_real

    # [Top-Right] Deletion Cost (N x N)
    # A real point x0[i] matches a dummy target.
    # We use a very high value for off-diagonals so each point has its own dummy.
    del_block = np.full((N, N), fill_value=1e8)
    np.fill_diagonal(del_block, c_del)
    aug_cost[:N, M:] = del_block

    # [Bottom-Left] Insertion Cost (M x M)
    # A real point x1[j] matches a dummy source.
    ins_block = np.full((M, M), fill_value=1e8)
    np.fill_diagonal(ins_block, c_ins)
    aug_cost[N:, :M] = ins_block

    # [Bottom-Right] Unused Dummies (M x N)
    # This block must be 0.0 to allow dummies to pair with each other for "free"
    aug_cost[N:, M:] = 0.0

    # 3. Solve
    row_ind, col_ind = linear_sum_assignment(aug_cost)

    return row_ind, col_ind


def match_branches(
    i_s: int,
    i_t: int,
    src_branches: list[list[int]],
    tgt_branches: list[list[int]],
    x0: np.ndarray,
    x1: np.ndarray,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Match substituent branches by spatial direction after Kabsch alignment.

    For each branch the root atom (index 0, the direct non-scaffold neighbour)
    defines the branch position.  The cost of matching source branch ``bi`` to
    target branch ``bj`` is the Euclidean distance between their root atoms
    (``x0[src_branches[bi][0]]`` and ``x1[tgt_branches[bj][0]]``).

    Because ``x0`` has already been Kabsch-aligned to ``x1`` before this
    function is called, "up" in source directly corresponds to "up" in target.

    Args:
        i_s: Scaffold atom index in source (used for the root-position lookup).
        i_t: Scaffold atom index in target (used for the root-position lookup).
        src_branches: List of branches for scaffold atom ``i_s``; each branch
            is a list of atom indices with the root atom first.
        tgt_branches: List of branches for scaffold atom ``i_t``; each branch
            is a list of atom indices with the root atom first.
        x0: Source atom positions (already Kabsch-aligned), shape (N, 3).
        x1: Target atom positions, shape (M, 3).

    Returns:
        matched: List of (src_branch_idx, tgt_branch_idx) pairs.
        unmatched_src: Source branch indices with no target partner.
        unmatched_tgt: Target branch indices with no source partner.
    """
    n_s = len(src_branches)
    n_t = len(tgt_branches)

    if n_s == 0 and n_t == 0:
        return [], [], []
    if n_s == 0:
        return [], [], list(range(n_t))
    if n_t == 0:
        return [], list(range(n_s)), []

    # Cost matrix: distance between branch root positions
    cost = np.zeros((n_s, n_t), dtype=float)
    for bi, branch_s in enumerate(src_branches):
        for bj, branch_t in enumerate(tgt_branches):
            cost[bi, bj] = np.linalg.norm(
                x0[branch_s[0]] - x1[branch_t[0]]
            )

    # Hungarian assignment on the smaller of the two sides
    row_ind, col_ind = linear_sum_assignment(cost)

    matched = list(zip(row_ind.tolist(), col_ind.tolist()))
    matched_src = set(row_ind.tolist())
    matched_tgt = set(col_ind.tolist())
    unmatched_src = [i for i in range(n_s) if i not in matched_src]
    unmatched_tgt = [j for j in range(n_t) if j not in matched_tgt]
    return matched, unmatched_src, unmatched_tgt


def substituent_constrained_assignment(
    x0,
    x1,
    a0,
    a1,
    scaffold_pairs: list[tuple[int, int]],
    src_subs: dict[int, list[list[int]]],
    tgt_subs: dict[int, list[list[int]]],
    c_move=1.0,
    c_sub=10.0,
    c_ins=5.0,
    c_del=10.0,
):
    """Run per-scaffold-position OT on substituent groups with fixed scaffold pairs.

    For each scaffold atom pair, runs independent partial OT on the corresponding
    substituent groups. Non-scaffold atoms not in any substituent group are
    processed as an unconstrained partial OT fallback.

    Args:
        x0, x1: positions
        a0, a1: atom types
        scaffold_pairs: fixed backbone correspondences
        src_subs: dict[scaffold_atom_idx -> list of residue branches] for source;
            each branch is a list of atom indices with the root atom first
        tgt_subs: same structure for target
        c_move, c_sub, c_ins, c_del: OT cost weights
    """
    N = x0.shape[0]
    M = x1.shape[0]

    # Keep only valid, one-to-one scaffold pairs
    valid_pairs = []
    used_src = set()
    used_tgt = set()
    for i_s, i_t in scaffold_pairs:
        if not (0 <= i_s < N and 0 <= i_t < M):
            continue
        if i_s in used_src or i_t in used_tgt:
            continue
        valid_pairs.append((i_s, i_t))
        used_src.add(i_s)
        used_tgt.add(i_t)

    global_pairs: list[tuple[int, int]] = []

    # Phase 1: Fix scaffold real-real pairs
    for i_s, i_t in valid_pairs:
        global_pairs.append((i_s, i_t))

    # Phase 2: Pair corresponding scaffold dummies with each other
    for i_s, i_t in valid_pairs:
        global_pairs.append((N + i_t, M + i_s))

    # Helper: run partial OT between one matched branch pair and record results
    # in global_pairs.  Captures N, M, x0, x1, a0, a1 and cost weights from
    # the enclosing scope.
    def _ot_branch(sub0, sub1):
        n_u, m_u = len(sub0), len(sub1)
        if n_u == 0 and m_u == 0:
            return
        if n_u == 0:
            global_pairs.extend((N + j, j) for j in sub1)
            return
        if m_u == 0:
            global_pairs.extend((i, M + i) for i in sub0)
            return
        row_u, col_u = distance_and_class_based_assignment(
            x0[np.array(sub0)],
            x1[np.array(sub1)],
            a0[np.array(sub0)],
            a1[np.array(sub1)],
            c_move=c_move,
            c_sub=c_sub,
            c_ins=c_ins,
            c_del=c_del,
        )
        for r_loc, c_loc in zip(
            row_u.tolist(), col_u.tolist(), strict=True
        ):
            r_glob = sub0[r_loc] if r_loc < n_u else N + sub1[r_loc - n_u]
            c_glob = sub1[c_loc] if c_loc < m_u else M + sub0[c_loc - m_u]
            global_pairs.append((r_glob, c_glob))

    # Phase 3: Per-residue OT — match branches by spatial direction, then OT
    # within each matched branch pair.
    for i_s, i_t in valid_pairs:
        src_branches = src_subs.get(i_s, [])
        tgt_branches = tgt_subs.get(i_t, [])

        matched, unmatched_src_b, unmatched_tgt_b = match_branches(
            i_s, i_t, src_branches, tgt_branches, x0, x1
        )

        for bi_s, bi_t in matched:
            sub0 = [
                a for a in src_branches[bi_s] if a not in used_src
            ]
            sub1 = [
                b for b in tgt_branches[bi_t] if b not in used_tgt
            ]
            used_src.update(sub0)
            used_tgt.update(sub1)
            _ot_branch(sub0, sub1)

        for bi_s in unmatched_src_b:
            sub0 = [a for a in src_branches[bi_s] if a not in used_src]
            used_src.update(sub0)
            global_pairs.extend((a, M + a) for a in sub0)

        for bi_t in unmatched_tgt_b:
            sub1 = [b for b in tgt_branches[bi_t] if b not in used_tgt]
            used_tgt.update(sub1)
            global_pairs.extend((N + b, b) for b in sub1)

    # Phase 4: Handle atoms not in any substituent (unowned atoms)
    unowned_src = [i for i in range(N) if i not in used_src]
    unowned_tgt = [j for j in range(M) if j not in used_tgt]
    n_u = len(unowned_src)
    m_u = len(unowned_tgt)

    if n_u > 0 and m_u > 0:
        row_u, col_u = distance_and_class_based_assignment(
            x0[np.array(unowned_src)],
            x1[np.array(unowned_tgt)],
            a0[np.array(unowned_src)],
            a1[np.array(unowned_tgt)],
            c_move=c_move,
            c_sub=c_sub,
            c_ins=c_ins,
            c_del=c_del,
        )

        for r_loc, c_loc in zip(row_u.tolist(), col_u.tolist(), strict=True):
            if r_loc < n_u:
                r_glob = unowned_src[r_loc]
            else:
                r_glob = N + unowned_tgt[r_loc - n_u]

            if c_loc < m_u:
                c_glob = unowned_tgt[c_loc]
            else:
                c_glob = M + unowned_src[c_loc - m_u]

            global_pairs.append((r_glob, c_glob))
    elif n_u == 0 and m_u > 0:
        # Only insertions remain
        global_pairs.extend((N + j, j) for j in unowned_tgt)
    elif n_u > 0 and m_u == 0:
        # Only deletions remain
        global_pairs.extend((i, M + i) for i in unowned_src)

    if len(global_pairs) != N + M:
        raise ValueError(
            "Substituent-constrained assignment did not produce a complete matching "
            f"(got {len(global_pairs)}, expected {N + M})."
        )

    row_ind = np.array([p[0] for p in global_pairs], dtype=int)
    col_ind = np.array([p[1] for p in global_pairs], dtype=int)

    # Safety checks: augmented assignment must be a permutation on both sides
    if len(np.unique(row_ind)) != (N + M) or len(np.unique(col_ind)) != (N + M):
        raise ValueError("Substituent-constrained assignment produced duplicate indices.")

    return row_ind, col_ind


def mcs_constrained_assignment(
    x0,
    x1,
    a0,
    a1,
    mcs_pairs: list[tuple[int, int]],
    c_move=1.0,
    c_sub=10.0,
    c_ins=5.0,
    c_del=10.0,
):
    """Run OT on non-MCS atoms and merge with fixed MCS pairs.

    This enforces backbone matches from ``mcs_pairs`` and computes all
    remaining assignments with the same partial OT routine used elsewhere.
    """
    N = x0.shape[0]
    M = x1.shape[0]

    # Keep only valid, one-to-one MCS pairs.
    valid_pairs = []
    used_src = set()
    used_tgt = set()
    for i_s, i_t in mcs_pairs:
        if not (0 <= i_s < N and 0 <= i_t < M):
            continue
        if i_s in used_src or i_t in used_tgt:
            continue
        valid_pairs.append((i_s, i_t))
        used_src.add(i_s)
        used_tgt.add(i_t)

    unmatched_src = [i for i in range(N) if i not in used_src]
    unmatched_tgt = [j for j in range(M) if j not in used_tgt]
    n_u = len(unmatched_src)
    m_u = len(unmatched_tgt)

    # Pair list in the augmented global index spaces:
    # source indices in [0, N+M), target indices in [0, M+N)
    global_pairs: list[tuple[int, int]] = []

    # 1) Force MCS real-real pairs.
    for i_s, i_t in valid_pairs:
        global_pairs.append((i_s, i_t))

    # 2) Pair corresponding MCS dummies with each other.
    #    source dummy for target i_t is N + i_t
    #    target dummy for source i_s is M + i_s
    for i_s, i_t in valid_pairs:
        global_pairs.append((N + i_t, M + i_s))

    # 3) Solve the reduced OT for non-MCS atoms only.
    if n_u > 0 and m_u > 0:
        row_u, col_u = distance_and_class_based_assignment(
            x0[np.array(unmatched_src)],
            x1[np.array(unmatched_tgt)],
            a0[np.array(unmatched_src)],
            a1[np.array(unmatched_tgt)],
            c_move=c_move,
            c_sub=c_sub,
            c_ins=c_ins,
            c_del=c_del,
        )

        for r_loc, c_loc in zip(row_u.tolist(), col_u.tolist(), strict=True):
            # Map reduced augmented indices to full augmented indices.
            if r_loc < n_u:
                r_glob = unmatched_src[r_loc]
            else:
                r_glob = N + unmatched_tgt[r_loc - n_u]

            if c_loc < m_u:
                c_glob = unmatched_tgt[c_loc]
            else:
                c_glob = M + unmatched_src[c_loc - m_u]

            global_pairs.append((r_glob, c_glob))
    elif n_u == 0 and m_u > 0:
        # Only insertions remain (all unmatched targets).
        global_pairs.extend((N + j, j) for j in unmatched_tgt)
    elif n_u > 0 and m_u == 0:
        # Only deletions remain (all unmatched sources).
        global_pairs.extend((i, M + i) for i in unmatched_src)

    if len(global_pairs) != N + M:
        raise ValueError(
            "MCS-constrained assignment did not produce a complete matching "
            f"(got {len(global_pairs)}, expected {N + M})."
        )

    row_ind = np.array([p[0] for p in global_pairs], dtype=int)
    col_ind = np.array([p[1] for p in global_pairs], dtype=int)

    # Safety checks: augmented assignment must be a permutation on both sides.
    if len(np.unique(row_ind)) != (N + M) or len(np.unique(col_ind)) != (N + M):
        raise ValueError("MCS-constrained assignment produced duplicate indices.")

    return row_ind, col_ind


def mcs_based_assignment_single(
    sample_mol: MoleculeData,
    target_mol: MoleculeData,
    smiles_sample: str | None = None,
    smiles_target: str | None = None,
    fixed_pairs: list[tuple[int, int]] | None = None,
    c_move: float = 1.0,
    c_sub: float = 10.0,
    c_ins: float = 5.0,
    c_del: float = 10.0,
    optimal_transport: str = "equivariant",
) -> tuple[AugmentedMoleculeData, AugmentedMoleculeData]:
    """MCS-guided partial optimal transport alignment.

    Finds the Maximum Common Substructure between the two SMILES, locks the
    MCS atom pairs in the assignment, then solves the remaining matching with
    the standard augmented-cost OT.  The interface mirrors
    ``partial_optimal_transport_single``.

    If ``fixed_pairs`` is provided, it is used directly as the locked atom
    correspondence (e.g. scaffold pairs) and the MCS computation is skipped;
    ``smiles_sample`` / ``smiles_target`` are then not required.

    Note: the atom ordering in each MoleculeData must match the RDKit ordering
    obtained via ``Chem.AddHs(Chem.MolFromSmiles(smiles))``.
    """
    sample = AugmentedMoleculeData.from_molecule(sample_mol)
    target = AugmentedMoleculeData.from_molecule(target_mol)

    # hydra.utils.log.info(f"[MCS] Running MCS-based assignment with {len(fixed_pairs) if fixed_pairs else 'auto-detected'} fixed pairs.")

    N, M = sample.x.shape[0], target.x.shape[0]
    x0 = sample.x.detach().cpu().numpy()
    x1 = target.x.detach().cpu().numpy()
    a0 = sample.a.detach().cpu().numpy()
    a1 = target.a.detach().cpu().numpy()

    if fixed_pairs is not None:
        mcs_pairs = fixed_pairs
    else:
        mcs_pairs = get_mcs_atom_mapping(smiles_sample, smiles_target)

        # Use heavy-atom MCS anchors only. Hydrogens should be resolved by OT in the
        # non-MCS stage, not hard-fixed by the backbone match.
        mol_s = Chem.AddHs(Chem.MolFromSmiles(smiles_sample))
        mol_t = Chem.AddHs(Chem.MolFromSmiles(smiles_target))
        if mol_s is not None and mol_t is not None:
            mcs_pairs = [
                (i_s, i_t)
                for i_s, i_t in mcs_pairs
                if mol_s.GetAtomWithIdx(i_s).GetAtomicNum() > 1
                and mol_t.GetAtomWithIdx(i_t).GetAtomicNum() > 1
            ]

    # Kabsch pre-alignment using MCS anchor points
    if len(mcs_pairs) >= 3:
        mcs_s_idxs = [p[0] for p in mcs_pairs]
        mcs_t_idxs = [p[1] for p in mcs_pairs]
        pts_s = torch.tensor(x0[mcs_s_idxs], dtype=torch.float32)
        pts_t = torch.tensor(x1[mcs_t_idxs], dtype=torch.float32)
        R, t = rigid_alignment(pts_s, pts_t)
        x0 = (torch.tensor(x0, dtype=torch.float32) @ R.T + t).numpy()
        sample.x = torch.tensor(x0, dtype=sample.x.dtype, device=sample.x.device)

    row_ind, col_ind = mcs_constrained_assignment(
        x0, x1, a0, a1, mcs_pairs, c_move, c_sub, c_ins, c_del
    )

    sample.pad(num_auxiliary=M).permute_nodes(row_ind)
    target.pad(num_auxiliary=N).permute_nodes(col_ind)

    sample_is_aux = sample.is_auxiliary.squeeze()
    target_is_aux = target.is_auxiliary.squeeze()
    valid_mask = ~(sample_is_aux & target_is_aux)

    if optimal_transport == "equivariant":
        match_mask = (~sample_is_aux) & (~target_is_aux)
        if match_mask.sum() > 0:
            R, t = rigid_alignment(sample.x[match_mask], target.x[match_mask])
            sample.x = sample.x @ R.T + t

    sample = filter_nodes(sample, valid_mask)
    target = filter_nodes(target, valid_mask)

    return sample, target


def substituent_based_assignment_single(
    sample_mol: MoleculeData,
    target_mol: MoleculeData,
    fixed_pairs: list[tuple[int, int]],
    src_subs: dict[int, list[list[int]]],
    tgt_subs: dict[int, list[list[int]]],
    c_move: float = 1.0,
    c_sub: float = 10.0,
    c_ins: float = 5.0,
    c_del: float = 10.0,
    optimal_transport: str = "equivariant",
) -> tuple[AugmentedMoleculeData, AugmentedMoleculeData]:
    """Substituent-based partial optimal transport alignment.

    Fixes scaffold atom pairs, then matches substituent branches by spatial
    direction (after Kabsch alignment) and runs independent partial OT within
    each matched branch pair.

    Args:
        sample_mol: Prior sample molecule.
        target_mol: Target molecule from the dataset.
        fixed_pairs: Scaffold atom correspondences (source_idx, target_idx).
        src_subs: dict mapping scaffold atom index to list of residue branches;
            each branch is a list of atom indices with the root atom first
            (source molecule).
        tgt_subs: dict mapping scaffold atom index to list of residue branches
            (target molecule).
        c_move, c_sub, c_ins, c_del: OT cost weights.
        optimal_transport: Alignment strategy ("equivariant" or other).

    Returns:
        Aligned (sample, target) as AugmentedMoleculeData with auxiliary
        flags and filtered auxiliary-auxiliary pairs.
    """
    sample = AugmentedMoleculeData.from_molecule(sample_mol)
    target = AugmentedMoleculeData.from_molecule(target_mol)

    N, M = sample.x.shape[0], target.x.shape[0]
    x0 = sample.x.detach().cpu().numpy()
    x1 = target.x.detach().cpu().numpy()
    a0 = sample.a.detach().cpu().numpy()
    a1 = target.a.detach().cpu().numpy()

    # Kabsch pre-alignment using scaffold anchor points
    if len(fixed_pairs) >= 3:
        mcs_s_idxs = [p[0] for p in fixed_pairs]
        mcs_t_idxs = [p[1] for p in fixed_pairs]
        pts_s = torch.tensor(x0[mcs_s_idxs], dtype=torch.float32)
        pts_t = torch.tensor(x1[mcs_t_idxs], dtype=torch.float32)
        R, t = rigid_alignment(pts_s, pts_t)
        x0 = (torch.tensor(x0, dtype=torch.float32) @ R.T + t).numpy()
        sample.x = torch.tensor(x0, dtype=sample.x.dtype, device=sample.x.device)

    row_ind, col_ind = substituent_constrained_assignment(
        x0, x1, a0, a1, fixed_pairs, src_subs, tgt_subs, c_move, c_sub, c_ins, c_del
    )

    sample.pad(num_auxiliary=M).permute_nodes(row_ind)
    target.pad(num_auxiliary=N).permute_nodes(col_ind)

    sample_is_aux = sample.is_auxiliary.squeeze()
    target_is_aux = target.is_auxiliary.squeeze()
    valid_mask = ~(sample_is_aux & target_is_aux)

    if optimal_transport == "equivariant":
        match_mask = (~sample_is_aux) & (~target_is_aux)
        if match_mask.sum() > 0:
            R, t = rigid_alignment(sample.x[match_mask], target.x[match_mask])
            sample.x = sample.x @ R.T + t

    sample = filter_nodes(sample, valid_mask)
    target = filter_nodes(target, valid_mask)

    return sample, target


def partial_optimal_transport_single(
    sample_mol: MoleculeData,
    target_mol: MoleculeData,
    c_move: float = 1.0,
    c_sub: float = 10.0,
    c_ins: float = 5.0,
    c_del: float = 10.0,
    optimal_transport: str = "equivariant",
    pre_align: bool = False,
) -> tuple[AugmentedMoleculeData, AugmentedMoleculeData]:
    """Single-pair partial optimal transport alignment.

    Args:
        sample_mol: Prior sample molecule.
        target_mol: Target molecule.
        c_move: Distance cost weight.
        c_sub: Substitution cost weight.
        c_ins: Insertion cost weight.
        optimal_transport: Alignment strategy ("equivariant" or other).
        pre_align: Whether to pre-align positions with iterated Kabsch.

    Returns:
        Aligned (sample, target) as AugmentedMoleculeData with auxiliary
        flags and filtered auxiliary-auxiliary pairs.
    """
    sample = AugmentedMoleculeData.from_molecule(sample_mol)
    target = AugmentedMoleculeData.from_molecule(target_mol)

    N, M = sample.x.shape[0], target.x.shape[0]
    x0, x1 = sample.x.detach().cpu().numpy(), target.x.detach().cpu().numpy()
    a0, a1 = sample.a.detach().cpu().numpy(), target.a.detach().cpu().numpy()

    if pre_align:
        x0, x1 = pre_align_positions(x0, x1, num_iterations=3)

    row_ind, col_ind = distance_and_class_based_assignment(
        x0,
        x1,
        a0,
        a1,
        c_move,
        c_sub,
        c_ins,
        c_del,
    )

    sample.pad(num_auxiliary=M).permute_nodes(row_ind)
    target.pad(num_auxiliary=N).permute_nodes(col_ind)

    sample_is_aux = sample.is_auxiliary.squeeze()
    target_is_aux = target.is_auxiliary.squeeze()
    valid_mask = ~(sample_is_aux & target_is_aux)

    if optimal_transport == "equivariant":
        match_mask = (~sample_is_aux) & (~target_is_aux)
        if match_mask.sum() > 0:
            R, t = rigid_alignment(sample.x[match_mask], target.x[match_mask])
            sample.x = sample.x @ R.T + t

    sample = filter_nodes(sample, valid_mask)
    target = filter_nodes(target, valid_mask)

    return sample, target


def partial_optimal_transport(
    samples_batched: MoleculeBatch,
    targets_batched: MoleculeBatch,
    c_move: float = 1.0,
    c_sub: float = 10.0,
    c_ins: float = 5.0,
    c_del: float = 10.0,
    optimal_transport: str = "equivariant",
    pre_align: bool = False,
) -> list[tuple[AugmentedMoleculeData, AugmentedMoleculeData]]:
    """Batch wrapper around partial_optimal_transport_single."""
    num_graphs = (
        targets_batched.batch_size
        if hasattr(targets_batched, "batch_size")
        else len(targets_batched)
    )
    return [
        partial_optimal_transport_single(
            samples_batched[b],
            targets_batched[b],
            c_move=c_move,
            c_sub=c_sub,
            c_ins=c_ins,
            c_del=c_del,
            optimal_transport=optimal_transport,
            pre_align=pre_align,
        )
        for b in range(num_graphs)
    ]
