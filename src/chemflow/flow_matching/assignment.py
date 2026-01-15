import numpy as np
from scipy.optimize import linear_sum_assignment
import torch

from chemflow.utils import rigid_alignment

from chemflow.dataset.molecule_data import (
    AugmentedMoleculeData,
    MoleculeBatch,
    MoleculeData,
    filter_nodes,
)


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


def assign_targets_batched(
    samples_batched,
    targets_batched,
    optimal_transport="equivariant",
):
    """
    Assigns targets from x1 to x0 using the Hungarian algorithm,
    handling batches with flexible graph sizes using batch_id.

    Args:
        samples_batched (Batch): Batch of samples
        targets_batched (Batch): Batch of targets
        optimal_transport (str): Optimal transport strategy

    Returns:
        tuple: A tuple of four lists (one per graph):
        - all_matched_x0 (list): List of B tensors, each shape (K_b, D).
        - all_matched_x1 (list): List of B tensors, each shape (K_b, D).
        - all_unmatched_x0 (list): List of B tensors, each shape (U0_b, D).
        - all_unmatched_x1 (list): List of B tensors, each shape (U1_b, D).

        Where K_b is the number of matches for graph b,
        U0_b is the number of unmatched x0 items for graph b,
        and U1_b is the number of unmatched x1 items for graph b.
    """
    matched_samples = []
    matched_targets = []

    unmatched_samples = []
    unmatched_targets = []

    device = samples_batched.x.device

    empty_mol = MoleculeData(
        x=torch.empty((0, 3), device=device, dtype=torch.float32),
        a=torch.empty((0), device=device, dtype=torch.long),
        c=torch.empty((0), device=device, dtype=torch.long),
        e=torch.empty((0), device=device, dtype=torch.long),
        edge_index=torch.empty((2, 0), device=device, dtype=torch.long),
    )

    for b in range(targets_batched.batch_size):
        sampled_mol = samples_batched[b]
        target_mol = targets_batched[b]

        x0 = sampled_mol.x.detach().cpu().numpy()
        x1 = target_mol.x.detach().cpu().numpy()

        N_x0 = x0.shape[0]
        N_x1 = x1.shape[0]

        # Handle edge cases where one or both sets are empty
        if N_x0 == 0:
            # No x0 items to match
            matched_samples.append(empty_mol.clone())
            matched_targets.append(empty_mol.clone())

            # All valid x1 are unmatched
            unmatched_samples.append(empty_mol.clone())
            unmatched_targets.append(target_mol.clone())
            continue

        if N_x1 == 0:
            # No x1 items to match
            matched_samples.append(empty_mol.clone())
            matched_targets.append(empty_mol.clone())

            # All valid x0 are unmatched
            unmatched_samples.append(sampled_mol)
            unmatched_targets.append(empty_mol.clone())
            continue

        # Assign targets using distance-based assignment
        row_ind, col_ind = distance_based_assignment(x0, x1)
        matched_sample_indices = torch.tensor(row_ind, dtype=torch.long).to(
            sampled_mol.x.device
        )
        matched_target_indices = torch.tensor(col_ind, dtype=torch.long).to(
            target_mol.x.device
        )

        # Get the matched items
        matched_sample = sampled_mol.get_permuted_subgraph(matched_sample_indices)
        matched_target = target_mol.get_permuted_subgraph(matched_target_indices)

        # Get the unmatched items
        # Find indices of valid items that were *not* in the assignment
        unmatched_sample_indices = np.setdiff1d(np.arange(N_x0), row_ind)
        unmatched_sample_indices = torch.tensor(
            unmatched_sample_indices, dtype=torch.long
        ).to(sampled_mol.x.device)

        unmatched_target_indices = np.setdiff1d(np.arange(N_x1), col_ind)
        unmatched_target_indices = torch.tensor(
            unmatched_target_indices, dtype=torch.long
        ).to(target_mol.x.device)

        unmatched_sample = sampled_mol.get_permuted_subgraph(unmatched_sample_indices)
        unmatched_target = target_mol.get_permuted_subgraph(unmatched_target_indices)

        if optimal_transport == "equivariant":
            # Align the matched items
            R, t = rigid_alignment(matched_sample.x, matched_target.x)
            matched_sample.x = matched_sample.x.mm(R.T) + t

            # rotate and translate the unmatched items
            unmatched_sample.x = unmatched_sample.x.mm(R.T) + t

        matched_samples.append(matched_sample)
        matched_targets.append(matched_target)

        unmatched_samples.append(unmatched_sample)
        unmatched_targets.append(unmatched_target)

    return matched_samples, matched_targets, unmatched_samples, unmatched_targets


def distance_and_class_based_assignment_old(
    x0,
    x1,
    a0,
    a1,
    C_dist=1.0,
    C_class=10.0,
    C_birth=5.0,
):
    """
    Assign targets from x1 to x0 using the Hungarian algorithm,
    handling batches with flexible graph sizes using batch_id.

    Special cases:
    1) Distance-based assignment
    C_dist=const, C_class=0, C_birth>>C_dist

    2) Class-based assignment.
    Note: This will not be useful for our application!
    C_dist=0, C_class=const, C_birth>>C_class

    3) Pure birth-death assignment - no move.
    C_dist>>C_birth, C_class=const, C_birth=const

    Args:
        x0 (np.ndarray): Shape (N_x0, D)
        x1 (np.ndarray): Shape (N_x1, D)
        a0 (np.ndarray): Shape (N_x0, C)
        a1 (np.ndarray): Shape (N_x1, C)
        C_dist (float): Distance cost weight
        C_class (float): Class cost weight
        C_birth (float): Birth cost weight

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
    cost_real = (C_dist * dist_matrix) + (C_class * class_diff_matrix)

    # 2. Construct Augmented Cost Matrix (N+M, N+M)
    total_size = N + M
    aug_cost = np.zeros((total_size, total_size))

    # [Top-Left] Real matches (N x M)
    aug_cost[:N, :M] = cost_real

    # [Top-Right] DEATH Cost (N x N)
    # A real point x0[i] matches a dummy target.
    # We use a very high value for off-diagonals so each point has its own dummy.
    death_block = np.full((N, N), fill_value=1e8)
    np.fill_diagonal(death_block, C_birth)
    aug_cost[:N, M:] = death_block

    # [Bottom-Left] BIRTH Cost (M x M)
    # A real point x1[j] matches a dummy source.
    birth_block = np.full((M, M), fill_value=1e8)
    np.fill_diagonal(birth_block, C_birth)
    aug_cost[N:, :M] = birth_block

    # [Bottom-Right] Unused Dummies (M x N)
    # This block must be 0.0 to allow dummies to pair with each other for "free"
    aug_cost[N:, M:] = 0.0

    # 3. Solve
    row_ind, col_ind = linear_sum_assignment(aug_cost)

    # 4. Filter for Real Matches
    valid_matches_mask = (row_ind < N) & (col_ind < M)
    return row_ind[valid_matches_mask], col_ind[valid_matches_mask]


def partial_optimal_transport_old(
    samples_batched,
    targets_batched,
    C_dist=1.0,
    C_class=10.0,
    C_birth=5.0,
    optimal_transport="equivariant",
):
    matched_samples = []
    matched_targets = []
    unmatched_samples = []
    unmatched_targets = []

    for b in range(targets_batched.batch_size):
        sampled_mol = samples_batched[b]
        target_mol = targets_batched[b]

        x0 = sampled_mol.x.detach().cpu().numpy()
        x1 = target_mol.x.detach().cpu().numpy()
        a0 = sampled_mol.a.detach().cpu().numpy()
        a1 = target_mol.a.detach().cpu().numpy()

        N_x0 = x0.shape[0]
        N_x1 = x1.shape[0]

        # Assign targets using distance-based assignment
        row_ind, col_ind = distance_and_class_based_assignment_old(
            x0, x1, a0, a1, C_dist, C_class, C_birth
        )
        matched_sample_indices = torch.tensor(row_ind, dtype=torch.long).to(
            sampled_mol.x.device
        )
        matched_target_indices = torch.tensor(col_ind, dtype=torch.long).to(
            target_mol.x.device
        )

        # Get the matched items
        matched_sample = sampled_mol.get_permuted_subgraph(matched_sample_indices)
        matched_sample.original_indices = matched_sample_indices

        matched_target = target_mol.get_permuted_subgraph(matched_target_indices)
        matched_target.original_indices = matched_target_indices

        # Get the unmatched items
        unmatched_sample_indices = np.setdiff1d(np.arange(N_x0), row_ind)
        unmatched_sample_indices = torch.tensor(
            unmatched_sample_indices, dtype=torch.long
        ).to(sampled_mol.x.device)

        unmatched_target_indices = np.setdiff1d(np.arange(N_x1), col_ind)
        unmatched_target_indices = torch.tensor(
            unmatched_target_indices, dtype=torch.long
        ).to(target_mol.x.device)

        unmatched_sample = sampled_mol.get_permuted_subgraph(unmatched_sample_indices)
        unmatched_sample.original_indices = unmatched_sample_indices

        unmatched_target = target_mol.get_permuted_subgraph(unmatched_target_indices)
        unmatched_target.original_indices = unmatched_target_indices

        if optimal_transport == "equivariant":
            # Align the matched samples with the matched targets
            R, t = rigid_alignment(matched_sample.x, matched_target.x)
            matched_sample.x = matched_sample.x.mm(R.T) + t

            # rotate and translate the unmatched samples
            unmatched_sample.x = unmatched_sample.x.mm(R.T) + t

        # get the closest nodes of the samples to unmatched targets
        if unmatched_target.num_nodes > 0:
            unmatched_x1 = unmatched_target.x.detach().cpu().numpy()
            _, col_ind = distance_based_assignment(x0, unmatched_x1)
            col_ind = torch.tensor(col_ind, dtype=torch.long).to(
                unmatched_target.x.device
            )

            # attach a marker to the unmatched targets for closest sample nodes
            unmatched_target.nearest_sample_idx = col_ind

        matched_samples.append(matched_sample)
        matched_targets.append(matched_target)
        unmatched_samples.append(unmatched_sample)
        unmatched_targets.append(unmatched_target)

    return matched_samples, matched_targets, unmatched_samples, unmatched_targets


def distance_and_class_based_assignment(
    x0,
    x1,
    a0,
    a1,
    c_move=1.0,
    c_sub=10.0,
    c_ins=5.0,
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

    3) Pure birth-death assignment - no move.
    c_move>>c_ins, c_sub=const, c_ins=const

    Args:
        x0 (np.ndarray): Shape (N_x0, D)
        x1 (np.ndarray): Shape (N_x1, D)
        a0 (np.ndarray): Shape (N_x0, C)
        a1 (np.ndarray): Shape (N_x1, C)
        c_move (float): Distance cost weight
        c_sub (float): Class cost weight
        c_ins (float): Birth cost weight

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

    # [Top-Right] DEATH Cost (N x N)
    # A real point x0[i] matches a dummy target.
    # We use a very high value for off-diagonals so each point has its own dummy.
    death_block = np.full((N, N), fill_value=1e8)
    np.fill_diagonal(death_block, c_sub)
    aug_cost[:N, M:] = death_block

    # [Bottom-Left] BIRTH Cost (M x M)
    # A real point x1[j] matches a dummy source.
    birth_block = np.full((M, M), fill_value=1e8)
    np.fill_diagonal(birth_block, c_ins)
    aug_cost[N:, :M] = birth_block

    # [Bottom-Right] Unused Dummies (M x N)
    # This block must be 0.0 to allow dummies to pair with each other for "free"
    aug_cost[N:, M:] = 0.0

    # 3. Solve
    row_ind, col_ind = linear_sum_assignment(aug_cost)

    return row_ind, col_ind


def partial_optimal_transport(
    samples_batched: MoleculeBatch,
    targets_batched: MoleculeBatch,
    c_move: float = 1.0,
    c_sub: float = 10.0,
    c_ins: float = 5.0,
    optimal_transport: str = "equivariant",
) -> list[tuple[AugmentedMoleculeData, AugmentedMoleculeData]]:
    results = []
    num_graphs = (
        targets_batched.batch_size
        if hasattr(targets_batched, "batch_size")
        else len(targets_batched)
    )

    for b in range(num_graphs):
        # 1. Inititalize the augmented molecule data objects
        sample = AugmentedMoleculeData.from_molecule(samples_batched[b])
        target = AugmentedMoleculeData.from_molecule(targets_batched[b])

        N, M = sample.x.shape[0], target.x.shape[0]

        # 2. Solve the POT assignment problem
        row_ind, col_ind = distance_and_class_based_assignment(
            sample.x.detach().cpu().numpy(),
            target.x.detach().cpu().numpy(),
            sample.a.detach().cpu().numpy(),
            target.a.detach().cpu().numpy(),
            c_move,
            c_sub,
            c_ins,
        )

        # 3. Augment with auxiliary nodes & permute based on the assignment
        sample.pad(num_auxiliary=M).permute_nodes(row_ind)
        target.pad(num_auxiliary=N).permute_nodes(col_ind)

        # 4. Filter out auxiliary-auxiliary pairs
        sample_is_aux = sample.is_auxiliary.squeeze()
        target_is_aux = target.is_auxiliary.squeeze()

        valid_mask = ~(sample_is_aux & target_is_aux)

        # 5. Align the matched items
        if optimal_transport == "equivariant":
            match_mask = (~sample_is_aux) & (~target_is_aux)
            if match_mask.sum() > 0:
                R, t = rigid_alignment(sample.x[match_mask], target.x[match_mask])
                sample.x = sample.x @ R.T + t

        sample = filter_nodes(sample, valid_mask)
        target = filter_nodes(target, valid_mask)

        results.append((sample, target))

    return results
