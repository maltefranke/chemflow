import numpy as np
from scipy.optimize import linear_sum_assignment
import torch

from chemflow.utils import rigid_alignment

from chemflow.dataset.molecule_data import MoleculeData


def distance_based_assignment(valid_x0, valid_x1):
    """
    Assign targets from valid_x1 to valid_x0 using the Hungarian algorithm.

    Args:
        valid_x0 (np.ndarray): Shape (N_valid, D)
        valid_x1 (np.ndarray): Shape (M_valid, D)

    Returns:
        tuple: A tuple of two arrays:
        - row_ind (np.ndarray): Shape (N_valid,)
        - col_ind (np.ndarray): Shape (M_valid,)
    """
    # Calculate the cost matrix *only* for valid items
    # (N_valid, 1, D) - (1, M_valid, D) => (N_valid, M_valid, D)
    cost_matrix_b = np.linalg.norm(
        valid_x0[:, np.newaxis] - valid_x1[np.newaxis, :], axis=2
    )
    # cost_matrix_b shape is (N_valid, M_valid)

    # Perform the assignment
    # row_ind and col_ind are indices *into* valid_x0 and valid_x1
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


def distance_and_class_based_assignment(
    valid_x0, valid_x1, class_x0, class_x1, C_dist=1.0, C_class=10.0, C_birth=5.0
):
    """
    Assigns targets considering distance, class consistency, and birth/death costs.

    Args:
        valid_x0 (np.ndarray): Source points / Previous frame (N, D)
        valid_x1 (np.ndarray): Target points / Current frame (M, D)
        class_x0 (np.ndarray): Source classes (N,)
        class_x1 (np.ndarray): Target classes (M,)
        C_dist (float): Multiplier for Euclidean distance cost.
        C_class (float): Penalty added if classes do not match.
        C_birth (float): Cost threshold. If matching costs > 2*C_birth,
                         the match is skipped.

    Returns:
        tuple:
        - row_ind (np.ndarray): Indices into valid_x0 for successful matches.
        - col_ind (np.ndarray): Indices into valid_x1 for successful matches.
    """
    N = valid_x0.shape[0]
    M = valid_x1.shape[0]

    if N == 0 or M == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    # --- 1. Calculate Real Costs ---
    dist_matrix = np.linalg.norm(
        valid_x0[:, np.newaxis] - valid_x1[np.newaxis, :], axis=2
    )
    class_diff_matrix = class_x0[:, np.newaxis] != class_x1[np.newaxis, :]
    cost_real = (C_dist * dist_matrix) + (C_class * class_diff_matrix)

    # --- 2. Construct Augmented Cost Matrix (N+M, N+M) ---
    aug_cost = np.full((N + M, N + M), fill_value=1e9)

    # [Top-Left] Real Matches: x0 matches x1
    aug_cost[:N, :M] = cost_real

    # [Top-Right] DEATH (Unmatched Rows):
    # If x0[i] matches here, it maps to a dummy.
    # Conceptually: x0[i] is "killed" or "lost" because matching was too expensive.
    aug_cost[:N, M : M + N] = np.eye(N) * C_birth

    # [Bottom-Left] BIRTH (Unmatched Cols):
    # If x1[j] matches here, it maps from a dummy.
    # Conceptually: x1[j] is "born" or "new" because it found no cheap x0 source.
    aug_cost[N : N + M, :M] = np.eye(M) * C_birth

    # [Bottom-Right] Unused Dummies:
    # Dummies matching dummies (free).
    aug_cost[N:, M:] = 0.0

    # --- 3. Solve Assignment ---
    full_row_ind, full_col_ind = linear_sum_assignment(aug_cost)

    # --- 4. Filter for Real Matches ---
    # We only return indices where both row < N and col < M.
    # Any row index in the result >= N implies a Dummy Source -> Real Target (Birth).
    # Any col index in the result >= M implies Real Source -> Dummy Target (Death).

    valid_matches_mask = (full_row_ind < N) & (full_col_ind < M)

    row_ind = full_row_ind[valid_matches_mask]
    col_ind = full_col_ind[valid_matches_mask]

    return row_ind, col_ind
