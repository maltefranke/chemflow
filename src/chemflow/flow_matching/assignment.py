import numpy as np
from scipy.optimize import linear_sum_assignment
import torch

from chemflow.utils import rigid_alignment


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
    x0,
    a0,
    # c0,
    edge_types0,
    x0_batch_id,
    x1,
    a1,
    # c1,
    edge_types1,
    x1_batch_id,
    optimal_transport="equivariant",
):
    """
    Assigns targets from x1 to x0 using the Hungarian algorithm,
    handling batches with flexible graph sizes using batch_id.

    Args:
        x0 (torch.Tensor): Shape (N_total, D) - concatenated nodes from all graphs
        c0 (torch.Tensor): Shape (N_total, M+1) - concatenated types from all graphs
        edge_types0 (torch.Tensor): Shape (N_total, N_total) - concatenated edge types from all graphs
        x0_batch_id (torch.Tensor): Shape (N_total,) - batch assignment for each x0 node
        x1 (torch.Tensor): Shape (M_total, D) - concatenated nodes from all graphs
        c1 (torch.Tensor): Shape (M_total, M+1) - concatenated types from all graphs
        edge_types1 (torch.Tensor): Shape (M_total, M_total) - concatenated edge features from all graphs
        x1_batch_id (torch.Tensor): Shape (M_total,) - batch assignment for each x1 node

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
    D = x0.shape[-1]  # dimension of the data
    M = a0.shape[-1] - 1  # number of classes
    # C = c0.shape[-1] - 1  # number of charge classes

    # Get number of unique graphs in the batch
    if len(x0_batch_id) > 0 or len(x1_batch_id) > 0:
        max_x0 = x0_batch_id.max().item() + 1 if len(x0_batch_id) > 0 else 0
        max_x1 = x1_batch_id.max().item() + 1 if len(x1_batch_id) > 0 else 0
        num_graphs = max(max_x0, max_x1)
        num_graphs = int(num_graphs)
    else:
        num_graphs = 0

    all_matched_x0 = []
    all_matched_x1 = []
    all_unmatched_x0 = []
    all_unmatched_x1 = []

    all_matched_a0 = []
    all_matched_a1 = []
    all_unmatched_a0 = []
    all_unmatched_a1 = []

    # all_matched_c0 = []
    # all_matched_c1 = []
    # all_unmatched_c0 = []
    # all_unmatched_c1 = []

    all_matched_edge_types0 = []
    all_matched_edge_types1 = []
    all_unmatched_edge_types0 = []
    all_unmatched_edge_types1 = []

    empty_x = torch.empty((0, D), device=x0.device, dtype=x0.dtype)
    # empty_c = torch.empty((0, C + 1), device=x0.device, dtype=x0.dtype)
    empty_a = torch.empty((0, M + 1), device=x0.device, dtype=x0.dtype)

    for b in range(num_graphs):
        # Filter nodes belonging to this graph
        x0_mask_b = x0_batch_id == b
        x1_mask_b = x1_batch_id == b

        valid_x0 = x0[x0_mask_b]  # Shape (N_b, D)
        valid_x1 = x1[x1_mask_b]  # Shape (M_b, D)
        valid_a0 = a0[x0_mask_b]  # Shape (N_b, M+1)
        valid_a1 = a1[x1_mask_b]  # Shape (M_b, M+1)
        # valid_c0 = c0[x0_mask_b]  # Shape (N_b, C+1)
        # valid_c1 = c1[x1_mask_b]  # Shape (M_b, C+1)

        x0_idx = torch.nonzero(x0_mask_b).squeeze()
        x1_idx = torch.nonzero(x1_mask_b).squeeze()

        valid_edge_types0 = edge_types0[x0_idx[:, None], x0_idx]
        valid_edge_types1 = edge_types1[x1_idx[:, None], x1_idx]

        # Convert to numpy for assignment algorithm
        valid_x0_np = valid_x0.detach().cpu().numpy()
        valid_x1_np = valid_x1.detach().cpu().numpy()

        N_valid = valid_x0.shape[0]
        M_valid = valid_x1.shape[0]

        D = x0.shape[1]
        # Handle edge cases where one or both sets are empty
        if N_valid == 0:
            # No x0 items to match
            all_matched_x0.append(empty_x)
            all_matched_x1.append(empty_x)

            # All valid x1 are unmatched
            all_unmatched_x0.append(empty_x)
            all_unmatched_x1.append(valid_x1)

            all_unmatched_a0.append(empty_a)
            all_unmatched_a1.append(valid_a1)
            # all_unmatched_c0.append(empty_c)
            # all_unmatched_c1.append(valid_c1)
            continue

        if M_valid == 0:
            # No x1 items to match
            all_matched_x0.append(empty_x)
            all_matched_x1.append(empty_x)

            # All valid x0 are unmatched
            all_unmatched_x0.append(valid_x0)
            all_unmatched_x1.append(empty_x)

            all_unmatched_a0.append(valid_a0)
            all_unmatched_a1.append(empty_a)
            # all_unmatched_c0.append(valid_c0)
            # all_unmatched_c1.append(empty_c)
            continue

        # Assign targets using distance-based assignment
        row_ind, col_ind = distance_based_assignment(valid_x0_np, valid_x1_np)

        # Get the matched items
        matched_x0_b = valid_x0[row_ind]
        matched_x1_b = valid_x1[col_ind]

        matched_a0_b = valid_a0[row_ind]
        matched_a1_b = valid_a1[col_ind]
        # matched_c0_b = valid_c0[row_ind]
        # matched_c1_b = valid_c1[col_ind]

        matched_edge_types0_b = valid_edge_types0[row_ind[:, None], row_ind]
        matched_edge_types1_b = valid_edge_types1[col_ind[:, None], col_ind]

        # Get the unmatched items
        # Find indices of valid items that were *not* in the assignment
        unmatched_i_x0 = np.setdiff1d(np.arange(N_valid), row_ind)
        unmatched_i_x1 = np.setdiff1d(np.arange(M_valid), col_ind)

        unmatched_x0_b = valid_x0[unmatched_i_x0]
        unmatched_x1_b = valid_x1[unmatched_i_x1]

        unmatched_a0_b = valid_a0[unmatched_i_x0]
        unmatched_a1_b = valid_a1[unmatched_i_x1]
        # unmatched_c0_b = valid_c0[unmatched_i_x0]
        # unmatched_c1_b = valid_c1[unmatched_i_x1]

        unmatched_edge_types0_b = valid_edge_types0[
            unmatched_i_x0[:, None], unmatched_i_x0
        ]
        unmatched_edge_types1_b = valid_edge_types1[
            unmatched_i_x1[:, None], unmatched_i_x1
        ]

        if optimal_transport == "equivariant":
            # Align the matched items
            R, t = rigid_alignment(matched_x0_b, matched_x1_b)
            matched_x0_b = matched_x0_b.mm(R.T) + t

            # rotate and translate the unmatched items
            unmatched_x0_b = unmatched_x0_b.mm(R.T) + t

        all_matched_x0.append(matched_x0_b)
        all_matched_x1.append(matched_x1_b)
        all_unmatched_x0.append(unmatched_x0_b)
        all_unmatched_x1.append(unmatched_x1_b)

        all_matched_a0.append(matched_a0_b)
        all_matched_a1.append(matched_a1_b)
        all_unmatched_a0.append(unmatched_a0_b)
        all_unmatched_a1.append(unmatched_a1_b)

        # all_matched_c0.append(matched_c0_b)
        # all_matched_c1.append(matched_c1_b)
        # all_unmatched_c0.append(unmatched_c0_b)
        # all_unmatched_c1.append(unmatched_c1_b)

        all_matched_edge_types0.append(matched_edge_types0_b)
        all_matched_edge_types1.append(matched_edge_types1_b)
        all_unmatched_edge_types0.append(unmatched_edge_types0_b)
        all_unmatched_edge_types1.append(unmatched_edge_types1_b)

    assigned_targets = {
        "matched": {
            "x": (all_matched_x0, all_matched_x1),
            "a": (all_matched_a0, all_matched_a1),
            # "c": (all_matched_c0, all_matched_c1),
            "edge_types": (all_matched_edge_types0, all_matched_edge_types1),
        },
        "unmatched": {
            "x": (all_unmatched_x0, all_unmatched_x1),
            "a": (all_unmatched_a0, all_unmatched_a1),
            # "c": (all_unmatched_c0, all_unmatched_c1),
            "edge_types": (all_unmatched_edge_types0, all_unmatched_edge_types1),
        },
    }

    return assigned_targets


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
