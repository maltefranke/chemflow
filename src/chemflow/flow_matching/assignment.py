import numpy as np
from scipy.optimize import linear_sum_assignment
import torch


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


def distance_and_class_based_assignment(valid_x0, valid_x1, class_x0, class_x1):
    pass


def assign_targets_batched(x0, c0, x0_batch_id, x1, c1, x1_batch_id):
    """
    Assigns targets from x1 to x0 using the Hungarian algorithm,
    handling batches with flexible graph sizes using batch_id.

    Args:
        x0 (torch.Tensor): Shape (N_total, D) - concatenated nodes from all graphs
        c0 (torch.Tensor): Shape (N_total, M+1) - concatenated types from all graphs
        x0_batch_id (torch.Tensor): Shape (N_total,) - batch assignment for each x0 node
        x1 (torch.Tensor): Shape (M_total, D) - concatenated nodes from all graphs
        c1 (torch.Tensor): Shape (M_total, M+1) - concatenated types from all graphs
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
    M = c0.shape[-1] - 1  # number of classes

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
    all_matched_c0 = []
    all_matched_c1 = []
    all_unmatched_c0 = []
    all_unmatched_c1 = []

    empty_x = torch.empty((0, D), device=x0.device, dtype=x0.dtype)
    empty_c = torch.empty((0, M + 1), device=x0.device, dtype=x0.dtype)

    for b in range(num_graphs):
        # Filter nodes belonging to this graph
        x0_mask_b = x0_batch_id == b
        x1_mask_b = x1_batch_id == b

        valid_x0 = x0[x0_mask_b]  # Shape (N_b, D)
        valid_x1 = x1[x1_mask_b]  # Shape (M_b, D)
        valid_c0 = c0[x0_mask_b]  # Shape (N_b, M+1)
        valid_c1 = c1[x1_mask_b]  # Shape (M_b, M+1)

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

            all_unmatched_c0.append(empty_c)
            all_unmatched_c1.append(valid_c1)
            continue

        if M_valid == 0:
            # No x1 items to match
            all_matched_x0.append(empty_x)
            all_matched_x1.append(empty_x)

            # All valid x0 are unmatched
            all_unmatched_x0.append(valid_x0)
            all_unmatched_x1.append(empty_x)

            all_unmatched_c0.append(valid_c0)
            all_unmatched_c1.append(empty_c)
            continue

        # Assign targets using distance-based assignment
        row_ind, col_ind = distance_based_assignment(valid_x0_np, valid_x1_np)

        # Get the matched items
        matched_x0_b = valid_x0[row_ind]
        matched_x1_b = valid_x1[col_ind]
        matched_c0_b = valid_c0[row_ind]
        matched_c1_b = valid_c1[col_ind]

        all_matched_x0.append(matched_x0_b)
        all_matched_x1.append(matched_x1_b)
        all_matched_c0.append(matched_c0_b)
        all_matched_c1.append(matched_c1_b)

        # Get the unmatched items
        # Find indices of valid items that were *not* in the assignment
        unmatched_indices_x0 = np.setdiff1d(np.arange(N_valid), row_ind)
        unmatched_indices_x1 = np.setdiff1d(np.arange(M_valid), col_ind)

        all_unmatched_x0.append(valid_x0[unmatched_indices_x0])
        all_unmatched_x1.append(valid_x1[unmatched_indices_x1])

        all_unmatched_c0.append(valid_c0[unmatched_indices_x0])
        all_unmatched_c1.append(valid_c1[unmatched_indices_x1])

    assigned_targets = {
        "matched": {
            "x": (all_matched_x0, all_matched_x1),
            "c": (all_matched_c0, all_matched_c1),
        },
        "unmatched": {
            "x": (all_unmatched_x0, all_unmatched_x1),
            "c": (all_unmatched_c0, all_unmatched_c1),
        },
    }

    return assigned_targets
