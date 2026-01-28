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
    np.fill_diagonal(del_block, c_sub)
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


def partial_optimal_transport(
    samples_batched: MoleculeBatch,
    targets_batched: MoleculeBatch,
    c_move: float = 1.0,
    c_sub: float = 10.0,
    c_ins: float = 5.0,
    optimal_transport: str = "equivariant",
    pre_align: bool = False,
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
        x0, x1 = sample.x.detach().cpu().numpy(), target.x.detach().cpu().numpy()
        a0, a1 = sample.a.detach().cpu().numpy(), target.a.detach().cpu().numpy()

        # 1.5. Pre-align the positions
        if pre_align:
            x0, x1 = pre_align_positions(x0, x1, num_iterations=3)

        # 2. Solve the POT assignment problem
        row_ind, col_ind = distance_and_class_based_assignment(
            x0,
            x1,
            a0,
            a1,
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
