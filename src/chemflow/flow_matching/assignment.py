import numpy as np
from scipy.optimize import linear_sum_assignment


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
