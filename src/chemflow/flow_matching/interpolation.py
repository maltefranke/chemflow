import numpy as np
import torch

from chemflow.flow_matching.assignment import distance_based_assignment
from chemflow.flow_matching.gmm import interpolate_gmm
from chemflow.flow_matching.sampling import sample_birth_locations, sample_deaths


def assign_targets_batched(x0, x0_batch_id, x1, x1_batch_id):
    """
    Assigns targets from x1 to x0 using the Hungarian algorithm,
    handling batches with flexible graph sizes using batch_id.

    Args:
        x0 (torch.Tensor): Shape (N_total, D) - concatenated nodes from all graphs
        x0_batch_id (torch.Tensor): Shape (N_total,) - batch assignment for each x0 node
        x1 (torch.Tensor): Shape (M_total, D) - concatenated nodes from all graphs
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

    for b in range(num_graphs):
        # Filter nodes belonging to this graph
        x0_mask_b = x0_batch_id == b
        x1_mask_b = x1_batch_id == b

        valid_x0 = x0[x0_mask_b]  # Shape (N_b, D)
        valid_x1 = x1[x1_mask_b]  # Shape (M_b, D)

        # Convert to numpy for assignment algorithm
        valid_x0_np = valid_x0.detach().cpu().numpy()
        valid_x1_np = valid_x1.detach().cpu().numpy()

        N_valid = valid_x0.shape[0]
        M_valid = valid_x1.shape[0]

        D = x0.shape[1]
        # Handle edge cases where one or both sets are empty
        if N_valid == 0:
            # No x0 items to match
            empty_x0 = torch.empty((0, D), device=x0.device, dtype=x0.dtype)
            empty_x1 = torch.empty((0, D), device=x1.device, dtype=x1.dtype)
            all_matched_x0.append(empty_x0)
            all_matched_x1.append(empty_x1)
            all_unmatched_x0.append(empty_x0)
            # All valid x1 are unmatched
            all_unmatched_x1.append(valid_x1)
            continue

        if M_valid == 0:
            # No x1 items to match
            empty_x0 = torch.empty((0, D), device=x0.device, dtype=x0.dtype)
            empty_x1 = torch.empty((0, D), device=x1.device, dtype=x1.dtype)
            all_matched_x0.append(empty_x0)
            all_matched_x1.append(empty_x1)
            # All valid x0 are unmatched
            all_unmatched_x0.append(valid_x0)
            all_unmatched_x1.append(empty_x1)
            continue

        # Assign targets using distance-based assignment
        row_ind, col_ind = distance_based_assignment(valid_x0_np, valid_x1_np)

        # Get the matched items
        matched_x0_b = valid_x0[row_ind]
        matched_x1_b = valid_x1[col_ind]
        all_matched_x0.append(matched_x0_b)
        all_matched_x1.append(matched_x1_b)

        # Get the unmatched items
        # Find indices of valid items that were *not* in the assignment
        unmatched_indices_x0 = np.setdiff1d(np.arange(N_valid), row_ind)
        unmatched_indices_x1 = np.setdiff1d(np.arange(M_valid), col_ind)

        all_unmatched_x0.append(valid_x0[unmatched_indices_x0])
        all_unmatched_x1.append(valid_x1[unmatched_indices_x1])

    return all_matched_x0, all_matched_x1, all_unmatched_x0, all_unmatched_x1


def interpolate_same_size(x0, x1, t):
    """
    x0 (N, D)
    x1 (N, D)
    t (float)
    """
    return x0 * (1 - t) + x1 * t


def interpolate_different_size(
    x0, x0_batch_id, x1, x1_batch_id, t, N_loc_samples=50
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    Interpolates between x0 and x1 for graphs with flexible sizes using batch_id.
    Handles matched nodes, death processes (unmatched x0), and birth processes.

    Args:
        x0: Shape (N_total, D) - concatenated nodes from all graphs
        x0_batch_id: Shape (N_total,) - batch assignment for each x0 node
        x1: Shape (M_total, D) - concatenated nodes from all graphs
        x1_batch_id: Shape (M_total,) - batch assignment for each x1 node
        t: Shape (num_graphs,) or scalar - time parameter for each graph

    Returns:
        tuple: A tuple of:
        - xt: Shape (K_total, D) - interpolated positions, concatenated
        - xt_batch_id: Shape (K_total,) - batch assignment for each xt node
        - target_vf: Shape (K_total, D) - velocity field targets
        - point_wise_death_targets: Shape (K_total, 1) - death targets
        - birth_rate_target: Shape (num_graphs, 1) - birth rate targets
        - death_rate_target: Shape (num_graphs, 1) - death rate targets
        - birth_next_locations: Shape (num_graphs, D) - GMM samples
    """
    D = x0.shape[1]
    device = x0.device
    dtype = x0.dtype

    # Get number of unique graphs
    max_x0 = x0_batch_id.max().item() + 1 if len(x0_batch_id) > 0 else 0
    max_x1 = x1_batch_id.max().item() + 1 if len(x1_batch_id) > 0 else 0
    num_graphs = max(max_x0, max_x1)
    num_graphs = int(num_graphs)

    # Handle scalar t
    if t.dim() == 0:
        t = t.expand(num_graphs)
    elif len(t) != num_graphs:
        raise ValueError(f"t must have length {num_graphs} but got {len(t)}")

    # 1. Assign targets
    matched_x0, matched_x1, unmatched_x0, unmatched_x1 = assign_targets_batched(
        x0, x0_batch_id, x1, x1_batch_id
    )

    # Initialize lists for collecting results per graph
    xt_per_graph = []
    target_vf_per_graph = []
    birth_rate_target = []
    death_rate_target = []
    birth_next_locations = []
    birth_next_locations_batch_ids = []
    point_wise_death_targets_per_graph = []

    # Process each graph separately
    for b in range(num_graphs):
        t_b = t[b] if t.dim() > 0 else t
        instantaneous_rate = 1 / torch.clamp(1 - t_b, min=1e-8)

        # 2.1 Interpolate the matched targets
        matched_x0_b = matched_x0[b]
        matched_x1_b = matched_x1[b]

        # Collect parts to concatenate (only non-empty tensors)
        xt_parts = []
        vf_parts = []
        num_matched = 0
        num_death = 0

        if matched_x0_b.shape[0] > 0:
            matched_xt_b = interpolate_same_size(matched_x0_b, matched_x1_b, t_b)
            matched_vf_b = matched_x1_b - matched_x0_b
            xt_parts.append(matched_xt_b)
            vf_parts.append(matched_vf_b)
            num_matched = matched_xt_b.shape[0]

        unmatched_x0_b = unmatched_x0[b]
        unmatched_x1_b = unmatched_x1[b]

        # 2.2 Handle unmatched samples / targets
        # 2.3 Death process
        if unmatched_x0_b.shape[0] > 0:
            # Create a sink state that all unmatched x0 will move towards
            if matched_x1_b.shape[0] > 0:
                x_sink = matched_x1_b.mean(dim=0).reshape(1, -1)
            else:
                # If no matched nodes, use a zero sink
                x_sink = torch.zeros((1, D), device=device, dtype=dtype)

            # Sample death times
            _, _, x0_alive_at_xt = sample_deaths(unmatched_x0_b, t_b)

            # Interpolate the unmatched x0 to the sink state
            if x0_alive_at_xt.shape[0] > 0:
                death_xt_b = interpolate_same_size(x0_alive_at_xt, x_sink, t_b)
                death_vf_b = x_sink.repeat(death_xt_b.shape[0], 1) - death_xt_b

                xt_parts.append(death_xt_b)
                vf_parts.append(death_vf_b)
                num_death = death_xt_b.shape[0]
                N_necessary_deaths = x0_alive_at_xt.shape[0]
                death_rate_target_b = N_necessary_deaths * instantaneous_rate
            else:
                death_rate_target_b = torch.tensor(0.0, device=device)

            death_rate_target.append(death_rate_target_b.view(1, 1))
            birth_rate_target.append(torch.zeros((1, 1), device=device))
            birth_next_locations.append(-1e3 * torch.ones((1, D), device=device))

        # 2.4 Birth process
        elif unmatched_x1_b.shape[0] > 0:
            # Sample birth times and locations
            birth_times, birth_location_t_birth, birth_mu, unborn_x1_b = (
                sample_birth_locations(unmatched_x1_b, t_b)
            )

            # Interpolate the birth locations to the birth times
            if birth_location_t_birth.shape[0] > 0:
                birth_xt_b = birth_location_t_birth + (
                    t_b.repeat(birth_times.shape[0]) - birth_times
                ).unsqueeze(-1) * (birth_mu - birth_location_t_birth)
                birth_vf_b = (birth_mu - birth_xt_b) * instantaneous_rate

                xt_parts.append(birth_xt_b)
                vf_parts.append(birth_vf_b)
                N_necessary_births = unmatched_x1_b.shape[0] - birth_xt_b.shape[0]
                birth_rate_target_b = N_necessary_births * instantaneous_rate
            else:
                birth_rate_target_b = torch.tensor(0.0, device=device)

            death_rate_target.append(torch.zeros((1, 1), device=device))
            birth_rate_target.append(birth_rate_target_b.view(1, 1))

            if unborn_x1_b.shape[0] > 0:
                # Store the GMM samples for NLL calculation during training
                birth_samples = interpolate_gmm(
                    unborn_x1_b, t_b, num_samples=N_loc_samples
                )
                birth_next_locations.append(birth_samples)
            else:
                # If no unborn x1, use placeholder
                birth_next_locations.append(-1e3 * torch.ones((1, D), device=device))

        # 2.5 No birth or death process, just movement
        else:
            death_rate_target.append(torch.zeros((1, 1), device=device))
            birth_rate_target.append(torch.zeros((1, 1), device=device))
            birth_next_locations.append(-1e3 * torch.ones((1, D), device=device))

        # Concatenate all parts for this graph (only non-empty tensors)
        if xt_parts:
            xt_graph = torch.cat(xt_parts, dim=0)
            vf_graph = torch.cat(vf_parts, dim=0)
        else:
            # Edge case: no nodes at all for this graph
            xt_graph = torch.empty((0, D), device=device, dtype=dtype)
            vf_graph = torch.empty((0, D), device=device, dtype=dtype)

        xt_per_graph.append(xt_graph)
        target_vf_per_graph.append(vf_graph)

        # Create point-wise death targets for this graph
        num_total = xt_graph.shape[0]
        death_targets_b = torch.zeros(num_total, 1, device=device)
        if num_death > 0:
            death_targets_b[num_matched : num_matched + num_death] = 1.0

        point_wise_death_targets_per_graph.append(death_targets_b)

        birth_next_locations_batch_id = torch.full(
            (birth_next_locations[-1].shape[0],), b, dtype=torch.long, device=device
        )
        birth_next_locations_batch_ids.append(birth_next_locations_batch_id)

    # 3. Concatenate across all graphs and create batch_id
    xt = torch.cat(xt_per_graph, dim=0)
    target_vf = torch.cat(target_vf_per_graph, dim=0)
    point_wise_death_targets = torch.cat(point_wise_death_targets_per_graph, dim=0)

    # Create batch_id for the final output
    xt_batch_id = torch.cat(
        [
            torch.full((xt_per_graph[i].shape[0],), i, dtype=torch.long, device=device)
            for i in range(num_graphs)
        ],
        dim=0,
    )

    # Stack rate targets
    if birth_rate_target:
        birth_rate_target = torch.cat(birth_rate_target, dim=0)
    else:
        birth_rate_target = torch.zeros((num_graphs, 1), device=device)

    if death_rate_target:
        death_rate_target = torch.cat(death_rate_target, dim=0)
    else:
        death_rate_target = torch.zeros((num_graphs, 1), device=device)

    if birth_next_locations:
        birth_next_locations = torch.cat(birth_next_locations, dim=0)
    else:
        birth_next_locations = -1e3 * torch.ones((num_graphs, D), device=device)

    birth_next_locations_batch_ids = torch.cat(birth_next_locations_batch_ids, dim=0)

    return (
        xt,
        xt_batch_id,
        target_vf,
        point_wise_death_targets,
        birth_rate_target,
        death_rate_target,
        birth_next_locations,
        birth_next_locations_batch_ids,
    )


if __name__ == "__main__":
    x0 = torch.randn(6, 3)
    x1 = torch.randn(8, 3)
    x0_batch_id = torch.zeros(6)
    x1_batch_id = torch.zeros(8)
    t = torch.tensor(0.5).view(1, 1)
    (
        xt,
        xt_batch_id,
        target_vf,
        point_wise_death_targets,
        birth_rate_target,
        death_rate_target,
        birth_next_locations,
        birth_next_locations_batch_ids,
    ) = interpolate_different_size(x0, x0_batch_id, x1, x1_batch_id, t)
    print(xt.shape)
    print(xt_batch_id.shape)
    print(target_vf.shape)
    print(point_wise_death_targets.shape)
    print(birth_rate_target.shape)
    print(death_rate_target.shape)
    print(birth_next_locations.shape)
    print(birth_next_locations_batch_ids.shape)
