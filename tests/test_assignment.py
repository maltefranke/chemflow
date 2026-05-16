import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from dataclasses import dataclass
from typing import List, Tuple, Optional
from chemflow.dataset.molecule_data import MoleculeData
from chemflow.flow_matching.assignment import distance_and_class_based_assignment
from chemflow.utils import rigid_alignment

# --- Data Structures ---


@dataclass
class AugmentedMoleculeData:
    """
    Data object for transport results.
    Mirrors your MoleculeData but guarantees is_auxiliary is present.
    """

    x: torch.Tensor
    a: torch.Tensor
    is_auxiliary: torch.Tensor
    # Optional attributes (topology and charge)
    c: Optional[torch.Tensor] = None
    e: Optional[torch.Tensor] = None
    edge_index: Optional[torch.Tensor] = None


# --- Helper Logic ---


def _permute_and_filter_adj(
    edge_index: torch.Tensor,
    edge_attr: Optional[torch.Tensor],
    perm_indices: np.ndarray,
    keep_mask: torch.Tensor,
    num_aug_nodes: int,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Re-indexes edges according to a permutation and a subsequent filtering mask.

    Args:
        edge_index: Original edge indices (2, E).
        perm_indices: Array where perm_indices[k] is the original index of the node
                      that moved to position k.
        keep_mask: Boolean mask of nodes to keep after permutation.
        num_aug_nodes: Total number of nodes before filtering.

    Returns:
        (new_edge_index, new_edge_attr) valid for the filtered graph.
    """
    if edge_index is None or edge_index.numel() == 0:
        return None, edge_attr

    device = edge_index.device

    # 1. Inverse Permutation Map
    # We have: permuted_array[k] = original_array[perm_indices[k]]
    # We need: map_old_to_perm[old_idx] = new_idx
    # such that perm_indices[new_idx] = old_idx
    map_old_to_perm = torch.full((num_aug_nodes,), -1, dtype=torch.long, device=device)
    perm_tensor = torch.from_numpy(perm_indices).to(device=device, dtype=torch.long)

    # Standard invert permutation: map[val] = index
    map_old_to_perm[perm_tensor] = torch.arange(len(perm_indices), device=device)

    # 2. Apply Permutation to Edges
    # Edges now refer to the indices in the permuted array (before filtering)
    permuted_edge_index = map_old_to_perm[edge_index]

    # 3. Handle Filtering
    # We are removing nodes where keep_mask is False.
    # We need a map: map_perm_to_final[perm_idx] = final_idx (or -1 if removed)

    # Indices of nodes that survive
    surviving_indices = torch.nonzero(keep_mask, as_tuple=True)[0]

    map_perm_to_final = torch.full(
        (len(keep_mask),), -1, dtype=torch.long, device=device
    )
    map_perm_to_final[surviving_indices] = torch.arange(
        len(surviving_indices), device=device
    )

    # 4. Apply Filtering Map to Edges
    final_edge_index = map_perm_to_final[permuted_edge_index]

    # 5. Remove edges that connected to dropped nodes
    # If any endpoint is -1, the edge is removed.
    valid_edges_mask = (final_edge_index[0] != -1) & (final_edge_index[1] != -1)

    final_edge_index = final_edge_index[:, valid_edges_mask]

    final_edge_attr = None
    if edge_attr is not None:
        final_edge_attr = edge_attr[valid_edges_mask]

    return final_edge_index, final_edge_attr


# --- Main Function ---


def partial_optimal_transport(
    samples_batched,
    targets_batched,
    c_dist: float = 1.0,
    c_class: float = 10.0,
    c_birth: float = 5.0,
    optimal_transport: str = "equivariant",
) -> List[Tuple[AugmentedMoleculeData, AugmentedMoleculeData]]:
    results = []

    # Handle batch iteration whether it's a list or a PyG Batch
    num_graphs = (
        targets_batched.batch_size
        if hasattr(targets_batched, "batch_size")
        else len(targets_batched)
    )

    for b in range(num_graphs):
        sample = samples_batched[b]
        target = targets_batched[b]

        # Extract numpy arrays for solver
        x0_np = sample.x.detach().cpu().numpy()
        a0_np = sample.a.detach().cpu().numpy()
        x1_np = target.x.detach().cpu().numpy()
        a1_np = target.a.detach().cpu().numpy()

        N, M = x0_np.shape[0], x1_np.shape[0]

        # --- 1. Solve Assignment ---
        row_ind, col_ind = distance_and_class_based_assignment(
            x0_np, x1_np, a0_np, a1_np, c_dist, c_class, c_birth
        )

        # --- 2. Augment Tensors (PyTorch) ---
        device = sample.x.device

        # Prepare padding blocks
        # x: (N/M, 3), a: (N/M, 1), c: (N/M, 1) usually
        pad_x0 = torch.zeros(
            (M, sample.x.shape[1]), device=device, dtype=sample.x.dtype
        )
        pad_a0 = torch.zeros(
            (M, sample.a.shape[1]), device=device, dtype=sample.a.dtype
        )
        pad_c0 = (
            torch.zeros((M, 1), device=device, dtype=sample.c.dtype)
            if sample.c is not None
            else None
        )

        pad_x1 = torch.zeros(
            (N, target.x.shape[1]), device=device, dtype=target.x.dtype
        )
        pad_a1 = torch.zeros(
            (N, target.a.shape[1]), device=device, dtype=target.a.dtype
        )
        pad_c1 = (
            torch.zeros((N, 1), device=device, dtype=target.c.dtype)
            if target.c is not None
            else None
        )

        # Augment Sample [Real; Dummy]
        x0_aug = torch.cat([sample.x, pad_x0], dim=0)
        a0_aug = torch.cat([sample.a, pad_a0], dim=0)
        c0_aug = torch.cat([sample.c, pad_c0], dim=0) if sample.c is not None else None
        is_dummy_0 = torch.cat(
            [
                torch.zeros(N, 1, dtype=torch.bool, device=device),
                torch.ones(M, 1, dtype=torch.bool, device=device),
            ],
            dim=0,
        )

        # Augment Target [Real; Dummy]
        x1_aug = torch.cat([target.x, pad_x1], dim=0)
        a1_aug = torch.cat([target.a, pad_a1], dim=0)
        c1_aug = torch.cat([target.c, pad_c1], dim=0) if target.c is not None else None
        is_dummy_1 = torch.cat(
            [
                torch.zeros(M, 1, dtype=torch.bool, device=device),
                torch.ones(N, 1, dtype=torch.bool, device=device),
            ],
            dim=0,
        )

        # --- 3. Permute Nodes ---
        # row_ind permutes Sample, col_ind permutes Target
        x0_perm = x0_aug[row_ind]
        a0_perm = a0_aug[row_ind]
        c0_perm = c0_aug[row_ind] if c0_aug is not None else None
        is_dummy_0_perm = is_dummy_0[row_ind]

        x1_perm = x1_aug[col_ind]
        a1_perm = a1_aug[col_ind]
        c1_perm = c1_aug[col_ind] if c1_aug is not None else None
        is_dummy_1_perm = is_dummy_1[col_ind]

        # --- 4. Determine Filter Mask (Remove Double Dummies) ---
        # Keep if NOT (Sample is dummy AND Target is dummy)
        keep_mask = ~(is_dummy_0_perm.squeeze() & is_dummy_1_perm.squeeze())

        # --- 5. Handle Topology (Re-index Edges) ---
        # We process edges *before* applying the final mask to the node tensors
        # so we can use the helper to do permute -> map -> filter.

        # Sample Topology
        e0_final_idx, e0_final_attr = _permute_and_filter_adj(
            sample.edge_index, sample.e, row_ind, keep_mask, num_aug_nodes=N + M
        )

        # Target Topology
        # Note: Original target edges are on indices 0..M-1.
        # The augmented target has size N+M.
        e1_final_idx, e1_final_attr = _permute_and_filter_adj(
            target.edge_index, target.e, col_ind, keep_mask, num_aug_nodes=N + M
        )

        # --- 6. Apply Filter to Nodes ---
        x0_final = x0_perm[keep_mask]
        a0_final = a0_perm[keep_mask]
        c0_final = c0_perm[keep_mask] if c0_perm is not None else None
        aux0_final = is_dummy_0_perm[keep_mask]

        x1_final = x1_perm[keep_mask]
        a1_final = a1_perm[keep_mask]
        c1_final = c1_perm[keep_mask] if c1_perm is not None else None
        aux1_final = is_dummy_1_perm[keep_mask]

        # --- 7. Equivariant Alignment ---
        # Align Sample to Target using only Real-Real matches
        matched_mask = (~aux0_final.squeeze()) & (~aux1_final.squeeze())

        if optimal_transport == "equivariant" and matched_mask.sum() > 0:
            R, t = rigid_alignment(x0_final[matched_mask], x1_final[matched_mask])
            x0_final = x0_final @ R.t() + t

        # --- 8. Pack Results ---
        res_sample = AugmentedMoleculeData(
            x=x0_final,
            a=a0_final,
            c=c0_final,
            is_auxiliary=aux0_final,
            edge_index=e0_final_idx,
            e=e0_final_attr,
        )

        res_target = AugmentedMoleculeData(
            x=x1_final,
            a=a1_final,
            c=c1_final,
            is_auxiliary=aux1_final,
            edge_index=e1_final_idx,
            e=e1_final_attr,
        )

        results.append((res_sample, res_target))

    return results


import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- Import your classes and function here ---
# from your_module import MoleculeData, partial_optimal_transport


def create_dummy_molecule(coords, atom_types, edge_indices=None):
    """Helper to create a MoleculeData object from raw lists."""
    x = torch.tensor(coords, dtype=torch.float)
    a = torch.tensor(atom_types, dtype=torch.float).view(-1, 1)

    # Simple charge placeholder
    c = torch.zeros((len(coords), 1), dtype=torch.float).view(-1, 1)

    edge_index = None
    e = None
    if edge_indices is not None:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        # Dummy edge features
        e = torch.ones((edge_index.shape[1], 1), dtype=torch.float)

    return MoleculeData(x=x, a=a, c=c, edge_index=edge_index, e=e)


def visualize_transport(sample, target, res_sample, res_target, title="Transport"):
    """
    Visualizes the transport plan.
    Left: Original Setup.
    Right: Transport Result (Aligned Sample vs Target).
    """
    fig = plt.figure(figsize=(16, 8))

    # --- Plot 1: Original Positions (Before Transport) ---
    ax1 = fig.add_subplot(121, projection="3d")

    # Original Sample (Blue)
    xs = sample.x.numpy()
    ax1.scatter(
        xs[:, 0], xs[:, 1], xs[:, 2], c="blue", s=100, label="Source (Original)"
    )
    for i, txt in enumerate(range(len(xs))):
        ax1.text(xs[i, 0], xs[i, 1], xs[i, 2], f"S{i}", color="blue")

    # Original Target (Red)
    xt = target.x.numpy()
    ax1.scatter(xt[:, 0], xt[:, 1], xt[:, 2], c="red", s=100, label="Target (Original)")
    for i, txt in enumerate(range(len(xt))):
        ax1.text(xt[i, 0], xt[i, 1], xt[i, 2], f"T{i}", color="red")

    ax1.set_title(f"{title} - Original State")
    ax1.legend()
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")

    # --- Plot 2: Transport Result (After Alignment & Matching) ---
    ax2 = fig.add_subplot(122, projection="3d")

    # Extract results
    x_res_s = res_sample.x.numpy()  # Aligned Source
    x_res_t = res_target.x.numpy()  # Target (re-ordered)
    aux_s = res_sample.is_auxiliary.numpy().flatten()
    aux_t = res_target.is_auxiliary.numpy().flatten()

    # 1. Plot Matches (Real Source -> Real Target)
    # These are indices where neither is auxiliary
    matches = (~aux_s) & (~aux_t)

    if np.any(matches):
        ax2.scatter(
            x_res_s[matches, 0],
            x_res_s[matches, 1],
            x_res_s[matches, 2],
            c="blue",
            s=100,
            label="Matched Source",
        )
        ax2.scatter(
            x_res_t[matches, 0],
            x_res_t[matches, 1],
            x_res_t[matches, 2],
            c="red",
            s=100,
            label="Matched Target",
        )

        # Draw lines connecting matches
        for i in np.where(matches)[0]:
            ax2.plot(
                [x_res_s[i, 0], x_res_t[i, 0]],
                [x_res_s[i, 1], x_res_t[i, 1]],
                [x_res_s[i, 2], x_res_t[i, 2]],
                "k--",
                alpha=0.5,
            )

    # 2. Plot Deaths (Real Source -> Dummy Target)
    # Source node had no match in target
    deaths = (~aux_s) & (aux_t)
    if np.any(deaths):
        ax2.scatter(
            x_res_s[deaths, 0],
            x_res_s[deaths, 1],
            x_res_s[deaths, 2],
            c="black",
            marker="x",
            s=100,
            label="Death (Unmatched Source)",
        )

    # 3. Plot Births (Dummy Source -> Real Target)
    # Target node had no match in source
    births = (aux_s) & (~aux_t)
    if np.any(births):
        # We plot the Target node that was "born"
        ax2.scatter(
            x_res_t[births, 0],
            x_res_t[births, 1],
            x_res_t[births, 2],
            c="green",
            marker="*",
            s=150,
            label="Birth (New Target)",
        )

    # Plot Edges (Optional: Visualize connectivity of the result)
    # Using the re-indexed edge_index from the result
    if res_target.edge_index is not None:
        edges = res_target.edge_index.numpy()
        for k in range(edges.shape[1]):
            start, end = edges[0, k], edges[1, k]
            # Draw edges on the target structure
            ax2.plot(
                [x_res_t[start, 0], x_res_t[end, 0]],
                [x_res_t[start, 1], x_res_t[end, 1]],
                [x_res_t[start, 2], x_res_t[end, 2]],
                "r-",
                alpha=0.2,
            )

    ax2.set_title("Transport Result (Aligned)")
    ax2.legend()
    plt.tight_layout()
    plt.show()


def test_pot_visual():
    print("Generating Synthetic Data...")

    # Scenario:
    # Source: A small triangle (3 atoms) at the origin.
    # Target: A larger square (4 atoms) shifted and rotated.
    # Expectation: 3 points match, 1 point is born (the 4th corner of square).

    # Source (Triangle)
    s_pos = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
    s_type = [1, 1, 1]  # All same type
    s_edges = [[0, 1], [1, 2], [2, 0]]  # Cycle

    # Target (Square) - Shifted by (5,5,5) and slightly larger
    t_pos = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 1]]
    t_type = [1, 1, 1, 1]
    t_edges = [[0, 1], [0, 2], [1, 3], [2, 3]]  # Grid edges

    source_mol = create_dummy_molecule(s_pos, s_type, s_edges)
    target_mol = create_dummy_molecule(t_pos, t_type, t_edges)

    # Wrap in a list (batch size = 1)
    batch_source = [source_mol]
    batch_target = [target_mol]

    print("Running Partial Optimal Transport...")
    results = partial_optimal_transport(
        batch_source,
        batch_target,
        c_dist=5000.0,  # Cost to move distance
        c_class=0.0,  # High cost to mismatch class
        c_birth=0.0,  # Low cost to create/destroy points (favors birth over far movement)
        c_del=0.0,
    )

    print("Visualizing...")
    res_sample, res_target = results[0]

    # Debug info
    print(f"Result Nodes: {res_sample.x.shape[0]}")
    print(f"Is Aux Sample: {res_sample.is_auxiliary.flatten()}")
    print(f"Is Aux Target: {res_target.is_auxiliary.flatten()}")

    visualize_transport(source_mol, target_mol, res_sample, res_target)


if __name__ == "__main__":
    test_pot_visual()
