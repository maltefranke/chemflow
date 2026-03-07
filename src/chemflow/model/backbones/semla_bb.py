from external_code.semla import EquiInvDynamics
import torch

from torch_geometric.utils import to_dense_batch, scatter


def dense_to_sparse_edges(edge_feats, atom_mask):
    """
    Recovers sparse edge_index and edge_attr from dense edge_feats.
    """

    # --- Step A: Create a Global Index Map ---
    # We create a container of the same shape as the mask, filled with placeholders (-1)
    # We then fill the valid spots with 0, 1, 2, ... N_total
    global_node_idx = torch.full_like(atom_mask, -1, dtype=torch.long)
    global_node_idx[atom_mask] = torch.arange(atom_mask.sum(), device=atom_mask.device)

    # --- Step B: Determine which edges exist ---
    # 1. Check where features are non-zero.
    #    If edge_feats is [B, N, N, F], we sum over F.
    #    If edge_feats is [B, N, N], we just take the absolute.
    if edge_feats.dim() == 4:
        has_edge = edge_feats.abs().sum(-1) > 0
    else:
        has_edge = edge_feats.abs() > 0

    # 2. IMPORTANT: Force edges involving padding nodes to False.
    #    (This cleans up any noise if edge_feats came from a model prediction)
    valid_nodes = atom_mask.unsqueeze(2) & atom_mask.unsqueeze(1)  # [B, N, N]
    edge_mask = has_edge & valid_nodes

    # --- Step C: Extract Indices and Attributes ---
    # 1. Get the coordinates of valid edges: (batch_idx, source_local, dest_local)
    b_idx, src_local, dst_local = edge_mask.nonzero(as_tuple=True)

    # 2. Map local indices to global indices
    src_global = global_node_idx[b_idx, src_local]
    dst_global = global_node_idx[b_idx, dst_local]

    # 3. Stack to form edge_index
    recovered_edge_index = torch.stack([src_global, dst_global], dim=0)

    # 4. Extract the attributes using the mask coordinates
    #    (This handles both [B,N,N] and [B,N,N,F] cases automatically)
    recovered_edge_attr = edge_feats[b_idx, src_local, dst_local]

    return recovered_edge_index, recovered_edge_attr


class SemlaBB(torch.nn.Module):
    def __init__(
        self,
        d_model,
        n_layers,
        n_attn_heads,
        d_message,
        d_message_hidden,
        d_edge,
        bond_refine,
        self_cond,
        coord_norm,
        n_coord_sets,
        eps,
    ):
        super().__init__()

        self.dynamics = EquiInvDynamics(
            d_model=d_model,
            n_coord_sets=n_coord_sets,
            d_message=d_message,
            n_layers=n_layers,
            n_attn_heads=n_attn_heads,
            d_message_hidden=d_message_hidden,
            d_edge=d_edge,
            bond_refine=bond_refine,
            self_cond=self_cond,
            coord_norm=coord_norm,
            eps=eps,
        )

    def forward(self, h, x, edges: tuple[torch.Tensor, torch.Tensor], edge_attr, batch):
        # batch_size as a data-dependent SymInt (same pattern as transformer.py)
        batch_size = batch.unique().shape[0]
        num_nodes = scatter(
            batch.new_ones(x.size(0)), batch, dim=0, dim_size=batch_size, reduce="sum"
        )
        # max_num_nodes as a 0-dim tensor (same pattern as dit.py)
        max_num_nodes = num_nodes.max()

        edges = torch.stack(edges, dim=0)

        coords, atom_mask = to_dense_batch(
            x, batch, batch_size=batch_size, max_num_nodes=max_num_nodes
        )
        inv_feats, _ = to_dense_batch(
            h, batch, batch_size=batch_size, max_num_nodes=max_num_nodes
        )

        # Build dense adj/edge tensors without to_dense_adj.
        # B and N are SymInts derived from tensor shapes, which torch.zeros accepts —
        # unlike to_dense_adj which calls torch.Size([flattened_size_tensor]) and crashes.
        B, N = atom_mask.shape[0], atom_mask.shape[1]

        # Compute each node's local index within its graph
        counts = scatter(
            batch.new_ones(batch.shape[0]),
            batch,
            dim=0,
            dim_size=batch_size,
            reduce="sum",
        )
        cum_counts = torch.cat([counts.new_zeros(1), counts.cumsum(0)[:-1]])
        local_idx = (
            torch.arange(batch.shape[0], device=batch.device) - cum_counts[batch]
        )

        row, col = edges[0], edges[1]
        b_idx = batch[row]
        local_row = local_idx[row]
        local_col = local_idx[col]

        adj_matrix = torch.zeros(B, N, N, device=x.device)
        adj_matrix[b_idx, local_row, local_col] = 1.0

        edge_feats = torch.zeros(
            B, N, N, edge_attr.shape[-1], dtype=edge_attr.dtype, device=edge_attr.device
        )
        edge_feats[b_idx, local_row, local_col] = edge_attr

        out_coords, inv_feats, edge_feats = self.dynamics(
            coords, inv_feats, adj_matrix, atom_mask, edge_feats
        )

        # Convert back to sparse format.
        # Index at the original input edge positions — no nonzero() call, no graph break.
        # The dynamics refines features at existing edge positions; we don't need to detect
        # new edges from the dense output (which would require nonzero() and break torch.compile).
        coords_out = out_coords[atom_mask]
        h_out = inv_feats[atom_mask]
        e_out = edge_feats[b_idx, local_row, local_col]

        return h_out, coords_out, e_out
