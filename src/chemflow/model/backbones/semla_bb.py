from external_code.semla import EquiInvDynamics
import torch

from torch_geometric.utils import to_dense_batch, to_dense_adj, dense_to_sparse

import torch

# --- 1. Setup: Assume you have these from previous steps ---
# atom_mask: [B, N_max] (True for real nodes, False for padding)
# edge_feats: [B, N_max, N_max, F] (Dense edge attributes)


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
        in_node_nf,
        n_layers,
        n_attn_heads,
        hidden_nf,
        in_edge_nf,
        bond_refine,
        self_cond,
        coord_norm,
        eps,
    ):
        # we only have one coord set
        n_coord_sets = 1
        self_cond = False

        super().__init__()
        self.dynamics = EquiInvDynamics(
            d_model=in_node_nf,
            n_coord_sets=n_coord_sets,
            d_message=hidden_nf,
            n_layers=n_layers,
            n_attn_heads=n_attn_heads,
            d_message_hidden=hidden_nf,
            d_edge=in_edge_nf,
            bond_refine=bond_refine,
            self_cond=self_cond,
            coord_norm=coord_norm,
            eps=eps,
        )

    def forward(self, h, x, edges: tuple[torch.Tensor, torch.Tensor], edge_attr, batch):
        edges = torch.stack(edges, dim=0)

        # transform to dense feats as required by semla
        coords, atom_mask = to_dense_batch(x, batch)
        # coords = coords.unsqueeze(1)
        # atom_mask = atom_mask.unsqueeze(1)
        inv_feats, _ = to_dense_batch(h, batch)
        adj_matrix = to_dense_adj(edges, batch)
        edge_feats = to_dense_adj(edges, batch, edge_attr)

        out_coords, inv_feats, edge_feats = self.dynamics(
            coords, inv_feats, adj_matrix, atom_mask, edge_feats
        )

        # transform back to our format
        coords_out = out_coords[atom_mask]
        h_out = inv_feats[atom_mask]
        _, e_out = dense_to_sparse_edges(edge_feats, atom_mask)

        return h_out, coords_out, e_out
