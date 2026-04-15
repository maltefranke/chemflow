import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F

from external_code.gvp import GVP, GVPConvLayer


class GVPBB(nn.Module):
    """
    GVP (Geometric Vector Perceptron) backbone.

    Matches the forward interface of SemlaBB / EGNNWithEdgeType:
        forward(h, x, edges, edge_attr, batch) -> (h_out, x_out, e_out)

    Equivariant coordinate updates come from GVP vector channels: the 3-D
    coordinate is lifted to a single vector channel, expanded to n_vectors
    equivariant channels through message passing, then collapsed back to a
    coordinate residual (x_out = x + delta_x).

    Edge embeddings are produced by an MLP that concatenates updated node
    features for the source/destination nodes with an RBF distance embedding.
    """

    def __init__(
        self,
        d_scalar: int,
        n_vectors: int,
        d_edge: int,
        n_layers: int,
        drop_rate: float = 0.1,
        rbf_embedding_args=None,
    ):
        super().__init__()
        self.d_scalar = d_scalar
        self.n_vectors = n_vectors
        self.d_edge = d_edge

        node_dims = (d_scalar, n_vectors)
        edge_dims = (d_edge, 0)  # scalar-only edge features

        # (d_scalar, 1) -> (d_scalar, n_vectors)
        # lifts coord as single vector channel and expands to n_vectors
        self.node_in = GVP(
            (d_scalar, 1),
            node_dims,
            activations=(F.relu, torch.sigmoid),
        )

        self.layers = nn.ModuleList([
            GVPConvLayer(node_dims, edge_dims, drop_rate=drop_rate)
            for _ in range(n_layers)
        ])

        # (d_scalar, n_vectors) -> (d_scalar, 1), no activation for coord residual
        self.node_out = GVP(
            node_dims,
            (d_scalar, 1),
            activations=(None, None),
        )

        # Edge MLP: concat(h_src, h_dst, rbf(dist)) -> [E, d_edge]
        self.rbf_embedding = hydra.utils.instantiate(rbf_embedding_args)
        rbf_out_dim = rbf_embedding_args.get(
            "out_dim", rbf_embedding_args.get("num_rbf", 16)
        )
        edge_in_dim = 2 * d_scalar + rbf_out_dim
        self.edge_out = nn.Sequential(
            nn.LayerNorm(edge_in_dim),
            nn.Linear(edge_in_dim, d_edge),
            nn.LayerNorm(d_edge),
            nn.GELU(),
        )

    def forward(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        edges: tuple[torch.Tensor, torch.Tensor],
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
    ):
        """
        h          [N, d_scalar]   invariant node features
        x          [N, 3]          3-D coordinates
        edges      (src, dst)      sparse edge index tensors
        edge_attr  [E, d_edge]     edge features
        batch      [N]             batch index (unused inside GVP, kept for interface parity)

        Returns:
          h_out    [N, d_scalar]
          x_out    [N, 3]
          e_out    [E, d_edge]
        """
        edge_index = torch.stack(edges, dim=0)  # [2, E]

        # Lift coord to single equivariant vector channel: [N, 3] -> [N, 1, 3]
        x_v = x.unsqueeze(-2)

        # Input projection: (d_scalar, 1) -> (d_scalar, n_vectors)
        h_gvp, v_gvp = self.node_in((h, x_v))

        # Edge features as GVP scalar-only tuple (no vector edge channels)
        e_zeros = torch.zeros(
            edge_attr.shape[0], 0, 3, dtype=h.dtype, device=h.device
        )
        e_gvp = (edge_attr, e_zeros)

        # GVP message passing layers
        for layer in self.layers:
            h_gvp, v_gvp = layer(h_gvp, v_gvp, edge_index, e_gvp)

        # Output projection: (d_scalar, n_vectors) -> (d_scalar, 1)
        h_out, v_out = self.node_out((h_gvp, v_gvp))

        # Equivariant coordinate residual update
        x_out = x + v_out.squeeze(-2)  # [N, 3]

        # Edge embeddings from updated node features + RBF distance
        rows, cols = edges
        dist = torch.norm(x_out[rows] - x_out[cols], dim=-1)  # [E]
        dist_emb = self.rbf_embedding(dist)                    # [E, rbf_out_dim]
        edge_inputs = torch.cat([h_out[rows], h_out[cols], dist_emb], dim=-1)
        e_out = self.edge_out(edge_inputs)                     # [E, d_edge]

        return h_out, x_out, e_out