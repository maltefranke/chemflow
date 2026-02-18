import torch
import torch.nn as nn
import hydra
from omegaconf import DictConfig

from torch_geometric.utils import to_dense_batch


class Transformer(nn.Module):
    def __init__(
        self,
        in_node_nf: int,
        out_node_nf: int,
        in_edge_nf: int,
        rbf_embedding_args: DictConfig = None,
        d_model: int = 512,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        activation: str = "gelu",
        dropout: float = 0.0,
        norm_first: bool = True,
        bias: bool = True,
        num_layers: int = 6,
    ):
        super().__init__()

        # RBF distance embedding for edge prediction
        self.rbf_embedding = hydra.utils.instantiate(
            rbf_embedding_args
        )
        rbf_out_dim = rbf_embedding_args.get(
            "out_dim", rbf_embedding_args.get("num_rbf", 16)
        )

        # Input / output projections to match backbone interface
        self.node_in = nn.Sequential(
            nn.Linear(in_node_nf, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.node_out = nn.Sequential(
            nn.Linear(d_model, out_node_nf),
            nn.SiLU(),
            nn.Linear(out_node_nf, out_node_nf),
        )

        self.pos_embedding = nn.Sequential(
            nn.Linear(3, d_model, bias=False),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # Edge output MLP:
        #   src + tgt node feats + input edge feats + RBF dist
        edge_in_dim = 2 * out_node_nf + in_edge_nf + rbf_out_dim
        self.edge_output = nn.Sequential(
            nn.LayerNorm(edge_in_dim),
            nn.Linear(edge_in_dim, out_node_nf),
            nn.LayerNorm(out_node_nf),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        self.model = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                activation=activation,
                dropout=dropout,
                batch_first=True,
                norm_first=norm_first,
                bias=bias,
            ),
            norm=nn.LayerNorm(d_model),
            num_layers=num_layers,
        )

        self.pos_out = nn.Sequential(
            nn.Linear(out_node_nf, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 3),
        )

    def forward(self, h, x, edges, edge_attr, batch):
        # Project input node features to transformer dimension
        h = self.node_in(h)
        h = h + self.pos_embedding(x)

        # Pad to dense batch and run transformer with padding mask
        h_padded, atom_mask = to_dense_batch(h, batch)
        h_padded = self.model(h_padded, src_key_padding_mask=(~atom_mask))
        h = h_padded[atom_mask]

        # Project back to output dimension
        h = self.node_out(h)

        x = self.pos_out(h)

        # Edge embeddings from src/tgt node feats + RBF distance
        rows, cols = edges
        h_i = h[rows]   # (E, out_node_nf)
        h_j = h[cols]   # (E, out_node_nf)

        dist_vec = x[rows] - x[cols]          # (E, 3)
        dist = torch.norm(dist_vec, dim=1)     # (E,)
        dist_emb = self.rbf_embedding(dist)    # (E, rbf_out_dim)

        edge_inputs = torch.cat(
            [h_i, h_j, edge_attr, dist_emb], dim=-1,
        )
        edge_emb = self.edge_output(edge_inputs)

        return h, x, edge_emb
