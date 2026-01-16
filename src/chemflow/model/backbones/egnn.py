from src.external_code.egnn import EGNN
from omegaconf import DictConfig
import torch
import torch.nn as nn

import hydra


class EGNNWithEdgeType(EGNN):
    def __init__(self, *args, rbf_embedding_args: DictConfig = None, **kwargs):
        super().__init__(*args, **kwargs)
        out_node_nf = kwargs.get("out_node_nf", 0)

        self.rbf_embedding = hydra.utils.instantiate(rbf_embedding_args)
        rbf_out_dim = rbf_embedding_args.get(
            "out_dim", rbf_embedding_args.get("num_rbf", 16)
        )

        edge_dim = 2 * out_node_nf + rbf_out_dim
        projection_dim = out_node_nf
        self.edge_embedding_out = nn.Sequential(
            nn.LayerNorm(edge_dim),
            nn.Linear(edge_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

    def forward(self, h, x, edges, edge_attr):
        h = self.embedding_in(h)
        for i in range(0, self.n_layers):
            h, x, _ = self._modules["gcl_%d" % i](h, edges, x, edge_attr=edge_attr)

        h = self.embedding_out(h)

        rows, cols = edges

        # Collect features for every edge
        h_i = h[rows]  # [E, hidden]
        h_j = h[cols]  # [E, hidden]

        # Calculate final distances (actual distance, not squared)
        dist_vec = x[rows] - x[cols]  # [E, 3]
        dist = torch.norm(dist_vec, dim=1)  # [E]

        # Embed distances using RBF
        dist_emb = self.rbf_embedding(dist)  # [E, rbf_out_dim]

        # Concatenate: [Source Node, Target Node, RBF Distance Embedding]
        edge_inputs = torch.cat([h_i, h_j, dist_emb], dim=-1)

        # Predict
        edge_emb = self.edge_embedding_out(edge_inputs)

        return h, x, edge_emb
