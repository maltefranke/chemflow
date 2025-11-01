from src.external_code.egnn import EGNN
from omegaconf import DictConfig
import torch.nn as nn
import torch
import hydra


class BaseEGNN(nn.Module):
    """Base class for EGNN models. Embeds the atom feats and passes the data through the EGNN."""

    def __init__(self, embedding_args: DictConfig, egnn_args: DictConfig):
        super().__init__()
        self.embedding = hydra.utils.instantiate(embedding_args)
        self.egnn = hydra.utils.instantiate(egnn_args)

    def forward(self, atom_feats, coord, edge_index, edge_attr=None, node_attr=None):
        # first embed the atom feats
        h = self.embedding(atom_feats)

        # then pass the data through the EGNN
        h, coord, _ = self.egnn(h, edge_index, coord, edge_attr, node_attr)

        return h, coord


class EGNNwithHeads(BaseEGNN):
    """EGNN model with heads. Passes the data through an embedding layer, an EGNN and then through the heads."""

    def __init__(
        self, embedding_args: DictConfig, egnn_args: DictConfig, heads_args: DictConfig
    ):
        super().__init__(embedding_args, egnn_args)
        self.heads = hydra.utils.instantiate(heads_args)

    def forward(
        self, atom_feats, coord, edge_index, edge_attr=None, node_attr=None, batch=None
    ):
        """
        Forward pass through EGNN with heads.

        Args:
            atom_feats: Node features
            coord: Node coordinates
            edge_index: Edge indices
            edge_attr: Edge attributes (optional)
            node_attr: Node attributes (optional)
            batch: Batch assignment for each node (required for graph-level heads)

        Returns:
            Dictionary mapping head names to their outputs
        """
        h, coord = super().forward(atom_feats, coord, edge_index, edge_attr, node_attr)

        # Pass through heads
        if batch is not None:
            return self.heads(h, batch)
        else:
            # If no batch info provided, only node-level heads will work
            return self.heads(h, batch=None)
