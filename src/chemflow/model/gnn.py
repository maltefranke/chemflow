from src.external_code.egnn import EGNN
from omegaconf import DictConfig
import torch.nn as nn
import torch
import hydra

import math


class SinusoidalEmbedding(nn.Module):
    """
    Applies sinusoidal (periodic) embeddings to a 1D tensor of integers.

    This module is commonly used for positional encoding, but here we adapt it
    to embed any scalar integer, such as the number of nodes in a graph.
    """

    def __init__(self, embedding_dim: int):
        """
        Initializes the SinusoidalEmbedding module.

        Args:
            embedding_dim (int): The dimension of the output embedding.
                                 Must be an even number.
        """
        super().__init__()

        if embedding_dim % 2 != 0:
            raise ValueError(f"Embedding dimension {embedding_dim} must be even.")

        self.embedding_dim = embedding_dim

        # Calculate the 'div_term' buffer: 1 / 10000^(2i/d)
        # This is done in log-space for numerical stability
        exponent = torch.arange(0, embedding_dim, 2).float() * (
            -math.log(10000.0) / embedding_dim
        )
        div_term = torch.exp(exponent)
        self.register_buffer("div_term", div_term)

    def forward(self, n: torch.Tensor) -> torch.Tensor:
        """
        Computes the sinusoidal embedding for the input tensor.

        Args:
            n (torch.Tensor): A 1D tensor of integers (e.g., node counts)
                              of shape (batch_size,).

        Returns:
            torch.Tensor: The sinusoidal embeddings of shape
                          (batch_size, embedding_dim).
        """

        # Ensure n is a float tensor and has the right shape
        # We need n to be (batch_size, 1) to multiply with div_term (1, d/2)
        # to get (batch_size, d/2)
        n_float = n.float().unsqueeze(1)

        # Calculate the arguments for sin and cos
        # Shape: (batch_size, embedding_dim / 2)
        arg = n_float * self.div_term

        # Initialize the embedding tensor
        # Shape: (batch_size, embedding_dim)
        pe = torch.zeros(n.shape[0], self.embedding_dim, device=n.device)

        # Apply sin to even indices (0, 2, 4, ...)
        pe[:, 0::2] = torch.sin(arg)

        # Apply cos to odd indices (1, 3, 5, ...)
        pe[:, 1::2] = torch.cos(arg)

        return pe


class EGNNWithEdgeType(EGNN):
    def forward(self, h, x, edges, edge_attr):
        h = self.embedding_in(h)
        for i in range(0, self.n_layers):
            h, x, edge_attr = self._modules["gcl_%d" % i](
                h, edges, x, edge_attr=edge_attr
            )
        h = self.embedding_out(h)
        edge_attr = self.embedding_out(edge_attr)
        return h, x, edge_attr


class BaseEGNN(nn.Module):
    """Base class for EGNN models. Embeds the atom feats and passes the data through the EGNN."""

    def __init__(
        self,
        atom_type_embedding_args: DictConfig,
        edge_type_embedding_args: DictConfig,
        egnn_args: DictConfig,
    ):
        super().__init__()
        self.atom_type_embedding = hydra.utils.instantiate(atom_type_embedding_args)
        self.edge_type_embedding = hydra.utils.instantiate(edge_type_embedding_args)
        self.egnn = hydra.utils.instantiate(egnn_args)

    def forward(self, atom_feats, coord, edge_index, edge_type_ids=None):
        # first embed the atom feats
        h = self.atom_type_embedding(atom_feats)

        if edge_type_ids is not None:
            edge_type_embeddings = self.edge_type_embedding(edge_type_ids)
            edge_attr = edge_type_embeddings
        else:
            edge_attr = None

        edge_index = (edge_index[0], edge_index[1])

        # then pass the data through the EGNN
        h, coord, edge_attr = self.egnn(h, coord, edge_index, edge_attr)

        return h, coord, edge_attr


class EGNNwithHeads(BaseEGNN):
    """EGNN model with heads. Passes the data through an embedding layer, an EGNN and then through the heads."""

    def __init__(
        self,
        atom_type_embedding_args: DictConfig,
        edge_type_embedding_args: DictConfig,
        egnn_args: DictConfig,
        heads_args: DictConfig,
    ):
        super().__init__(atom_type_embedding_args, edge_type_embedding_args, egnn_args)
        self.time_embedding = nn.Sequential(
            nn.Linear(1, atom_type_embedding_args["out_nf"]),
            nn.SiLU(),
            nn.Linear(
                atom_type_embedding_args["out_nf"], atom_type_embedding_args["out_nf"]
            ),
        )
        self.sinusoidal_embedding = SinusoidalEmbedding(
            atom_type_embedding_args["out_nf"]
        )
        self.heads = hydra.utils.instantiate(heads_args)

    def forward(self, atom_feats, coord, edge_index, t, batch, edge_type_ids=None):
        """
        Forward pass through EGNN with heads.

        Args:
            atom_feats: Node features
            coord: Node coordinates
            edge_index: Edge indices
            edge_type_ids: Edge type ids (optional)
            batch: Batch assignment for each node (required for graph-level heads)

        Returns:
            Dictionary mapping head names to their outputs
        """
        N_nodes = torch.bincount(batch)

        h = self.atom_type_embedding(atom_feats)

        # calculate conditioning embeddings
        N_nodes_embedding = self.sinusoidal_embedding(N_nodes)[batch]
        t_embedding = self.time_embedding(t)[batch]
        h = h + N_nodes_embedding + t_embedding

        edge_index = (edge_index[0], edge_index[1])

        if edge_type_ids is not None:
            edge_attr = self.edge_type_embedding(edge_type_ids)
        else:
            edge_attr = None

        # then pass the data through the EGNN
        h, coord, edge_attr = self.egnn(h, coord, edge_index, edge_attr=edge_attr)

        # Pass through heads
        if batch is not None:
            out_dict = self.heads(h, batch, edge_attr=edge_attr)
        else:
            # If no batch info provided, only node-level heads will work
            out_dict = self.heads(h, batch=None, edge_attr=edge_attr)

        out_dict["pos_head"] = coord

        return out_dict
