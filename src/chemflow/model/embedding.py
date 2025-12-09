import torch.nn as nn
import torch
import math


class Embedding(nn.Module):
    """Simple embedding layer for node features."""

    def __init__(self, in_nf: int, out_nf: int):
        super().__init__()
        self.emb = nn.Embedding(in_nf, out_nf)

    def forward(self, x):
        return self.emb(x)


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
