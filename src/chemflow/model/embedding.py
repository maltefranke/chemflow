import torch.nn as nn
import torch
import math


class Embedding(nn.Module):
    """
    Advanced embedding layer for node features.

    Structure:
    Input Indices -> Lookup -> Linear -> Norm -> Activation -> Dropout -> Output
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        out_dim: int = None,
        dropout: float = 0.1,
    ):
        """
        Args:
            num_embeddings: Vocabulary size (e.g., number of unique node types).
            embedding_dim: Size of the initial lookup vector.
            out_dim: Size of the output vector (if None, typically matches embedding_dim).
            dropout: Probability of dropout.
        """
        super().__init__()

        # If out_dim isn't specified, keep dimensions constant
        if out_dim is None:
            out_dim = embedding_dim

        # 1. The Base Lookup
        self.emb = nn.Embedding(num_embeddings, embedding_dim)

        # 2. The Feature Projection Block (MLP)
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Optional: Initialize weights specifically for embeddings if needed
        self._init_weights()

    def _init_weights(self):
        # Xavier initialization often works better than default for projections
        nn.init.xavier_uniform_(self.emb.weight)
        for module in self.projection.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)

    def forward(self, x):
        # x shape: [batch_size, num_nodes] or [num_nodes]

        # Get raw vectors
        x_emb = self.emb(x)

        # Apply non-linearity and normalization
        x_out = self.projection(x_emb)

        return x_out


class SinusoidalEncoding(nn.Module):
    """
    The mathematical engine for sinusoidal embeddings.
    Computes: [sin(x*w0), cos(x*w0), sin(x*w1), cos(x*w1)...]
    """

    def __init__(self, embedding_dim: int, max_period: float = 10000.0):
        super().__init__()
        if embedding_dim % 2 != 0:
            raise ValueError(f"Embedding dimension {embedding_dim} must be even.")

        self.embedding_dim = embedding_dim
        half_dim = embedding_dim // 2

        # Calculate frequencies (DDPM style / Vaswani style)
        # We want frequencies ranging from 1 to 1/max_period
        # The formula creates a geometric progression of frequencies
        lambda_max = max_period

        # Derived from: exp(-2 * i / d * ln(10000))
        # This matches the JAX 'half_dim - 1' logic for exact boundary alignment
        exponent = -math.log(lambda_max) * torch.arange(
            start=0, end=half_dim, dtype=torch.float32
        )
        exponent = exponent / (half_dim - 1)

        freqs = torch.exp(exponent)

        # Register as fixed buffer (not a learnable parameter)
        self.register_buffer("freqs", freqs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [Batch, 1] (values can be float or int)
        Returns:
            Tensor of shape [Batch, embedding_dim]
        """
        # x: [Batch, 1], freqs: [Half_Dim]
        # args: [Batch, Half_Dim]
        args = x.float() * self.freqs.unsqueeze(0)

        # Concatenate sin and cos -> [Batch, Dim]
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

        return embedding


class CountEmbedding(nn.Module):
    """
    Embeds integer counts (e.g., Number of Nodes).
    Input:  Long Tensor [Batch] (e.g., [5, 20, 10])
    Output: Float Tensor [Batch, Dim]
    """

    def __init__(
        self, embedding_dim: int, out_dim: int = None, max_period: float = 10000.0
    ):
        super().__init__()
        if out_dim is None:
            out_dim = embedding_dim
        self.encoder = SinusoidalEncoding(embedding_dim, max_period)

        # Optional: A small MLP to adapt the fixed features to the task
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    def forward(self, counts: torch.Tensor) -> torch.Tensor:
        # Ensure input is [Batch, 1] for the encoder
        if counts.ndim == 1:
            counts = counts.unsqueeze(-1)

        x_enc = self.encoder(counts)
        return self.projection(x_enc)


class TimeEmbedding(nn.Module):
    """
    Embeds continuous time/noise levels.
    Input:  Float Tensor [Batch, 1] (e.g., values 0.0 to 1.0)
    Output: Float Tensor [Batch, Dim]
    """

    def __init__(self, embedding_dim: int, out_dim: int = None):
        super().__init__()
        if out_dim is None:
            out_dim = embedding_dim
        # For time in [0,1], max_period=10000 ensures high precision
        # for even very small time steps.
        self.encoder = SinusoidalEncoding(embedding_dim, max_period=10000.0)

        # Standard MLP block for Diffusion/Flow models
        # Projects the fixed sinusoids into the learnable semantic space
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.SiLU(),  # SiLU is standard for time
            nn.Linear(embedding_dim, out_dim),
        )

    def forward(self, times: torch.Tensor) -> torch.Tensor:
        # Ensure input is [Batch, 1]
        if times.ndim == 1:
            times = times.unsqueeze(-1)

        t_enc = self.encoder(times)
        return self.mlp(t_enc)


class RBFEncoding(nn.Module):
    """
    Radial Basis Function (RBF) encoding for distances.
    Computes RBF features using Gaussian basis functions.

    Input:  Float Tensor [N] (distances)
    Output: Float Tensor [N, num_rbf]
    """

    def __init__(self, num_rbf: int = 16, rbf_dmax: float = 10.0):
        """
        Args:
            num_rbf: Number of RBF basis functions
            rbf_dmax: Maximum distance for RBF
        """
        super().__init__()
        self.num_rbf = num_rbf
        self.rbf_dmax = rbf_dmax

        # RBF centers for distance encoding
        self.register_buffer("rbf_centers", torch.linspace(0, rbf_dmax, num_rbf))
        self.rbf_gamma = 1.0 / (rbf_dmax / num_rbf)

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """
        Compute RBF features for distances.

        Args:
            distances: Tensor of shape [N] or [N, 1] containing distances

        Returns:
            Tensor of shape [N, num_rbf] containing RBF encodings
        """
        # Ensure distances is [N]
        if distances.ndim > 1:
            distances = distances.squeeze(-1)

        # Compute RBF features: exp(-gamma * (dist - centers)^2)
        # distances: [N], rbf_centers: [num_rbf]
        # distances.unsqueeze(-1): [N, 1], rbf_centers: [num_rbf]
        # Broadcasting: [N, 1] - [num_rbf] -> [N, num_rbf]
        return torch.exp(
            -self.rbf_gamma * (distances.unsqueeze(-1) - self.rbf_centers) ** 2
        )


class RBFEmbedding(nn.Module):
    """
    Radial Basis Function (RBF) embedding for distances with projection layer.
    Wraps RBFEncoding with a learnable projection MLP.

    Structure:
    Input Distances -> RBFEncoding -> Linear -> Norm -> Activation -> Dropout -> Output

    Input:  Float Tensor [N] (distances)
    Output: Float Tensor [N, out_dim]
    """

    def __init__(
        self,
        num_rbf: int = 16,
        rbf_dmax: float = 10.0,
        out_dim: int = None,
        dropout: float = 0.0,
    ):
        """
        Args:
            num_rbf: Number of RBF basis functions
            rbf_dmax: Maximum distance for RBF
            out_dim: Output dimension (if None, uses num_rbf)
            dropout: Dropout probability
        """
        super().__init__()

        # If out_dim isn't specified, keep dimensions constant
        if out_dim is None:
            out_dim = num_rbf

        # Store out_dim for easy access
        self.out_dim = out_dim

        # RBF encoding layer
        self.rbf_encoding = RBFEncoding(num_rbf=num_rbf, rbf_dmax=rbf_dmax)

        # Simple projection block (matching Embedding class style)
        self.projection = nn.Sequential(
            nn.Linear(num_rbf, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for projection layers."""
        for module in self.projection.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """
        Compute RBF embeddings for distances.

        Args:
            distances: Tensor of shape [N] or [N, 1] containing distances

        Returns:
            Tensor of shape [N, out_dim] containing RBF embeddings
        """
        # Get RBF encoding
        rbf_enc = self.rbf_encoding(distances)  # [N, num_rbf]

        # Apply projection
        return self.projection(rbf_enc)  # [N, out_dim]
