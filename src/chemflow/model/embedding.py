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
        dropout: float = 0.0,
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
        self, embedding_dim: int, out_dim: int = None, max_period: float = 100.0
    ):
        super().__init__()
        if out_dim is None:
            out_dim = embedding_dim

        self.embedding_dim = embedding_dim
        self.out_dim = out_dim

        self.encoder = SinusoidalEncoding(embedding_dim, max_period)

        # Optional: A small MLP to adapt the fixed features to the task
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, out_dim),
            # nn.LayerNorm(out_dim),
            # nn.GELU(),
        )

    def forward(self, counts: torch.Tensor) -> torch.Tensor:
        # Ensure input is [Batch, 1] for the encoder
        if counts.ndim == 1:
            counts = counts.unsqueeze(-1)

        x_enc = self.encoder(counts)
        return self.projection(x_enc)


class ExtrapolatableCountEmbedding(nn.Module):
    """Count embedding designed to extrapolate slightly outside the training range.

    The standard :class:`CountEmbedding` concatenates sinusoidal features whose
    shortest period is ``2π`` (the geometric series of frequencies starts at
    ``freq=1``).  For small integer counts (e.g. QM9 with counts in ``[3, 29]``)
    this produces aliased, non-monotonic features: neighbouring counts map to
    unrelated sin/cos patterns, and counts just outside the training support
    land on patterns the downstream linear projection has never seen — so it
    cannot extrapolate.

    This module fixes both issues:

    * It uses a **low-frequency** Fourier basis whose shortest period is larger
      than the expected training support (``min_period``), so every dimension
      varies smoothly over the training range and its immediate neighbourhood.
    * It concatenates a **raw normalised scalar** ``count / max_count``, giving
      the MLP a strictly linear (monotonic, unbounded) axis to extrapolate
      along — the Fourier basis only contributes local precision within range.

    With defaults tuned for QM9 (``min_period=64``, ``max_period=512``,
    ``max_count=32``), counts in ``[0, 32]`` map to half a period at most on
    any Fourier component, and ``count=30`` — just outside the training
    support — produces an embedding that is a smooth continuation of the
    ``count=29`` embedding rather than a novel aliased fingerprint.

    Input:  Long / Float Tensor ``[Batch]`` or ``[Batch, 1]``.
    Output: Float Tensor ``[Batch, out_dim]``.
    """

    def __init__(
        self,
        embedding_dim: int,
        out_dim: int = None,
        min_period: float = 64.0,
        max_period: float = 512.0,
        max_count: float = 32.0,
    ):
        super().__init__()
        if out_dim is None:
            out_dim = embedding_dim
        if embedding_dim % 2 != 0:
            raise ValueError(
                f"embedding_dim {embedding_dim} must be even (sin/cos pairs)."
            )
        if min_period <= 0 or max_period < min_period:
            raise ValueError(
                f"Expected 0 < min_period <= max_period, got "
                f"min_period={min_period}, max_period={max_period}."
            )
        if max_count <= 0:
            raise ValueError(f"max_count must be > 0, got {max_count}.")

        self.embedding_dim = embedding_dim
        self.out_dim = out_dim
        self.min_period = float(min_period)
        self.max_period = float(max_period)
        self.max_count = float(max_count)

        # Log-spaced periods in [min_period, max_period].  Highest frequency
        # has the *shortest* period (= min_period), ensuring every component
        # stays in a monotonic half-cycle across [0, max_count].
        half_dim = embedding_dim // 2
        if half_dim == 1:
            periods = torch.tensor([min_period], dtype=torch.float32)
        else:
            log_pmin = math.log(min_period)
            log_pmax = math.log(max_period)
            periods = torch.exp(torch.linspace(log_pmin, log_pmax, half_dim))
        self.register_buffer("freqs", 2 * math.pi / periods)

        # MLP takes [fourier_features, raw_scalar] and mixes them.  The raw
        # scalar is the path responsible for linear extrapolation.
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim + 1, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, counts: torch.Tensor) -> torch.Tensor:
        if counts.ndim == 1:
            counts = counts.unsqueeze(-1)

        x = counts.float()
        args = x * self.freqs.unsqueeze(0)
        fourier = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        scalar = x / self.max_count
        return self.projection(torch.cat([fourier, scalar], dim=-1))


class ExtrapolatableScalarEmbedding(nn.Module):
    """Scalar (continuous, possibly negative) embedding designed to extrapolate
    slightly outside the training range.

    Continuous-valued analogue of :class:`ExtrapolatableCountEmbedding`.  Suited
    for signals like RDKit Crippen logP, molecular weight, or any other
    real-valued conditioning scalar where:

    * Training values can be negative.
    * The desired query range at inference exceeds the training support
      (e.g. QM9 trains at logP ∈ [-3, 4] but you want to query logP=6).

    The naive ``SinusoidalEncoding(... , max_period=20)`` + MLP path used by
    the legacy :class:`UnifiedCFGEmbedding` logp/mw branches has two failure
    modes:

    * Its highest-frequency components have wavelengths comparable to (or
      shorter than) the training range, so they alias inside-distribution and
      produce never-seen sin/cos quadrants for slight OOD queries.
    * Every output dim is bounded in ``[-1, 1]``, so there is no monotonic /
      unbounded direction the downstream MLP can ride to "go further".

    This module fixes both:

    * Uses a low-frequency Fourier basis whose **shortest period
      (``min_period``)** is larger than the expected training-range span
      (``value_max - value_min``), so every Fourier dim varies smoothly across
      the training range and its immediate neighbourhood.
    * Concatenates a **raw centred-and-normalised scalar**
      ``(value - mid) / half_range`` (``mid = (value_min + value_max) / 2``,
      ``half_range = (value_max - value_min) / 2``) so the projection MLP has
      a strictly linear, unbounded axis to extrapolate along — the Fourier
      basis only contributes local precision within range.

    Defaults are tuned for QM9 logP (training span ~``[-3, 4]``) with query
    headroom up to logP=8: ``min_period=16`` (> training-span 7), plus
    ``value_min=-4``, ``value_max=8`` so ``value=8`` maps to scalar=+1 and
    ``value=14`` would map to scalar=+2 (extrapolation direction).

    Args:
        embedding_dim: Number of Fourier features (must be even — sin/cos
            pairs).
        out_dim: Output embedding size.  Defaults to ``embedding_dim``.
        min_period: Shortest Fourier wavelength in input units.  Must
            exceed ``(value_max - value_min)`` so every dim is monotonic
            across the training range.
        max_period: Longest Fourier wavelength.
        value_min: Lower end of the expected (training) value range.  Used
            only to centre/normalise the raw scalar pass-through.
        value_max: Upper end of the expected (training) value range.

    Input:  Float Tensor ``[Batch]`` or ``[Batch, 1]``.
    Output: Float Tensor ``[Batch, out_dim]``.
    """

    def __init__(
        self,
        embedding_dim: int,
        out_dim: int = None,
        min_period: float = 16.0,
        max_period: float = 64.0,
        value_min: float = -4.0,
        value_max: float = 8.0,
    ):
        super().__init__()
        if out_dim is None:
            out_dim = embedding_dim
        if embedding_dim % 2 != 0:
            raise ValueError(
                f"embedding_dim {embedding_dim} must be even (sin/cos pairs)."
            )
        if min_period <= 0 or max_period < min_period:
            raise ValueError(
                f"Expected 0 < min_period <= max_period, got "
                f"min_period={min_period}, max_period={max_period}."
            )
        if value_max <= value_min:
            raise ValueError(
                f"Expected value_max > value_min, got "
                f"value_min={value_min}, value_max={value_max}."
            )

        self.embedding_dim = embedding_dim
        self.out_dim = out_dim
        self.min_period = float(min_period)
        self.max_period = float(max_period)
        self.value_min = float(value_min)
        self.value_max = float(value_max)
        self.value_mid = 0.5 * (self.value_min + self.value_max)
        self.value_half_range = 0.5 * (self.value_max - self.value_min)

        half_dim = embedding_dim // 2
        if half_dim == 1:
            periods = torch.tensor([min_period], dtype=torch.float32)
        else:
            log_pmin = math.log(min_period)
            log_pmax = math.log(max_period)
            periods = torch.exp(torch.linspace(log_pmin, log_pmax, half_dim))
        self.register_buffer("freqs", 2 * math.pi / periods)

        # MLP takes [fourier_features, raw_centred_scalar] and mixes them.
        # The raw scalar is the path responsible for linear extrapolation.
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim + 1, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim == 1:
            value = value.unsqueeze(-1)

        x = value.float()
        args = x * self.freqs.unsqueeze(0)
        fourier = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        scalar = (x - self.value_mid) / self.value_half_range
        return self.projection(torch.cat([fourier, scalar], dim=-1))


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
        self.encoder = SinusoidalEncoding(embedding_dim, max_period=5000.0)

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

        times = times * 1000.0
        t_enc = self.encoder(times)
        return self.mlp(t_enc)


class RBFEncoding(nn.Module):
    """
    Trainable Radial Basis Function (RBF) encoding.
    Both the centers and the widths (sigmas) of the Gaussians are learnable.
    """

    def __init__(
        self, num_rbf: int = 16, rbf_dmax: float = 10.0, trainable: bool = True
    ):
        super().__init__()
        self.num_rbf = num_rbf
        self.rbf_dmax = rbf_dmax

        # 1. Initialize centers evenly spaced (same as before)
        initial_centers = torch.linspace(0, rbf_dmax, num_rbf)

        # 2. Initialize sigmas (widths)
        # The gap between centers is dmax / num_rbf.
        # A good default sigma is often the gap size itself or slightly smaller.
        initial_sigma = (rbf_dmax / num_rbf) * torch.ones(num_rbf)

        if trainable:
            # Wrap in nn.Parameter to enable gradient descent
            self.rbf_centers = nn.Parameter(initial_centers)
            self.rbf_sigma = nn.Parameter(initial_sigma)
        else:
            # Keep them fixed if desired
            self.register_buffer("rbf_centers", initial_centers)
            self.register_buffer("rbf_sigma", initial_sigma)

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """Returns: [N, num_rbf]"""
        if distances.ndim > 1:
            distances = distances.squeeze(-1)

        # Broadcasting logic:
        # distances:       [N, 1]
        # centers/sigmas:  [num_rbf]

        # We calculate gamma dynamically based on the current learnable sigma
        # Gamma = 1 / sigma^2
        # We use abs() or clamp() on sigma to ensure widths stay positive and non-zero
        sigmas = torch.abs(self.rbf_sigma) + 1e-5  # Stability trick
        gamma = 1.0 / (sigmas**2)

        # Formula: exp( -gamma * (x - mu)^2 )
        return torch.exp(-gamma * (distances.unsqueeze(-1) - self.rbf_centers) ** 2)


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
        trainable: bool = True,
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
        self.rbf_encoding = RBFEncoding(
            num_rbf=num_rbf, rbf_dmax=rbf_dmax, trainable=trainable
        )

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
