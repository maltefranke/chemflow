import torch.nn as nn
import torch
import math

from rdkit.Chem import GetPeriodicTable

_PERIODIC_TABLE = GetPeriodicTable()


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


class PropertyEmbedding(nn.Module):
    """
    Embeds continuous molecular property vectors for classifier-free guidance.

    Includes a learnable null embedding for the unconditional case (CFG dropout).
    Properties are graph-level features that get broadcast to all nodes in the graph.

    Input:  Float Tensor [Batch, num_properties] (e.g., QM9 has 19 properties)
    Output: Float Tensor [Batch, out_dim]
    """

    def __init__(
        self,
        num_properties: int,
        embedding_dim: int,
        out_dim: int = None,
    ):
        super().__init__()
        if out_dim is None:
            out_dim = embedding_dim

        self.out_dim = out_dim

        self.mlp = nn.Sequential(
            nn.Linear(num_properties, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, out_dim),
        )

        self.null_embedding = nn.Parameter(torch.randn(out_dim) * 0.02)
        self._init_weights()

    def _init_weights(self):
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        properties: torch.Tensor | None,
        drop_mask: torch.Tensor | None = None,
        batch_size: int | None = None,
    ) -> torch.Tensor:
        """
        Args:
            properties: [batch_size, num_properties] or None.
                        If None, returns null embeddings.
            drop_mask:  [batch_size] bool. True = use null (unconditional).
            batch_size: Explicit batch size for the null embedding when
                        both properties and drop_mask are None.
        """
        if properties is None:
            if batch_size is None:
                batch_size = drop_mask.shape[0] if drop_mask is not None else 1
            return self.null_embedding.unsqueeze(0).expand(batch_size, -1)

        emb = self.mlp(properties)

        if drop_mask is not None:
            # Always use torch.where without .any() guard so null_embedding is always
            # in the computation graph. When drop_mask is all-False, null_embedding
            # receives a zero gradient (not None), satisfying DDP's requirement that
            # every requires_grad parameter participates in every backward pass.
            null = self.null_embedding.unsqueeze(0).expand_as(emb)
            emb = torch.where(drop_mask.unsqueeze(-1), null, emb)

        return emb


class NAtomsCFGEmbedding(nn.Module):
    """Embeds a target atom count with classifier-free guidance dropout.

    Wraps :class:`CountEmbedding` (sinusoidal encoding + learned MLP) and adds
    a learnable null embedding for the unconditional case.  API mirrors
    :class:`PropertyEmbedding` so the two can coexist independently.

    Input:  Long/Float Tensor [Batch] (target atom counts, e.g. [5, 20, 10])
    Output: Float Tensor [Batch, out_dim]
    """

    def __init__(
        self,
        embedding_dim: int,
        out_dim: int = None,
        max_period: float = 100.0,
    ):
        super().__init__()
        if out_dim is None:
            out_dim = embedding_dim
        self.out_dim = out_dim

        self.count_embedding = CountEmbedding(
            embedding_dim=embedding_dim, out_dim=out_dim, max_period=max_period
        )
        self.null_embedding = nn.Parameter(torch.randn(out_dim) * 0.02)

    def forward(
        self,
        target_n_atoms: torch.Tensor | None,
        drop_mask: torch.Tensor | None = None,
        batch_size: int | None = None,
    ) -> torch.Tensor:
        """
        Args:
            target_n_atoms: [batch_size] integer counts, or None for fully
                            unconditional (returns null embeddings).
            drop_mask:      [batch_size] bool.  True → use null (unconditional).
            batch_size:     Explicit size when both inputs are None.
        """
        if target_n_atoms is None:
            if batch_size is None:
                batch_size = drop_mask.shape[0] if drop_mask is not None else 1
            return self.null_embedding.unsqueeze(0).expand(batch_size, -1)

        emb = self.count_embedding(target_n_atoms)

        if drop_mask is not None:
            # Always use torch.where without .any() guard so null_embedding is always
            # in the computation graph. When drop_mask is all-False, null_embedding
            # receives a zero gradient (not None), satisfying DDP's requirement that
            # every requires_grad parameter participates in every backward pass.
            null = self.null_embedding.unsqueeze(0).expand_as(emb)
            emb = torch.where(drop_mask.unsqueeze(-1), null, emb)

        return emb


def compute_molecular_weight(
    atom_indices: torch.Tensor,
    atom_tokens: list[str],
    batch: torch.Tensor | None = None,
    num_graphs: int | None = None,
) -> torch.Tensor:
    """Compute molecular weight per graph from atom token indices.

    Uses RDKit's periodic table for accurate atomic weights.

    Args:
        atom_indices: (N,) integer tensor of atom type indices.
        atom_tokens: ordered list of element symbols (vocab).
        batch: (N,) graph membership for each atom.
        num_graphs: number of graphs in the batch.

    Returns:
        (num_graphs,) tensor of molecular weights in Daltons.
        If batch / num_graphs are not provided, returns a scalar.
    """
    weights = torch.tensor(
        [_PERIODIC_TABLE.GetAtomicWeight(tok) for tok in atom_tokens],
        dtype=torch.float,
        device=atom_indices.device,
    )
    per_atom_mw = weights[atom_indices.long()]

    if batch is not None and num_graphs is not None:
        mw = torch.zeros(num_graphs, device=atom_indices.device)
        mw.scatter_add_(0, batch, per_atom_mw)
        return mw

    return per_atom_mw.sum().unsqueeze(0)


def _encode_signal(
    value: torch.Tensor | None,
    encoder: nn.Module,
    null_emb: nn.Parameter,
    drop_mask: torch.Tensor | None,
    batch_size: int,
) -> torch.Tensor:
    """Shared logic for encoding a single CFG signal with dropout."""
    if value is None:
        return null_emb.unsqueeze(0).expand(batch_size, -1)
    if value.ndim == 1:
        value = value.unsqueeze(-1) if value.dtype.is_floating_point else value
    emb = encoder(value)
    if drop_mask is not None:
        null = null_emb.unsqueeze(0).expand_as(emb)
        emb = torch.where(drop_mask.unsqueeze(-1), null, emb)
    return emb


class UnifiedCFGEmbedding(nn.Module):
    """Unified classifier-free guidance embedding.

    Bundles property, n_atoms, and molecular weight conditioning into a
    single module.  Each signal has its own sub-encoder and learnable null
    embedding; the sub-embeddings are concatenated and projected to
    ``out_dim``.

    Accepts a ``cfg_inputs`` dict so callers do not need separate kwargs
    per signal.

    Input dict keys (all optional):
        properties, property_drop_mask,
        target_n_atoms, natoms_drop_mask,
        target_mw, mw_drop_mask

    Output: Float Tensor [Batch, out_dim]
    """

    def __init__(
        self,
        out_dim: int,
        # Property conditioning (num_properties=0 disables)
        num_properties: int = 0,
        property_hidden_dim: int = 128,
        # N-atoms conditioning
        use_natoms: bool = False,
        natoms_sinusoidal_dim: int = 64,
        natoms_max_period: float = 100.0,
        # MW conditioning
        use_mw: bool = False,
        mw_sinusoidal_dim: int = 64,
        mw_max_period: float = 1000.0,
    ):
        super().__init__()
        self.out_dim = out_dim
        internal_dim = 0

        self.property_encoder = None
        self._property_null = None
        if num_properties > 0:
            prop_out = property_hidden_dim
            self.property_encoder = nn.Sequential(
                nn.Linear(num_properties, property_hidden_dim),
                nn.LayerNorm(property_hidden_dim),
                nn.SiLU(),
                nn.Linear(property_hidden_dim, prop_out),
            )
            self._property_null = nn.Parameter(torch.randn(prop_out) * 0.02)
            internal_dim += prop_out

        self.natoms_encoder = None
        self._natoms_null = None
        if use_natoms:
            self.natoms_encoder = CountEmbedding(
                embedding_dim=natoms_sinusoidal_dim,
                out_dim=natoms_sinusoidal_dim,
                max_period=natoms_max_period,
            )
            self._natoms_null = nn.Parameter(torch.randn(natoms_sinusoidal_dim) * 0.02)
            internal_dim += natoms_sinusoidal_dim

        self.mw_encoder = None
        self._mw_null = None
        if use_mw:
            self._mw_sinusoidal = SinusoidalEncoding(
                mw_sinusoidal_dim, max_period=mw_max_period
            )
            self.mw_encoder = nn.Sequential(
                nn.Linear(mw_sinusoidal_dim, mw_sinusoidal_dim),
                nn.SiLU(),
                nn.Linear(mw_sinusoidal_dim, mw_sinusoidal_dim),
            )
            self._mw_null = nn.Parameter(torch.randn(mw_sinusoidal_dim) * 0.02)
            internal_dim += mw_sinusoidal_dim

        if internal_dim > 0:
            self.projection = nn.Sequential(
                nn.Linear(internal_dim, out_dim),
                nn.SiLU(),
                nn.Linear(out_dim, out_dim),
            )
        else:
            self.projection = None

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _encode_mw(self, value, drop_mask, batch_size):
        if value is None:
            return self._mw_null.unsqueeze(0).expand(batch_size, -1)
        if value.ndim == 1:
            value = value.unsqueeze(-1)
        emb = self.mw_encoder(self._mw_sinusoidal(value))
        if drop_mask is not None:
            null = self._mw_null.unsqueeze(0).expand_as(emb)
            emb = torch.where(drop_mask.unsqueeze(-1), null, emb)
        return emb

    def forward(
        self,
        cfg_inputs: dict,
        batch_size: int,
    ) -> torch.Tensor:
        parts: list[torch.Tensor] = []

        if self.property_encoder is not None:
            parts.append(
                _encode_signal(
                    cfg_inputs.get("properties"),
                    self.property_encoder,
                    self._property_null,
                    cfg_inputs.get("property_drop_mask"),
                    batch_size,
                )
            )

        if self.natoms_encoder is not None:
            parts.append(
                _encode_signal(
                    cfg_inputs.get("target_n_atoms"),
                    self.natoms_encoder,
                    self._natoms_null,
                    cfg_inputs.get("natoms_drop_mask"),
                    batch_size,
                )
            )

        if self.mw_encoder is not None:
            parts.append(
                self._encode_mw(
                    cfg_inputs.get("target_mw"),
                    cfg_inputs.get("mw_drop_mask"),
                    batch_size,
                )
            )

        if not parts:
            device = next(self.parameters()).device
            return torch.zeros(
                batch_size, self.out_dim, device=device,
            )

        return self.projection(torch.cat(parts, dim=-1))


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


class BondDegreeEmbedding(nn.Module):
    """Per-atom structural embedding based on bond-type degree histogram.

    For each atom, counts how many edges of each type it has, then embeds
    each count through a type-specific ``CountEmbedding`` (sinusoidal
    encoding + learned projection).  This gives transformers explicit
    awareness of local graph structure (e.g. detecting unattached atoms,
    saturated vs unsaturated centres).
    """

    def __init__(
        self,
        n_edge_types: int,
        out_dim: int,
        per_type_dim: int = 16,
    ):
        super().__init__()
        self.n_edge_types = n_edge_types

        self.count_embeddings = nn.ModuleList(
            [
                CountEmbedding(
                    embedding_dim=per_type_dim, out_dim=per_type_dim, max_period=100
                )
                for _ in range(n_edge_types)
            ]
        )
        total_dim = n_edge_types * per_type_dim
        self.proj = nn.Sequential(
            nn.Linear(total_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        e: torch.Tensor,
        edge_index_row: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        """
        Args:
            e: (E,) edge type indices
            edge_index_row: (E,) source node index for each edge
            num_nodes: total number of nodes in the batch

        Returns:
            (N, out_dim) structural embeddings per atom
        """
        e_onehot = torch.nn.functional.one_hot(
            e.long(), self.n_edge_types
        )  # (E, n_edge_types)
        bond_hist = torch.zeros(
            num_nodes, self.n_edge_types, dtype=torch.long, device=e.device
        )
        bond_hist.scatter_add_(
            0,
            edge_index_row.unsqueeze(1).expand_as(e_onehot),
            e_onehot.long(),
        )

        type_embeds = [
            self.count_embeddings[k](bond_hist[:, k]) for k in range(self.n_edge_types)
        ]
        combined = torch.cat(type_embeds, dim=-1)
        return self.proj(combined)
