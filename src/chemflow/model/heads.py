import torch
import torch.nn as nn
from typing import Dict, Optional
from omegaconf import DictConfig
from src.external_code.egnn import unsorted_segment_sum, unsorted_segment_mean
import torch.nn.functional as F
import hydra


class OutputHead(nn.Module):
    """
    A standard output MLP head for GNNs.

    Structure per hidden layer:
    Linear -> Norm -> Activation -> Dropout

    Final layer:
    Linear (Output Logits)
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int | None = None,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = input_dim

        layers = []

        # --- Build Hidden Layers ---
        # If num_layers > 1, we add hidden layers with norm/act/dropout
        cur_dim = input_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(cur_dim, hidden_dim))

            # Normalization (Crucial for deep GNNs)
            layers.append(nn.LayerNorm(hidden_dim))

            # Activation
            layers.append(nn.SiLU())

            # Dropout (Regularization)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            cur_dim = hidden_dim

        # --- Final Projection ---
        layers.append(nn.Linear(cur_dim, output_dim))

        self.mlp = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for better convergence."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.mlp(h)


class MultiHeadModule(nn.Module):
    """Module that combines multiple heads (both node-level and graph-level)."""

    def __init__(self, heads_configs: DictConfig):
        super().__init__()
        self.heads = nn.ModuleDict()
        self.head_types = {}

        # Store aggregation method for graph heads (default to "mean")
        self.graph_aggregation = heads_configs.get("graph_aggregation", "mean")

        # Create node heads
        if "node_heads" in heads_configs:
            for name, config in heads_configs.node_heads.items():
                self.heads[name] = OutputHead(
                    input_dim=config.input_dim,
                    output_dim=config.output_dim,
                    hidden_dim=config.get("hidden_dim", None),
                )
                self.head_types[name] = "node"

        # Create graph heads
        self.has_graph_heads = False
        if "graph_heads" in heads_configs:
            self.has_graph_heads = True
            for name, config in heads_configs.graph_heads.items():
                self.heads[name] = OutputHead(
                    input_dim=config.input_dim,
                    output_dim=config.output_dim,
                    hidden_dim=config.get("hidden_dim", None),
                )
                self.head_types[name] = "graph"

        # Create edge heads
        if "edge_heads" in heads_configs:
            for name, config in heads_configs.edge_heads.items():
                self.heads[name] = OutputHead(
                    input_dim=config.input_dim,
                    output_dim=config.output_dim,
                    hidden_dim=config.get("hidden_dim", None),
                )
                self.head_types[name] = "edge"

    def forward(
        self,
        h: torch.Tensor,
        batch: Optional[torch.Tensor],
        edge_attr: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            h: Node features of shape (num_nodes, input_dim)
            batch: Batch assignment for each node (required for graph heads)
            edge_attr: Edge features of shape (num_edges, input_dim)
                (required for edge heads)
        Returns:
            Dictionary mapping head names to their outputs
        """
        outputs = {}

        # Aggregate once for all graph heads if they exist
        graph_features = None
        if self.has_graph_heads:
            num_graphs = batch.max() + 1

            # Aggregate node features to graph level using the chosen aggregation method
            if self.graph_aggregation == "mean":
                graph_features = unsorted_segment_mean(h, batch, num_graphs)
            elif self.graph_aggregation == "sum":
                graph_features = unsorted_segment_sum(h, batch, num_graphs)
            else:
                raise ValueError(f"Unknown aggregation: {self.graph_aggregation}")

        for name, head in self.heads.items():
            if self.head_types[name] == "node":
                outputs[name] = head(h)

            elif self.head_types[name] == "graph":
                # Use pre-aggregated features
                outputs[name] = head(graph_features)

            elif self.head_types[name] == "edge":
                if edge_attr is None:
                    raise ValueError(f"Edge head '{name}' requires edge_attr")

                outputs[name] = head(edge_attr)

        return outputs


class EquivariantGMMHead(nn.Module):
    def __init__(self, hidden_dim, K, N_a, N_c, rbf_embedding_args: DictConfig = None):
        """
        Args:
            hidden_dim: Input dimension of invariant features h.
            K: Number of Gaussian components.
            N_a: Number of atom types.
            N_c: Number of charge types.
        """
        super().__init__()
        self.K = K
        self.N_a = N_a
        self.N_c = N_c
        self.rbf_embedding = hydra.utils.instantiate(rbf_embedding_args)
        dist_feat_dim = rbf_embedding_args.get(
            "out_dim", rbf_embedding_args.get("num_rbf", 16)
        )

        # --- 1. Invariant Scalar Head (MLP) ---
        # Predicts: Pi (K), Sigma (K), AtomProbs (K*N_a), ChargeProbs (K*N_c)
        # --- 2. Invariant Scalar Head (MLP) ---
        scalar_out_dim = K + K + (K * N_a) + (K * N_c)
        self.scalar_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # Stabilizer
            nn.SiLU(),
            nn.Linear(hidden_dim, scalar_out_dim),
        )

        # --- 3. Equivariant Vector Head (Edge Processor) ---
        # Input: h_i (dim) + h_j (dim) + rbf_feat (dim)
        input_dim_coord = hidden_dim * 2 + dist_feat_dim

        self.coord_mlp = nn.Sequential(
            nn.Linear(input_dim_coord, hidden_dim),
            nn.LayerNorm(hidden_dim),  # Stabilizer
            nn.SiLU(),
            nn.Linear(hidden_dim, K, bias=False),
        )

        self.coord_scale = nn.Parameter(torch.tensor(1.0))

        self.reset_parameters()

    def reset_parameters(self):
        # 1. Standard Init for Hidden Layers
        # (Note: RBFEmbedding has its own init, so we skip it here)
        for m in [self.scalar_mlp[0], self.coord_mlp[0]]:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

        # 2. "Near-Zero" Init for Coordinate Output
        # CRITICAL: Ensures mu starts as current position
        last_coord = self.coord_mlp[-1]
        nn.init.uniform_(last_coord.weight, -1e-4, 1e-4)
        # nn.init.zeros_(last_coord.bias)

        # 3. Stable Init for Scalar Output
        last_scalar = self.scalar_mlp[-1]
        nn.init.xavier_uniform_(last_scalar.weight, gain=0.1)
        nn.init.zeros_(last_scalar.bias)

    def forward(self, h, x, edge_index):
        """
        Args:
            h: [N, hidden_dim] Final invariant features from EGNN.
            x: [N, 3] Final equivariant coordinates from EGNN.
            edge_index: [2, E] Graph connectivity.
        """
        row, col = edge_index
        N = h.shape[0]

        # ----------------------------------------
        # Part A: Predict Scalar Parameters (Pi, Sigma, Types)
        # ----------------------------------------
        scalar_pred = self.scalar_mlp(h)

        # ... (Slicing logic same as previous answer) ...
        # 1. Pi
        pi = F.softmax(scalar_pred[:, : self.K], dim=-1)
        # 2. Sigma
        sigma = F.softplus(scalar_pred[:, self.K : 2 * self.K]) + 1e-5
        # 3. Atom Types
        start_a = 2 * self.K
        end_a = start_a + self.K * self.N_a
        a_probs = F.softmax(
            scalar_pred[:, start_a:end_a].view(N, self.K, self.N_a), dim=-1
        )
        # 4. Charge Types
        start_c = end_a
        end_c = start_c + self.K * self.N_c
        c_probs = F.softmax(
            scalar_pred[:, start_c:end_c].view(N, self.K, self.N_c), dim=-1
        )

        # ----------------------------------------
        # Part B: Predict Equivariant Means (The "Minimal" Fix)
        # ----------------------------------------
        # Calculate relative differences
        diff = x[row] - x[col]  # [E, 3]
        dist_sq = torch.sum(diff**2, dim=1, keepdim=True)  # [E, 1]
        dist = torch.sqrt(dist_sq + 1e-8)

        # 1. Embed distances
        dist_feat = self.rbf_embedding(dist)

        # 2. Prepare edge features for the weighting function
        # Concatenate: h_i, h_j, dist_feat
        edge_feat = torch.cat(
            [h[row], h[col], dist_feat], dim=1
        )  # [E, 2*dim + dist_feat_dim]

        # 3. Predict weights for K distinct vectors
        # Weights shape: [E, K]
        weights = self.coord_mlp(edge_feat)
        weights = F.tanh(weights) * self.coord_scale

        # 4. Weighted Sum (Equivariant Aggregation)
        # We need to broadcast diff to K channels: [E, 1, 3] * [E, K, 1] -> [E, K, 3]
        weighted_diff = diff.unsqueeze(1) * weights.unsqueeze(2)

        # Aggregate over neighbors (scatter_add)
        # Result: [N, K, 3]
        shift = torch.zeros(N, self.K, 3, device=x.device)
        shift.index_add_(0, row, weighted_diff)

        # If an atom has 50 neighbors, the sum is huge. Divide by degree.
        # Calculate degree
        deg = torch.zeros(N, 1, device=x.device)
        deg.index_add_(0, row, torch.ones(row.shape[0], 1, device=x.device))
        deg = deg.clamp(min=1.0)  # Avoid division by zero

        # Normalize shift (Broadcasting: [N, K, 3] / [N, 1, 1])
        shift = shift / deg.unsqueeze(-1)

        # 5. Final Equivariant Means
        # The mean is the current position + the calculated shift
        mu = x.unsqueeze(1) + shift

        return {
            "pi": pi,
            "mu": mu,  # [N, K, 3]
            "sigma": sigma,
            "a_probs": a_probs,
            "c_probs": c_probs,
        }


class InsertionEdgeHead(nn.Module):
    """
    Predicts edge types between newly inserted nodes and existing nodes.

    For each insertion (spawned from a spawn node), predicts edge probabilities
    to all other existing nodes in the same graph.

    The prediction is based on:
    1. Inserted node position/type/charge provided by caller
    2. Features/positions of existing nodes
    3. Features of the spawn node
    """

    def __init__(
        self,
        hidden_dim: int,
        n_edge_types: int,
        n_atom_types: int,
        n_charge_types: int,
        rbf_embedding_args: DictConfig = None,
    ):
        """
        Args:
            hidden_dim: Dimension of node features
            n_edge_types: Number of edge types to predict
            n_atom_types: Number of atom types (for inserted node embedding)
            n_charge_types: Number of charge types (for inserted node embedding)
            use_distance_features: Whether to use RBF distance features
            rbf_embedding_args: Configuration for RBFEmbedding (DictConfig)
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_edge_types = n_edge_types
        self.n_atom_types = n_atom_types
        self.n_charge_types = n_charge_types

        self.rbf_embedding = hydra.utils.instantiate(rbf_embedding_args)
        dist_feat_dim = rbf_embedding_args.get(
            "out_dim", rbf_embedding_args.get("num_rbf", 16)
        )

        # One-hot atom/charge features for inserted nodes.
        ins_type_feat_dim = n_atom_types + n_charge_types

        # MLP for edge prediction
        # Input: [h_spawn, h_existing, distance_features, ins_atom_one_hot, ins_charge_one_hot]
        input_dim = hidden_dim * 2 + dist_feat_dim + ins_type_feat_dim
        self.edge_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, n_edge_types),
        )

        self._init_weights()

    def _init_weights(self):
        """Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        spawn_node_idx: torch.Tensor,
        existing_node_idx: torch.Tensor,
        ins_x: torch.Tensor,
        ins_a: torch.Tensor,
        ins_c: torch.Tensor,
    ) -> torch.Tensor:
        if spawn_node_idx.numel() == 0:
            return torch.empty((0, self.n_edge_types), device=h.device)

        insertion_pos = ins_x.detach()
        ins_atom_type = F.one_hot(ins_a.long(), num_classes=self.n_atom_types).to(
            dtype=h.dtype, device=h.device
        )
        ins_charge_type = F.one_hot(ins_c.long(), num_classes=self.n_charge_types).to(
            dtype=h.dtype, device=h.device
        )

        # ------------------------------------------------------------------
        # 3. EDGE PREDICTION
        # ------------------------------------------------------------------

        # Use fixed insertion attributes to decouple edge prediction from GMM sampling.
        x = x.detach()
        existing_pos = x[existing_node_idx]
        distances = torch.norm(insertion_pos - existing_pos, dim=-1).clamp(min=1e-6)
        dist_features = self.rbf_embedding(distances)

        h_spawn = h[spawn_node_idx]
        # "existing" refers to the already-present node endpoint in current state.
        h_existing = h[existing_node_idx]

        # Concatenate features.
        edge_features = torch.cat(
            [h_spawn, h_existing, dist_features, ins_atom_type, ins_charge_type],
            dim=-1,
        )

        edge_logits = self.edge_mlp(edge_features)

        return edge_logits

    def predict_edges_for_insertion(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        batch: torch.Tensor,
        insertion_mask: torch.Tensor,
        ins_x: torch.Tensor,
        ins_a: torch.Tensor,
        ins_c: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict edges for all insertions to all other nodes in the same graph.

        This is used during inference to get edge predictions.

        Args:
            h: [N, hidden_dim] - Node features
            x: [N, 3] - Node positions
            batch: [N] - Batch assignment
            insertion_mask: [N] - Boolean mask of nodes that will spawn insertions
            ins_x: [N_ins, 3] - Inserted node positions aligned with insertion_mask True indices
            ins_a: [N_ins] - Inserted atom types aligned with insertion_mask True indices
            ins_c: [N_ins] - Inserted charge types aligned with insertion_mask True indices

        Returns:
            spawn_idx: [E_ins] - Spawn node indices
            existing_idx: [E_ins] - Existing endpoint node indices
            edge_logits: [E_ins, n_edge_types] - Edge type logits
        """
        device = h.device
        insertion_indices = torch.where(insertion_mask)[0]

        if insertion_indices.numel() == 0:
            return (
                torch.empty(0, dtype=torch.long, device=device),
                torch.empty(0, dtype=torch.long, device=device),
                torch.empty(0, self.n_edge_types, device=device),
            )

        # Build pairs: (spawn_node, all nodes in same graph including spawn)
        spawn_list = []
        existing_list = []
        ins_x_list = []
        ins_a_list = []
        ins_c_list = []

        for local_idx, ins_idx in enumerate(insertion_indices):
            graph_id = batch[ins_idx]
            same_graph_mask = batch == graph_id
            existing_indices = torch.where(same_graph_mask)[0]
            n_existing = len(existing_indices)

            spawn_list.append(ins_idx.expand(n_existing))
            existing_list.append(existing_indices)
            ins_x_list.append(ins_x[local_idx].unsqueeze(0).expand(n_existing, -1))
            ins_a_list.append(ins_a[local_idx].expand(n_existing))
            ins_c_list.append(ins_c[local_idx].expand(n_existing))

        spawn_idx = torch.cat(spawn_list)
        existing_idx = torch.cat(existing_list)
        ins_x_expanded = torch.cat(ins_x_list, dim=0)
        ins_a_expanded = torch.cat(ins_a_list, dim=0)
        ins_c_expanded = torch.cat(ins_c_list, dim=0)

        # Predict edge types
        edge_logits = self.forward(
            h,
            x,
            spawn_idx,
            existing_idx,
            ins_x_expanded,
            ins_a_expanded,
            ins_c_expanded,
        )

        return spawn_idx, existing_idx, edge_logits
