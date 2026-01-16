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
            layers.append(nn.GELU())

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
    1. The predicted insertion position (from GMM mean)
    2. The predicted atom type and charge of the inserted node (from GMM)
    3. The features/positions of existing nodes
    4. The features of the spawn node
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

        # Embeddings for inserted node's atom type and charge (from GMM probabilities)
        # We use the probability vectors directly as soft embeddings
        ins_type_feat_dim = n_atom_types + n_charge_types

        # MLP for edge prediction
        # Input: [h_spawn, h_existing, distance_features, ins_atom_probs, ins_charge_probs]
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
        gmm_dict: dict,
        batch: torch.Tensor,
        spawn_node_idx: torch.Tensor,
        target_node_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict edge types between insertion points and existing nodes.

        Args:
            h: [N, hidden_dim] - Node features for all nodes
            x: [N, 3] - Node positions for all nodes
            gmm_dict: dict with 'mu' [N, K, 3], 'a_probs' [N, K, n_atom],
                      'c_probs' [N, K, n_charge] - GMM parameters
            batch: [N] - Batch assignment for each node
            spawn_node_idx: [E_ins] - Indices of spawn nodes (nodes predicting insertions)
            target_node_idx: [E_ins] - Indices of existing nodes to predict edges to

        Returns:
            edge_logits: [E_ins, n_edge_types] - Edge type logits for each pair
        """
        if spawn_node_idx.numel() == 0:
            return torch.empty((0, self.n_edge_types), device=h.device, dtype=h.dtype)

        # Get GMM parameters for spawn nodes
        mu = gmm_dict["mu"]  # [N, K, 3]
        pi = gmm_dict["pi"]  # [N, K]
        a_probs = gmm_dict["a_probs"]  # [N, K, n_atom_types]
        c_probs = gmm_dict["c_probs"]  # [N, K, n_charge_types]

        # Sample GMM component for each spawn node based on mixture weights
        pi_spawn = pi[spawn_node_idx]  # [E_ins, K]
        k_samples = torch.distributions.Categorical(probs=pi_spawn).sample()  # [E_ins]

        # Gather the sampled component's parameters
        E_ins = spawn_node_idx.shape[0]
        batch_idx = torch.arange(E_ins, device=h.device)

        # [E_ins, 3]
        insertion_pos = mu[spawn_node_idx][batch_idx, k_samples, :]

        # [E_ins, n_atom]
        ins_atom_probs = a_probs[spawn_node_idx][batch_idx, k_samples, :]

        # [E_ins, n_charge]
        ins_charge_probs = c_probs[spawn_node_idx][batch_idx, k_samples, :]

        # Get positions of target nodes
        target_pos = x[target_node_idx]  # [E_ins, 3]

        # Compute distances with numerical stability
        distances = torch.norm(insertion_pos - target_pos, dim=-1).clamp(min=1e-6)

        # Compute distance features
        dist_features = self.rbf_embedding(distances)  # [E_ins, num_rbf]

        # Get node features
        h_spawn = h[spawn_node_idx]  # [E_ins, hidden_dim]
        h_target = h[target_node_idx]  # [E_ins, hidden_dim]

        # Concatenate all features including inserted node type info
        edge_features = torch.cat(
            [h_spawn, h_target, dist_features, ins_atom_probs, ins_charge_probs],
            dim=-1,
        )

        # Predict edge types
        edge_logits = self.edge_mlp(edge_features)  # [E_ins, n_edge_types]

        return edge_logits

    def predict_edges_for_insertion(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        gmm_dict: dict,
        batch: torch.Tensor,
        insertion_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict edges for all insertions to all other nodes in the same graph.

        This is used during inference to get edge predictions.

        Args:
            h: [N, hidden_dim] - Node features
            x: [N, 3] - Node positions
            gmm_dict: dict with GMM parameters
            batch: [N] - Batch assignment
            insertion_mask: [N] - Boolean mask of nodes that will spawn insertions

        Returns:
            spawn_idx: [E_ins] - Spawn node indices
            target_idx: [E_ins] - Target node indices
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

        # Build pairs: (spawn_node, all other nodes in same graph)
        spawn_list = []
        target_list = []

        for ins_idx in insertion_indices:
            graph_id = batch[ins_idx]
            # Find all nodes in the same graph (excluding the spawn node itself)
            same_graph_mask = (batch == graph_id) & (
                torch.arange(len(batch), device=device) != ins_idx
            )
            target_indices = torch.where(same_graph_mask)[0]

            # Create pairs
            spawn_list.append(ins_idx.expand(len(target_indices)))
            target_list.append(target_indices)

        spawn_idx = torch.cat(spawn_list)
        target_idx = torch.cat(target_list)

        # Predict edge types
        edge_logits = self.forward(h, x, gmm_dict, batch, spawn_idx, target_idx)

        return spawn_idx, target_idx, edge_logits
