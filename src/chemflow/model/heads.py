import torch
import torch.nn as nn
from typing import Dict, Optional
from omegaconf import DictConfig
from src.external_code.egnn import unsorted_segment_sum, unsorted_segment_mean


class OutputHead(nn.Module):
    """A generic head that operates on features (node-level, graph-level, or edge-level)."""

    def __init__(
        self, input_dim: int, output_dim: int, hidden_dim: Optional[int] = None
    ):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = input_dim

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim, bias=False),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: Features of shape (num_items, input_dim) where num_items can be
               num_nodes, num_graphs, or num_edges depending on the head type.
        Returns:
            Predictions of shape (num_items, output_dim)
        """
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
            elif self.graph_aggregation == "max":
                graph_features = self._unsorted_segment_max(h, batch, num_graphs)
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

    def _unsorted_segment_max(
        self, data: torch.Tensor, segment_ids: torch.Tensor, num_segments: int
    ) -> torch.Tensor:
        """Custom implementation of unsorted_segment_max using pure PyTorch."""
        result_shape = (num_segments, data.size(1))
        result = data.new_full(result_shape, float("-inf"))
        segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
        result.scatter_reduce_(0, segment_ids, data, reduce="max")
        return result
