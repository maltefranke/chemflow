import torch
import torch.nn as nn
from typing import Dict, Optional
from omegaconf import DictConfig
from src.external_code.egnn import unsorted_segment_sum, unsorted_segment_mean


class NodeHead(nn.Module):
    """Head that operates on individual node features."""

    def __init__(
        self, input_dim: int, output_dim: int, hidden_dim: Optional[int] = None
    ):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = input_dim

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: Node features of shape (num_nodes, input_dim)
        Returns:
            Node-level predictions of shape (num_nodes, output_dim)
        """
        return self.mlp(h)


class GraphHead(nn.Module):
    """Head that operates on graph-level features (aggregated from node features)."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: Optional[int] = None,
        aggregation: str = "mean",
    ):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = input_dim

        self.aggregation = aggregation
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, h: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: Node features of shape (num_nodes, input_dim)
            batch: Batch assignment for each node of shape (num_nodes,)
        Returns:
            Graph-level predictions of shape (num_graphs, output_dim)
        """
        # Get number of graphs
        num_graphs = batch.max().item() + 1

        # Aggregate node features to graph level using EGNN utilities
        if self.aggregation == "mean":
            graph_features = unsorted_segment_mean(h, batch, num_graphs)
        elif self.aggregation == "sum":
            graph_features = unsorted_segment_sum(h, batch, num_graphs)
        elif self.aggregation == "max":
            # For max aggregation, we need to implement it manually
            graph_features = self._unsorted_segment_max(h, batch, num_graphs)
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

        return self.mlp(graph_features)

    def _unsorted_segment_max(
        self, data: torch.Tensor, segment_ids: torch.Tensor, num_segments: int
    ) -> torch.Tensor:
        """Custom implementation of unsorted_segment_max using pure PyTorch."""
        result_shape = (num_segments, data.size(1))
        result = data.new_full(result_shape, float("-inf"))
        segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
        result.scatter_reduce_(0, segment_ids, data, reduce="max")
        return result


class MultiHeadModule(nn.Module):
    """Module that combines multiple heads (both node-level and graph-level)."""

    def __init__(self, heads_configs: DictConfig):
        super().__init__()
        self.heads = nn.ModuleDict()
        self.head_types = {}

        # Create node heads
        if "node_heads" in heads_configs:
            for name, config in heads_configs.node_heads.items():
                self.heads[name] = NodeHead(
                    input_dim=config.input_dim,
                    output_dim=config.output_dim,
                    hidden_dim=config.get("hidden_dim", None),
                )
                self.head_types[name] = "node"

        # Create graph heads
        if "graph_heads" in heads_configs:
            for name, config in heads_configs.graph_heads.items():
                self.heads[name] = GraphHead(
                    input_dim=config.input_dim,
                    output_dim=config.output_dim,
                    hidden_dim=config.get("hidden_dim", None),
                    aggregation=config.get("aggregation", "mean"),
                )
                self.head_types[name] = "graph"

    def forward(
        self, h: torch.Tensor, batch: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            h: Node features of shape (num_nodes, input_dim)
            batch: Batch assignment for each node (required for graph heads)
        Returns:
            Dictionary mapping head names to their outputs
        """
        outputs = {}

        for name, head in self.heads.items():
            if self.head_types[name] == "node":
                outputs[name] = head(h)

            elif self.head_types[name] == "graph":
                if batch is None:
                    raise ValueError(f"Graph head '{name}' requires batch information")

                outputs[name] = head(h, batch)

        return outputs
