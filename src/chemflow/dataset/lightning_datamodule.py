import pytorch_lightning as pl
from torch.utils.data import DataLoader
from omegaconf import DictConfig
import hydra
import torch


class LightningDataModule(pl.LightningDataModule):
    def __init__(
        self,
        datasets: DictConfig,
        num_workers: DictConfig,
        batch_size: DictConfig,
    ):
        super().__init__()

        self.datasets = datasets
        self.num_workers = num_workers
        self.batch_size = batch_size

        # will be set later
        self.train_dataset = None
        self.val_datasets = None
        self.test_datasets = None

    def setup(self, stage=None):
        """Construct datasets and assign data scalers."""
        self.train_dataset = hydra.utils.instantiate(self.datasets.train)
        self.val_datasets = [
            hydra.utils.instantiate(dataset_cfg) for dataset_cfg in self.datasets.val
        ]

        self.test_datasets = [
            hydra.utils.instantiate(dataset_cfg) for dataset_cfg in self.datasets.test
        ]

    def collate_fn(self, batch):
        """
        Collate function for graph datasets.
        Each item in batch is a dictionary with keys: atom_feats, coord, edge_index,
        and optionally edge_attr and graph_attr.
        Returns a batched graph with proper node and edge indexing.
        """
        # Extract required components
        atom_feats_list = [item["atom_feats"] for item in batch]
        coord_list = [item["coord"] for item in batch]
        edge_index_list = [item["edge_index"] for item in batch]

        # Check if optional components exist in the first item
        has_edge_attr = "edge_attr" in batch[0] and batch[0]["edge_attr"] is not None
        has_graph_attr = "graph_attr" in batch[0] and batch[0]["graph_attr"] is not None

        # Extract optional components if they exist
        edge_attr_list = (
            [item.get("edge_attr") for item in batch] if has_edge_attr else None
        )
        graph_attr_list = (
            [item.get("graph_attr") for item in batch] if has_graph_attr else None
        )

        # Calculate cumulative node counts for batch indexing
        num_nodes = [feats.size(0) for feats in atom_feats_list]
        cumsum_nodes = torch.cumsum(torch.tensor([0] + num_nodes[:-1]), dim=0)

        # Concatenate node features and coordinates
        batched_atom_feats = torch.cat(atom_feats_list, dim=0)
        batched_coord = torch.cat(coord_list, dim=0)

        # Adjust edge indices by adding cumulative node counts
        batched_edge_index = []
        for i, edge_index in enumerate(edge_index_list):
            # Add offset to edge indices
            offset = cumsum_nodes[i]
            adjusted_edge_index = edge_index + offset
            batched_edge_index.append(adjusted_edge_index)

        # Concatenate edge indices
        batched_edge_index = torch.cat(batched_edge_index, dim=1)

        # Handle optional edge attributes
        if has_edge_attr and edge_attr_list[0] is not None:
            batched_edge_attr = torch.cat(edge_attr_list, dim=0)
        else:
            batched_edge_attr = None

        # Handle optional graph attributes
        if has_graph_attr and graph_attr_list[0] is not None:
            batched_graph_attr = torch.stack(graph_attr_list, dim=0)
        else:
            batched_graph_attr = None

        # Create batch index tensor to track which nodes belong to which graph
        batch_index = torch.cat(
            [
                torch.full((num_nodes[i],), i, dtype=torch.long)
                for i in range(len(batch))
            ]
        )

        # Build result dictionary with only present attributes
        result = {
            "atom_feats": batched_atom_feats,
            "coord": batched_coord,
            "edge_index": batched_edge_index,
            "batch_index": batch_index,
            "num_nodes": torch.tensor(num_nodes),
        }

        # Add optional attributes if they exist
        if batched_edge_attr is not None:
            result["edge_attr"] = batched_edge_attr
        if batched_graph_attr is not None:
            result["graph_attr"] = batched_graph_attr

        return result

    def train_dataloader(self):
        # Return a DataLoader for training
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.batch_size.train,
            num_workers=self.num_workers.train,
            pin_memory=True,
            drop_last=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        # Return a DataLoader for validation
        return [
            DataLoader(
                dataset,
                shuffle=False,
                batch_size=self.batch_size.val,
                num_workers=self.num_workers.val,
                pin_memory=True,
                drop_last=True,
                collate_fn=self.collate_fn,
            )
            for dataset in self.val_datasets
        ]

    def test_dataloader(self):
        # Return a DataLoader for testing
        return [
            DataLoader(
                dataset,
                shuffle=False,
                batch_size=self.batch_size.test,
                num_workers=self.num_workers.test,
                pin_memory=True,
                collate_fn=self.collate_fn,
            )
            for dataset in self.test_datasets
        ]

    def predict_dataloader(self) -> list[DataLoader]:
        return self.test_dataloader()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"{self.datasets=}, "
            f"{self.num_workers=}, "
            f"{self.batch_size=})"
        )
