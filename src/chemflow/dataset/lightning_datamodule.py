import pytorch_lightning as pl
from torch.utils.data import DataLoader
from omegaconf import DictConfig
import hydra
import torch

from chemflow.flow_matching.sampling import sample_prior_graph
from chemflow.utils import token_to_index


class LightningDataModule(pl.LightningDataModule):
    def __init__(
        self,
        datasets: DictConfig,
        num_workers: DictConfig,
        batch_size: DictConfig,
        typed_gmm: bool = True,
    ):
        self.datasets = datasets
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.typed_gmm = typed_gmm

        # Will be set via setter methods
        self.tokens = None
        self.atom_type_distribution = None
        self.edge_type_distribution = None
        self.n_atoms_distribution = None
        self.mask_token = None

        # will be set later
        self.train_dataset = None
        self.val_datasets = None
        self.test_datasets = None

        super().__init__()

    def set_tokens_and_distributions(
        self,
        tokens: list[str],
        atom_type_distribution: torch.Tensor,
        edge_type_distribution: torch.Tensor,
        n_atoms_distribution: torch.Tensor,
    ):
        """Set tokens and distributions after initialization."""
        self.tokens = tokens
        self.atom_type_distribution = atom_type_distribution
        self.edge_type_distribution = edge_type_distribution
        self.n_atoms_distribution = n_atoms_distribution
        self.mask_token = token_to_index(self.tokens, "<MASK>")

    def setup(self, stage=None):
        """Construct datasets and assign data scalers."""
        # Check that tokens and distributions are set
        if self.tokens is None:
            raise ValueError(
                "tokens and distributions must be set before calling setup(). "
                "Call set_tokens_and_distributions() first."
            )

        # Inject tokens and distributions into dataset configs
        train_cfg = self.datasets.train.copy()
        self.train_dataset = hydra.utils.instantiate(train_cfg)

        self.val_datasets = []
        for dataset_cfg in self.datasets.val:
            val_cfg = dataset_cfg.copy()
            self.val_datasets.append(hydra.utils.instantiate(val_cfg))

        self.test_datasets = []
        for dataset_cfg in self.datasets.test:
            test_cfg = dataset_cfg.copy()
            self.test_datasets.append(hydra.utils.instantiate(test_cfg))

    def collate_graphs(self, graph_dicts: list[dict]):
        """
        Collate function for graph datasets.
        Each item in batch is a dictionary with keys: atom_feats, coord, edge_index,
        and optionally edge_attr and graph_attr.
        Returns a batched graph with proper node and edge indexing.
        """
        # Extract required components
        atom_types_list = [item["atom_types"] for item in graph_dicts]
        coord_list = [item["coord"] for item in graph_dicts]
        edge_types_list = [item["edge_types"] for item in graph_dicts]

        # Check if optional components exist in the first item
        has_edge_attr = (
            "edge_attr" in graph_dicts[0] and graph_dicts[0]["edge_attr"] is not None
        )
        has_graph_attr = (
            "graph_attr" in graph_dicts[0] and graph_dicts[0]["graph_attr"] is not None
        )

        # Extract optional components if they exist
        edge_attr_list = (
            [item.get("edge_attr") for item in graph_dicts] if has_edge_attr else None
        )
        graph_attr_list = (
            [item.get("graph_attr") for item in graph_dicts] if has_graph_attr else None
        )

        # Calculate node counts for batch indexing
        N_atoms = torch.tensor([feats.size(0) for feats in atom_types_list])

        # Concatenate node features, coordinates, and edge types
        batched_atom_types = torch.cat(atom_types_list, dim=0)
        batched_coord = torch.cat(coord_list, dim=0)
        batched_edge_types = torch.cat(edge_types_list, dim=0)

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
                torch.full(
                    (N_atoms[i],), i, dtype=torch.long, device=batched_atom_types.device
                )
                for i in range(len(graph_dicts))
            ]
        )

        N_triu_edges = (N_atoms**2 - N_atoms) // 2
        edge_type_batch_index = torch.cat(
            [
                torch.full(
                    (N_triu_edges[i],),
                    i,
                    dtype=torch.long,
                    device=batched_edge_types.device,
                )
                for i in range(len(graph_dicts))
            ]
        )

        # Build result dictionary with only present attributes
        result = {
            "atom_types": batched_atom_types,
            "coord": batched_coord,
            "edge_types": batched_edge_types,
            "batch_index": batch_index,
            "edge_type_batch_index": edge_type_batch_index,
            "N_atoms": torch.tensor(N_atoms),
            "N_triu_edges": torch.tensor(N_triu_edges),
        }

        # Add optional attributes if they exist
        if batched_edge_attr is not None:
            result["edge_attr"] = batched_edge_attr
        if batched_graph_attr is not None:
            result["graph_attr"] = batched_graph_attr

        return result

    def collate_fn(self, batch):
        targets = batch

        samples = [
            sample_prior_graph(
                self.atom_type_distribution,
                self.edge_type_distribution,
                self.n_atoms_distribution,
                self.typed_gmm,
                self.mask_token,
            )
            for _ in range(len(targets))
        ]

        samples_batched = self.collate_graphs(samples)
        targets_batched = self.collate_graphs(targets)
        return samples_batched, targets_batched

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
