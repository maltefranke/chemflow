import pytorch_lightning as pl
from torch.utils.data import DataLoader
from omegaconf import DictConfig
import hydra
import torch

from chemflow.dataset.molecule_data import MoleculeBatch
from chemflow.flow_matching.sampling import sample_prior_graph
from chemflow.utils import token_to_index


class LightningDataModule(pl.LightningDataModule):
    def __init__(
        self,
        datasets: DictConfig,
        num_workers: DictConfig,
        batch_size: DictConfig,
        cat_strategy: str = "uniform-sample",
        n_atoms_strategy: str = "flexible",
        optimal_transport: str = "equivariant",
    ):
        self.datasets = datasets
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.cat_strategy = cat_strategy
        self.n_atoms_strategy = n_atoms_strategy
        self.optimal_transport = optimal_transport

        # Will be set via setter methods
        self.atom_tokens = None
        self.edge_tokens = None
        self.atom_type_distribution = None
        self.edge_type_distribution = None
        self.charge_type_distribution = None
        self.n_atoms_distribution = None
        self.charge_tokens = None
        self.atom_mask_token_index = None
        self.edge_mask_token_index = None
        self.coord_std = None

        # will be set later
        self.train_dataset = None
        self.val_datasets = None
        self.test_datasets = None

        super().__init__()

    def set_tokens_and_distributions(
        self,
        atom_tokens: list[str],
        edge_tokens: list[str],
        charge_tokens: list[str],
        atom_type_distribution: torch.Tensor,
        edge_type_distribution: torch.Tensor,
        charge_type_distribution: torch.Tensor,
        n_atoms_distribution: torch.Tensor,
        coord_std: torch.Tensor = None,
    ):
        """Set tokens and distributions after initialization."""
        self.atom_tokens = atom_tokens
        self.edge_tokens = edge_tokens
        self.charge_tokens = charge_tokens
        self.atom_type_distribution = atom_type_distribution
        self.edge_type_distribution = edge_type_distribution
        self.charge_type_distribution = charge_type_distribution
        self.n_atoms_distribution = n_atoms_distribution

        if self.n_atoms_strategy != "fixed":
            self.death_token_index = token_to_index(self.atom_tokens, "<DEATH>")

            if self.cat_strategy == "uniform-sample":
                # We never want to sample a death token
                self.atom_type_distribution[self.death_token_index] = 0.0

        elif self.cat_strategy == "mask":
            self.atom_mask_token_index = token_to_index(self.atom_tokens, "<MASK>")
            self.edge_mask_token_index = token_to_index(self.edge_tokens, "<MASK>")

            self.atom_type_distribution = torch.zeros_like(self.atom_type_distribution)
            # always sample a mask token
            self.atom_type_distribution[self.atom_mask_token_index] = 1.0

            self.edge_type_distribution = torch.zeros_like(self.edge_type_distribution)
            # always sample a mask token
            self.edge_type_distribution[self.edge_mask_token_index] = 1.0

        self.coord_std = coord_std if coord_std is not None else None

    def setup(self, stage=None):
        """Construct datasets and assign data scalers."""
        # Check that tokens and distributions are set
        if self.atom_tokens is None:
            raise ValueError(
                "atom_tokens and distributions must be set before calling setup(). "
                "Call set_tokens_and_distributions() first."
            )

        # Inject tokens and distributions into dataset configs
        train_cfg = self.datasets.train.copy()
        train_cfg.coord_std = self.coord_std
        self.train_dataset = hydra.utils.instantiate(train_cfg)

        self.val_datasets = []
        for dataset_cfg in self.datasets.val:
            val_cfg = dataset_cfg.copy()
            val_cfg.coord_std = self.coord_std
            self.val_datasets.append(hydra.utils.instantiate(val_cfg))

        self.test_datasets = []
        for dataset_cfg in self.datasets.test:
            test_cfg = dataset_cfg.copy()
            test_cfg.coord_std = self.coord_std
            self.test_datasets.append(hydra.utils.instantiate(test_cfg))

    def collate_fn(self, batch):
        targets = batch
        if self.n_atoms_strategy == "fixed":
            n_atoms = [target.num_nodes for target in targets]
        else:
            n_atoms = [None for target in targets]

        samples = [
            sample_prior_graph(
                self.atom_type_distribution,
                self.edge_type_distribution,
                self.charge_type_distribution,
                self.n_atoms_distribution,
                n_atoms=n_atoms_i,
                coord_std=self.coord_std,
            )
            for n_atoms_i in n_atoms
        ]
        samples_batched = MoleculeBatch.from_data_list(samples)
        targets_batched = MoleculeBatch.from_data_list(targets)
        return samples_batched, targets_batched

    def train_dataloader(self):
        # Return a DataLoader for training
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.batch_size.train,
            num_workers=self.num_workers.train,
            persistent_workers=True,
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
                persistent_workers=True,
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
                persistent_workers=True,
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
