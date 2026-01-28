import pytorch_lightning as pl
from torch.utils.data import DataLoader
from omegaconf import DictConfig
import hydra
import torch
from functools import partial

from chemflow.dataset.molecule_data import MoleculeBatch
from chemflow.flow_matching.sampling import sample_prior_graph
from chemflow.utils import token_to_index, validate_no_cross_batch_edges


class LightningDataModule(pl.LightningDataModule):
    def __init__(
        self,
        vocab: DictConfig,
        distributions: DictConfig,
        datasets: DictConfig,
        interpolator: DictConfig,
        num_workers: DictConfig,
        batch_size: DictConfig,
        cat_strategy: str = "uniform-sample",
        n_atoms_strategy: str = "flexible",
        optimal_transport: str = "equivariant",
        time_dist: DictConfig = None,
    ):
        self.vocab = vocab
        self.distributions = distributions
        self.datasets = datasets
        self.interpolator = interpolator
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.cat_strategy = cat_strategy
        self.n_atoms_strategy = n_atoms_strategy
        self.optimal_transport = optimal_transport

        if time_dist is None:
            time_dist = DictConfig(
                {
                    "_target_": "torch.distributions.Uniform",
                    "low": 0.0,
                    "high": 1.0,
                }
            )
        self.time_dist = hydra.utils.instantiate(time_dist)

        # Will be set via setter methods
        # will be set later
        self.train_dataset = None
        self.val_datasets = None
        self.test_datasets = None

        if self.cat_strategy == "mask":
            self.atom_mask_token_index = token_to_index(
                self.vocab.atom_tokens, "<MASK>"
            )
            self.edge_mask_token_index = token_to_index(
                self.vocab.edge_tokens, "<MASK>"
            )

            self.distributions.atom_type_distribution = torch.zeros_like(
                self.distributions.atom_type_distribution
            )
            # always sample a mask token
            self.distributions.atom_type_distribution[self.atom_mask_token_index] = 1.0

            self.distributions.edge_type_distribution = torch.zeros_like(
                self.distributions.edge_type_distribution
            )
            # always sample a mask token
            self.distributions.edge_type_distribution[self.edge_mask_token_index] = 1.0

        self.distributions.coordinate_std = (
            self.distributions.coordinate_std
            if self.distributions.coordinate_std is not None
            else None
        )

        self.interpolator = hydra.utils.instantiate(
            interpolator, distributions=self.distributions
        )

        super().__init__()

    def setup(self, stage=None):
        """Construct datasets and assign data scalers."""
        # Check that tokens and distributions are set
        if self.vocab is None:
            raise ValueError(
                "vocab and distributions must be set before calling setup(). "
                "Call set_tokens_and_distributions() first."
            )

        # Inject tokens and distributions into dataset configs
        train_cfg = self.datasets.train.copy()
        self.train_dataset = hydra.utils.instantiate(
            train_cfg, vocab=self.vocab, distributions=self.distributions
        )

        self.val_datasets = []
        for dataset_cfg in self.datasets.val:
            val_cfg = dataset_cfg.copy()
            self.val_datasets.append(
                hydra.utils.instantiate(
                    val_cfg, vocab=self.vocab, distributions=self.distributions
                )
            )

        self.test_datasets = []
        for dataset_cfg in self.datasets.test:
            test_cfg = dataset_cfg.copy()
            self.test_datasets.append(
                hydra.utils.instantiate(
                    test_cfg, vocab=self.vocab, distributions=self.distributions
                )
            )

    def collate_fn(self, batch, stage="train"):
        targets = batch
        if self.n_atoms_strategy == "fixed":
            n_atoms = [target.num_nodes for target in targets]
        elif self.n_atoms_strategy == "approx":
            # sample normal distribution with mean 0 and std 2 and add to target.num_nodes
            n_atoms = [
                target.num_nodes + int((torch.randn(1) * 2).round().item())
                for target in targets
            ]
            # floor to minimum of 3
            n_atoms = [max(3, n_atoms_i) for n_atoms_i in n_atoms]

        else:
            n_atoms = [None for target in targets]

        samples = [
            sample_prior_graph(
                self.distributions,
                n_atoms=n_atoms_i,
            )
            for n_atoms_i in n_atoms
        ]
        samples_batched = MoleculeBatch.from_data_list(samples)
        targets_batched = MoleculeBatch.from_data_list(targets)

        if stage == "train":
            batch_size = targets_batched.batch_size
            device = targets_batched.x.device

            # interpolate
            t = self.time_dist.sample((batch_size,)).to(device)
            # clip t_max
            t = torch.clamp(t, min=0.0, max=1 - 1e-6)
            mols_t, mols_1, ins_targets = self.interpolator.interpolate_different_size(
                samples_batched,
                targets_batched,
                t,
            )

            # Validate: check for cross-batch edges after interpolation
            validate_no_cross_batch_edges(
                mols_t.edge_index, mols_t.batch, "collate_fn train mols_t"
            )
            validate_no_cross_batch_edges(
                mols_1.edge_index, mols_1.batch, "collate_fn train mols_1"
            )

            return mols_t, mols_1, ins_targets, t

        else:
            # Validate: check for cross-batch edges in validation batches
            validate_no_cross_batch_edges(
                samples_batched.edge_index,
                samples_batched.batch,
                "collate_fn val samples",
            )
            validate_no_cross_batch_edges(
                targets_batched.edge_index,
                targets_batched.batch,
                "collate_fn val targets",
            )
            return samples_batched, targets_batched

    def train_dataloader(self):
        # Return a DataLoader for training
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.batch_size.train,
            num_workers=self.num_workers.train,
            persistent_workers=True,
            pin_memory=False,
            drop_last=True,
            collate_fn=partial(self.collate_fn, stage="train"),
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
                pin_memory=False,
                drop_last=True,
                collate_fn=partial(self.collate_fn, stage="val"),
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
                pin_memory=False,
                collate_fn=partial(self.collate_fn, stage="test"),
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
