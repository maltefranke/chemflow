import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from omegaconf import DictConfig
import hydra

from chemflow.dataset.flow_matching_wrapper import (
    FlowMatchingDatasetWrapper,
    train_collate_fn,
    eval_collate_fn,
    worker_init_fn,
)
from chemflow.dataset.representation import Representation


class LightningDataModule(pl.LightningDataModule):
    def __init__(
        self,
        vocab: DictConfig,
        distributions: DictConfig,
        datasets: DictConfig,
        interpolator: DictConfig,
        num_workers: DictConfig,
        batch_size: DictConfig,
        representation: str | Representation = Representation.GEOMETRIC_GRAPH,
        n_atoms_strategy: str = "flexible",
        optimal_transport: str = "equivariant",
        time_dist: DictConfig = None,
        rotate: bool = False,
        n_augmentations: int = 1,
    ):
        self.vocab = vocab
        self.distributions = distributions
        self.datasets = datasets
        self.interpolator = interpolator
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.representation = Representation(representation)
        self.n_atoms_strategy = n_atoms_strategy
        self.optimal_transport = optimal_transport
        self.rotate = rotate
        self.n_augmentations = max(1, int(n_augmentations))

        if self.n_augmentations > 1 and self.batch_size.train % self.n_augmentations != 0:
            raise ValueError(
                f"batch_size.train ({self.batch_size.train}) must be divisible by "
                f"n_augmentations ({self.n_augmentations})."
            )

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
        if self.vocab is None:
            raise ValueError(
                "vocab and distributions must be set before calling setup(). "
                "Call set_tokens_and_distributions() first."
            )

        # Inject tokens and distributions into dataset configs
        train_cfg = self.datasets.train.copy()
        base_train = hydra.utils.instantiate(
            train_cfg, vocab=self.vocab, distributions=self.distributions, split="train"
        )
        self.train_dataset = FlowMatchingDatasetWrapper(
            base_dataset=base_train,
            distributions=self.distributions,
            interpolator=self.interpolator,
            vocab=self.vocab,
            representation=self.representation,
            n_atoms_strategy=self.n_atoms_strategy,
            time_dist=self.time_dist,
            stage="train",
            rotate=self.rotate,
            n_augmentations=self.n_augmentations,
        )

        self.val_datasets = []
        for dataset_cfg in self.datasets.val:
            val_cfg = dataset_cfg.copy()
            base_val = hydra.utils.instantiate(
                val_cfg, vocab=self.vocab, distributions=self.distributions, split="val"
            )
            self.val_datasets.append(
                FlowMatchingDatasetWrapper(
                    base_dataset=base_val,
                    distributions=self.distributions,
                    interpolator=self.interpolator,
                    vocab=self.vocab,
                    representation=self.representation,
                    n_atoms_strategy=self.n_atoms_strategy,
                    time_dist=self.time_dist,
                    stage="val",
                )
            )

        self.test_datasets = []
        for dataset_cfg in self.datasets.test:
            test_cfg = dataset_cfg.copy()
            base_test = hydra.utils.instantiate(
                test_cfg,
                vocab=self.vocab,
                distributions=self.distributions,
                split="test",
            )
            self.test_datasets.append(
                FlowMatchingDatasetWrapper(
                    base_dataset=base_test,
                    distributions=self.distributions,
                    interpolator=self.interpolator,
                    vocab=self.vocab,
                    representation=self.representation,
                    n_atoms_strategy=self.n_atoms_strategy,
                    time_dist=self.time_dist,
                    stage="test",
                )
            )

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        """Move batch to *device* with ``non_blocking=True``.

        Lightning's default path uses ``apply_to_collection`` keyed on
        ``Tensor``, which does not recurse into PyG ``Data``/``Batch``
        objects.  We handle the concrete batch formats produced by
        ``train_collate_fn`` and ``eval_collate_fn`` explicitly so that
        every tensor attribute inside the PyG objects benefits from the
        asynchronous DMA transfer enabled by ``pin_memory=True``.
        """
        if isinstance(batch, (tuple, list)):
            return type(batch)(
                item.to(device, non_blocking=True)
                if isinstance(item, torch.Tensor) or hasattr(item, "stores")
                else item
                for item in batch
            )
        if isinstance(batch, torch.Tensor) or hasattr(batch, "stores"):
            return batch.to(device, non_blocking=True)
        return batch

    def train_dataloader(self):
        nw = self.num_workers.train
        # Each __getitem__ produces n_augmentations training items, so we
        # request batch_size.train // n_augmentations unique molecules per
        # batch and the wrapper expands them to the configured effective size.
        per_load_batch_size = self.batch_size.train // self.n_augmentations
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=per_load_batch_size,
            num_workers=nw,
            persistent_workers=nw > 0,
            pin_memory=True,
            drop_last=True,
            collate_fn=train_collate_fn,
            worker_init_fn=worker_init_fn,
            prefetch_factor=2 if nw > 0 else None,
        )

    def val_dataloader(self):
        nw = self.num_workers.val
        return [
            DataLoader(
                dataset,
                shuffle=False,
                batch_size=self.batch_size.val,
                num_workers=nw,
                persistent_workers=nw > 0,
                pin_memory=True,
                drop_last=True,
                collate_fn=eval_collate_fn,
                worker_init_fn=worker_init_fn,
                prefetch_factor=2 if nw > 0 else None,
            )
            for dataset in self.val_datasets
        ]

    def test_dataloader(self):
        nw = self.num_workers.test
        return [
            DataLoader(
                dataset,
                shuffle=False,
                batch_size=self.batch_size.test,
                num_workers=nw,
                persistent_workers=nw > 0,
                pin_memory=True,
                collate_fn=eval_collate_fn,
                worker_init_fn=worker_init_fn,
                prefetch_factor=2 if nw > 0 else None,
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
