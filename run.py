import hydra
import omegaconf
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from omegaconf import DictConfig
from omegaconf import OmegaConf

from chemflow.utils import build_callbacks

OmegaConf.register_new_resolver("oc.eval", eval)
OmegaConf.register_new_resolver("len", lambda x: len(x))
OmegaConf.register_new_resolver("if", lambda cond, t, f: t if cond else f)
OmegaConf.register_new_resolver("eq", lambda x, y: x == y)

torch.set_float32_matmul_precision("medium")

pl.seed_everything(42)


def run(cfg: DictConfig):
    # Instantiate preprocessing to compute distributions from training dataset
    hydra.utils.log.info("Instantiating preprocessing...")
    preprocessing = hydra.utils.instantiate(cfg.data.preprocessing)

    # Extract tokens and distributions from preprocessing
    atom_tokens = preprocessing.atom_tokens
    edge_tokens = preprocessing.edge_tokens
    charge_tokens = preprocessing.charge_tokens
    atom_type_distribution = preprocessing.atom_type_distribution
    edge_type_distribution = preprocessing.edge_type_distribution
    charge_type_distribution = preprocessing.charge_type_distribution
    n_atoms_distribution = preprocessing.n_atoms_distribution
    coordinate_std = preprocessing.coordinate_std

    OmegaConf.update(cfg.data, "atom_tokens", atom_tokens)
    OmegaConf.update(cfg.data, "edge_tokens", edge_tokens)
    OmegaConf.update(cfg.data, "charge_tokens", charge_tokens)

    hydra.utils.log.info(
        f"Preprocessing complete. Found {len(atom_tokens)} tokens: {atom_tokens}"
    )
    hydra.utils.log.info("Distributions computed from training dataset.")

    # Instantiate datamodule
    hydra.utils.log.info(f"Instantiating <{cfg.data.datamodule._target_}>")
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(
        cfg.data.datamodule,
        _recursive_=False,
    )
    # Set tokens and distributions after initialization
    datamodule.set_tokens_and_distributions(
        atom_tokens=atom_tokens,
        edge_tokens=edge_tokens,
        charge_tokens=charge_tokens,
        atom_type_distribution=atom_type_distribution,
        edge_type_distribution=edge_type_distribution,
        charge_type_distribution=charge_type_distribution,
        n_atoms_distribution=n_atoms_distribution,
        coord_std=coordinate_std,
        cat_strategy=cfg.data.cat_strategy,
    )
    # Call setup to create datasets with tokens and distributions
    datamodule.setup()

    # Instantiate module
    hydra.utils.log.info(f"Instantiating <{cfg.model.module._target_}>")
    module: pl.LightningModule = hydra.utils.instantiate(
        cfg.model.module,
        _recursive_=False,
    )
    # Set tokens and distribution after initialization
    module.set_tokens_and_distribution(
        atom_tokens=atom_tokens,
        edge_tokens=edge_tokens,
        charge_tokens=charge_tokens,
        atom_type_distribution=atom_type_distribution,
        edge_type_distribution=edge_type_distribution,
        charge_type_distribution=charge_type_distribution,
    )
    # module.compile()

    # Setup logging and callbacks
    wandb_logger = WandbLogger(**cfg.logging)
    callbacks = build_callbacks(cfg)
    lr_monitor = LearningRateMonitor(logging_interval="step")

    callbacks.append(lr_monitor)
    # Instantiate trainer
    trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=callbacks,
        **cfg.trainer.trainer,
    )

    # Train the model
    trainer.fit(
        module,
        datamodule=datamodule,
    )

    results = trainer.predict(
        module,
        dataloaders=datamodule.test_dataloader(),
    )
    torch.save(results, "results.pt")


@hydra.main(
    config_path="configs",
    config_name="default",
    version_base="1.1",
)
def main(cfg: omegaconf.DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
