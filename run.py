import hydra
import omegaconf
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from omegaconf import DictConfig

from yourprojectname.utils import build_callbacks


def run(cfg: DictConfig):
    # Instantiate datamodule given the datamolecule
    hydra.utils.log.info(f"Instantiating <{cfg.data.datamodule._target_}>")
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(
        cfg.data.datamodule, _recursive_=False
    )

    # Instantiate module
    hydra.utils.log.info(f"Instantiating <{cfg.model.module._target_}>")
    module: pl.LightningModule = hydra.utils.instantiate(
        cfg.model.module, _recursive_=False
    )

    # Setup logging and callbacks
    wandb_logger = WandbLogger(**cfg.logging)
    callbacks = build_callbacks(cfg)

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


@hydra.main(
    config_path="configs",
    config_name="default",
    version_base="1.1",
)
def main(cfg: omegaconf.DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
