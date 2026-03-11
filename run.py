import hydra
import omegaconf
import torch
from copy import deepcopy
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from omegaconf import DictConfig
from omegaconf import OmegaConf
from rdkit import RDLogger
from pytorch_lightning.strategies import DDPStrategy

from chemflow.utils.utils  import build_callbacks, init_uniform_prior
from chemflow.model.lightning_module import LightningModuleRates

# resolvers for more complex config expressions
OmegaConf.register_new_resolver("oc.eval", eval)
OmegaConf.register_new_resolver("len", lambda x: len(x))
OmegaConf.register_new_resolver("if", lambda cond, t, f: t if cond else f)
OmegaConf.register_new_resolver("eq", lambda x, y: x == y)

torch.set_float32_matmul_precision("medium")

# Disable only the warning level (most common choice)
RDLogger.DisableLog("rdApp.*")

pl.seed_everything(42)


def run(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)

    # Instantiate preprocessing to compute distributions from training dataset
    hydra.utils.log.info("Instantiating preprocessing...")
    preprocessing = hydra.utils.instantiate(cfg.data.preprocessing)

    # Extract tokens and distributions from preprocessing
    vocab = preprocessing.vocab
    distributions = preprocessing.distributions

    # Keep training-frequency distributions for loss weighting.
    loss_weight_distributions = deepcopy(distributions)

    token_prior_distribution = init_uniform_prior(distributions)

    cfg.data.vocab = vocab
    print(cfg)

    hydra.utils.log.info(
        f"Preprocessing complete.\n"
        f"Found {len(vocab.atom_tokens)} atom tokens: {vocab.atom_tokens}\n"
        f"Found {len(vocab.edge_tokens)} edge tokens: {vocab.edge_tokens}\n"
        f"Found {len(vocab.charge_tokens)} charge tokens: {vocab.charge_tokens}"
    )

    hydra.utils.log.info("Distributions computed from training dataset.")

    # Instantiate datamodule
    hydra.utils.log.info(f"Instantiating <{cfg.data.datamodule._target_}>")
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(
        cfg.data.datamodule,
        _recursive_=False,
        vocab=vocab,
        distributions=token_prior_distribution,
    )
    # Call setup to create datasets with tokens and distributions
    datamodule.setup()

    # Instantiate module
    hydra.utils.log.info(f"Instantiating <{cfg.model.module._target_}>")
    module: pl.LightningModule = hydra.utils.instantiate(
        cfg.model.module,
        _recursive_=False,
        distributions=token_prior_distribution,
        loss_weight_distributions=loss_weight_distributions,
    )

    # module.compile()

    # Setup logging and callbacks
    wandb_logger = WandbLogger(**cfg.logging)
    callbacks = build_callbacks(cfg)
    lr_monitor = LearningRateMonitor(logging_interval="step")

    callbacks.append(lr_monitor)
    # Instantiate trainer
    trainer = pl.Trainer(
        # strategy=DDPStrategy(find_unused_parameters=True),
        logger=wandb_logger,
        callbacks=callbacks,
        **cfg.trainer.trainer,
    )

    ckpt_path = None
    # ckpt_path = "/capstor/store/cscs/swissai/a131/frankem/chemflow/logs/wandb/sched_learnable_w/chemflow/asoc4re8/checkpoints/epoch=1999-step=32000.ckpt"

    # Train the model
    trainer.fit(
        module,
        datamodule=datamodule,
        ckpt_path=ckpt_path,
    )

    """trainer.validate(
        module,
        dataloaders=datamodule.val_dataloader(),
        ckpt_path=ckpt_path,
    )
    exit()"""

    valid_mols, invalid_mols = trainer.predict(
        module,
        dataloaders=datamodule.test_dataloader(),
        ckpt_path=ckpt_path,
    )

    torch.save(valid_mols, "valid_mols.pt")
    torch.save(invalid_mols, "invalid_mols.pt")


@hydra.main(
    config_path="configs",
    config_name="default",
    version_base="1.1",
)
def main(cfg: omegaconf.DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
