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

from chemflow.utils import build_callbacks, remove_token_from_distribution
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

    if cfg.data.cat_strategy != "mask":
        # remove <MASK> token from the atom_type_distribution and edge_type_distribution
        atom_tokens, atom_type_distribution = remove_token_from_distribution(
            vocab.atom_tokens, distributions.atom_type_distribution, "<MASK>"
        )
        edge_tokens, edge_type_distribution = remove_token_from_distribution(
            vocab.edge_tokens, distributions.edge_type_distribution, "<MASK>"
        )
        vocab.atom_tokens = atom_tokens
        distributions.atom_type_distribution = atom_type_distribution
        loss_weight_distributions.atom_type_distribution = atom_type_distribution
        vocab.edge_tokens = edge_tokens
        distributions.edge_type_distribution = edge_type_distribution
        loss_weight_distributions.edge_type_distribution = edge_type_distribution

    if cfg.data.cat_strategy == "uniform-sample":
        # Sampling prior uses uniform categorical distributions.
        distributions.atom_type_distribution = torch.ones_like(
            distributions.atom_type_distribution
        )
        distributions.atom_type_distribution = (
            distributions.atom_type_distribution
            / distributions.atom_type_distribution.sum()
        )
        distributions.edge_type_distribution = torch.ones_like(
            distributions.edge_type_distribution
        )
        distributions.edge_type_distribution = (
            distributions.edge_type_distribution
            / distributions.edge_type_distribution.sum()
        )
        distributions.charge_type_distribution = torch.ones_like(
            distributions.charge_type_distribution
        )
        distributions.charge_type_distribution = (
            distributions.charge_type_distribution
            / distributions.charge_type_distribution.sum()
        )

    cfg.data.vocab = vocab
    # cfg.data.distributions = distributions

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
        distributions=distributions,
    )
    # Call setup to create datasets with tokens and distributions
    datamodule.setup()

    # Instantiate module
    hydra.utils.log.info(f"Instantiating <{cfg.model.module._target_}>")
    module: pl.LightningModule = hydra.utils.instantiate(
        cfg.model.module,
        _recursive_=False,
        distributions=distributions,
        loss_weight_distributions=loss_weight_distributions,
    )

    module.compile()

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
    # ckpt_path = "/cluster/project/krause/frankem/chemflow/outputs/2026-02-28/10-54-28/logs/chemflow/8u2u45g9/checkpoints/epoch=469-step=10340.ckpt"

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

    results = trainer.predict(
        module,
        dataloaders=datamodule.test_dataloader(),
        ckpt_path=ckpt_path,
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
