import hydra
import omegaconf
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from omegaconf import DictConfig
from omegaconf import OmegaConf
from rdkit import RDLogger

from chemflow.utils import build_callbacks, remove_token_from_distribution

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
        vocab.edge_tokens = edge_tokens
        distributions.edge_type_distribution = edge_type_distribution

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
    )

    module.compile()

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
        # ckpt_path="/cluster/project/krause/frankem/chemflow/outputs/2026-01-13/10-43-06/checkpoints/epoch=306-step=78234.ckpt",
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
