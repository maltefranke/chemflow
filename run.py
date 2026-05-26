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

from chemflow.utils.utils import build_callbacks, init_uniform_prior, bootstrap_run_id
from chemflow.dataset.vocab import setup_token_weights
from chemflow.dataset.representation import (
    Representation,
    project_distributions_to_representation,
    validate_representation,
)
from chemflow.model.lightning_module import LightningModuleRates
from chemflow.utils.metrics import init_metrics
from rdkit import Chem

# resolvers for more complex config expressions
OmegaConf.register_new_resolver("oc.eval", eval)
OmegaConf.register_new_resolver("len", lambda x: len(x))
OmegaConf.register_new_resolver("if", lambda cond, t, f: t if cond else f)
OmegaConf.register_new_resolver("eq", lambda x, y: x == y)

torch.set_float32_matmul_precision("medium")

# Disable only the warning level (most common choice)
RDLogger.DisableLog("rdApp.*")

pl.seed_everything(42)


def setup(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)

    # Validate the run's representation against the chosen dataset's class-level
    # CAPABILITIES before doing any data work. Imports the class only — no
    # instantiation, no I/O.
    representation = Representation(cfg.representation)
    dataset_cls = hydra.utils.get_class(cfg.data.datamodule.datasets.train._target_)
    validate_representation(dataset_cls.CAPABILITIES, representation)

    # Instantiate preprocessing to compute distributions from training dataset
    hydra.utils.log.info("Instantiating preprocessing...")
    preprocessing = hydra.utils.instantiate(cfg.data.preprocessing)

    # Extract tokens and distributions from preprocessing
    vocab = preprocessing.vocab
    distributions = preprocessing.distributions

    # Channel A — canonical real frequencies, for loss weighting + metrics.
    loss_weight_distributions = deepcopy(distributions)

    # Channel B — priors used by sample_prior_graph / interpolator / integrator.
    # Project edge/charge priors onto the representation so sampled tokens match
    # the all-zero targets produced by project_molecule_to_representation
    # (otherwise non-topology runs would draw real bond tokens at t=0 and trip
    # the edge-substitution path the gating relies on being inactive).
    token_prior_distribution = init_uniform_prior(distributions)
    token_prior_distribution = project_distributions_to_representation(
        token_prior_distribution, vocab, representation
    )

    cfg.data.vocab = vocab

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

    # Compute token weights for loss weighting (using training-frequency distributions)
    tw = cfg.model.token_weighting
    atom_type_weights, edge_token_weights, charge_token_weights = setup_token_weights(
        vocab=vocab,
        distributions=loss_weight_distributions,
        weight_alpha=tw.weight_alpha,
        type_loss_token_weights=tw.type_loss_token_weights,
    )

    # Whether charged molecules are considered valid for this dataset.
    # Defaults to False (e.g. QM9) if the data config does not set it.
    allow_charged = bool(cfg.data.get("allow_charged", False))

    # Build metrics (including novelty against the training set). Fetched
    # whenever the dataset can provide SMILES, independent of representation, so
    # downstream RDKit-side metrics (e.g. Novelty) work even if bonds are later
    # inferred in non-topology modes. A pointcloud-only dataset like TMQM may not
    # implement get_all_smiles() at all, so fetch defensively.
    base_dataset = datamodule.train_dataset.base_dataset
    train_smiles = (
        base_dataset.get_all_smiles()
        if hasattr(base_dataset, "get_all_smiles")
        else None
    )
    metrics, stability_metrics, distribution_metrics, batch_metrics = init_metrics(
        train_smiles=train_smiles,
        target_n_atoms_distribution=loss_weight_distributions.n_atoms_distribution,
        atom_type_distribution=loss_weight_distributions.atom_type_distribution,
        edge_type_distribution=loss_weight_distributions.edge_type_distribution,
        charge_type_distribution=loss_weight_distributions.charge_type_distribution,
        atom_tokens=list(vocab.atom_tokens),
        edge_tokens=list(vocab.edge_tokens),
        charge_tokens=list(vocab.charge_tokens),
        allow_charged=allow_charged,
        distributions=loss_weight_distributions,
        representation=representation,
    )

    # Instantiate module
    hydra.utils.log.info(f"Instantiating <{cfg.model.module._target_}>")
    module: pl.LightningModule = hydra.utils.instantiate(
        cfg.model.module,
        _recursive_=False,
        distributions=token_prior_distribution,
        loss_weight_distributions=loss_weight_distributions,
        atom_type_weights=atom_type_weights,
        edge_token_weights=edge_token_weights,
        charge_token_weights=charge_token_weights,
        metrics=metrics,
        stability_metrics=stability_metrics,
        distribution_metrics=distribution_metrics,
        batch_metrics=batch_metrics,
        allow_charged=allow_charged,
        log_grad_norms_every_n_steps=0,  # disable loss gradient checking
        representation=representation.value,
    )

    module.compile()

    # Setup logging and callbacks. Skip wandb when we are not training so
    # validate/predict-only runs don't spawn a run.
    logger = WandbLogger(**cfg.logging) if cfg.trainer.do_train else False
    callbacks = build_callbacks(cfg)
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)

    # Instantiate trainer
    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        **cfg.trainer.trainer,
    )

    return module, datamodule, trainer


def train(module, datamodule, trainer, ckpt_path=None):
    trainer.fit(
        module,
        datamodule=datamodule,
        ckpt_path=ckpt_path,
    )


def validate(module, datamodule, trainer, ckpt_path=None):
    trainer.validate(
        module,
        dataloaders=datamodule.val_dataloader(),
        ckpt_path=ckpt_path,
    )


def predict(module, datamodule, trainer, ckpt_path=None):

    predictions = trainer.predict(
        module,
        dataloaders=datamodule.test_dataloader(),
        ckpt_path=ckpt_path,
    )

    # Flatten the lists directly in your main script
    all_valid_mols = []
    all_invalid_mols = []
    all_invalid_mols_rdkit = []

    for batch_output in predictions:
        all_valid_mols.extend(batch_output["valid_mols"])
        all_invalid_mols.extend(batch_output["invalid_mols"])
        all_invalid_mols_rdkit.extend(batch_output["invalid_mols_rdkit"])

    torch.save(all_valid_mols, "valid_mols.pt")
    torch.save(all_invalid_mols, "invalid_mols.pt")

    writer = Chem.SDWriter("invalid_molecules.sdf")

    for mol in all_invalid_mols_rdkit:
        # Optional: Catch sanitization errors during writing just in case
        if mol is None:
            continue

        try:
            writer.write(mol)
        except Exception as e:
            print(f"Could not write molecule to SDF: {e}")

    writer.close()


@hydra.main(
    config_path="configs",
    config_name="default",
    version_base="1.1",
)
def main(cfg: omegaconf.DictConfig):
    ckpt_path = cfg.trainer.checkpoint_path

    module, datamodule, trainer = setup(cfg)

    if cfg.trainer.do_train:
        train(module, datamodule, trainer, ckpt_path)
    if cfg.trainer.do_validate:
        validate(module, datamodule, trainer, ckpt_path)
    if cfg.trainer.do_predict:
        predict(module, datamodule, trainer, ckpt_path)


if __name__ == "__main__":
    bootstrap_run_id()
    main()
