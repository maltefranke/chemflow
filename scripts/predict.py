"""Run inference on the test set from a saved checkpoint.

Usage:
    python predict.py ckpt_path=/path/to/checkpoint.ckpt

Outputs (written to the Hydra working directory):
    valid_mols.pt          — list of trajectories for valid molecules
    invalid_mols.pt        — list of trajectories for invalid molecules
    invalid_molecules.sdf  — SDF file of invalid molecules
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import hydra
import omegaconf
import torch
from copy import deepcopy
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from rdkit import RDLogger, Chem

from chemflow.utils.utils import build_callbacks, init_uniform_prior
from chemflow.dataset.vocab import setup_token_weights
from chemflow.model.lightning_module import LightningModuleRates

OmegaConf.register_new_resolver("oc.eval", eval)
OmegaConf.register_new_resolver("len", lambda x: len(x))
OmegaConf.register_new_resolver("if", lambda cond, t, f: t if cond else f)
OmegaConf.register_new_resolver("eq", lambda x, y: x == y)

torch.set_float32_matmul_precision("medium")
RDLogger.DisableLog("rdApp.*")
pl.seed_everything(42)


CKPT_PATH = "/cluster/project/jorner/schmiste/flexflow/chemflow/outputs/2026-04-04/00-00-38/logs/chemflow/gq3aqksl/checkpoints/epoch=1569-step=59660.ckpt"


def predict(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)

    ckpt_path = CKPT_PATH

    # ── preprocessing ──
    hydra.utils.log.info("Instantiating preprocessing...")
    preprocessing = hydra.utils.instantiate(cfg.data.preprocessing)

    vocab = preprocessing.vocab
    distributions = preprocessing.distributions
    loss_weight_distributions = deepcopy(distributions)
    token_prior_distribution = init_uniform_prior(distributions)

    cfg.data.vocab = vocab

    hydra.utils.log.info(
        f"Preprocessing complete.\n"
        f"Found {len(vocab.atom_tokens)} atom tokens: {vocab.atom_tokens}\n"
        f"Found {len(vocab.edge_tokens)} edge tokens: {vocab.edge_tokens}\n"
        f"Found {len(vocab.charge_tokens)} charge tokens: {vocab.charge_tokens}"
    )

    # ── datamodule ──
    hydra.utils.log.info(f"Instantiating <{cfg.data.datamodule._target_}>")
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(
        cfg.data.datamodule,
        _recursive_=False,
        vocab=vocab,
        distributions=token_prior_distribution,
    )
    datamodule.setup()

    # ── module ──
    tw = cfg.model.token_weighting
    atom_type_weights, edge_token_weights, charge_token_weights = setup_token_weights(
        vocab=vocab,
        distributions=loss_weight_distributions,
        weight_alpha=tw.weight_alpha,
        type_loss_token_weights=tw.type_loss_token_weights,
    )

    hydra.utils.log.info(f"Instantiating <{cfg.model.module._target_}>")
    module: pl.LightningModule = hydra.utils.instantiate(
        cfg.model.module,
        _recursive_=False,
        distributions=token_prior_distribution,
        loss_weight_distributions=loss_weight_distributions,
        atom_type_weights=atom_type_weights,
        edge_token_weights=edge_token_weights,
        charge_token_weights=charge_token_weights,
    )

    # ── load checkpoint ──
    hydra.utils.log.info(f"Loading checkpoint from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"]
    state_dict = {k.replace("._orig_mod", ""): v for k, v in state_dict.items()}
    module.load_state_dict(state_dict)

    module.predict_return_traj = True

    # ── trainer ──
    trainer = pl.Trainer(
        logger=False,
        callbacks=[],
        enable_checkpointing=False,
        **cfg.trainer.trainer,
    )

    # ── predict ──
    hydra.utils.log.info("Running prediction on test set...")
    predictions = trainer.predict(
        module,
        dataloaders=datamodule.test_dataloader(),
    )

    # ── collect results ──
    all_valid_mols = []
    all_invalid_mols = []
    all_invalid_mols_rdkit = []

    for batch_output in predictions:
        all_valid_mols.extend(batch_output["valid_mols"])
        all_invalid_mols.extend(batch_output["invalid_mols"])
        all_invalid_mols_rdkit.extend(batch_output["invalid_mols_rdkit"])

    hydra.utils.log.info(
        f"Done. Valid: {len(all_valid_mols)}, Invalid: {len(all_invalid_mols)}"
    )

    torch.save(all_valid_mols, "valid_mols.pt")
    torch.save(all_invalid_mols, "invalid_mols.pt")

    writer = Chem.SDWriter("invalid_molecules.sdf")
    for mol in all_invalid_mols_rdkit:
        if mol is None:
            continue
        try:
            writer.write(mol)
        except Exception as e:
            print(f"Could not write molecule to SDF: {e}")
    writer.close()

    hydra.utils.log.info("Saved valid_mols.pt, invalid_mols.pt, invalid_molecules.sdf")


@hydra.main(
    config_path="../configs",
    config_name="default",
    version_base="1.1",
)
def main(cfg: omegaconf.DictConfig):
    predict(cfg)


if __name__ == "__main__":
    main()
