#!/usr/bin/env python3
import os
import sys
import math
import argparse
from copy import deepcopy

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in [_PROJECT_ROOT, os.path.join(_PROJECT_ROOT, "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import hydra
from hydra import compose, initialize_config_dir
import pytorch_lightning as pl
from omegaconf import OmegaConf

# Same custom resolvers as `run.py`; YAML uses ${eq:...}, ${if:...}, ${len:...}, ${oc.eval:...}.
OmegaConf.register_new_resolver("oc.eval", eval)
OmegaConf.register_new_resolver("len", lambda x: len(x))
OmegaConf.register_new_resolver("if", lambda cond, t, f: t if cond else f)
OmegaConf.register_new_resolver("eq", lambda x, y: x == y)

from chemflow.utils.utils import init_uniform_prior
from chemflow.dataset.vocab import setup_token_weights
from chemflow.utils.metrics import init_metrics


def compose_cfg(config_path: str, config_name: str, overrides: list[str]):
    # Absolute path: `hydra.initialize(relpath)` resolves against *cwd*, so running
    # from `rl/` or a mis-inferred root breaks. `initialize_config_dir` does not.
    cfg_dir = os.path.abspath(os.path.expanduser(config_path))
    with initialize_config_dir(config_dir=cfg_dir, version_base="1.1"):
        cfg = compose(config_name=config_name, overrides=overrides)
    OmegaConf.set_struct(cfg, False)
    return cfg


def build_module_and_datamodule(cfg):
    preprocessing = hydra.utils.instantiate(cfg.data.preprocessing)
    vocab = preprocessing.vocab
    distributions = preprocessing.distributions
    loss_weight_distributions = deepcopy(distributions)
    token_prior = init_uniform_prior(distributions)
    cfg.data.vocab = vocab
    datamodule = hydra.utils.instantiate(
        cfg.data.datamodule,
        _recursive_=False,
        vocab=vocab,
        distributions=token_prior,
    )
    datamodule.setup()
    tw = cfg.model.token_weighting
    atom_type_weights, edge_token_weights, charge_token_weights = setup_token_weights(
        vocab=vocab,
        distributions=loss_weight_distributions,
        weight_alpha=tw.weight_alpha,
        type_loss_token_weights=tw.type_loss_token_weights,
    )
    train_smiles = datamodule.train_dataset.base_dataset.get_all_smiles()
    metrics, stability_metrics, distribution_metrics = init_metrics(
        train_smiles=train_smiles,
        target_n_atoms_distribution=loss_weight_distributions.n_atoms_distribution,
        atom_type_distribution=loss_weight_distributions.atom_type_distribution,
        edge_type_distribution=loss_weight_distributions.edge_type_distribution,
        charge_type_distribution=loss_weight_distributions.charge_type_distribution,
        atom_tokens=list(vocab.atom_tokens),
        edge_tokens=list(vocab.edge_tokens),
        charge_tokens=list(vocab.charge_tokens),
    )
    module = hydra.utils.instantiate(
        cfg.model.module,
        _recursive_=False,
        distributions=token_prior,
        loss_weight_distributions=loss_weight_distributions,
        atom_type_weights=atom_type_weights,
        edge_token_weights=edge_token_weights,
        charge_token_weights=charge_token_weights,
        metrics=metrics,
        stability_metrics=stability_metrics,
        distribution_metrics=distribution_metrics,
    )
    return module, datamodule


def load_ckpt_into_module(module, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"]
    state_dict = {k.replace("._orig_mod", ""): v for k, v in state_dict.items()}
    module.load_state_dict(state_dict, strict=True)
    return module


def _count_atoms_from_traj(traj) -> int:
    """Count total atoms in the final state of a generated trajectory."""
    try:
        return int(traj[-1].a.numel())
    except Exception:
        return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ckpt",
        default=os.path.join(_PROJECT_ROOT, ".pretrained_model", "epoch=499-step=48500.ckpt"),
        help="Path to .ckpt",
    )
    ap.add_argument("--n_mols", type=int, default=100)
    ap.add_argument("--config_path", default=os.path.join(_PROJECT_ROOT, "configs"))
    ap.add_argument("--config_name", default="default")
    ap.add_argument(
        "--out",
        default=os.path.join(_PROJECT_ROOT, "pretrained_validity.pt"),
        help="Where to save results (.pt)",
    )
    ap.add_argument(
        "overrides",
        nargs="*",
        help="Hydra overrides, e.g. model=semla cfg=uncond model.gmm_params.K=10",
    )
    args = ap.parse_args()
    cfg = compose_cfg(args.config_path, args.config_name, overrides=list(args.overrides))
    # Ensure we generate at least n_mols molecules.
    bs = int(cfg.data.datamodule.batch_size.test)
    cfg.trainer.trainer.limit_predict_batches = int(math.ceil(args.n_mols / max(bs, 1)))
    module, datamodule = build_module_and_datamodule(cfg)
    module = load_ckpt_into_module(module, args.ckpt)
    trainer = pl.Trainer(
        logger=False,
        callbacks=[],
        enable_checkpointing=False,
        **OmegaConf.to_container(cfg.trainer.trainer, resolve=True),
    )
    preds = trainer.predict(module, dataloaders=datamodule.test_dataloader())
    valid = []
    invalid = []
    invalid_rdkit = []
    for out in preds:
        valid.extend(out["valid_mols"])
        invalid.extend(out["invalid_mols"])
        invalid_rdkit.extend(out["invalid_mols_rdkit"])
    total = len(valid) + len(invalid)
    if total > args.n_mols:
        # Prefer keeping all invalid RDKit molecules aligned with invalid_mols length.
        keep = args.n_mols
        if len(valid) >= keep:
            valid = valid[:keep]
            invalid = []
            invalid_rdkit = []
        else:
            keep_invalid = keep - len(valid)
            invalid = invalid[:keep_invalid]
            invalid_rdkit = invalid_rdkit[:keep_invalid]
    total = len(valid) + len(invalid)
    validity = (len(valid) / total) if total else 0.0
    valid_n_atoms = [_count_atoms_from_traj(traj) for traj in valid]
    invalid_n_atoms = [_count_atoms_from_traj(traj) for traj in invalid]
    all_n_atoms = valid_n_atoms + invalid_n_atoms
    out = dict(
        ckpt=args.ckpt,
        n_requested=args.n_mols,
        n_generated=total,
        n_valid=len(valid),
        n_invalid=len(invalid),
        validity=validity,
        valid_mols=valid,
        invalid_mols=invalid,
        invalid_mols_rdkit=invalid_rdkit,
        valid_n_atoms=valid_n_atoms,
        invalid_n_atoms=invalid_n_atoms,
        all_n_atoms=all_n_atoms,
        n_atoms_mean_valid=(sum(valid_n_atoms) / len(valid_n_atoms)) if valid_n_atoms else 0.0,
        n_atoms_max_valid=max(valid_n_atoms) if valid_n_atoms else 0,
        n_atoms_mean_all=(sum(all_n_atoms) / len(all_n_atoms)) if all_n_atoms else 0.0,
        n_atoms_max_all=max(all_n_atoms) if all_n_atoms else 0,
        hydra_overrides=list(args.overrides),
    )
    torch.save(out, args.out)
    print(f"saved: {args.out}")
    print(f"generated: {total} | valid: {len(valid)} | validity: {validity:.4f}")
    print(
        "n_atoms: "
        f"mean_valid={out['n_atoms_mean_valid']:.2f} "
        f"max_valid={out['n_atoms_max_valid']} "
        f"mean_all={out['n_atoms_mean_all']:.2f} "
        f"max_all={out['n_atoms_max_all']}"
    )


if __name__ == "__main__":
    main()
