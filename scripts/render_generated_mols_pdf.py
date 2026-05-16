#!/usr/bin/env python3
"""Generate molecules from a checkpoint and save them as SDF files.

All generations are written to two multi-molecule SDFs in ``--output-dir``:

* ``valid.sdf``   - molecules that passed ``mol_is_valid``
* ``invalid.sdf`` - everything else

Each entry has a ``_Name`` (``mol_XXXX``) and a ``valid`` SD tag (``1``/``0``)
so you can colour / filter them in PyMOL via ``load valid.sdf, multi=1`` or
``load invalid.sdf``.

Example
-------
    python scripts/render_generated_mols_pdf.py \\
        --checkpoint path/to/model.ckpt \\
        --output-dir eval_outputs/mol_sdfs \\
        --n-mols 200
"""

from __future__ import annotations

import argparse
import math
from copy import deepcopy
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from rdkit import Chem, RDLogger

from chemflow.dataset.vocab import setup_token_weights
from chemflow.utils.metrics import init_metrics
from chemflow.utils.utils import init_uniform_prior


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"


_KEKULIZE_EXCEPTIONS = tuple(
    exc
    for exc in (
        getattr(Chem, "AtomKekulizeException", None),
        getattr(Chem, "KekulizeException", None),
    )
    if exc is not None
)


def _register_resolvers() -> None:
    resolvers = {
        "oc.eval": eval,
        "len": lambda x: len(x),
        "if": lambda cond, t, f: t if cond else f,
        "eq": lambda x, y: x == y,
    }
    for name, fn in resolvers.items():
        if not OmegaConf.has_resolver(name):
            OmegaConf.register_new_resolver(name, fn)


def _setup_components(cfg, predict_batch_size: int):
    OmegaConf.set_struct(cfg, False)
    cfg.data.datamodule.batch_size.test = int(predict_batch_size)

    preprocessing = hydra.utils.instantiate(cfg.data.preprocessing)
    vocab = preprocessing.vocab
    distributions = preprocessing.distributions
    loss_weight_distributions = deepcopy(distributions)
    token_prior_distribution = init_uniform_prior(distributions)

    cfg.data.vocab = vocab
    datamodule = hydra.utils.instantiate(
        cfg.data.datamodule,
        _recursive_=False,
        vocab=vocab,
        distributions=token_prior_distribution,
    )
    datamodule.setup()

    allow_charged = bool(cfg.data.get("allow_charged", False))
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
        allow_charged=allow_charged,
    )

    tw = cfg.model.token_weighting
    atom_type_weights, edge_token_weights, charge_token_weights = setup_token_weights(
        vocab=vocab,
        distributions=loss_weight_distributions,
        weight_alpha=tw.weight_alpha,
        type_loss_token_weights=tw.type_loss_token_weights,
    )

    module = hydra.utils.instantiate(
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
        allow_charged=allow_charged,
    )

    test_dl = datamodule.test_dataloader()
    if isinstance(test_dl, list):
        test_dl = test_dl[0]
    return module, test_dl, vocab


def _load_checkpoint(module, checkpoint_path: Path) -> None:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    state_dict = {k.replace("._orig_mod", ""): v for k, v in state_dict.items()}
    module.load_state_dict(state_dict)


def _final_mol(traj):
    if isinstance(traj, list) and traj:
        return traj[-1]
    return traj


def _to_rdkit(mol, vocab):
    try:
        return mol.to_rdkit_mol(
            vocab.atom_tokens, vocab.edge_tokens, vocab.charge_tokens
        )
    except Exception:
        return None


def _kekulize_safe_writer(path: Path) -> Chem.SDWriter:
    """SDWriter that won't blow up on aromatic-perception failures."""
    writer = Chem.SDWriter(str(path))
    writer.SetKekulize(False)
    return writer


def _write_mol(writer: Chem.SDWriter, mol: Chem.Mol, name: str, is_valid: bool) -> bool:
    if mol is None:
        return False
    mol.SetProp("_Name", name)
    mol.SetProp("valid", "1" if is_valid else "0")
    try:
        writer.write(mol)
        return True
    except _KEKULIZE_EXCEPTIONS:
        # Downgrade aromatic bonds and retry once.
        fallback = Chem.RWMol(mol)
        for atom in fallback.GetAtoms():
            atom.SetIsAromatic(False)
        for bond in fallback.GetBonds():
            if bond.GetBondType() == Chem.BondType.AROMATIC or bond.GetIsAromatic():
                bond.SetBondType(Chem.BondType.SINGLE)
            bond.SetIsAromatic(False)
        fallback.SetProp("_Name", name)
        fallback.SetProp("valid", "1" if is_valid else "0")
        try:
            writer.write(fallback)
            return True
        except Exception as e:
            print(f"  skipping {name}: {e}")
            return False
    except Exception as e:
        print(f"  skipping {name}: {e}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("eval_outputs/mol_sdfs"),
    )
    parser.add_argument("--n-mols", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--predict-batch-size", type=int, default=64)
    parser.add_argument(
        "--overrides", nargs="*",
        default=["data=qm9", "model=semla", "cfg=uncond"],
        help="Hydra overrides; must match the loaded checkpoint's training config.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    valid_path = args.output_dir / "valid.sdf"
    invalid_path = args.output_dir / "invalid.sdf"

    _register_resolvers()
    torch.set_float32_matmul_precision("medium")
    RDLogger.DisableLog("rdApp.*")

    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.1"):
        cfg = compose(config_name="default", overrides=args.overrides)

    module, test_dl, vocab = _setup_components(cfg, args.predict_batch_size)
    _load_checkpoint(module, args.checkpoint)

    pl.seed_everything(args.seed, workers=True)

    n_predict_batches = max(
        1, math.ceil(args.n_mols / max(1, args.predict_batch_size))
    )
    trainer_kwargs = dict(cfg.trainer.trainer)
    trainer_kwargs["strategy"] = "auto"
    trainer_kwargs["limit_predict_batches"] = n_predict_batches

    trainer = pl.Trainer(
        logger=False, enable_checkpointing=False, **trainer_kwargs,
    )

    module.predict_return_traj = True
    module.predict_overrides = None

    print(f"Generating ~{args.n_mols} molecules...")
    predictions = trainer.predict(module, dataloaders=test_dl)

    valid_writer = _kekulize_safe_writer(valid_path)
    invalid_writer = _kekulize_safe_writer(invalid_path)

    n_valid = n_invalid = idx = 0
    try:
        for pred in predictions or []:
            if not isinstance(pred, dict):
                continue
            for traj in pred.get("valid_mols", []) or []:
                if idx >= args.n_mols:
                    break
                rdmol = _to_rdkit(_final_mol(traj), vocab)
                if _write_mol(valid_writer, rdmol, f"mol_{idx:04d}", True):
                    n_valid += 1
                idx += 1
            for traj in pred.get("invalid_mols", []) or []:
                if idx >= args.n_mols:
                    break
                rdmol = _to_rdkit(_final_mol(traj), vocab)
                if _write_mol(invalid_writer, rdmol, f"mol_{idx:04d}", False):
                    n_invalid += 1
                idx += 1
            if idx >= args.n_mols:
                break
    finally:
        valid_writer.close()
        invalid_writer.close()

    print(f"\nWrote {n_valid} valid molecules   -> {valid_path}")
    print(f"Wrote {n_invalid} invalid molecules -> {invalid_path}")
    print(
        "\nIn PyMOL:\n"
        f"  load {valid_path}, valid_mols, multi=1\n"
        f"  load {invalid_path}, invalid_mols, multi=1\n"
        "  set all_states, on    # show all conformers at once\n"
    )


if __name__ == "__main__":
    main()
