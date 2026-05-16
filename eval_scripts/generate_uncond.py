#!/usr/bin/env python3
"""Unconditional trajectory generation.

Loads an unconditional checkpoint (cfg=uncond) and dumps a small batch of
generated trajectories in the same on-disk format produced by
``eval_natoms_cfg.py`` (``valid_trajectories`` / ``invalid_trajectories``
lists of per-step ``MoleculeData``), so the existing analysis tooling
(``scripts/dump_insert_delete_trajectory_sdf.py``,
``src/chemflow/utils/diagnostics.py``, the GIF notebook) works as-is.

Usage:
    python eval_scripts/generate_uncond.py \\
        --checkpoint path/to/uncond.ckpt \\
        --output-dir eval_outputs/uncond \\
        --n-mols 64 \\
        --predict-batch-size 64
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
from rdkit import RDLogger

from chemflow.dataset.vocab import setup_token_weights
from chemflow.utils.metrics import init_metrics
from chemflow.utils.utils import init_uniform_prior


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"


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


def _setup(cfg, predict_batch_size: int):
    OmegaConf.set_struct(cfg, False)
    cfg.data.datamodule.batch_size.test = int(predict_batch_size)

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

    allow_charged = bool(cfg.data.get("allow_charged", False))
    train_smiles = datamodule.train_dataset.base_dataset.get_all_smiles()
    metrics, stab_metrics, dist_metrics = init_metrics(
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
    aw, ew, cw = setup_token_weights(
        vocab=vocab,
        distributions=loss_weight_distributions,
        weight_alpha=tw.weight_alpha,
        type_loss_token_weights=tw.type_loss_token_weights,
    )

    module = hydra.utils.instantiate(
        cfg.model.module,
        _recursive_=False,
        distributions=token_prior,
        loss_weight_distributions=loss_weight_distributions,
        atom_type_weights=aw,
        edge_token_weights=ew,
        charge_token_weights=cw,
        metrics=metrics,
        stability_metrics=stab_metrics,
        distribution_metrics=dist_metrics,
        allow_charged=allow_charged,
    )

    test_dl = datamodule.test_dataloader()
    if isinstance(test_dl, list):
        test_dl = test_dl[0]
    return module, test_dl, vocab


def _load_checkpoint(module, checkpoint_path: Path) -> None:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt)
    sd = {k.replace("._orig_mod", ""): v for k, v in sd.items()}
    module.load_state_dict(sd)


def _final_n_atoms(traj) -> int:
    last = traj[-1] if isinstance(traj, list) and traj else traj
    n = getattr(last, "num_nodes", None)
    if n is not None:
        return int(n)
    return int(last.a.shape[0])


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Unconditional trajectory generation for QM9/Semla-style models."
    )
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument(
        "--output-dir", type=Path, default=Path("eval_outputs/uncond"),
        help="Directory to save trajectories.",
    )
    p.add_argument("--n-mols", type=int, default=64)
    p.add_argument("--predict-batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--overrides", nargs="*",
        default=["data=qm9", "model=semla", "cfg=uncond"],
        help="Hydra overrides used to compose config.",
    )
    p.add_argument(
        "--name", default="uncond",
        help="Stem of the output trajectories file: <output-dir>/trajectories/<name>.pt",
    )
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trajectories_dir = args.output_dir / "trajectories"
    trajectories_dir.mkdir(parents=True, exist_ok=True)

    _register_resolvers()
    torch.set_float32_matmul_precision("medium")
    RDLogger.DisableLog("rdApp.*")
    pl.seed_everything(args.seed)

    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.1"):
        cfg = compose(config_name="default", overrides=args.overrides)

    module, test_dl, vocab = _setup(cfg, args.predict_batch_size)
    _load_checkpoint(module, args.checkpoint)
    module.predict_return_traj = True
    module.predict_overrides = None

    n_predict_batches = max(1, math.ceil(args.n_mols / max(1, args.predict_batch_size)))
    trainer_kwargs = dict(cfg.trainer.trainer)
    trainer_kwargs["strategy"] = "auto"
    trainer_kwargs["limit_predict_batches"] = n_predict_batches
    trainer = pl.Trainer(
        logger=False, callbacks=[], enable_checkpointing=False, **trainer_kwargs,
    )

    print(
        f"\nGenerating {args.n_mols} unconditional trajectories "
        f"(batches={n_predict_batches}, batch_size={args.predict_batch_size})..."
    )
    predictions = trainer.predict(module, dataloaders=test_dl)

    valid_trajs: list = []
    invalid_trajs: list = []
    for pred in predictions or []:
        if not isinstance(pred, dict):
            continue
        valid_trajs.extend(pred.get("valid_mols") or [])
        invalid_trajs.extend(pred.get("invalid_mols") or [])

    total = len(valid_trajs) + len(invalid_trajs)
    if total > args.n_mols:
        # Keep the first n_mols across both buckets, biased towards valid first.
        keep_valid = min(len(valid_trajs), args.n_mols)
        valid_trajs = valid_trajs[:keep_valid]
        invalid_trajs = invalid_trajs[: max(0, args.n_mols - keep_valid)]
        total = len(valid_trajs) + len(invalid_trajs)

    out_path = trajectories_dir / f"{args.name}.pt"
    torch.save(
        {
            "guidance_scale": 0.0,
            "target_n_atoms": None,
            "valid_trajectories": valid_trajs,
            "invalid_trajectories": invalid_trajs,
            "valid_n_atoms": [_final_n_atoms(t) for t in valid_trajs],
            "invalid_n_atoms": [_final_n_atoms(t) for t in invalid_trajs],
        },
        out_path,
    )
    print(
        f"Saved {len(valid_trajs)} valid + {len(invalid_trajs)} invalid trajectories "
        f"({total} total) to {out_path}"
    )

    # Dump tokens alongside so chemflow.utils.diagnostics CLI can read this dir.
    (args.output_dir / "atom_tokens.txt").write_text(
        "\n".join(vocab.atom_tokens) + "\n", encoding="utf-8"
    )
    (args.output_dir / "edge_tokens.txt").write_text(
        "\n".join(vocab.edge_tokens) + "\n", encoding="utf-8"
    )
    (args.output_dir / "charge_tokens.txt").write_text(
        "\n".join(vocab.charge_tokens) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
