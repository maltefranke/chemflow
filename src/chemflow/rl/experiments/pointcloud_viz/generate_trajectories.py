#!/usr/bin/env python3
"""Generate N trajectories from a single checkpoint (e.g. a GRPO RL .pt) and save them
in the same on-disk format the natoms notebook reads:

    {"records": [{"final_n_atoms": int, "valid": bool, "trajectory": [MoleculeData, ...]}, ...],
     "ckpt": str, "hydra_overrides": [...], "n_mols_requested": int, "validity": float}

Throwaway script — drop or rename when done.

Example
-------
    python experiments/pointcloud_viz/generate_trajectories.py \
        --ckpt $PROJECT_ROOT/.rl_ckpts/grpo_best.pt \
        --n_mols 200 \
        --out experiments/pointcloud_viz/trajectories.pt \
        data=qm9 model=dit cfg=uncond representation=pointcloud
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Any

_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")
)
for _p in [_PROJECT_ROOT, os.path.join(_PROJECT_ROOT, "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf

from chemflow.rl.run_grpo import (
    Representation,
    build_module_and_datamodule,
    compose_cfg,
    load_ckpt_into_module,
)


def _final_n_atoms(traj: list[Any]) -> int:
    return int(traj[-1].num_nodes)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="Path to checkpoint (.pt or .ckpt)")
    ap.add_argument("--n_mols", type=int, default=200)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "trajectories.pt"))
    ap.add_argument("--max_atoms", type=int, default=100,
                    help="integrator.max_atoms after ckpt load (match the training run)")
    ap.add_argument("--config_path", default=os.path.join(_PROJECT_ROOT, "configs"))
    ap.add_argument("--config_name", default="default")
    ap.add_argument("overrides", nargs="*", help="Hydra overrides (must match training)")
    args = ap.parse_args()

    os.environ.setdefault("PROJECT_ROOT", _PROJECT_ROOT)
    cfg = compose_cfg(args.config_path, args.config_name, overrides=list(args.overrides))
    representation = Representation(cfg.representation)

    module, dm = build_module_and_datamodule(cfg)
    print(f"loading ckpt: {os.path.abspath(args.ckpt)}")
    module = load_ckpt_into_module(module, args.ckpt, representation=representation)
    module.integrator.max_atoms = args.max_atoms

    bs = int(cfg.data.datamodule.batch_size.test)
    cfg.trainer.trainer.limit_predict_batches = int(math.ceil(args.n_mols / max(bs, 1)))
    trainer_kwargs = OmegaConf.to_container(cfg.trainer.trainer, resolve=True)
    # Force single-process: training config sets ddp, which hangs outside srun/torchrun.
    trainer_kwargs["strategy"] = "auto"
    trainer_kwargs["devices"] = 1
    trainer = pl.Trainer(
        logger=False,
        callbacks=[],
        enable_checkpointing=False,
        **trainer_kwargs,
    )

    preds = trainer.predict(module, dataloaders=dm.test_dataloader())
    # For pointcloud reps `valid_mols` is always empty by design — predict_step
    # skips the RDKit validity check when the representation has no topology.
    # Treat valid + invalid as a single bag of generated samples.
    all_trajs: list = []
    for out in preds:
        all_trajs.extend(out.get("valid_mols") or [])
        all_trajs.extend(out.get("invalid_mols") or [])
    if len(all_trajs) > args.n_mols:
        all_trajs = all_trajs[: args.n_mols]

    records = [{"final_n_atoms": _final_n_atoms(t), "trajectory": t} for t in all_trajs]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    torch.save(
        {
            "ckpt": args.ckpt,
            "n_mols_requested": args.n_mols,
            "n_generated": len(records),
            "integrator_max_atoms": args.max_atoms,
            "representation": representation.value,
            "hydra_overrides": list(args.overrides),
            "records": records,
        },
        args.out,
    )
    print(f"saved {len(records)} records -> {args.out}")


if __name__ == "__main__":
    main()
