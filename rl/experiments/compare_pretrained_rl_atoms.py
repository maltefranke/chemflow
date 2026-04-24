#!/usr/bin/env python3
"""Generate molecules from pretrained vs RL checkpoints, plot atom-count histograms, save large trajectories.

Run from repository root::

    python -m rl.experiments.compare_pretrained_rl_atoms

Uses the same Hydra stack and checkpoint loading as ``rl/eval_pretrained_validity.py``.
Trailing CLI args are forwarded as Hydra overrides (must match how the RL run was trained).

Outputs (under ``rl/experiments/`` by default):

* ``atom_hist_pretrained_vs_rl.png`` — grouped bar chart (grey = base trained, blue = RL);
  **only RDKit-valid** molecules enter the histogram (invalid samples are excluded).
* ``rl_valid_trajectories.pt`` — **only** this torch file: full generation trajectories for
  **RDKit-valid RL** samples whose final frame has at least ``--min_atoms`` atoms (default 32).

Atom count convention: we report ``rd.GetNumAtoms()`` on the RDKit mol built from the
final trajectory frame via ``MoleculeData.to_rdkit_mol`` — the *same* quantity used by
``rl/rewards.py::n_atoms_reward``. This includes explicit hydrogens if ``"H"`` is in
the vocab (no ``RemoveHs`` is applied), and equals the graph's ``num_nodes`` for valid
mols since ``to_rdkit_mol`` adds one RDKit atom per node and sanitize preserves them.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Any

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _p in [_PROJECT_ROOT, os.path.join(_PROJECT_ROOT, "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf

from rl.eval_pretrained_validity import (
    build_module_and_datamodule,
    compose_cfg,
    load_ckpt_into_module,
)


def _default_pretrained_ckpt() -> str:
    return os.path.join(_PROJECT_ROOT, ".pretrained_model", "epoch=499-step=48500.ckpt")


def _default_rl_ckpt(default_name: str) -> str:
    return os.path.join(_PROJECT_ROOT, ".rl_ckpts", default_name)


def _final_n_atoms(traj: list[Any], vocab) -> int:
    """RDKit ``GetNumAtoms()`` of the final frame — matches ``n_atoms_reward``."""
    rd = traj[-1].to_rdkit_mol(vocab.atom_tokens, vocab.edge_tokens, vocab.charge_tokens)
    return int(rd.GetNumAtoms()) if rd is not None else 0


def _gather_predict_outputs(module, datamodule, cfg, ckpt_path: str, n_mols: int):
    module = load_ckpt_into_module(module, ckpt_path)
    bs = int(cfg.data.datamodule.batch_size.test)
    cfg.trainer.trainer.limit_predict_batches = int(math.ceil(n_mols / max(bs, 1)))
    # Match `rl/eval_pretrained_validity.py` (Hydra trainer block includes accelerator/strategy).
    trainer = pl.Trainer(
        logger=False,
        callbacks=[],
        enable_checkpointing=False,
        **OmegaConf.to_container(cfg.trainer.trainer, resolve=True),
    )
    preds = trainer.predict(module, dataloaders=datamodule.test_dataloader())
    valid: list = []
    invalid: list = []
    invalid_rdkit: list = []
    for out in preds:
        valid.extend(out["valid_mols"])
        invalid.extend(out["invalid_mols"])
        invalid_rdkit.extend(out["invalid_mols_rdkit"])
    total = len(valid) + len(invalid)
    if total > n_mols:
        keep = n_mols
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
    return {
        "ckpt": ckpt_path,
        "n_requested": n_mols,
        "n_generated": total,
        "n_valid": len(valid),
        "validity": validity,
        "valid_mols": valid,
        "invalid_mols": invalid,
        "invalid_mols_rdkit": invalid_rdkit,
    }


def main(default_rl_ckpt_name: str = "grpo_best.pt"):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pretrained_ckpt", default=_default_pretrained_ckpt())
    ap.add_argument(
        "--rl_ckpt",
        default=None,
        help=f"RL fine-tuned weights (.pt or .ckpt). Default: .rl_ckpts/{default_rl_ckpt_name}",
    )
    ap.add_argument("--n_mols", type=int, default=300)
    ap.add_argument("--config_path", default=os.path.join(_PROJECT_ROOT, "configs"))
    ap.add_argument("--config_name", default="default")
    ap.add_argument(
        "--out_dir",
        default=os.path.join(os.path.dirname(__file__)),
        help="Directory for the figure and rl_valid_trajectories.pt",
    )
    ap.add_argument(
        "--min_atoms",
        type=int,
        default=32,
        help="Save RL trajectories when final n_atoms >= this (valid RL only)",
    )
    ap.add_argument(
        "--trajectories_out",
        default=None,
        help="Path for rl trajectory .pt (default: <out_dir>/rl_valid_trajectories.pt)",
    )
    ap.add_argument("overrides", nargs="*", help="Hydra overrides, e.g. data.n_atoms_strategy=fixed")
    args = ap.parse_args()
    rl_ckpt = args.rl_ckpt or _default_rl_ckpt(default_rl_ckpt_name)
    print("pretrained ckpt:", os.path.abspath(args.pretrained_ckpt))
    print("RL ckpt:        ", os.path.abspath(rl_ckpt))

    os.makedirs(args.out_dir, exist_ok=True)
    cfg = compose_cfg(args.config_path, args.config_name, overrides=list(args.overrides))

    # --- Pretrained ---
    module_pt, dm = build_module_and_datamodule(cfg)
    print("loading pretrained ckpt:", os.path.abspath(args.pretrained_ckpt))
    out_pt = _gather_predict_outputs(module_pt, dm, cfg, args.pretrained_ckpt, args.n_mols)

    # --- RL (fresh module so weights are not mixed) ---
    module_rl, dm_rl = build_module_and_datamodule(cfg)
    print("loading RL ckpt:", os.path.abspath(rl_ckpt))
    out_rl = _gather_predict_outputs(module_rl, dm_rl, cfg, rl_ckpt, args.n_mols)

    # --- Histogram: valid molecules only (same bin edges for both) ---
    valid_n_pt = [_final_n_atoms(t, module_pt.vocab) for t in out_pt["valid_mols"]]
    valid_n_rl = [_final_n_atoms(t, module_rl.vocab) for t in out_rl["valid_mols"]]
    all_vals = valid_n_pt + valid_n_rl
    if not all_vals:
        n_min, n_max = 0, 1
        centers = np.arange(n_min, n_max + 1)
        hist_pt = np.zeros_like(centers, dtype=float)
        hist_rl = np.zeros_like(centers, dtype=float)
    else:
        n_min, n_max = min(all_vals), max(all_vals)
        if n_min == n_max:
            n_max = n_min + 1
        bins = np.arange(n_min, n_max + 2) - 0.5
        hist_pt, _ = np.histogram(valid_n_pt, bins=bins)
        hist_rl, _ = np.histogram(valid_n_rl, bins=bins)
        centers = np.arange(n_min, n_max + 1)
    x = np.arange(len(centers))
    w = 0.36

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w / 2, hist_pt, width=w, color="#9e9e9e", label="base trained", align="center")
    ax.bar(x + w / 2, hist_rl, width=w, color="#1f77b4", label="RL finetuned", align="center")
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(c)) for c in centers])
    ax.set_xlabel("Number of atoms (final frame)")
    ax.set_ylabel("Count (valid only)")
    ax.set_title(
        "Atom counts among RDKit-valid molecules (requested n = %d per model)"
        % args.n_mols
    )
    ax.legend()
    rl_valid_pct = 100.0 * out_rl["validity"]
    ax.text(
        0.98,
        0.98,
        "RL finetuned RDKit validity: %.1f%%" % rl_valid_pct,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=11,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )
    fig.tight_layout()
    fig_path = os.path.join(args.out_dir, "atom_hist_pretrained_vs_rl_alpha01.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print("saved figure:", fig_path)

    traj_records = []
    for traj, n_at in zip(out_rl["valid_mols"], valid_n_rl):
        if n_at >= args.min_atoms:
            traj_records.append({"final_n_atoms": n_at, "trajectory": traj})

    traj_path = args.trajectories_out or os.path.join(args.out_dir, "rl_valid_trajectories_alpha01.pt")
    torch.save(
        {
            "rl_ckpt": rl_ckpt,
            "pretrained_ckpt": args.pretrained_ckpt,
            "n_mols_requested": args.n_mols,
            "min_atoms": args.min_atoms,
            "rl_validity": out_rl["validity"],
            "n_saved": len(traj_records),
            "records": traj_records,
            "hydra_overrides": list(args.overrides),
        },
        traj_path,
    )
    print(
        "saved %d valid RL trajectories (final n_atoms >= %d): %s"
        % (len(traj_records), args.min_atoms, traj_path)
    )


if __name__ == "__main__":
    main(default_rl_ckpt_name="grpo_natoms_seed0_alpha0.1_best.pt")
