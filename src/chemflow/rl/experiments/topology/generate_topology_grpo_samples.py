#!/usr/bin/env python3
"""Generate molecules from pretrained vs topology-GRPO checkpoints.

Run from the repository root:

    python -m chemflow.rl.experiments.topology.generate_topology_grpo_samples

Writes ``topology_grpo_samples.pt`` under
``src/chemflow/rl/experiments/topology/<rl_ckpt_stem>/`` by default. Trailing CLI args are
forwarded as Hydra overrides and should match the model config used for
training.
"""

from __future__ import annotations

import argparse
import os
from typing import Any
import sys

_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")
)
for _p in [_PROJECT_ROOT, os.path.join(_PROJECT_ROOT, "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("PROJECT_ROOT", _PROJECT_ROOT)

import torch

from chemflow.rl.experiments.tanimoto.generate_tanimoto_grpo_samples import (
    _PROJECT_ROOT,
    _default_pretrained_ckpt,
    _default_rl_ckpt,
    _gather_predict,
    _vocab_dict,
)
from chemflow.rl.run_grpo import build_module_and_datamodule, compose_cfg


DEFAULT_RL_CKPT = (
    "grpo_topology_seed0_sig0p05_g8_mu2_kl0.02_lr1e-4_"
    "hydra_continue_maxa100-omitposkl_scaff_b10_p0p5_w50_canonsmi_best.pt"
)


def _resolve_rl_ckpt(path_or_name: str) -> str:
    if os.path.isabs(path_or_name) or os.path.dirname(path_or_name):
        candidates = [path_or_name]
    else:
        candidates = [_default_rl_ckpt(path_or_name)]
    if not path_or_name.endswith((".pt", ".ckpt")):
        candidates.append(candidates[0] + ".pt")
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[-1]


def main(default_rl_ckpt_name: str = DEFAULT_RL_CKPT) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--pretrained_ckpt", default=_default_pretrained_ckpt())
    ap.add_argument(
        "--rl_ckpt",
        default=None,
        help=f"RL weights (.pt / .ckpt). Default: .rl_ckpts/{default_rl_ckpt_name}",
    )
    ap.add_argument("--n_mols", type=int, default=300)
    ap.add_argument(
        "--max_atoms",
        type=int,
        default=300,
        help="integrator.max_atoms after load; should match the GRPO run.",
    )
    ap.add_argument("--config_path", default=os.path.join(_PROJECT_ROOT, "configs"))
    ap.add_argument("--config_name", default="default")
    ap.add_argument(
        "--out_dir",
        default=os.path.dirname(os.path.abspath(__file__)),
        help="Base directory; artifact goes to <out_dir>/<stem>/topology_grpo_samples.pt",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Full path to output .pt (overrides default under <out_dir>/<stem>/).",
    )
    ap.add_argument(
        "--no-save-invalid",
        action="store_true",
        help="Omit invalid trajectories from the file.",
    )
    ap.add_argument(
        "overrides",
        nargs="*",
        help="Hydra overrides, e.g. data.datamodule.batch_size.test=128",
    )
    args = ap.parse_args()

    save_invalid = not args.no_save_invalid
    rl_ckpt = _resolve_rl_ckpt(args.rl_ckpt or default_rl_ckpt_name)

    overrides = ["data.datamodule.batch_size.test=128"]
    for override in args.overrides:
        if override not in overrides:
            overrides.append(override)

    ckpt_stem = os.path.splitext(os.path.basename(rl_ckpt))[0]
    run_dir = os.path.join(args.out_dir, ckpt_stem)
    os.makedirs(run_dir, exist_ok=True)
    out_path = args.out or os.path.join(run_dir, "topology_grpo_samples.pt")

    print("pretrained ckpt:", os.path.abspath(args.pretrained_ckpt))
    print("RL ckpt:        ", os.path.abspath(rl_ckpt))
    print("Hydra overrides:", overrides)
    print("output:         ", os.path.abspath(out_path))

    cfg = compose_cfg(args.config_path, args.config_name, overrides=overrides)

    module_base, dm_base = build_module_and_datamodule(cfg)
    print("predict: base (pretrained)...")
    module_base, valid_base, inv_base = _gather_predict(
        module_base,
        dm_base,
        cfg,
        args.pretrained_ckpt,
        args.n_mols,
        args.max_atoms,
    )
    vocab_base = _vocab_dict(module_base)

    module_rl, dm_rl = build_module_and_datamodule(cfg)
    print("predict: RL...")
    module_rl, valid_rl, inv_rl = _gather_predict(
        module_rl,
        dm_rl,
        cfg,
        rl_ckpt,
        args.n_mols,
        args.max_atoms,
    )
    vocab_rl = _vocab_dict(module_rl)
    if vocab_base != vocab_rl:
        print(
            "warning: vocab differs between base and RL modules; "
            "saved file uses RL vocab for `vocab` key."
        )

    payload: dict[str, Any] = {
        "format_version": 1,
        "meta": {
            "pretrained_ckpt": os.path.abspath(args.pretrained_ckpt),
            "rl_ckpt": os.path.abspath(rl_ckpt),
            "n_mols_requested": args.n_mols,
            "integrator_max_atoms": args.max_atoms,
            "hydra_overrides": overrides,
        },
        "vocab": vocab_rl,
        "base": {"valid_mols": valid_base},
        "rl": {"valid_mols": valid_rl},
    }
    if save_invalid:
        payload["base"]["invalid_mols"] = inv_base
        payload["rl"]["invalid_mols"] = inv_rl

    torch.save(payload, out_path)
    print(
        f"saved: n_base valid={len(valid_base)} invalid={len(inv_base)} | "
        f"n_rl valid={len(valid_rl)} invalid={len(inv_rl)}"
    )


if __name__ == "__main__":
    main()
