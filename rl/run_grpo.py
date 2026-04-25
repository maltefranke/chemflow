#!/usr/bin/env python3
"""Minimal CLI driver for GRPO fine-tuning (Phase 1: no ins/del).

Reuses the setup from `eval_pretrained_validity.py` (same config + ckpt loading)
and plugs the module into `rl.grpo.train`.

Example:
    python -m rl.run_grpo \
        --ckpt .pretrained_model/epoch=499-step=48500.ckpt \
        --n_updates 200 \
        --num_steps 40 \
        data.n_atoms_strategy=fixed

Any trailing args are Hydra overrides, exactly as in `eval_pretrained_validity.py`.
"""

import argparse
import os
import random
import sys
from copy import deepcopy

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in [_PROJECT_ROOT, os.path.join(_PROJECT_ROOT, "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import hydra  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from hydra import compose, initialize_config_dir  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402


def seed_everything(seed: int) -> None:
    """Seed python/numpy/torch (CPU+CUDA) for reproducibility.

    We deliberately *do not* set deterministic cuDNN flags: GRPO rollouts
    already carry big stochastic draws (SDE noise, categorical samples, shuffled
    dataloader), so exact cross-run reproducibility isn't the goal -- the goal
    is to make "seed=0" actually mean the same thing every time.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

OmegaConf.register_new_resolver("oc.eval", eval)
OmegaConf.register_new_resolver("len", lambda x: len(x))
OmegaConf.register_new_resolver("if", lambda cond, t, f: t if cond else f)
OmegaConf.register_new_resolver("eq", lambda x, y: x == y)

from chemflow.dataset.vocab import setup_token_weights  # noqa: E402
from chemflow.utils.metrics import init_metrics  # noqa: E402
from chemflow.utils.utils import init_uniform_prior  # noqa: E402

from rl.grpo import DEFAULT_VAR_FLOOR, GRPOConfig, train  # noqa: E402
from rl.rewards import REWARDS  # noqa: E402


def compose_cfg(config_path: str, config_name: str, overrides: list[str]):
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


def first_test_dataloader(datamodule):
    """`test_dataloader()` returns a list -- grab the first one."""
    dls = datamodule.test_dataloader()
    if isinstance(dls, list):
        if not dls:
            raise RuntimeError("test_dataloader() returned an empty list")
        return dls[0]
    return dls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ckpt",
        default=os.path.join(_PROJECT_ROOT, ".pretrained_model", "epoch=499-step=48500.ckpt"),
    )
    ap.add_argument("--config_path", default=os.path.join(_PROJECT_ROOT, "configs"))
    ap.add_argument("--config_name", default="default")
    ap.add_argument("--n_updates", type=int, default=100)
    ap.add_argument("--num_steps", type=int, default=None,
                    help="Integration steps per rollout; default = module's own setting")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--a_sde", type=float, default=0.1)
    ap.add_argument(
        "--var_floor",
        type=float,
        default=DEFAULT_VAR_FLOOR,
        help="Floor on position Gaussian variance in log-prob (see GRPOConfig.var_floor, DEPARTURES.md).",
    )
    ap.add_argument("--sigma_noise", type=float, default=0.2)
    ap.add_argument("--clip_eps", type=float, default=0.2)
    ap.add_argument("--max_grad_norm", type=float, default=1.0,
                    help="Global-norm gradient clip threshold; pass 0 or negative to disable")
    ap.add_argument("--group_size", type=int, default=1,
                    help="GRPO group size G: number of rollouts per shared prompt. "
                         "G=1 recovers batch-relative advantages (the old default). "
                         "G>1 replicates each prompt G times in-place, so the "
                         "effective unique-prompt count per update is batch_size // G.")
    ap.add_argument("--seed", type=int, default=0,
                    help="Seed for python/numpy/torch RNGs. Does not pin cuDNN.")
    ap.add_argument("--kl_coef", type=float, default=0.0,
                    help="β for reverse-KL to frozen ref (k3 per channel). 0 = disabled "
                         "(no second model copy, no extra forward).")
    ap.add_argument(
        "--per_element_logp_mean",
        action="store_true",
        help="Use mean within each RL channel before summing channels (positions: per "
             "Cartesian coord, i.e. denom 3×surviving atoms). May require a larger "
             "--kl_coef when using ref KL. Default: sum within each channel.",
    )
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--log_every", type=int, default=1)
    ap.add_argument(
        "--reward", default="validity", choices=sorted(REWARDS),
        help="Reward name (see rl/rewards.py::REWARDS).",
    )
    ap.add_argument("--wandb", action="store_true", help="Log metrics to wandb.")
    ap.add_argument("--wandb_project", default="chemflow-grpo")
    ap.add_argument("--wandb_name", default=None)
    ap.add_argument(
        "--wandb_group",
        default=None,
        help="Optional W&B run group (sweep / compare runs in the project UI).",
    )
    ap.add_argument("--save", default=None, help="Optional path to dump final module state_dict")
    ap.add_argument("--save_best", default=None,
                    help="Optional path to save the best (smoothed) reward checkpoint during training")
    ap.add_argument("--best_ema_beta", type=float, default=0.9,
                    help="EMA smoothing coefficient for reward (higher = smoother, ~1/(1-beta) effective window)")
    ap.add_argument("--best_warmup_steps", type=int, default=3,
                    help="Skip best-ckpt updates for the first N steps")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    seed_everything(args.seed)

    cfg = compose_cfg(args.config_path, args.config_name, overrides=list(args.overrides))
    module, datamodule = build_module_and_datamodule(cfg)
    module = load_ckpt_into_module(module, args.ckpt)

    if args.group_size < 1:
        raise ValueError(f"--group_size must be >= 1, got {args.group_size}")

    grpo_cfg = GRPOConfig(
        sigma_noise=args.sigma_noise,
        a_sde=args.a_sde,
        var_floor=args.var_floor,
        clip_eps=args.clip_eps,
        num_integration_steps=args.num_steps,
        max_grad_norm=(args.max_grad_norm if args.max_grad_norm and args.max_grad_norm > 0 else None),
        group_size=args.group_size,
        kl_coef=args.kl_coef,
        per_element_logp_mean=args.per_element_logp_mean,
    )

    dataloader = first_test_dataloader(datamodule)

    if args.wandb:
        import wandb
        w_init = dict(
            project=args.wandb_project, name=args.wandb_name, config=vars(args)
        )
        if args.wandb_group:
            w_init["group"] = args.wandb_group
        wandb.init(**w_init)

    try:
        train(
            module,
            dataloader,
            grpo_cfg,
            n_updates=args.n_updates,
            lr=args.lr,
            device=args.device,
            log_every=args.log_every,
            reward_fn=REWARDS[args.reward],
            best_save_path=args.save_best,
            best_ema_beta=args.best_ema_beta,
            best_warmup_steps=args.best_warmup_steps,
        )
    finally:
        if args.wandb:
            import wandb
            wandb.finish()

    if args.save is not None:
        os.makedirs(os.path.dirname(os.path.abspath(args.save)) or ".", exist_ok=True)
        torch.save({"state_dict": module.state_dict()}, args.save)
        print(f"saved: {args.save}")


if __name__ == "__main__":
    main()
