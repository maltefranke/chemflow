#!/usr/bin/env python3
"""Hydra driver for GRPO fine-tuning.

Owns the shared RL setup helpers (Hydra config composition, module/datamodule
construction, and checkpoint loading) and plugs the module into `rl.grpo.train`.

Example:
    python -m chemflow.rl.run_grpo \
        'rl.ckpt=".pretrained_model/epoch=499-step=48500.ckpt"' \
        rl.n_updates=200 \
        rl.grpo.num_integration_steps=40 \
        data.n_atoms_strategy=fixed

Any trailing args are Hydra overrides.
"""

import os
import random
import sys
from copy import deepcopy

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
for _p in [_PROJECT_ROOT, os.path.join(_PROJECT_ROOT, "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault("PROJECT_ROOT", _PROJECT_ROOT)

import hydra  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from hydra import compose, initialize_config_dir  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402


def seed_everything(seed: int) -> None:
    """Seed python/numpy/torch (CPU+CUDA) for reproducibility.

    We deliberately *do not* set deterministic cuDNN flags: GRPO rollouts
    already carry big stochastic draws (exploration noise, categorical samples, shuffled
    dataloader), so exact cross-run reproducibility isn't the goal -- the goal
    is to make "seed=0" actually mean the same thing every time.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def register_resolvers() -> None:
    """Register custom resolvers used by ChemFlow configs."""
    resolvers = {
        "oc.eval": eval,
        "len": lambda x: len(x),
        "if": lambda cond, t, f: t if cond else f,
        "eq": lambda x, y: x == y,
    }
    for name, fn in resolvers.items():
        if not OmegaConf.has_resolver(name):
            OmegaConf.register_new_resolver(name, fn)


register_resolvers()

from chemflow.dataset.representation import (  # noqa: E402
    Representation,
    project_distributions_to_representation,
    validate_representation,
)
from chemflow.dataset.vocab import setup_token_weights  # noqa: E402
from chemflow.utils.metrics import init_metrics  # noqa: E402
from chemflow.utils.utils import bootstrap_run_id, init_uniform_prior  # noqa: E402

from chemflow.rl.grpo import GRPOConfig, _sync_model_ema_to_live, train  # noqa: E402
from chemflow.rl.rewards import build_reward as build_reward_from_spec  # noqa: E402


def compose_cfg(config_path: str, config_name: str, overrides: list[str]):
    cfg_dir = os.path.abspath(os.path.expanduser(config_path))
    with initialize_config_dir(config_dir=cfg_dir, version_base="1.1"):
        cfg = compose(config_name=config_name, overrides=overrides)
    OmegaConf.set_struct(cfg, False)
    return cfg


def build_module_and_datamodule(cfg):
    representation = Representation(cfg.representation)
    dataset_cls = hydra.utils.get_class(cfg.data.datamodule.datasets.train._target_)
    validate_representation(dataset_cls.CAPABILITIES, representation)

    preprocessing = hydra.utils.instantiate(cfg.data.preprocessing)
    vocab = preprocessing.vocab
    distributions = preprocessing.distributions
    loss_weight_distributions = deepcopy(distributions)
    token_prior = init_uniform_prior(distributions)
    token_prior = project_distributions_to_representation(
        token_prior, vocab, representation
    )
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
    base_dataset = datamodule.train_dataset.base_dataset
    train_smiles = (
        base_dataset.get_all_smiles()
        if hasattr(base_dataset, "get_all_smiles")
        else None
    )
    allow_charged = bool(cfg.data.get("allow_charged", False))
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
        batch_metrics=batch_metrics,
        allow_charged=allow_charged,
        representation=representation.value,
    )
    return module, datamodule


def load_ckpt_into_module(module, ckpt_path: str, *, representation: Representation):
    """Load weights into ``module`` after enforcing checkpoint↔RL rep equality.

    RL must run under the same representation the checkpoint was trained on:
    going up (e.g. POINTCLOUD-trained → MOLECULE RL) leaves edge / charge
    heads at near-init values and silently produces nonsense; going down
    (MOLECULE-trained → POINTCLOUD RL) rots the pretrained heads we'd be
    discarding. Checkpoints with no ``representation`` field error out — no
    silent fallback.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hp = ckpt.get("hyper_parameters", {}) or {}
    ckpt_rep_raw = hp.get("representation")
    if ckpt_rep_raw is None:
        raise ValueError(
            f"Checkpoint {ckpt_path!r} has no 'representation' in "
            "hyper_parameters. Re-save the checkpoint with the field set "
            "before using it for RL."
        )
    ckpt_rep = Representation(ckpt_rep_raw)

    if ckpt_rep != representation:
        raise ValueError(
            f"Checkpoint was trained with representation={ckpt_rep.value!r} "
            f"but RL is running with representation={representation.value!r}. "
            "Representations must match exactly — re-pretrain or change the RL config."
        )
    state_dict = ckpt["state_dict"]
    state_dict = {k.replace("._orig_mod", ""): v for k, v in state_dict.items()}
    module.load_state_dict(state_dict, strict=True)

    # GRPO trains the *live* model. Lightning's `_get_model()` returns
    # `model_ema` whenever `use_ema_for_eval=True` (the default), which puts
    # every gradient step on `model_ema` while `self.model` stays at init.
    # On pretrained ckpts the good weights live in `model_ema`; copy them into
    # `model`. Then sync `model_ema <- model` so any future reload finds them
    # consistent.
    ema_was_source = getattr(module, "use_ema_for_eval", False)
    module.use_ema_for_eval = False

    if getattr(module, "model_ema", None) is not None:
        if ema_was_source:
            module.model.load_state_dict(module.model_ema.state_dict())
            print("[grpo] RL: copied model_ema -> model (pretrained init).", flush=True)
        module.model_ema.load_state_dict(module.model.state_dict())
        module.model_ema.eval()
        for p in module.model_ema.parameters():
            p.requires_grad_(False)
    return module


def first_test_dataloader(datamodule):
    """`test_dataloader()` returns a list -- grab the first one."""
    dls = datamodule.test_dataloader()
    if isinstance(dls, list):
        if not dls:
            raise RuntimeError("test_dataloader() returned an empty list")
        return dls[0]
    return dls


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _project_root() -> str:
    return os.path.abspath(os.environ.get("PROJECT_ROOT", _PROJECT_ROOT))


def _resolve_path(path: str | None) -> str | None:
    if path is None:
        return None
    path = os.path.expanduser(str(path))
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(_project_root(), path))


def _hydra_output_dir() -> str | None:
    try:
        from hydra.core.hydra_config import HydraConfig

        if HydraConfig.initialized():
            return str(HydraConfig.get().runtime.output_dir)
    except Exception:
        return None
    return None


def build_grpo_config(grpo_cfg: DictConfig) -> GRPOConfig:
    kwargs = dict(OmegaConf.to_container(grpo_cfg, resolve=True))
    max_grad_norm = kwargs.get("max_grad_norm")
    if max_grad_norm is not None and float(max_grad_norm) <= 0:
        kwargs["max_grad_norm"] = None
    return GRPOConfig(**kwargs)


def build_reward(
    reward_cfg: DictConfig,
    *,
    representation: Representation,
    extra_kwargs: dict[str, dict] | None = None,
):
    """Declarative wrapper composition.

    ``reward.wrappers`` is a name-keyed mapping; each entry carries an
    ``enabled`` flag plus the wrapper-specific kwargs (passed through verbatim
    to ``WrapperSpec.make``). Order of application matches insertion order in
    the config — validity gates should come before diversity gates.

    ``extra_kwargs`` is a per-wrapper dict of runtime objects (e.g. the train
    dataset, a tokenizer) that aren't expressible in Hydra config. They're
    merged on top of the config-supplied kwargs before instantiation.
    """
    name = str(reward_cfg.name)
    wrappers_cfg = reward_cfg.get("wrappers", {}) or {}
    extra_kwargs = extra_kwargs or {}

    wrappers: list[tuple[str, dict]] = []
    for w_name, w_cfg in wrappers_cfg.items():
        w_cfg_dict = OmegaConf.to_container(w_cfg, resolve=True) or {}
        if not bool(w_cfg_dict.pop("enabled", False)):
            continue
        w_cfg_dict.update(extra_kwargs.get(str(w_name), {}))
        wrappers.append((str(w_name), w_cfg_dict))

    return build_reward_from_spec(name, representation=representation, wrappers=wrappers)


def _init_wandb(cfg: DictConfig, config_payload: dict):
    if not bool(cfg.rl.wandb.enabled):
        return None
    import wandb

    w_init = {
        "project": str(cfg.rl.wandb.project),
        "name": None if cfg.rl.wandb.name is None else str(cfg.rl.wandb.name),
        "config": config_payload,
    }
    if cfg.rl.wandb.group:
        w_init["group"] = str(cfg.rl.wandb.group)
    return wandb.init(**w_init)


def run_from_cfg(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)

    seed_everything(int(cfg.rl.seed))

    ckpt_path = _resolve_path(cfg.rl.ckpt)
    save_path = _resolve_path(cfg.rl.save)
    save_best_path = _resolve_path(cfg.rl.save_best)
    wandb_config = OmegaConf.to_container(cfg, resolve=False)

    print(
        "[grpo] runtime "
        f"cwd={os.getcwd()} project_root={_project_root()} "
        f"hydra_output_dir={_hydra_output_dir() or '<none>'}",
        flush=True,
    )
    print(
        f"[grpo] paths ckpt={ckpt_path} save={save_path} save_best={save_best_path}",
        flush=True,
    )

    print(
        "[grpo] building datamodule + metrics (slow on cold cache / NFS) …",
        flush=True,
    )
    module, datamodule = build_module_and_datamodule(cfg)
    representation = Representation(cfg.representation)
    print(f"[grpo] loading checkpoint: {ckpt_path}", flush=True)
    module = load_ckpt_into_module(module, ckpt_path, representation=representation)
    module.integrator.max_atoms = int(cfg.rl.max_atoms)

    if int(cfg.rl.grpo.group_size) < 1:
        raise ValueError(f"rl.grpo.group_size must be >= 1, got {cfg.rl.grpo.group_size}")
    if int(cfg.rl.grpo.update_passes) < 1:
        raise ValueError(
            f"rl.grpo.update_passes must be >= 1, got {cfg.rl.grpo.update_passes}"
        )

    grpo_cfg = build_grpo_config(cfg.rl.grpo)

    dataloader = first_test_dataloader(datamodule)
    reward_fn = build_reward(cfg.rl.reward, representation=representation)
    device = _resolve_device(str(cfg.rl.device))

    if bool(cfg.rl.wandb.enabled):
        print("[grpo] initializing wandb …", flush=True)
    wandb_run = _init_wandb(cfg, wandb_config)
    print("[grpo] starting updates (first log line is after step 0 completes)", flush=True)
    try:
        train(
            module,
            dataloader,
            grpo_cfg,
            n_updates=int(cfg.rl.n_updates),
            lr=float(cfg.rl.lr),
            device=device,
            log_every=int(cfg.rl.log_every),
            reward_fn=reward_fn,
            best_save_path=save_best_path,
            best_ema_beta=float(cfg.rl.best.ema_beta),
            best_warmup_steps=int(cfg.rl.best.warmup_steps),
            diagnostics_every=int(cfg.rl.get("diagnostics_every", 0)),
        )
    finally:
        if wandb_run is not None:
            wandb_run.finish()

    if save_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)) or ".", exist_ok=True)
        _sync_model_ema_to_live(module)
        torch.save(
            {
                "state_dict": module.state_dict(),
                "hyper_parameters": {
                    "representation": representation.value,
                    "rl_checkpoint": True,
                    "use_ema_for_eval": False,
                },
            },
            save_path,
        )
        print(f"saved: {save_path}")


@hydra.main(config_path="../../../configs", config_name="rl/grpo", version_base="1.1")
def main(cfg: DictConfig) -> None:
    run_from_cfg(cfg)


if __name__ == "__main__":
    os.environ.setdefault("PROJECT_ROOT", _PROJECT_ROOT)
    bootstrap_run_id()
    main()
