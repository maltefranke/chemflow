#!/usr/bin/env python3
"""General evaluation of the n-atoms conditional model.

Draws ``target_n_atoms`` for every sample from the empirical training
distribution, generates a large pool of molecules (default 10k), computes the
full training-time metric stack plus PoseBusters, saves the trajectories, and
finally runs :mod:`chemflow.utils.diagnostics` on the saved trajectories to
report self-correction statistics.

Usage:
    python eval_scripts/eval_natoms_general.py \
        --checkpoint path/to/model.ckpt \
        --output-dir eval_outputs/natoms_general \
        --n-mols 10000 \
        --predict-batch-size 128
"""

from __future__ import annotations

import argparse
import json
import math
from copy import deepcopy
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from rdkit import RDLogger

from chemflow.dataset.vocab import setup_token_weights
from chemflow.utils.diagnostics import (
    _mol_to_frame,
    detect_self_correction_events_batch,
    summarize_self_corrections,
)
from chemflow.utils.metrics import (
    calc_atom_stabilities,
    calc_posebusters_metrics,
    init_metrics,
)
from chemflow.utils.utils import init_uniform_prior


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"


# ---------------------------------------------------------------------------
# Config / component setup (mirrors eval_natoms_cfg.py)
# ---------------------------------------------------------------------------


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


def _setup_eval_components(cfg, predict_batch_size: int):
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

    test_dataloaders = datamodule.test_dataloader()
    test_dl = (
        test_dataloaders[0] if isinstance(test_dataloaders, list) else test_dataloaders
    )
    return (
        module,
        test_dl,
        distributions,
        vocab,
        metrics,
        stability_metrics,
        distribution_metrics,
    )


def _load_checkpoint(module, checkpoint_path: Path) -> None:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    clean_state_dict = {k.replace("._orig_mod", ""): v for k, v in state_dict.items()}
    module.load_state_dict(clean_state_dict)


# ---------------------------------------------------------------------------
# Target-n_atoms sampling: draw per-batch from empirical training distribution
# ---------------------------------------------------------------------------


class EmpiricalNAtomsSampler(pl.Callback):
    """Per-batch target n_atoms sampler from an empirical distribution.

    Before each predict batch, samples a ``(batch_size,)`` tensor from
    ``n_atoms_distribution`` and writes it to
    ``module.predict_target_n_atoms_override`` (the signal read by
    ``LightningModuleRates.predict_step``).
    """

    def __init__(self, n_atoms_distribution: torch.Tensor, seed: int | None = None):
        super().__init__()
        probs = n_atoms_distribution.detach().to(dtype=torch.float32).clone()
        probs = probs.clamp_min(0.0)
        total = probs.sum()
        if float(total) <= 0.0:
            raise ValueError("n_atoms_distribution sums to zero; cannot sample.")
        self.probs = probs / total
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(int(seed))
        self.sampled_targets: list[int] = []

    def _batch_size(self, batch) -> int:
        if isinstance(batch, (tuple, list)) and len(batch) >= 1:
            mol_t = batch[0]
        else:
            mol_t = batch
        batch_size = getattr(mol_t, "batch_size", None)
        if batch_size is None:
            batch_attr = getattr(mol_t, "batch", None)
            if batch_attr is None:
                raise RuntimeError("Cannot infer batch_size from predict batch.")
            batch_size = int(batch_attr.max().item()) + 1
        return int(batch_size)

    def on_predict_batch_start(
        self, trainer, pl_module, batch, batch_idx, dataloader_idx=0
    ):
        batch_size = self._batch_size(batch)
        idx = torch.multinomial(
            self.probs,
            num_samples=batch_size,
            replacement=True,
            generator=self.generator,
        )
        targets = idx.to(dtype=torch.long, device=pl_module.device)
        pl_module.predict_target_n_atoms_override = targets
        pl_module.predict_target_mw_override = None
        self.sampled_targets.extend(int(x) for x in idx.tolist())


# ---------------------------------------------------------------------------
# Trajectory / metric helpers
# ---------------------------------------------------------------------------


def _mol_n_atoms(mol) -> int | None:
    if mol is None:
        return None
    n_atoms = getattr(mol, "num_nodes", None)
    if n_atoms is not None:
        return int(n_atoms)
    atom_tokens = getattr(mol, "a", None)
    if atom_tokens is not None:
        return int(atom_tokens.shape[0])
    return None


def _final_mol(traj):
    if isinstance(traj, list) and traj:
        return traj[-1]
    return traj


def _install_ordered_predict_step(module) -> None:
    """Monkey-patch ``module.predict_step`` to preserve generation order.

    The stock ``LightningModuleRates.predict_step`` buckets outputs into
    ``valid_mols`` / ``invalid_mols``, which loses the original batch
    ordering needed to align per-sample target atom counts with per-sample
    realised atom counts. Here we replace it with an equivalent step that
    additionally returns ``gen_mols`` (in generation order) and a boolean
    ``is_valid`` mask.
    """
    from chemflow.utils import rdkit_utils as chemflowRD

    def predict_step(self, batch, batch_idx):
        return_traj = bool(getattr(self, "predict_return_traj", True))
        target_override = getattr(self, "predict_target_n_atoms_override", None)
        target_mw_override = getattr(self, "predict_target_mw_override", None)
        gen_mols = self.sample(
            batch,
            batch_idx,
            return_traj=return_traj,
            target_n_atoms_override=target_override,
            target_mw_override=target_mw_override,
        )

        last_rdkit = [
            traj[-1].to_rdkit_mol(
                self.vocab.atom_tokens,
                self.vocab.edge_tokens,
                self.vocab.charge_tokens,
            )
            for traj in gen_mols
        ]
        is_valid: list[bool] = []
        for mol in last_rdkit:
            if mol is None:
                is_valid.append(False)
                continue
            try:
                is_valid.append(
                    chemflowRD.mol_is_valid(
                        mol, allow_charged=getattr(self, "allow_charged", False)
                    )
                )
            except Exception:
                is_valid.append(False)

        valid_mols = [t for t, v in zip(gen_mols, is_valid) if v]
        invalid_mols = [t for t, v in zip(gen_mols, is_valid) if not v]
        return {
            "gen_mols": gen_mols,
            "is_valid": is_valid,
            "valid_mols": valid_mols,
            "invalid_mols": invalid_mols,
        }

    import types

    module.predict_step = types.MethodType(predict_step, module)


def _collect_from_predict_output(pred) -> tuple[list, list, list, list]:
    """Return (valid_trajs, invalid_trajs, all_trajs_in_gen_order, is_valid_mask)."""
    valid_trajs: list = []
    invalid_trajs: list = []
    all_trajs: list = []
    is_valid_mask: list[bool] = []
    if not isinstance(pred, dict):
        return valid_trajs, invalid_trajs, all_trajs, is_valid_mask

    gen_mols = pred.get("gen_mols")
    mask = pred.get("is_valid")
    if gen_mols is not None and mask is not None:
        for traj, valid in zip(gen_mols, mask):
            all_trajs.append(traj)
            is_valid_mask.append(bool(valid))
            if valid:
                valid_trajs.append(traj)
            else:
                invalid_trajs.append(traj)
        return valid_trajs, invalid_trajs, all_trajs, is_valid_mask

    # Fallback for plain predict_step output; alignment info is lost.
    for traj in pred.get("valid_mols", []) or []:
        valid_trajs.append(traj)
        all_trajs.append(traj)
        is_valid_mask.append(True)
    for traj in pred.get("invalid_mols", []) or []:
        invalid_trajs.append(traj)
        all_trajs.append(traj)
        is_valid_mask.append(False)
    return valid_trajs, invalid_trajs, all_trajs, is_valid_mask


def _trajectories_to_rdkit(trajectories, vocab) -> list:
    rdkit_mols: list = []
    for traj in trajectories:
        final_mol = _final_mol(traj)
        if final_mol is None:
            rdkit_mols.append(None)
            continue
        try:
            rdkit_mols.append(
                final_mol.to_rdkit_mol(
                    vocab.atom_tokens,
                    vocab.edge_tokens,
                    vocab.charge_tokens,
                )
            )
        except Exception:
            rdkit_mols.append(None)
    return rdkit_mols


def _compute_all_metrics(
    rdkit_mols: list,
    metrics,
    stability_metrics,
    distribution_metrics,
) -> dict:
    results: dict[str, float] = {}

    metrics.reset()
    metrics.update(rdkit_mols)
    for k, v in metrics.compute().items():
        results[k] = v.item() if isinstance(v, torch.Tensor) else v

    stabilities = []
    for mol in rdkit_mols:
        if mol is None:
            continue
        try:
            stabilities.append(calc_atom_stabilities(mol))
        except Exception:
            continue
    if stabilities:
        stability_metrics.reset()
        stability_metrics.update(stabilities)
        for k, v in stability_metrics.compute().items():
            results[k] = v.item() if isinstance(v, torch.Tensor) else v
    else:
        for key in ("atom-stability", "molecule-stability"):
            results[key] = float("nan")

    if distribution_metrics is not None:
        distribution_metrics.reset()
        distribution_metrics.update(rdkit_mols)
        for k, v in distribution_metrics.compute().items():
            results[k] = v.item() if isinstance(v, torch.Tensor) else v

    return {
        k: float(v) if isinstance(v, (int, float)) else v for k, v in results.items()
    }


# ---------------------------------------------------------------------------
# Plotting: target (= empirical) vs. realised n_atoms
# ---------------------------------------------------------------------------


def _plot_natoms_distribution(
    sampled_targets: list[int],
    realised_all: list[int],
    realised_valid: list[int],
    train_distribution: torch.Tensor,
    output_path: Path,
    title: str,
) -> None:
    train_vals = train_distribution.detach().cpu().numpy()

    max_x = max(
        int(train_vals.shape[0]),
        max(realised_all) + 1 if realised_all else 0,
        max(sampled_targets) + 1 if sampled_targets else 0,
    )
    bins = np.arange(0, max_x + 2) - 0.5
    x = np.arange(0, max_x + 1)

    def _freq(values: list[int]) -> np.ndarray:
        if not values:
            return np.zeros_like(x, dtype=float)
        hist, _ = np.histogram(values, bins=bins)
        return hist / max(hist.sum(), 1)

    train_freq = np.zeros_like(x, dtype=float)
    train_freq[: train_vals.shape[0]] = train_vals

    target_freq = _freq(sampled_targets)
    all_freq = _freq(realised_all)
    valid_freq = _freq(realised_valid)

    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    ax.bar(
        x, train_freq, color="#8C8C8C", alpha=0.45, width=0.9, label="train (empirical)"
    )
    ax.plot(
        x,
        target_freq,
        color="#2CA02C",
        lw=1.2,
        marker="o",
        ms=3,
        label="sampled targets",
    )
    ax.plot(
        x, all_freq, color="#4C72B0", lw=1.4, marker="o", ms=3, label="generated (all)"
    )
    ax.plot(
        x,
        valid_freq,
        color="#D62728",
        lw=1.4,
        marker="o",
        ms=3,
        label="generated (valid)",
    )

    ax.set_xlabel("n_atoms")
    ax.set_ylabel("frequency")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _natoms_accuracy_stats(
    sampled_targets: list[int],
    realised_counts: list[int | None],
) -> dict:
    """Per-sample target vs realised atom-count stats.

    ``sampled_targets[i]`` is the target for the ``i``-th generated molecule;
    ``realised_counts[i]`` is the realised atom count (``None`` when we could
    not infer it). Only positions where we have both are used.
    """
    pairs = [
        (int(t), int(r))
        for t, r in zip(sampled_targets, realised_counts)
        if r is not None
    ]
    if not pairs:
        return {
            "n": 0,
            "exact_match_rate": 0.0,
            "within_1_rate": 0.0,
            "within_2_rate": 0.0,
            "mean_abs_error": float("nan"),
        }
    t = torch.tensor([p[0] for p in pairs], dtype=torch.float)
    r = torch.tensor([p[1] for p in pairs], dtype=torch.float)
    diff = (r - t).abs()
    return {
        "n": int(len(pairs)),
        "exact_match_rate": (diff == 0).float().mean().item(),
        "within_1_rate": (diff <= 1).float().mean().item(),
        "within_2_rate": (diff <= 2).float().mean().item(),
        "mean_abs_error": diff.mean().item(),
    }


# ---------------------------------------------------------------------------
# Token dump for diagnostics
# ---------------------------------------------------------------------------


def _dump_tokens(vocab, output_dir: Path) -> None:
    (output_dir / "atom_tokens.txt").write_text(
        "\n".join(vocab.atom_tokens) + "\n", encoding="utf-8"
    )
    (output_dir / "edge_tokens.txt").write_text(
        "\n".join(vocab.edge_tokens) + "\n", encoding="utf-8"
    )
    (output_dir / "charge_tokens.txt").write_text(
        "\n".join(vocab.charge_tokens) + "\n", encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Diagnostics driver
# ---------------------------------------------------------------------------


def _process_trajectories_to_frames(
    raw_trajectories: list,
    atom_tokens: list[str],
    edge_tokens: list[str],
    charge_tokens: list[str],
) -> list[list[dict]]:
    processed: list[list[dict]] = []
    for mol_traj in raw_trajectories:
        if not isinstance(mol_traj, (list, tuple)):
            continue
        frames = [
            _mol_to_frame(m, atom_tokens, edge_tokens, charge_tokens) for m in mol_traj
        ]
        processed.append(frames)
    return processed


def _run_diagnostics(
    valid_trajs: list,
    invalid_trajs: list,
    vocab,
    output_dir: Path,
    min_invalid_duration: int = 1,
    max_invalid_duration: int | None = None,
) -> dict:
    print("\nRunning self-correction diagnostics on generated trajectories...")
    atom_tokens = list(vocab.atom_tokens)
    edge_tokens = list(vocab.edge_tokens)
    charge_tokens = list(vocab.charge_tokens)

    valid_frames = _process_trajectories_to_frames(
        valid_trajs, atom_tokens, edge_tokens, charge_tokens
    )
    invalid_frames = _process_trajectories_to_frames(
        invalid_trajs, atom_tokens, edge_tokens, charge_tokens
    )

    valid_diags = detect_self_correction_events_batch(
        valid_frames,
        min_invalid_duration=min_invalid_duration,
        max_invalid_duration=max_invalid_duration,
    )
    invalid_diags = detect_self_correction_events_batch(
        invalid_frames,
        min_invalid_duration=min_invalid_duration,
        max_invalid_duration=max_invalid_duration,
    )

    valid_summary = summarize_self_corrections(valid_diags)
    invalid_summary = summarize_self_corrections(invalid_diags)
    combined_summary = summarize_self_corrections(valid_diags + invalid_diags)

    diagnostics_path = output_dir / "diagnostics.pt"
    torch.save(
        {
            "valid": [d.to_dict() for d in valid_diags],
            "invalid": [d.to_dict() for d in invalid_diags],
            "summary": {
                "valid": valid_summary,
                "invalid": invalid_summary,
                "combined": combined_summary,
            },
        },
        diagnostics_path,
    )
    print(f"Saved diagnostics to: {diagnostics_path}")

    # Also dump the raw trajectories that exhibit self-correction so they can
    # be inspected in the notebooks without re-running detection.
    def _filter(raw: list, diags: list) -> list:
        return [traj for traj, d in zip(raw, diags) if d.has_self_correction]

    sc_path = output_dir / "self_correction_trajectories.pt"
    torch.save(
        {
            "valid_trajectories": _filter(valid_trajs, valid_diags),
            "invalid_trajectories": _filter(invalid_trajs, invalid_diags),
        },
        sc_path,
    )
    print(f"Saved self-correcting trajectories to: {sc_path}")

    def _print(tag: str, s: dict) -> None:
        print(f"\n[{tag}] self-correction summary")
        print(f"  trajectories              : {s['n_trajectories']}")
        print(
            f"  with self-correction      : {s['n_with_self_correction']} "
            f"({s['fraction_with_self_correction']:.1%})"
        )
        print(f"  events (total)            : {s['n_events_total']}")
        print(f"  events / trajectory       : {s['events_per_trajectory']:.3f}")
        print(
            f"  mean / max event duration : {s['mean_event_duration']:.2f} "
            f"/ {s['max_event_duration']}"
        )
        print(f"  edit-kind counts          : {s['edit_kind_counts']}")
        print(f"  onset-reason counts       : {s['onset_reason_counts']}")

    if valid_diags:
        _print("valid", valid_summary)
    if invalid_diags:
        _print("invalid", invalid_summary)
    _print("combined", combined_summary)

    return {
        "valid": valid_summary,
        "invalid": invalid_summary,
        "combined": combined_summary,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "General evaluation for the n-atoms conditional model: sample "
            "10k molecules with targets drawn from the empirical n_atoms "
            "distribution, compute all eval metrics including PoseBusters, "
            "then run trajectory self-correction diagnostics."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to model checkpoint (.ckpt).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("eval_outputs/natoms_general"),
        help="Directory to save evaluation artifacts.",
    )
    parser.add_argument(
        "--n-mols",
        type=int,
        default=10_000,
        help="Approximate total number of molecules to sample (default: 10000).",
    )
    parser.add_argument(
        "--predict-batch-size",
        type=int,
        default=128,
        help="Predict dataloader batch size.",
    )
    parser.add_argument(
        "--natoms-guidance-scale",
        type=float,
        default=None,
        help=(
            "If provided, override the checkpoint's n_atoms CFG guidance "
            "scale. Property / MW CFG scales are always forced to 0."
        ),
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--overrides",
        nargs="*",
        default=[
            "data=qm9",
            "model=semla",
            "cfg=natoms",
            # Match the training-time overrides from
            # submission_scripts/euler/qm9_semla_natoms_extrapolatable.sh so
            # the instantiated backbone + CFG encoder match the checkpoint.
            "model.node_count_embedding._target_=chemflow.model.embedding.ExtrapolatableCountEmbedding",
            "++model.node_count_embedding.min_period=64.0",
            "++model.node_count_embedding.max_period=512.0",
            "++model.node_count_embedding.max_count=32.0",
            "cfg.cfg.natoms_encoder._target_=chemflow.model.embedding.ExtrapolatableCountEmbedding",
            "++cfg.cfg.natoms_encoder.min_period=64.0",
            "++cfg.cfg.natoms_encoder.max_period=512.0",
            "++cfg.cfg.natoms_encoder.max_count=32.0",
        ],
        help="Hydra overrides used to compose config.",
    )
    parser.add_argument(
        "--no-posebusters",
        action="store_true",
        help="Disable PoseBusters (it can be slow on 10k molecules).",
    )
    parser.add_argument(
        "--no-diagnostics",
        action="store_true",
        help="Disable the self-correction diagnostics pass.",
    )
    parser.add_argument(
        "--no-save-trajectories",
        action="store_true",
        help="Disable saving generated molecule trajectories.",
    )
    parser.add_argument(
        "--diagnostics-min-invalid-duration",
        type=int,
        default=1,
        help="Minimum invalid run length to count as a self-correction event.",
    )
    parser.add_argument(
        "--diagnostics-max-invalid-duration",
        type=int,
        default=None,
        help="Maximum invalid run length to count (default: no cap).",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    _register_resolvers()
    torch.set_float32_matmul_precision("medium")
    RDLogger.DisableLog("rdApp.*")
    pl.seed_everything(args.seed)

    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.1"):
        cfg = compose(config_name="default", overrides=args.overrides)

    (
        module,
        test_dl,
        distributions,
        vocab,
        metrics,
        stability_metrics,
        distribution_metrics,
    ) = _setup_eval_components(cfg, args.predict_batch_size)
    _load_checkpoint(module, args.checkpoint)
    _install_ordered_predict_step(module)

    adapter = module.cfg_adapter
    if not adapter._has_natoms_cfg:
        raise RuntimeError("Loaded model does not expose n-atoms CFG conditioning.")

    # Force property / MW CFG off; only the n_atoms branch is exercised here.
    original_property_scale = adapter.cfg_guidance_scale
    original_mw_scale = adapter.mw_cfg_guidance_scale
    original_natoms_scale = adapter.natoms_cfg_guidance_scale
    adapter.cfg_guidance_scale = 0.0
    adapter.mw_cfg_guidance_scale = 0.0
    if args.natoms_guidance_scale is not None:
        adapter.natoms_cfg_guidance_scale = float(args.natoms_guidance_scale)

    # Save tokens so the diagnostics CLI can be re-run on the saved trajectories.
    _dump_tokens(vocab, args.output_dir)

    # One batch per ceil(n_mols / batch_size). May need to loop through the
    # dataloader multiple times if the dataset is smaller than n_mols.
    n_predict_batches = max(1, math.ceil(args.n_mols / max(1, args.predict_batch_size)))

    trainer_kwargs = dict(cfg.trainer.trainer)
    trainer_kwargs["strategy"] = "auto"
    trainer_kwargs["limit_predict_batches"] = n_predict_batches

    sampler = EmpiricalNAtomsSampler(
        n_atoms_distribution=distributions.n_atoms_distribution,
        seed=args.seed,
    )
    trainer = pl.Trainer(
        logger=False,
        callbacks=[sampler],
        enable_checkpointing=False,
        **trainer_kwargs,
    )

    module.predict_return_traj = True

    all_trajs: list = []
    is_valid_mask: list[bool] = []

    print(
        f"\nSampling up to {args.n_mols} molecules "
        f"(batches={n_predict_batches}, batch_size={args.predict_batch_size}) "
        f"with target n_atoms drawn from the empirical training distribution..."
    )
    try:
        predictions = trainer.predict(module, dataloaders=test_dl)
    finally:
        adapter.cfg_guidance_scale = original_property_scale
        adapter.mw_cfg_guidance_scale = original_mw_scale
        adapter.natoms_cfg_guidance_scale = original_natoms_scale
        module.predict_target_n_atoms_override = None
        module.predict_target_mw_override = None

    for pred in predictions or []:
        _, _, trajs_in_order, mask_in_order = _collect_from_predict_output(pred)
        all_trajs.extend(trajs_in_order)
        is_valid_mask.extend(mask_in_order)

    # Cap at n_mols (in generation order to keep alignment with sampled targets).
    if len(all_trajs) > args.n_mols:
        all_trajs = all_trajs[: args.n_mols]
        is_valid_mask = is_valid_mask[: args.n_mols]

    # Trim targets to match trajectories: ``sampler.sampled_targets`` is
    # recorded for every sample the dataloader yielded, which may be >= the
    # number actually captured here (e.g. if we truncate to n_mols).
    sampled_targets = sampler.sampled_targets[: len(all_trajs)]

    n_total = len(all_trajs)
    n_valid = sum(1 for v in is_valid_mask if v)

    valid_trajs = [t for t, v in zip(all_trajs, is_valid_mask) if v]
    invalid_trajs = [t for t, v in zip(all_trajs, is_valid_mask) if not v]
    valid_targets = [tg for tg, v in zip(sampled_targets, is_valid_mask) if v]

    valid_counts_raw = [_mol_n_atoms(_final_mol(t)) for t in valid_trajs]
    valid_counts = [c for c in valid_counts_raw if c is not None]
    all_counts_raw = [_mol_n_atoms(_final_mol(t)) for t in all_trajs]
    all_counts = [c for c in all_counts_raw if c is not None]

    # ---------------------------------------------------------------------
    # Metric computation
    # ---------------------------------------------------------------------
    print(
        f"\nGenerated {n_total} molecules ({n_valid} valid, "
        f"{n_total - n_valid} invalid). Computing metrics..."
    )
    rdkit_mols = _trajectories_to_rdkit(all_trajs, vocab)
    training_metrics = _compute_all_metrics(
        rdkit_mols=rdkit_mols,
        metrics=metrics,
        stability_metrics=stability_metrics,
        distribution_metrics=distribution_metrics,
    )

    posebusters_metrics: dict = {}
    if not args.no_posebusters:
        try:
            print("Running PoseBusters checks...")
            posebusters_metrics = calc_posebusters_metrics(rdkit_mols)
        except Exception as e:
            print(f"PoseBusters failed: {e}")
            posebusters_metrics = {}

    # n_atoms_counts are in generation order and aligned with sampled_targets.
    natoms_stats_all = _natoms_accuracy_stats(sampled_targets, all_counts_raw)
    natoms_stats_valid = _natoms_accuracy_stats(valid_targets, valid_counts_raw)

    summary = {
        "n_total": n_total,
        "n_valid": n_valid,
        "n_invalid": n_total - n_valid,
        "validity_rate": (n_valid / n_total) if n_total else 0.0,
        "n_atoms": {
            "mean_all": float(np.mean(all_counts)) if all_counts else float("nan"),
            "mean_valid": float(np.mean(valid_counts))
            if valid_counts
            else float("nan"),
            "target_accuracy_all": natoms_stats_all,
            "target_accuracy_valid": natoms_stats_valid,
        },
        "training_metrics": training_metrics,
        "posebusters_metrics": posebusters_metrics,
    }

    # ---------------------------------------------------------------------
    # Persist results
    # ---------------------------------------------------------------------
    json_path = args.output_dir / "results.json"
    pt_path = args.output_dir / "results.pt"
    plot_path = args.output_dir / "natoms_distribution.png"

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    torch.save(
        {
            **summary,
            "sampled_targets": sampled_targets,
            "all_n_atoms": all_counts,
            "valid_n_atoms": valid_counts,
            "n_atoms_distribution": distributions.n_atoms_distribution.detach().cpu(),
        },
        pt_path,
    )

    _plot_natoms_distribution(
        sampled_targets=sampled_targets,
        realised_all=all_counts,
        realised_valid=valid_counts,
        train_distribution=distributions.n_atoms_distribution,
        output_path=plot_path,
        title=(
            f"n_atoms distributions (N={n_total}, "
            f"validity={summary['validity_rate']:.1%})"
        ),
    )

    trajectories_path: Path | None = None
    if not args.no_save_trajectories:
        trajectories_path = args.output_dir / "trajectories.pt"
        torch.save(
            {
                "valid_trajectories": valid_trajs,
                "invalid_trajectories": invalid_trajs,
                "valid_n_atoms": valid_counts,
                "invalid_n_atoms": [_mol_n_atoms(_final_mol(t)) for t in invalid_trajs],
                "sampled_targets": sampled_targets,
            },
            trajectories_path,
        )
        print(f"Saved trajectories to:           {trajectories_path}")

    print(f"Saved JSON summary to:           {json_path}")
    print(f"Saved tensor results to:         {pt_path}")
    print(f"Saved n_atoms plot to:           {plot_path}")

    # ---------------------------------------------------------------------
    # Report headline numbers
    # ---------------------------------------------------------------------
    tm = training_metrics
    print("\n" + "=" * 60)
    print("General evaluation summary")
    print("=" * 60)
    print(
        f"  n_total / n_valid        : {n_total} / {n_valid} "
        f"(validity={summary['validity_rate']:.1%})"
    )
    print(f"  novelty                  : {tm.get('novelty', float('nan')):.1%}")
    print(f"  uniqueness               : {tm.get('uniqueness', float('nan')):.1%}")
    print(f"  atom-stability           : {tm.get('atom-stability', float('nan')):.1%}")
    print(
        f"  molecule-stability       : {tm.get('molecule-stability', float('nan')):.1%}"
    )
    print(f"  energy-validity          : {tm.get('energy-validity', float('nan')):.1%}")
    print(
        f"  mean n_atoms (all/valid) : "
        f"{summary['n_atoms']['mean_all']:.2f} / {summary['n_atoms']['mean_valid']:.2f}"
    )
    print(
        "  n_atoms target accuracy  : "
        f"exact={natoms_stats_all['exact_match_rate']:.1%}, "
        f"±1={natoms_stats_all['within_1_rate']:.1%}, "
        f"±2={natoms_stats_all['within_2_rate']:.1%}"
    )
    if posebusters_metrics:
        print("  posebusters (mean pass rates):")
        for k, v in sorted(posebusters_metrics.items()):
            print(f"    - {k}: {v:.3f}")

    # ---------------------------------------------------------------------
    # Self-correction diagnostics on the trajectories
    # ---------------------------------------------------------------------
    if not args.no_diagnostics:
        diag_summary = _run_diagnostics(
            valid_trajs=valid_trajs,
            invalid_trajs=invalid_trajs,
            vocab=vocab,
            output_dir=args.output_dir,
            min_invalid_duration=args.diagnostics_min_invalid_duration,
            max_invalid_duration=args.diagnostics_max_invalid_duration,
        )
        # Add diagnostics summary into the JSON for downstream tooling.
        summary["self_correction_summary"] = diag_summary
        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)

    print("\nDone.")


if __name__ == "__main__":
    main()
