#!/usr/bin/env python3
"""Standalone evaluation for n-atoms classifier-free guidance on QM9.

Loads a trained checkpoint, sweeps target atom counts and guidance scales,
reports how closely generated molecules match the target (and their validity),
and saves the generated molecule trajectories for later inspection.
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
from matplotlib.animation import FuncAnimation, PillowWriter
from omegaconf import OmegaConf
from rdkit import RDLogger

from chemflow.dataset.vocab import setup_token_weights
from chemflow.utils.metrics import calc_atom_stabilities, init_metrics
from chemflow.utils.utils import init_uniform_prior


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"

X_AXIS_PADDING = 5


def _register_resolvers() -> None:
    """Register OmegaConf resolvers used by the project configs."""
    resolvers = {
        "oc.eval": eval,
        "len": lambda x: len(x),
        "if": lambda cond, t, f: t if cond else f,
        "eq": lambda x, y: x == y,
    }
    for name, fn in resolvers.items():
        if not OmegaConf.has_resolver(name):
            OmegaConf.register_new_resolver(name, fn)


def _parse_int_list(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def _parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def _mol_n_atoms(mol) -> int | None:
    """Pull the atom count off a single MoleculeData / batched object."""
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
    """predict_step returns trajectories (list[MoleculeData]) per generated mol."""
    if isinstance(traj, list) and traj:
        return traj[-1]
    return traj


def _natoms_trajectories(trajectories) -> list[list[int]]:
    """For every generation, return per-integration-step atom count.

    Returns a ``list[list[int]]`` of length ``len(trajectories)``. ``-1`` is
    used as a sentinel for frames whose atom count cannot be inferred (kept
    defensive; should not happen in practice).
    """
    out: list[list[int]] = []
    for traj in trajectories:
        if not isinstance(traj, (list, tuple)):
            n = _mol_n_atoms(traj)
            out.append([n if n is not None else -1])
            continue
        per_step: list[int] = []
        for frame in traj:
            n = _mol_n_atoms(frame)
            per_step.append(n if n is not None else -1)
        out.append(per_step)
    return out


def _save_natoms_trajectories(
    valid_trajs: list,
    invalid_trajs: list,
    target: int,
    scale: float,
    output_path: Path,
) -> None:
    """Save per-step atom-count trajectories for a single (scale, target) cell.

    Saved payload:

        {
            "guidance_scale": float,
            "target_n_atoms": int,
            "valid":  Tensor[n_valid, n_steps]   (or ragged list[list[int]]),
            "invalid": Tensor[n_invalid, n_steps] (or ragged list[list[int]]),
            "is_valid_final": Tensor[n_valid + n_invalid] bool, valid first,
            "shape_note": str,
        }
    """
    valid_n = _natoms_trajectories(valid_trajs)
    invalid_n = _natoms_trajectories(invalid_trajs)

    def _stack_or_ragged(seqs: list[list[int]]):
        if not seqs:
            return torch.empty((0, 0), dtype=torch.long)
        lengths = {len(s) for s in seqs}
        if len(lengths) == 1:
            return torch.tensor(seqs, dtype=torch.long)
        return seqs

    valid_payload = _stack_or_ragged(valid_n)
    invalid_payload = _stack_or_ragged(invalid_n)
    is_valid_final = torch.tensor(
        [True] * len(valid_n) + [False] * len(invalid_n),
        dtype=torch.bool,
    )
    is_tensor = isinstance(valid_payload, torch.Tensor) and isinstance(
        invalid_payload, torch.Tensor
    )

    torch.save(
        {
            "guidance_scale": float(scale),
            "target_n_atoms": int(target),
            "valid": valid_payload,
            "invalid": invalid_payload,
            "is_valid_final": is_valid_final,
            "shape_note": (
                "rows = generations, cols = integration steps"
                if is_tensor
                else "ragged; one inner list per generation"
            ),
        },
        output_path,
    )


def _collect_from_predict_output(
    pred,
) -> tuple[list[int], list[int], list, list]:
    """Extract (valid_n_atoms, invalid_n_atoms, valid_trajs, invalid_trajs)."""
    valid_counts: list[int] = []
    invalid_counts: list[int] = []
    valid_trajs: list = []
    invalid_trajs: list = []

    if pred is None or not isinstance(pred, dict):
        return valid_counts, invalid_counts, valid_trajs, invalid_trajs

    for traj in pred.get("valid_mols", []) or []:
        n = _mol_n_atoms(_final_mol(traj))
        if n is not None:
            valid_counts.append(n)
            valid_trajs.append(traj)

    for traj in pred.get("invalid_mols", []) or []:
        n = _mol_n_atoms(_final_mol(traj))
        if n is not None:
            invalid_counts.append(n)
            invalid_trajs.append(traj)

    return valid_counts, invalid_counts, valid_trajs, invalid_trajs


def _natoms_stats(counts: list[int], target: int) -> dict:
    if not counts:
        return {
            "mean_n_atoms": float("nan"),
            "std_n_atoms": float("nan"),
            "median_n_atoms": float("nan"),
            "exact_match_rate": 0.0,
            "within_1_rate": 0.0,
            "within_2_rate": 0.0,
            "n_molecules": 0,
        }
    t = torch.tensor(counts, dtype=torch.float)
    return {
        "mean_n_atoms": t.mean().item(),
        "std_n_atoms": t.std().item() if t.numel() > 1 else 0.0,
        "median_n_atoms": t.median().item(),
        "exact_match_rate": (t == target).float().mean().item(),
        "within_1_rate": ((t - target).abs() <= 1).float().mean().item(),
        "within_2_rate": ((t - target).abs() <= 2).float().mean().item(),
        "n_molecules": int(t.numel()),
    }


def _summarize(
    predictions,
    target_n_atoms: int,
    max_samples: int,
) -> tuple[dict, list, list]:
    valid_counts: list[int] = []
    invalid_counts: list[int] = []
    valid_trajs: list = []
    invalid_trajs: list = []

    for pred in predictions:
        v_c, i_c, v_t, i_t = _collect_from_predict_output(pred)
        valid_counts.extend(v_c)
        invalid_counts.extend(i_c)
        valid_trajs.extend(v_t)
        invalid_trajs.extend(i_t)

    # Keep the ordering consistent between counts and trajectories when we cap.
    total = len(valid_counts) + len(invalid_counts)
    if total > max_samples:
        # Truncate while preserving interleaving as best we can: we just cap
        # the valid list first, then the invalid list.
        keep_valid = min(len(valid_counts), max_samples)
        valid_counts = valid_counts[:keep_valid]
        valid_trajs = valid_trajs[:keep_valid]
        remaining = max_samples - keep_valid
        invalid_counts = invalid_counts[:remaining]
        invalid_trajs = invalid_trajs[:remaining]

    all_counts = valid_counts + invalid_counts
    total = len(all_counts)
    n_valid = len(valid_counts)

    summary = {
        "target_n_atoms": int(target_n_atoms),
        "n_total": total,
        "n_valid": n_valid,
        "n_invalid": total - n_valid,
        "validity_rate": (n_valid / total) if total else 0.0,
        "all_stats": _natoms_stats(all_counts, target_n_atoms),
        "valid_stats": _natoms_stats(valid_counts, target_n_atoms),
        "all_n_atoms": all_counts,
        "valid_n_atoms": valid_counts,
        "invalid_n_atoms": invalid_counts,
    }
    return summary, valid_trajs, invalid_trajs


def _compute_x_bounds(
    all_values: list[int],
    train_dist: torch.Tensor,
    padding: int,
) -> tuple[int, int]:
    """Bounds span the union of observed generated counts and train support."""
    train_nonzero = (train_dist > 0).nonzero(as_tuple=True)[0]
    candidates_min: list[int] = []
    candidates_max: list[int] = []
    if all_values:
        candidates_min.append(min(all_values))
        candidates_max.append(max(all_values))
    if train_nonzero.numel() > 0:
        candidates_min.append(int(train_nonzero.min().item()))
        candidates_max.append(int(train_nonzero.max().item()))
    if not candidates_min:
        return 0, int(train_dist.numel())
    lo = max(0, min(candidates_min) - padding)
    hi = max(candidates_max) + padding
    if hi <= lo:
        hi = lo + 1
    return lo, hi


def _plot_natoms_grid(
    all_results: dict[tuple[float, int], dict],
    train_distribution: torch.Tensor,
    output_path: Path,
    value_key: str,
    title: str,
) -> None:
    """Plot a grid (rows=targets, cols=scales) of generated vs train n_atoms.

    Uses ``value_key`` on each cell's result dict to choose the generated counts
    (e.g. ``"all_n_atoms"`` or ``"valid_n_atoms"``).
    """
    if not all_results:
        return

    guidance_scales = sorted({key[0] for key in all_results})
    target_n_atoms = sorted({key[1] for key in all_results})

    # Shared x-bounds across every cell so visual comparison is meaningful.
    flat_counts: list[int] = []
    for result in all_results.values():
        flat_counts.extend(result.get(value_key, []))
    x_lo, x_hi = _compute_x_bounds(flat_counts, train_distribution, X_AXIS_PADDING)

    bins = np.arange(x_lo, x_hi + 2) - 0.5
    x_support = np.arange(x_lo, x_hi + 1)
    train_vals = train_distribution.detach().cpu().numpy()
    train_max_idx = train_vals.shape[0]
    train_bar_heights = np.array(
        [train_vals[i] if 0 <= i < train_max_idx else 0.0 for i in x_support],
        dtype=float,
    )

    n_rows = len(target_n_atoms)
    n_cols = len(guidance_scales)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.4 * max(1, n_cols), 2.8 * max(1, n_rows)),
        squeeze=False,
    )

    for row_idx, target in enumerate(target_n_atoms):
        for col_idx, scale in enumerate(guidance_scales):
            ax = axes[row_idx][col_idx]
            result = all_results.get((scale, target), {})
            values = result.get(value_key, [])

            if values:
                counts, _ = np.histogram(values, bins=bins)
                freqs = counts / max(counts.sum(), 1)
            else:
                freqs = np.zeros_like(x_support, dtype=float)

            ax.bar(
                x_support,
                train_bar_heights,
                color="#8C8C8C",
                alpha=0.45,
                edgecolor="none",
                width=0.9,
                label="train",
            )
            ax.bar(
                x_support,
                freqs,
                color="#4C72B0",
                alpha=0.8,
                edgecolor="black",
                linewidth=0.4,
                width=0.6,
                label="generated",
            )

            if not values:
                ax.text(
                    0.5,
                    0.5,
                    "no data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=9,
                    color="gray",
                )

            ax.axvline(
                target,
                color="#D62728",
                linestyle="--",
                linewidth=1.3,
                label="target",
            )

            n_mols = len(values)
            validity = result.get("validity_rate")
            training_metrics = result.get("training_metrics", {}) or {}
            novelty = training_metrics.get("novelty")
            subtitle_bits = [f"n={n_mols}"]
            if validity is not None and result.get("n_total", 0) > 0:
                subtitle_bits.append(f"valid={validity:.0%}")
            if novelty is not None and not math.isnan(float(novelty)):
                subtitle_bits.append(f"novel={novelty:.0%}")
            ax.text(
                0.98,
                0.95,
                " ".join(subtitle_bits),
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                color="#222",
            )

            if row_idx == 0:
                ax.set_title(f"scale={scale}", fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(f"target={target}\nfrequency", fontsize=9)
            if row_idx == n_rows - 1:
                ax.set_xlabel("n_atoms", fontsize=9)

            ax.set_xlim(x_lo - 0.5, x_hi + 0.5)
            ax.tick_params(labelsize=8)
            if row_idx == 0 and col_idx == n_cols - 1:
                ax.legend(fontsize=7, loc="upper left", frameon=False)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _plot_natoms_gif(
    trajectories: list,
    output_path: Path,
    target: int,
    scale: float,
    padding: int = X_AXIS_PADDING,
    fps: int = 10,
) -> None:
    """Animate the n_atoms marginal over integration time t ∈ [0, 1]."""
    if not trajectories:
        return

    n_frames = min(len(traj) for traj in trajectories if traj)
    if n_frames == 0:
        return

    per_step_counts: list[list[int]] = [[] for _ in range(n_frames)]
    for traj in trajectories:
        for step_idx in range(n_frames):
            n = _mol_n_atoms(traj[step_idx])
            if n is not None:
                per_step_counts[step_idx].append(n)

    flat = [n for counts in per_step_counts for n in counts]
    if not flat:
        return
    x_lo = max(0, min(flat) - padding)
    x_hi = max(flat) + padding
    if x_hi <= x_lo:
        x_hi = x_lo + 1

    bins = np.arange(x_lo, x_hi + 2) - 0.5
    x_support = np.arange(x_lo, x_hi + 1)

    per_step_freqs: list[np.ndarray] = []
    for counts in per_step_counts:
        hist, _ = np.histogram(counts, bins=bins)
        freqs = hist / max(hist.sum(), 1)
        per_step_freqs.append(freqs)

    y_max = max((f.max() for f in per_step_freqs if f.size), default=0.1)
    y_max = max(float(y_max) * 1.15, 0.05)

    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    bars = ax.bar(
        x_support,
        per_step_freqs[0],
        color="#4C72B0",
        edgecolor="black",
        linewidth=0.4,
        width=0.8,
    )
    ax.axvline(
        target,
        color="#D62728",
        linestyle="--",
        linewidth=1.3,
        label="target",
    )
    ax.set_xlim(x_lo - 0.5, x_hi + 0.5)
    ax.set_ylim(0, y_max)
    ax.set_xlabel("n_atoms", fontsize=9)
    ax.set_ylabel("frequency", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=8, loc="upper right", frameon=False)
    title = ax.set_title("", fontsize=10)

    def _frame(frame_idx: int):
        freqs = per_step_freqs[frame_idx]
        for bar, h in zip(bars, freqs):
            bar.set_height(h)
        t_val = frame_idx / max(n_frames - 1, 1)
        title.set_text(f"scale={scale}, target={target}, t={t_val:.2f}")
        return (*bars, title)

    anim = FuncAnimation(
        fig,
        _frame,
        frames=n_frames,
        interval=1000.0 / fps,
        blit=False,
    )
    anim.save(output_path, writer=PillowWriter(fps=fps))
    plt.close(fig)


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
        metrics,
        stability_metrics,
        distribution_metrics,
    )


def _trajectories_to_rdkit(trajectories, vocab) -> list:
    """Convert the final MoleculeData in each trajectory to an RDKit mol."""
    rdkit_mols: list = []
    for traj in trajectories:
        final_mol = traj[-1] if isinstance(traj, list) and traj else traj
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
    """Compute every training-time metric collection on a list of RDKit mols.

    Mirrors ``calc_metrics_`` and the validation-step flow in
    ``LightningModuleRates``. Returns a flat dict of Python floats.
    """
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

    return {k: float(v) if isinstance(v, (int, float)) else v for k, v in results.items()}


def _load_checkpoint(module, checkpoint_path: Path) -> None:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    clean_state_dict = {k.replace("._orig_mod", ""): v for k, v in state_dict.items()}
    module.load_state_dict(clean_state_dict)


def _results_for_json(all_results: dict[tuple[float, int], dict]) -> list[dict]:
    drop_keys = {"all_n_atoms", "valid_n_atoms", "invalid_n_atoms"}
    rows: list[dict] = []
    for (scale, target), result in sorted(all_results.items()):
        row = {k: v for k, v in result.items() if k not in drop_keys}
        training_metrics = row.pop("training_metrics", None)
        if training_metrics:
            for k, v in training_metrics.items():
                row[f"metric/{k}"] = v
        if "per_seed" in row:
            row["per_seed"] = [
                {k: v for k, v in ps.items() if k not in drop_keys}
                for ps in row["per_seed"]
            ]
        row["guidance_scale"] = scale
        row["target_n_atoms"] = target
        rows.append(row)
    return rows


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate n-atoms CFG steering on QM9."
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
        default=Path("eval_outputs/natoms_cfg"),
        help="Directory to save evaluation artifacts.",
    )
    parser.add_argument(
        "--targets",
        type=str,
        default="10,18,28",
        help="Comma-separated target atom counts.",
    )
    parser.add_argument(
        "--guidance-scales",
        type=str,
        default="1.0,5.0,10.0",
        help="Comma-separated n-atoms CFG guidance scales.",
    )
    parser.add_argument(
        "--n-mols", type=int, default=500, help="Approximate molecules per sweep cell."
    )
    parser.add_argument(
        "--predict-batch-size",
        type=int,
        default=128,
        help="Predict dataloader batch size.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (used when --seeds not given).")
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated list of seeds. When set, each (scale, target) cell is run once per seed and aggregated.",
    )
    parser.add_argument(
        "--overrides",
        nargs="*",
        default=["data=qm9", "model=semla", "cfg=natoms"],
        help="Hydra overrides used to compose config.",
    )
    parser.add_argument(
        "--no-save-trajectories",
        action="store_true",
        help="Disable saving generated molecule trajectories.",
    )
    parser.add_argument(
        "--no-save-natoms-trajectories",
        action="store_true",
        help=(
            "Disable saving the per-step atom-count trajectory of every "
            "generation (one .pt per (scale, target) cell, written to "
            "<output-dir>/natoms_trajectories/)."
        ),
    )
    parser.add_argument(
        "--no-save-gifs",
        action="store_true",
        help="Disable saving per-cell GIFs of the n_atoms marginal over time.",
    )
    parser.add_argument(
        "--gif-fps",
        type=int,
        default=10,
        help="Frames per second for the integration-time GIFs.",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trajectories_dir = args.output_dir / "trajectories"
    if not args.no_save_trajectories:
        trajectories_dir.mkdir(parents=True, exist_ok=True)
    natoms_traj_dir = args.output_dir / "natoms_trajectories"
    if not args.no_save_natoms_trajectories:
        natoms_traj_dir.mkdir(parents=True, exist_ok=True)
    gifs_dir = args.output_dir / "natoms_gifs"
    if not args.no_save_gifs:
        gifs_dir.mkdir(parents=True, exist_ok=True)

    _register_resolvers()
    torch.set_float32_matmul_precision("medium")
    RDLogger.DisableLog("rdApp.*")
    seeds = _parse_int_list(args.seeds) if args.seeds else [args.seed]
    pl.seed_everything(seeds[0])

    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.1"):
        cfg = compose(config_name="default", overrides=args.overrides)

    (
        module,
        test_dl,
        distributions,
        metrics,
        stability_metrics,
        distribution_metrics,
    ) = _setup_eval_components(cfg, args.predict_batch_size)
    _load_checkpoint(module, args.checkpoint)

    adapter = module.cfg_adapter
    if not adapter._has_natoms_cfg:
        raise RuntimeError("Loaded model does not expose n-atoms CFG conditioning.")

    n_predict_batches = max(1, math.ceil(args.n_mols / max(1, args.predict_batch_size)))
    trainer_kwargs = dict(cfg.trainer.trainer)
    trainer_kwargs["strategy"] = "auto"
    trainer_kwargs["limit_predict_batches"] = n_predict_batches
    trainer = pl.Trainer(
        logger=False,
        callbacks=[],
        enable_checkpointing=False,
        **trainer_kwargs,
    )

    target_n_atoms = _parse_int_list(args.targets)
    guidance_scales = _parse_float_list(args.guidance_scales)
    module.predict_return_traj = True

    original_property_scale = adapter.cfg_guidance_scale
    original_mw_scale = adapter.mw_cfg_guidance_scale
    all_results: dict[tuple[float, int], dict] = {}

    try:
        adapter.cfg_guidance_scale = 0.0
        adapter.mw_cfg_guidance_scale = 0.0

        for scale in guidance_scales:
            adapter.natoms_cfg_guidance_scale = float(scale)
            print(f"\n{'=' * 60}\nn_atoms guidance scale={scale}\n{'=' * 60}")

            for target in target_n_atoms:
                module.predict_target_n_atoms_override = int(target)
                module.predict_target_mw_override = None

                per_seed: list[dict] = []
                valid_trajs: list = []
                invalid_trajs: list = []
                for seed in seeds:
                    pl.seed_everything(int(seed))
                    predictions = trainer.predict(module, dataloaders=test_dl)
                    seed_summary, vt, it = _summarize(
                        predictions=predictions,
                        target_n_atoms=int(target),
                        max_samples=args.n_mols,
                    )
                    per_seed.append(
                        {
                            "seed": int(seed),
                            "n_total": seed_summary["n_total"],
                            "n_valid": seed_summary["n_valid"],
                            "n_invalid": seed_summary["n_invalid"],
                            "validity_rate": seed_summary["validity_rate"],
                            "all_stats": seed_summary["all_stats"],
                            "valid_stats": seed_summary["valid_stats"],
                            "all_n_atoms": list(seed_summary["all_n_atoms"]),
                            "valid_n_atoms": list(seed_summary["valid_n_atoms"]),
                            "invalid_n_atoms": list(seed_summary["invalid_n_atoms"]),
                        }
                    )
                    valid_trajs.extend(vt)
                    invalid_trajs.extend(it)

                valid_counts = [c for ps in per_seed for c in ps["valid_n_atoms"]]
                invalid_counts = [c for ps in per_seed for c in ps["invalid_n_atoms"]]
                all_counts = valid_counts + invalid_counts
                n_total = len(all_counts)
                n_valid = len(valid_counts)
                summary = {
                    "target_n_atoms": int(target),
                    "n_total": n_total,
                    "n_valid": n_valid,
                    "n_invalid": n_total - n_valid,
                    "validity_rate": (n_valid / n_total) if n_total else 0.0,
                    "all_stats": _natoms_stats(all_counts, int(target)),
                    "valid_stats": _natoms_stats(valid_counts, int(target)),
                    "all_n_atoms": all_counts,
                    "valid_n_atoms": valid_counts,
                    "invalid_n_atoms": invalid_counts,
                    "guidance_scale": float(scale),
                    "seeds": [int(s) for s in seeds],
                    "per_seed": per_seed,
                }

                rdkit_mols = _trajectories_to_rdkit(
                    valid_trajs + invalid_trajs,
                    module.vocab,
                )
                training_metrics = _compute_all_metrics(
                    rdkit_mols=rdkit_mols,
                    metrics=metrics,
                    stability_metrics=stability_metrics,
                    distribution_metrics=distribution_metrics,
                )
                summary["training_metrics"] = training_metrics
                all_results[(float(scale), int(target))] = summary

                if summary["n_total"] == 0:
                    print(f"target={target}: no molecules")
                else:
                    stats = summary["all_stats"]
                    valid_stats = summary["valid_stats"]
                    tm = training_metrics
                    print(
                        f"target={target}: "
                        f"validity={summary['validity_rate']:.1%} "
                        f"({summary['n_valid']}/{summary['n_total']}), "
                        f"novelty={tm.get('novelty', float('nan')):.1%}, "
                        f"uniqueness={tm.get('uniqueness', float('nan')):.1%}, "
                        f"atom-stab={tm.get('atom-stability', float('nan')):.1%}, "
                        f"mol-stab={tm.get('molecule-stability', float('nan')):.1%}, "
                        f"exact(all)={stats['exact_match_rate']:.1%}, "
                        f"exact(valid)={valid_stats['exact_match_rate']:.1%}, "
                        f"within±1(valid)={valid_stats['within_1_rate']:.1%}, "
                        f"mean(all)={stats['mean_n_atoms']:.2f}"
                    )

                if not args.no_save_trajectories:
                    traj_path = trajectories_dir / f"scale={scale}_target={target}.pt"
                    torch.save(
                        {
                            "guidance_scale": float(scale),
                            "target_n_atoms": int(target),
                            "valid_trajectories": valid_trajs,
                            "invalid_trajectories": invalid_trajs,
                            "valid_n_atoms": summary["valid_n_atoms"],
                            "invalid_n_atoms": summary["invalid_n_atoms"],
                        },
                        traj_path,
                    )

                if not args.no_save_natoms_trajectories:
                    natoms_path = (
                        natoms_traj_dir / f"scale={scale}_target={target}.pt"
                    )
                    _save_natoms_trajectories(
                        valid_trajs=valid_trajs,
                        invalid_trajs=invalid_trajs,
                        target=int(target),
                        scale=float(scale),
                        output_path=natoms_path,
                    )

                if not args.no_save_gifs:
                    gif_path = gifs_dir / f"scale={scale}_target={target}.gif"
                    _plot_natoms_gif(
                        trajectories=valid_trajs + invalid_trajs,
                        output_path=gif_path,
                        target=int(target),
                        scale=float(scale),
                        fps=args.gif_fps,
                    )
    finally:
        adapter.cfg_guidance_scale = original_property_scale
        adapter.mw_cfg_guidance_scale = original_mw_scale
        module.predict_target_n_atoms_override = None
        module.predict_target_mw_override = None

    pt_path = args.output_dir / "natoms_cfg_results.pt"
    json_path = args.output_dir / "natoms_cfg_results.json"
    plot_all_path = args.output_dir / "natoms_cfg_distributions_all.png"
    plot_valid_path = args.output_dir / "natoms_cfg_distributions_valid.png"

    if pt_path.exists():
        try:
            existing = torch.load(pt_path, map_location="cpu", weights_only=False)
            if isinstance(existing, dict):
                merged = dict(existing)
                overlapped = sorted(set(merged) & set(all_results))
                if overlapped:
                    print(
                        f"[append] replacing {len(overlapped)} existing cell(s): "
                        f"{overlapped}"
                    )
                merged.update(all_results)
                all_results = merged
                print(
                    f"[append] merged with {len(existing)} existing cell(s) from "
                    f"{pt_path}; total now {len(all_results)}."
                )
            else:
                print(f"[append] existing {pt_path} is not a dict; overwriting.")
        except Exception as exc:
            print(f"[append] failed to load existing results ({exc}); overwriting.")

    torch.save(all_results, pt_path)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(_results_for_json(all_results), handle, indent=2)
    _plot_natoms_grid(
        all_results,
        distributions.n_atoms_distribution,
        plot_all_path,
        value_key="all_n_atoms",
        title="Generated n_atoms distributions (all generations)",
    )
    _plot_natoms_grid(
        all_results,
        distributions.n_atoms_distribution,
        plot_valid_path,
        value_key="valid_n_atoms",
        title="Generated n_atoms distributions (valid generations only)",
    )

    print(f"\nSaved tensor results to:        {pt_path}")
    print(f"Saved JSON summary to:          {json_path}")
    print(f"Saved all-generations plot to:  {plot_all_path}")
    print(f"Saved valid-only plot to:       {plot_valid_path}")
    if not args.no_save_trajectories:
        print(f"Saved trajectories under:       {trajectories_dir}")
    if not args.no_save_natoms_trajectories:
        print(f"Saved n_atoms trajectories under: {natoms_traj_dir}")
    if not args.no_save_gifs:
        print(f"Saved n_atoms GIFs under:       {gifs_dir}")


if __name__ == "__main__":
    main()
