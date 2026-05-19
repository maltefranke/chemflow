#!/usr/bin/env python3
"""Standalone evaluation for logP classifier-free guidance on QM9.

Loads a logP-CFG checkpoint, sweeps target logP values (and optionally
guidance scales), measures the realised RDKit Crippen logP of each generated
molecule, and reports how closely the model hits the requested target.

Usage example::

    python eval_scripts/eval_logp_cfg.py \
        --checkpoint /path/to/epoch=499-step=12500.ckpt \
        --output-dir eval_outputs/logp_cfg \
        --targets -2,2,4,5,6 \
        --guidance-scales 1.0 \
        --n-mols 200 \
        --predict-batch-size 100

Caveat
------
``configs/cfg/logp.yaml`` trains with ``logp_dropout_prob=0`` and
``logp_guidance_scale=1.0`` — the model never sees the unconditional branch,
so at inference only ``logp_cfg_guidance_scale=1.0`` is well-defined.  Scales
> 1 interpolate against an untrained null token and are not meaningful unless
the model was retrained with non-zero ``logp_dropout_prob``.
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
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen

from chemflow.dataset.vocab import setup_token_weights
from chemflow.utils.metrics import calc_atom_stabilities, init_metrics
from chemflow.utils.utils import init_uniform_prior


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"


# ---------------------------------------------------------------------------
# Resolvers / arg parsing
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


def _parse_int_list(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def _parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


# ---------------------------------------------------------------------------
# Per-molecule logP extraction (RDKit Crippen, the same metric used in
# ``chemflow.model.cfg.compute_logp`` so target/realised are commensurable).
# ---------------------------------------------------------------------------


def _final_mol(traj):
    if isinstance(traj, list) and traj:
        return traj[-1]
    return traj


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


def _rdkit_logp(mol) -> float | None:
    """Robust Crippen MolLogP. Returns ``None`` if the molecule is unusable."""
    if mol is None:
        return None
    try:
        return float(Crippen.MolLogP(mol))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Result aggregation
# ---------------------------------------------------------------------------


def _collect_from_predict_output(
    pred,
    vocab,
) -> tuple[
    list[float], list[int],
    list[float], list[int],
    list, list,
    list, list,
]:
    """Extract per-cell aggregated lists.

    Returns
    -------
    ``(valid_logps, valid_natoms,
       invalid_logps, invalid_natoms,
       valid_trajs, invalid_trajs,
       valid_rdkit, invalid_rdkit)``

    The ``logps`` / ``natoms`` lists are **paired** index-wise (same length,
    only molecules where *both* are computable contribute).  The ``trajs`` /
    ``rdkit`` lists are full-coverage (one entry per generated molecule,
    including those with unrecoverable RDKit objects) and are re-used
    downstream for the training-metric stack.
    """
    valid_logps: list[float] = []
    valid_natoms: list[int] = []
    invalid_logps: list[float] = []
    invalid_natoms: list[int] = []
    valid_trajs: list = []
    invalid_trajs: list = []
    valid_rdkit: list = []
    invalid_rdkit: list = []

    if pred is None or not isinstance(pred, dict):
        return (
            valid_logps,
            valid_natoms,
            invalid_logps,
            invalid_natoms,
            valid_trajs,
            invalid_trajs,
            valid_rdkit,
            invalid_rdkit,
        )

    atom_tok = vocab.atom_tokens
    edge_tok = vocab.edge_tokens
    charge_tok = vocab.charge_tokens

    for traj in pred.get("valid_mols", []) or []:
        final = _final_mol(traj)
        try:
            rd = final.to_rdkit_mol(atom_tok, edge_tok, charge_tok)
        except Exception:
            rd = None
        valid_rdkit.append(rd)
        valid_trajs.append(traj)
        n = _mol_n_atoms(final)
        v = _rdkit_logp(rd)
        if v is not None and n is not None:
            valid_logps.append(v)
            valid_natoms.append(int(n))

    for traj in pred.get("invalid_mols", []) or []:
        final = _final_mol(traj)
        try:
            rd = final.to_rdkit_mol(atom_tok, edge_tok, charge_tok)
        except Exception:
            rd = None
        invalid_rdkit.append(rd)
        invalid_trajs.append(traj)
        n = _mol_n_atoms(final)
        v = _rdkit_logp(rd)
        if v is not None and n is not None:
            invalid_logps.append(v)
            invalid_natoms.append(int(n))

    return (
        valid_logps,
        valid_natoms,
        invalid_logps,
        invalid_natoms,
        valid_trajs,
        invalid_trajs,
        valid_rdkit,
        invalid_rdkit,
    )


def _logp_stats(values: list[float], target: float) -> dict:
    if not values:
        return {
            "mean_logp": float("nan"),
            "std_logp": float("nan"),
            "median_logp": float("nan"),
            "mae": float("nan"),
            "within_0p5_rate": 0.0,
            "within_1p0_rate": 0.0,
            "within_2p0_rate": 0.0,
            "n_molecules": 0,
        }
    t = torch.tensor(values, dtype=torch.float)
    diff = (t - target).abs()
    return {
        "mean_logp": t.mean().item(),
        "std_logp": t.std().item() if t.numel() > 1 else 0.0,
        "median_logp": t.median().item(),
        "mae": diff.mean().item(),
        "within_0p5_rate": (diff <= 0.5).float().mean().item(),
        "within_1p0_rate": (diff <= 1.0).float().mean().item(),
        "within_2p0_rate": (diff <= 2.0).float().mean().item(),
        "n_molecules": int(t.numel()),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _plot_logp_grid(
    all_results: dict[tuple[float, float], dict],
    output_path: Path,
    value_key: str,
    title: str,
    n_bins: int = 30,
) -> None:
    """Histogram grid: rows=target logP, cols=guidance scales."""
    if not all_results:
        return

    guidance_scales = sorted({key[0] for key in all_results})
    target_logps = sorted({key[1] for key in all_results})

    flat: list[float] = []
    for result in all_results.values():
        flat.extend(result.get(value_key, []))
    if not flat:
        x_lo, x_hi = -4.0, 8.0
    else:
        x_lo = min(flat + target_logps) - 0.5
        x_hi = max(flat + target_logps) + 0.5
        if x_hi - x_lo < 1.0:
            x_hi = x_lo + 1.0

    bins = np.linspace(x_lo, x_hi, n_bins + 1)

    n_rows = len(target_logps)
    n_cols = len(guidance_scales)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.4 * max(1, n_cols), 2.8 * max(1, n_rows)),
        squeeze=False,
    )

    for row_idx, target in enumerate(target_logps):
        for col_idx, scale in enumerate(guidance_scales):
            ax = axes[row_idx][col_idx]
            result = all_results.get((scale, target), {})
            values = result.get(value_key, [])

            if values:
                ax.hist(
                    values,
                    bins=bins,
                    color="#4C72B0",
                    alpha=0.85,
                    edgecolor="black",
                    linewidth=0.4,
                )
            else:
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
            tm = result.get("training_metrics", {}) or {}
            novelty = tm.get("novelty")
            stats = result.get(
                "valid_stats" if value_key == "valid_logps" else "all_stats",
                {},
            )
            mae = stats.get("mae")
            subtitle_bits = [f"n={n_mols}"]
            if validity is not None and result.get("n_total", 0) > 0:
                subtitle_bits.append(f"valid={validity:.0%}")
            if mae is not None and not math.isnan(float(mae)):
                subtitle_bits.append(f"mae={mae:.2f}")
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
                ax.set_ylabel(f"target={target:g}\ncount", fontsize=9)
            if row_idx == n_rows - 1:
                ax.set_xlabel("RDKit Crippen logP", fontsize=9)

            ax.set_xlim(x_lo, x_hi)
            ax.tick_params(labelsize=8)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _plot_logp_natoms_scatter(
    all_results: dict[tuple[float, float], dict],
    output_path: Path,
    title: str = "Generated logP vs n_atoms",
) -> None:
    """Scatter grid: rows=target logP, cols=guidance scales.

    Each cell shows one point per generated molecule (x=realised RDKit
    Crippen logP, y=heavy-atom count).  Valid molecules are blue, invalid
    are light grey.  The target logP is drawn as a vertical dashed red line.

    Shared x/y limits across all cells make visual comparison meaningful.
    """
    if not all_results:
        return

    guidance_scales = sorted({key[0] for key in all_results})
    target_logps = sorted({key[1] for key in all_results})

    flat_logp: list[float] = []
    flat_natoms: list[int] = []
    for result in all_results.values():
        flat_logp.extend(result.get("valid_logps", []))
        flat_logp.extend(result.get("invalid_logps", []))
        flat_natoms.extend(result.get("valid_natoms", []))
        flat_natoms.extend(result.get("invalid_natoms", []))

    if flat_logp:
        x_lo = min(flat_logp + target_logps) - 0.5
        x_hi = max(flat_logp + target_logps) + 0.5
        if x_hi - x_lo < 1.0:
            x_hi = x_lo + 1.0
    else:
        x_lo, x_hi = -4.0, 8.0

    if flat_natoms:
        y_lo = max(0, min(flat_natoms) - 1)
        y_hi = max(flat_natoms) + 1
    else:
        y_lo, y_hi = 0, 32

    n_rows = len(target_logps)
    n_cols = len(guidance_scales)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.6 * max(1, n_cols), 3.0 * max(1, n_rows)),
        squeeze=False,
    )

    legend_handles: list = []
    for row_idx, target in enumerate(target_logps):
        for col_idx, scale in enumerate(guidance_scales):
            ax = axes[row_idx][col_idx]
            result = all_results.get((scale, target), {})
            v_lp = list(result.get("valid_logps", []))
            v_n = list(result.get("valid_natoms", []))
            i_lp = list(result.get("invalid_logps", []))
            i_n = list(result.get("invalid_natoms", []))

            # Defensive: ``valid_natoms`` / ``invalid_natoms`` were added to
            # the per-cell dict after the first version of this script;
            # cells loaded from an older ``.pt`` via the ``[append]`` merge
            # path won't have them, so the lists may be empty / mismatched.
            # Treat any length mismatch as "no paired data" for that subset.
            valid_paired = len(v_lp) > 0 and len(v_lp) == len(v_n)
            invalid_paired = len(i_lp) > 0 and len(i_lp) == len(i_n)
            mismatched = (
                (len(v_lp) > 0 and len(v_lp) != len(v_n))
                or (len(i_lp) > 0 and len(i_lp) != len(i_n))
            )

            if invalid_paired:
                inv_handle = ax.scatter(
                    i_lp,
                    i_n,
                    s=14,
                    c="#BBBBBB",
                    alpha=0.55,
                    edgecolors="none",
                    label="invalid",
                )
                if all(h.get_label() != "invalid" for h in legend_handles):
                    legend_handles.append(inv_handle)
            if valid_paired:
                val_handle = ax.scatter(
                    v_lp,
                    v_n,
                    s=18,
                    c="#4C72B0",
                    alpha=0.85,
                    edgecolors="black",
                    linewidths=0.3,
                    label="valid",
                )
                if all(h.get_label() != "valid" for h in legend_handles):
                    legend_handles.append(val_handle)
            if not valid_paired and not invalid_paired:
                msg = (
                    "no paired n_atoms data\n(legacy cell — re-run to repopulate)"
                    if mismatched
                    else "no data"
                )
                ax.text(
                    0.5,
                    0.5,
                    msg,
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
            )

            n_v = len(v_lp)
            n_i = len(i_lp)
            stats = result.get("valid_stats", {})
            mae = stats.get("mae")
            mean_n_v = (
                float(np.mean(v_n)) if v_n else float("nan")
            )
            subtitle_bits = [f"valid n={n_v}", f"inv n={n_i}"]
            if mae is not None and not math.isnan(float(mae)):
                subtitle_bits.append(f"mae={mae:.2f}")
            if not math.isnan(mean_n_v):
                subtitle_bits.append(f"mean atoms={mean_n_v:.1f}")
            ax.text(
                0.98,
                0.97,
                "\n".join(subtitle_bits),
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=7,
                color="#222",
            )

            if row_idx == 0:
                ax.set_title(f"scale={scale}", fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(f"target={target:g}\nn_atoms", fontsize=9)
            if row_idx == n_rows - 1:
                ax.set_xlabel("RDKit Crippen logP", fontsize=9)

            ax.set_xlim(x_lo, x_hi)
            ax.set_ylim(y_lo, y_hi)
            ax.grid(True, color="#EEEEEE", linewidth=0.6)
            ax.set_axisbelow(True)
            ax.tick_params(labelsize=8)

    if legend_handles:
        fig.legend(
            handles=legend_handles,
            loc="upper right",
            fontsize=9,
            frameon=False,
            bbox_to_anchor=(0.99, 0.995),
        )
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 0.96, 0.96])
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Eval-component setup (shared with eval_natoms_cfg.py)
# ---------------------------------------------------------------------------


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
    metrics, stability_metrics, distribution_metrics, _batch_metrics = init_metrics(
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


def _load_checkpoint(module, checkpoint_path: Path) -> None:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    clean_state_dict = {k.replace("._orig_mod", ""): v for k, v in state_dict.items()}
    module.load_state_dict(clean_state_dict)


def _compute_all_metrics(
    rdkit_mols: list,
    metrics,
    stability_metrics,
    distribution_metrics,
) -> dict:
    """Mirror of the metric stack used in eval_natoms_cfg.py."""
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
        for key in ("rdkit-atom-stability", "rdkit-molecule-stability"):
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
# JSON serialisation
# ---------------------------------------------------------------------------


def _save_logp_natoms_csv(
    all_results: dict[tuple[float, float], dict],
    output_path: Path,
) -> int:
    """Persist the per-molecule rows that back the logP × n_atoms scatter.

    Writes a flat CSV with one row per generated molecule and columns
    ``guidance_scale, target_logp, logp, n_atoms, is_valid`` — directly
    loadable with ``pd.read_csv`` for custom downstream analysis.

    Pairing is enforced index-wise: only molecules where both the realised
    RDKit Crippen logP and the heavy-atom count were successfully inferred
    contribute a row.  Cells with mismatched-length / missing ``*_natoms``
    lists (legacy entries pre-dating n_atoms tracking) are silently skipped.

    Returns the number of rows written.
    """
    import csv as _csv

    rows_written = 0
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = _csv.writer(handle)
        writer.writerow(
            [
                "guidance_scale",
                "target_logp",
                "logp",
                "n_atoms",
                "is_valid",
            ]
        )
        for (scale, target), result in sorted(all_results.items()):
            v_lp = list(result.get("valid_logps", []) or [])
            v_n = list(result.get("valid_natoms", []) or [])
            i_lp = list(result.get("invalid_logps", []) or [])
            i_n = list(result.get("invalid_natoms", []) or [])

            # Skip subsets where pairing is broken (legacy cells).
            if v_lp and len(v_lp) == len(v_n):
                for lp, n in zip(v_lp, v_n):
                    writer.writerow([scale, target, lp, int(n), 1])
                    rows_written += 1
            if i_lp and len(i_lp) == len(i_n):
                for lp, n in zip(i_lp, i_n):
                    writer.writerow([scale, target, lp, int(n), 0])
                    rows_written += 1
    return rows_written


def _results_for_json(all_results: dict[tuple[float, float], dict]) -> list[dict]:
    drop_keys = {
        "all_logps",
        "valid_logps",
        "invalid_logps",
        "all_natoms",
        "valid_natoms",
        "invalid_natoms",
        "valid_smiles",
        "invalid_smiles",
    }
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
        row["target_logp"] = target
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate RDKit-Crippen-logP CFG steering on QM9."
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
        default=Path("eval_outputs/logp_cfg"),
        help="Directory to save evaluation artifacts.",
    )
    parser.add_argument(
        "--targets",
        type=str,
        default="-2,2,4,5,6",
        help="Comma-separated target logP values.",
    )
    parser.add_argument(
        "--guidance-scales",
        type=str,
        default="1.0",
        help=(
            "Comma-separated logP CFG guidance scales. "
            "Note: scales other than 1.0 only make sense if the checkpoint "
            "was trained with logp_dropout_prob > 0."
        ),
    )
    parser.add_argument(
        "--n-mols",
        type=int,
        default=500,
        help="Approximate molecules per (scale, target) cell.",
    )
    parser.add_argument(
        "--predict-batch-size",
        type=int,
        default=128,
        help="Predict dataloader batch size.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (used when --seeds not given).",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help=(
            "Comma-separated list of seeds. When set, each (scale, target) "
            "cell is run once per seed and aggregated."
        ),
    )
    parser.add_argument(
        "--overrides",
        nargs="*",
        default=["data=qm9", "model=semla", "cfg=logp"],
        help="Hydra overrides used to compose config.",
    )
    parser.add_argument(
        "--no-save-trajectories",
        action="store_true",
        help="Disable saving generated molecule trajectories.",
    )
    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = _build_arg_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trajectories_dir = args.output_dir / "trajectories"
    if not args.no_save_trajectories:
        trajectories_dir.mkdir(parents=True, exist_ok=True)

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
        _distributions,
        metrics,
        stability_metrics,
        distribution_metrics,
    ) = _setup_eval_components(cfg, args.predict_batch_size)
    _load_checkpoint(module, args.checkpoint)

    guidance = module.cfg_guidance
    logp_signal = guidance.get_signal("logp")
    if logp_signal is None:
        raise RuntimeError(
            "Loaded model does not expose logP CFG conditioning. "
            "Check that --overrides includes `cfg=logp` and that the "
            "checkpoint was trained with that config."
        )

    n_predict_batches = max(
        1,
        math.ceil(args.n_mols / max(1, args.predict_batch_size)),
    )
    trainer_kwargs = dict(cfg.trainer.trainer)
    trainer_kwargs["strategy"] = "auto"
    trainer_kwargs["limit_predict_batches"] = n_predict_batches
    trainer = pl.Trainer(
        logger=False,
        callbacks=[],
        enable_checkpointing=False,
        **trainer_kwargs,
    )

    target_logps = _parse_float_list(args.targets)
    guidance_scales = _parse_float_list(args.guidance_scales)
    module.predict_return_traj = True

    # Force every other CFG signal off; only logP is exercised here.
    original_scales = {s.name: s.guidance_scale for s in guidance.signals}
    all_results: dict[tuple[float, float], dict] = {}

    try:
        for s in guidance.signals:
            if s.name != "logp":
                s.guidance_scale = 0.0

        for scale in guidance_scales:
            logp_signal.guidance_scale = float(scale)
            print(
                f"\n{'=' * 60}\n"
                f"logP guidance scale={scale}\n"
                f"{'=' * 60}"
            )

            for target in target_logps:
                module.predict_overrides = {"logp": float(target)}

                per_seed: list[dict] = []
                valid_trajs: list = []
                invalid_trajs: list = []
                valid_rdkit: list = []
                invalid_rdkit: list = []
                valid_logps: list[float] = []
                invalid_logps: list[float] = []
                valid_natoms: list[int] = []
                invalid_natoms: list[int] = []

                for seed in seeds:
                    pl.seed_everything(int(seed))
                    predictions = trainer.predict(module, dataloaders=test_dl)

                    seed_valid_logps: list[float] = []
                    seed_invalid_logps: list[float] = []
                    seed_valid_natoms: list[int] = []
                    seed_invalid_natoms: list[int] = []
                    for pred in predictions or []:
                        (
                            v_lp,
                            v_n,
                            i_lp,
                            i_n,
                            v_t,
                            i_t,
                            v_rd,
                            i_rd,
                        ) = _collect_from_predict_output(pred, module.vocab)
                        seed_valid_logps.extend(v_lp)
                        seed_invalid_logps.extend(i_lp)
                        seed_valid_natoms.extend(v_n)
                        seed_invalid_natoms.extend(i_n)
                        valid_trajs.extend(v_t)
                        invalid_trajs.extend(i_t)
                        valid_rdkit.extend(v_rd)
                        invalid_rdkit.extend(i_rd)

                    valid_logps.extend(seed_valid_logps)
                    invalid_logps.extend(seed_invalid_logps)
                    valid_natoms.extend(seed_valid_natoms)
                    invalid_natoms.extend(seed_invalid_natoms)
                    n_valid = len(seed_valid_logps)
                    n_total = n_valid + len(seed_invalid_logps)
                    per_seed.append(
                        {
                            "seed": int(seed),
                            "n_total": n_total,
                            "n_valid": n_valid,
                            "n_invalid": n_total - n_valid,
                            "validity_rate": (n_valid / n_total) if n_total else 0.0,
                            "all_stats": _logp_stats(
                                seed_valid_logps + seed_invalid_logps,
                                target,
                            ),
                            "valid_stats": _logp_stats(seed_valid_logps, target),
                            "all_logps": list(
                                seed_valid_logps + seed_invalid_logps
                            ),
                            "valid_logps": list(seed_valid_logps),
                            "invalid_logps": list(seed_invalid_logps),
                            "all_natoms": list(
                                seed_valid_natoms + seed_invalid_natoms
                            ),
                            "valid_natoms": list(seed_valid_natoms),
                            "invalid_natoms": list(seed_invalid_natoms),
                        }
                    )

                # Cap aggregated outputs at args.n_mols (preserve interleave).
                # logps and natoms are paired index-wise; trajs / rdkit are
                # full-coverage so they may have more entries than logps —
                # we cap them by their own length.
                if len(valid_logps) + len(invalid_logps) > args.n_mols:
                    keep_valid = min(len(valid_logps), args.n_mols)
                    valid_logps = valid_logps[:keep_valid]
                    valid_natoms = valid_natoms[:keep_valid]
                    remaining = args.n_mols - keep_valid
                    invalid_logps = invalid_logps[:remaining]
                    invalid_natoms = invalid_natoms[:remaining]

                if len(valid_trajs) + len(invalid_trajs) > args.n_mols:
                    keep_valid = min(len(valid_trajs), args.n_mols)
                    valid_trajs = valid_trajs[:keep_valid]
                    valid_rdkit = valid_rdkit[:keep_valid]
                    remaining = args.n_mols - keep_valid
                    invalid_trajs = invalid_trajs[:remaining]
                    invalid_rdkit = invalid_rdkit[:remaining]

                all_logps = valid_logps + invalid_logps
                all_natoms = valid_natoms + invalid_natoms
                n_total = len(all_logps)
                n_valid = len(valid_logps)
                summary = {
                    "target_logp": float(target),
                    "n_total": n_total,
                    "n_valid": n_valid,
                    "n_invalid": n_total - n_valid,
                    "validity_rate": (n_valid / n_total) if n_total else 0.0,
                    "all_stats": _logp_stats(all_logps, target),
                    "valid_stats": _logp_stats(valid_logps, target),
                    "all_logps": all_logps,
                    "valid_logps": valid_logps,
                    "invalid_logps": invalid_logps,
                    "all_natoms": all_natoms,
                    "valid_natoms": valid_natoms,
                    "invalid_natoms": invalid_natoms,
                    "guidance_scale": float(scale),
                    "seeds": [int(s) for s in seeds],
                    "per_seed": per_seed,
                }

                training_metrics = _compute_all_metrics(
                    rdkit_mols=valid_rdkit + invalid_rdkit,
                    metrics=metrics,
                    stability_metrics=stability_metrics,
                    distribution_metrics=distribution_metrics,
                )
                summary["training_metrics"] = training_metrics
                all_results[(float(scale), float(target))] = summary

                if summary["n_total"] == 0:
                    print(f"target={target}: no molecules")
                else:
                    stats_v = summary["valid_stats"]
                    stats_a = summary["all_stats"]
                    tm = training_metrics
                    print(
                        f"target={target}: "
                        f"validity={summary['validity_rate']:.1%} "
                        f"({summary['n_valid']}/{summary['n_total']}), "
                        f"novelty={tm.get('novelty', float('nan')):.1%}, "
                        f"unique={tm.get('uniqueness', float('nan')):.1%}, "
                        f"mol-stab={tm.get('rdkit-molecule-stability', float('nan')):.1%}, "
                        f"mae(valid)={stats_v['mae']:.2f}, "
                        f"±0.5(valid)={stats_v['within_0p5_rate']:.1%}, "
                        f"±1(valid)={stats_v['within_1p0_rate']:.1%}, "
                        f"mean(valid)={stats_v['mean_logp']:.2f}, "
                        f"mean(all)={stats_a['mean_logp']:.2f}"
                    )

                if not args.no_save_trajectories:
                    traj_path = (
                        trajectories_dir / f"scale={scale}_target={target}.pt"
                    )
                    torch.save(
                        {
                            "guidance_scale": float(scale),
                            "target_logp": float(target),
                            "valid_trajectories": valid_trajs,
                            "invalid_trajectories": invalid_trajs,
                            "valid_logps": list(valid_logps),
                            "invalid_logps": list(invalid_logps),
                            "valid_natoms": list(valid_natoms),
                            "invalid_natoms": list(invalid_natoms),
                            "valid_smiles": [
                                Chem.MolToSmiles(m) if m is not None else ""
                                for m in valid_rdkit
                            ],
                            "invalid_smiles": [
                                Chem.MolToSmiles(m) if m is not None else ""
                                for m in invalid_rdkit
                            ],
                        },
                        traj_path,
                    )
    finally:
        for s in guidance.signals:
            s.guidance_scale = original_scales[s.name]
        module.predict_overrides = None

    pt_path = args.output_dir / "logp_cfg_results.pt"
    json_path = args.output_dir / "logp_cfg_results.json"
    plot_all_path = args.output_dir / "logp_cfg_distributions_all.png"
    plot_valid_path = args.output_dir / "logp_cfg_distributions_valid.png"
    scatter_path = args.output_dir / "logp_natoms_scatter.png"
    scatter_data_path = args.output_dir / "logp_natoms_data.csv"

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
                    f"[append] merged with {len(existing)} existing cell(s) "
                    f"from {pt_path}; total now {len(all_results)}."
                )
            else:
                print(f"[append] existing {pt_path} is not a dict; overwriting.")
        except Exception as exc:
            print(f"[append] failed to load existing results ({exc}); overwriting.")

    torch.save(all_results, pt_path)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(_results_for_json(all_results), handle, indent=2)
    _plot_logp_grid(
        all_results,
        plot_all_path,
        value_key="all_logps",
        title="Generated logP distributions (all generations)",
    )
    _plot_logp_grid(
        all_results,
        plot_valid_path,
        value_key="valid_logps",
        title="Generated logP distributions (valid generations only)",
    )
    _plot_logp_natoms_scatter(
        all_results,
        scatter_path,
        title="Generated logP vs n_atoms (rows=target logP, cols=guidance scale)",
    )
    n_scatter_rows = _save_logp_natoms_csv(all_results, scatter_data_path)

    print(f"\nSaved tensor results to:        {pt_path}")
    print(f"Saved JSON summary to:          {json_path}")
    print(f"Saved all-generations plot to:  {plot_all_path}")
    print(f"Saved valid-only plot to:       {plot_valid_path}")
    print(f"Saved logP\u00d7n_atoms scatter to: {scatter_path}")
    print(
        f"Saved logP\u00d7n_atoms data to:    {scatter_data_path} "
        f"({n_scatter_rows} rows)"
    )
    if not args.no_save_trajectories:
        print(f"Saved trajectories under:       {trajectories_dir}")


if __name__ == "__main__":
    main()
