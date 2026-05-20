#!/usr/bin/env python3
"""Standalone evaluation for joint logP × QED classifier-free guidance on QM9.

Loads a logP+QED-CFG checkpoint and conditions on a list of ``(logp, qed)``
target tuples (the model is always given both values at the same time — there
is no independent sweep over each signal).  For every tuple the script
measures the realised RDKit Crippen logP, RDKit QED, and heavy-atom count of
each generated molecule, computes the full training-time metric stack, and
persists per-molecule rows + per-cell summaries.

Usage example::

    python eval_scripts/eval_logp_qed_cfg.py \
        --checkpoint /path/to/epoch=499-step=12500.ckpt \
        --output-dir eval_outputs/logp_qed_cfg \
        --targets="-2:0.3,2:0.5,4:0.7,5:0.85" \
        --guidance-scales 1.0 \
        --n-mols 200 \
        --predict-batch-size 100

Each ``--targets`` entry is ``logp:qed`` (a single colon).  ``--guidance-scales``
applies to *both* logP and QED signals (joint CFG).  Scales != 1.0 only
make sense if the checkpoint was trained with non-zero
``logp_dropout_prob`` / ``qed_dropout_prob`` — otherwise the model never saw
the unconditional branch and CFG amplification interpolates against an
untrained null token.

Outputs (under ``--output-dir``):

* ``logp_qed_cfg_results.pt``  — full per-cell dict (target, scale → summary).
* ``logp_qed_cfg_results.json`` — slim summary (one row per cell) with all
  training metrics inlined as ``metric/<name>``.
* ``logp_qed_natoms_data.csv`` — flat per-molecule rows
  (``guidance_scale, target_logp, target_qed, logp, qed, n_atoms, is_valid,
  smiles``) for downstream plotting.
* ``logp_qed_cfg_distributions_logp.png`` /
  ``logp_qed_cfg_distributions_qed.png`` — histogram grids over generated
  logP / QED, one row per target tuple, one column per guidance scale.
* ``logp_qed_scatter.png`` — generated logP vs QED scatter, colour-coded
  by validity, with each target tuple drawn as a red ✕.
* ``trajectories/scale=...,logp=...,qed=....pt`` — full trajectories per cell
  (skip with ``--no-save-trajectories``).
"""

from __future__ import annotations

import argparse
import csv as _csv
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
from rdkit.Chem.QED import qed as _rdkit_qed

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


def _parse_target_pairs(raw: str) -> list[tuple[float, float]]:
    """Parse ``"logp:qed,logp:qed,..."`` into a list of ``(logp, qed)``."""
    pairs: list[tuple[float, float]] = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise ValueError(
                f"Bad --targets entry {chunk!r}: expected ``logp:qed``."
            )
        lp_str, qed_str = chunk.split(":", 1)
        lp = float(lp_str.strip())
        qed = float(qed_str.strip())
        if not (0.0 <= qed <= 1.0):
            raise ValueError(
                f"QED target {qed} for entry {chunk!r} is outside [0, 1]."
            )
        pairs.append((lp, qed))
    if not pairs:
        raise ValueError("--targets is empty")
    return pairs


# ---------------------------------------------------------------------------
# Per-molecule property extraction (RDKit Crippen logP + QED + heavy-atom
# count, the same metrics used at training time so target / realised are
# commensurable).
# ---------------------------------------------------------------------------


def _final_mol(traj):
    if isinstance(traj, list) and traj:
        return traj[-1]
    return traj


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


def _rdkit_logp(mol) -> float | None:
    if mol is None:
        return None
    try:
        return float(Crippen.MolLogP(mol))
    except Exception:
        return None


def _rdkit_qed_safe(mol) -> float | None:
    if mol is None:
        return None
    try:
        return float(_rdkit_qed(mol))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Result aggregation
# ---------------------------------------------------------------------------


def _collect_from_predict_output(pred, vocab):
    """Extract per-cell aggregated lists from a single predict-step output.

    Returns
    -------
    dict with keys (each list is full-coverage; ``logps`` / ``qeds`` /
    ``natoms`` / ``smiles`` are paired index-wise per validity-bucket — only
    molecules where all four are recoverable contribute):

        valid_logps, valid_qeds, valid_natoms, valid_smiles,
        invalid_logps, invalid_qeds, invalid_natoms, invalid_smiles,
        valid_trajs, invalid_trajs,
        valid_rdkit, invalid_rdkit
    """
    out = {
        "valid_logps": [], "valid_qeds": [], "valid_natoms": [], "valid_smiles": [],
        "invalid_logps": [], "invalid_qeds": [], "invalid_natoms": [], "invalid_smiles": [],
        "valid_trajs": [], "invalid_trajs": [],
        "valid_rdkit": [], "invalid_rdkit": [],
    }

    if pred is None or not isinstance(pred, dict):
        return out

    atom_tok = vocab.atom_tokens
    edge_tok = vocab.edge_tokens
    charge_tok = vocab.charge_tokens

    def _process(trajs, kind: str) -> None:
        for traj in trajs or []:
            final = _final_mol(traj)
            try:
                rd = final.to_rdkit_mol(atom_tok, edge_tok, charge_tok)
            except Exception:
                rd = None
            out[f"{kind}_rdkit"].append(rd)
            out[f"{kind}_trajs"].append(traj)
            n = _mol_n_atoms(final)
            lp = _rdkit_logp(rd)
            q = _rdkit_qed_safe(rd)
            if lp is not None and q is not None and n is not None:
                out[f"{kind}_logps"].append(lp)
                out[f"{kind}_qeds"].append(q)
                out[f"{kind}_natoms"].append(int(n))
                try:
                    out[f"{kind}_smiles"].append(
                        Chem.MolToSmiles(rd) if rd is not None else ""
                    )
                except Exception:
                    out[f"{kind}_smiles"].append("")

    _process(pred.get("valid_mols"), "valid")
    _process(pred.get("invalid_mols"), "invalid")
    return out


def _stats_1d(values: list[float], target: float) -> dict:
    if not values:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "median": float("nan"),
            "mae": float("nan"),
            "n_molecules": 0,
        }
    t = torch.tensor(values, dtype=torch.float)
    diff = (t - target).abs()
    return {
        "mean": t.mean().item(),
        "std": t.std().item() if t.numel() > 1 else 0.0,
        "median": t.median().item(),
        "mae": diff.mean().item(),
        "n_molecules": int(t.numel()),
    }


def _pair_stats(
    logps: list[float],
    qeds: list[float],
    target_logp: float,
    target_qed: float,
) -> dict:
    """Joint stats — index-wise pairing of logP and QED is assumed."""
    s_lp = _stats_1d(logps, target_logp)
    s_q = _stats_1d(qeds, target_qed)
    out = {f"logp_{k}": v for k, v in s_lp.items()}
    out.update({f"qed_{k}": v for k, v in s_q.items()})
    if logps and qeds and len(logps) == len(qeds):
        lp = torch.tensor(logps, dtype=torch.float)
        q = torch.tensor(qeds, dtype=torch.float)
        d_lp = (lp - target_logp).abs()
        d_q = (q - target_qed).abs()
        out["within_logp0p5_qed0p1_rate"] = (
            ((d_lp <= 0.5) & (d_q <= 0.1)).float().mean().item()
        )
        out["within_logp1p0_qed0p2_rate"] = (
            ((d_lp <= 1.0) & (d_q <= 0.2)).float().mean().item()
        )
    else:
        out["within_logp0p5_qed0p1_rate"] = 0.0
        out["within_logp1p0_qed0p2_rate"] = 0.0
    return out


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _plot_marginal_grid(
    all_results: dict,
    output_path: Path,
    value_key: str,
    target_key: str,
    axis_label: str,
    title: str,
    n_bins: int = 30,
    fallback_range: tuple[float, float] = (-4.0, 8.0),
) -> None:
    if not all_results:
        return

    guidance_scales = sorted({key[0] for key in all_results})
    target_pairs = sorted({(key[1], key[2]) for key in all_results})

    flat: list[float] = []
    targets_for_axis: list[float] = []
    for (lp, qed) in target_pairs:
        targets_for_axis.append(lp if target_key == "target_logp" else qed)
    for result in all_results.values():
        flat.extend(result.get(value_key, []) or [])
    if not flat:
        x_lo, x_hi = fallback_range
    else:
        x_lo = min(flat + targets_for_axis) - 0.5
        x_hi = max(flat + targets_for_axis) + 0.5
        if x_hi - x_lo < 1.0:
            x_hi = x_lo + 1.0
    bins = np.linspace(x_lo, x_hi, n_bins + 1)

    n_rows = len(target_pairs)
    n_cols = len(guidance_scales)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.4 * max(1, n_cols), 2.8 * max(1, n_rows)),
        squeeze=False,
    )

    for row_idx, (lp_target, qed_target) in enumerate(target_pairs):
        target_value = lp_target if target_key == "target_logp" else qed_target
        for col_idx, scale in enumerate(guidance_scales):
            ax = axes[row_idx][col_idx]
            result = all_results.get((scale, lp_target, qed_target), {})
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
                    0.5, 0.5, "no data",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=9, color="gray",
                )

            ax.axvline(
                target_value,
                color="#D62728",
                linestyle="--",
                linewidth=1.3,
            )

            n_mols = len(values)
            validity = result.get("validity_rate")
            tm = result.get("training_metrics", {}) or {}
            novelty = tm.get("novelty")
            stats = result.get("valid_stats", {}) or {}
            mae_key = f"{'logp' if target_key == 'target_logp' else 'qed'}_mae"
            mae = stats.get(mae_key)
            subtitle_bits = [f"n={n_mols}"]
            if validity is not None and result.get("n_total", 0) > 0:
                subtitle_bits.append(f"valid={validity:.0%}")
            if mae is not None and not math.isnan(float(mae)):
                subtitle_bits.append(f"mae={mae:.2f}")
            if novelty is not None and not math.isnan(float(novelty)):
                subtitle_bits.append(f"novel={novelty:.0%}")
            ax.text(
                0.98, 0.95,
                " ".join(subtitle_bits),
                transform=ax.transAxes,
                ha="right", va="top",
                fontsize=8, color="#222",
            )

            if row_idx == 0:
                ax.set_title(f"scale={scale}", fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(
                    f"target=(lp={lp_target:g}, qed={qed_target:g})\ncount",
                    fontsize=8,
                )
            if row_idx == n_rows - 1:
                ax.set_xlabel(axis_label, fontsize=9)

            ax.set_xlim(x_lo, x_hi)
            ax.tick_params(labelsize=8)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _plot_logp_qed_scatter(
    all_results: dict,
    output_path: Path,
    title: str = "Generated logP vs QED",
) -> None:
    """Scatter grid: rows=target tuple, cols=guidance scales.

    Valid molecules are blue, invalid are light grey.  The target tuple is
    drawn as a red ✕ with a vertical (logP) and horizontal (QED) reference
    line so over/undershoot is visually obvious.
    """
    if not all_results:
        return

    guidance_scales = sorted({key[0] for key in all_results})
    target_pairs = sorted({(key[1], key[2]) for key in all_results})

    flat_lp: list[float] = []
    flat_q: list[float] = []
    for result in all_results.values():
        flat_lp.extend(result.get("valid_logps", []) or [])
        flat_lp.extend(result.get("invalid_logps", []) or [])
        flat_q.extend(result.get("valid_qeds", []) or [])
        flat_q.extend(result.get("invalid_qeds", []) or [])
    target_lps = [p[0] for p in target_pairs]
    target_qs = [p[1] for p in target_pairs]

    if flat_lp:
        x_lo = min(flat_lp + target_lps) - 0.5
        x_hi = max(flat_lp + target_lps) + 0.5
        if x_hi - x_lo < 1.0:
            x_hi = x_lo + 1.0
    else:
        x_lo, x_hi = -4.0, 8.0
    if flat_q:
        y_lo = max(0.0, min(flat_q + target_qs) - 0.05)
        y_hi = min(1.05, max(flat_q + target_qs) + 0.05)
        if y_hi - y_lo < 0.1:
            y_hi = y_lo + 0.1
    else:
        y_lo, y_hi = 0.0, 1.0

    n_rows = len(target_pairs)
    n_cols = len(guidance_scales)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.6 * max(1, n_cols), 3.0 * max(1, n_rows)),
        squeeze=False,
    )

    legend_handles: list = []
    for row_idx, (lp_target, qed_target) in enumerate(target_pairs):
        for col_idx, scale in enumerate(guidance_scales):
            ax = axes[row_idx][col_idx]
            result = all_results.get((scale, lp_target, qed_target), {})
            v_lp = list(result.get("valid_logps", []) or [])
            v_q = list(result.get("valid_qeds", []) or [])
            i_lp = list(result.get("invalid_logps", []) or [])
            i_q = list(result.get("invalid_qeds", []) or [])

            valid_paired = len(v_lp) > 0 and len(v_lp) == len(v_q)
            invalid_paired = len(i_lp) > 0 and len(i_lp) == len(i_q)

            if invalid_paired:
                inv_handle = ax.scatter(
                    i_lp, i_q,
                    s=14, c="#BBBBBB", alpha=0.55, edgecolors="none",
                    label="invalid",
                )
                if all(h.get_label() != "invalid" for h in legend_handles):
                    legend_handles.append(inv_handle)
            if valid_paired:
                val_handle = ax.scatter(
                    v_lp, v_q,
                    s=18, c="#4C72B0", alpha=0.85,
                    edgecolors="black", linewidths=0.3,
                    label="valid",
                )
                if all(h.get_label() != "valid" for h in legend_handles):
                    legend_handles.append(val_handle)
            if not valid_paired and not invalid_paired:
                ax.text(
                    0.5, 0.5, "no data",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=9, color="gray",
                )

            ax.axvline(lp_target, color="#D62728", linestyle="--", linewidth=1.0)
            ax.axhline(qed_target, color="#D62728", linestyle="--", linewidth=1.0)
            ax.scatter(
                [lp_target], [qed_target],
                marker="x", s=80, c="#D62728", linewidths=2.0, label="target",
            )

            n_v = len(v_lp)
            n_i = len(i_lp)
            stats = result.get("valid_stats", {}) or {}
            mae_lp = stats.get("logp_mae")
            mae_q = stats.get("qed_mae")
            subtitle_bits = [f"valid n={n_v}", f"inv n={n_i}"]
            if mae_lp is not None and not math.isnan(float(mae_lp)):
                subtitle_bits.append(f"mae(lp)={mae_lp:.2f}")
            if mae_q is not None and not math.isnan(float(mae_q)):
                subtitle_bits.append(f"mae(qed)={mae_q:.2f}")
            ax.text(
                0.98, 0.97,
                "\n".join(subtitle_bits),
                transform=ax.transAxes,
                ha="right", va="top",
                fontsize=7, color="#222",
            )

            if row_idx == 0:
                ax.set_title(f"scale={scale}", fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(
                    f"target=(lp={lp_target:g}, qed={qed_target:g})\nQED",
                    fontsize=8,
                )
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
            fontsize=9, frameon=False,
            bbox_to_anchor=(0.99, 0.995),
        )
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 0.96, 0.96])
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Eval-component setup (mirrors eval_logp_cfg.py)
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
# Persistence helpers
# ---------------------------------------------------------------------------


_CSV_FIELDS = (
    "guidance_scale",
    "target_logp",
    "target_qed",
    "logp",
    "qed",
    "n_atoms",
    "is_valid",
    "smiles",
)


def _save_per_molecule_csv(all_results: dict, output_path: Path) -> int:
    """One row per generated molecule.

    Pairing is index-wise: a row is written only when logp / qed / n_atoms
    are all recoverable for that molecule.  Cells with broken pairing
    (length mismatch — should not happen for fresh runs) are skipped.
    """
    rows_written = 0
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = _csv.writer(handle)
        writer.writerow(_CSV_FIELDS)

        for (scale, lp_target, q_target), result in sorted(all_results.items()):
            for kind, valid_flag in (("valid", 1), ("invalid", 0)):
                lps = list(result.get(f"{kind}_logps", []) or [])
                qs = list(result.get(f"{kind}_qeds", []) or [])
                ns = list(result.get(f"{kind}_natoms", []) or [])
                smis = list(result.get(f"{kind}_smiles", []) or [])
                if not lps:
                    continue
                if not (len(lps) == len(qs) == len(ns)):
                    print(
                        f"[csv] skipping cell scale={scale} "
                        f"target=(lp={lp_target}, qed={q_target}) {kind}: "
                        f"length mismatch (lp={len(lps)} qed={len(qs)} "
                        f"natoms={len(ns)})"
                    )
                    continue
                # Smiles list is best-effort; pad if short.
                if len(smis) < len(lps):
                    smis = smis + [""] * (len(lps) - len(smis))
                for lp, q, n, s in zip(lps, qs, ns, smis):
                    writer.writerow(
                        [scale, lp_target, q_target, lp, q, int(n), valid_flag, s]
                    )
                    rows_written += 1
    return rows_written


_DROP_KEYS_FOR_JSON = {
    "valid_logps", "valid_qeds", "valid_natoms", "valid_smiles",
    "invalid_logps", "invalid_qeds", "invalid_natoms", "invalid_smiles",
    "all_logps", "all_qeds", "all_natoms",
}


def _results_for_json(all_results: dict) -> list[dict]:
    rows: list[dict] = []
    for (scale, lp_target, q_target), result in sorted(all_results.items()):
        row = {k: v for k, v in result.items() if k not in _DROP_KEYS_FOR_JSON}
        training_metrics = row.pop("training_metrics", None)
        if training_metrics:
            for k, v in training_metrics.items():
                row[f"metric/{k}"] = v
        if "per_seed" in row:
            row["per_seed"] = [
                {k: v for k, v in ps.items() if k not in _DROP_KEYS_FOR_JSON}
                for ps in row["per_seed"]
            ]
        row["guidance_scale"] = scale
        row["target_logp"] = lp_target
        row["target_qed"] = q_target
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate joint logP × QED CFG steering on QM9.",
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
        default=Path("eval_outputs/logp_qed_cfg"),
        help="Directory to save evaluation artifacts.",
    )
    parser.add_argument(
        "--targets",
        type=str,
        default="-2:0.3,2:0.5,4:0.7,5:0.85",
        help=(
            "Comma-separated ``logp:qed`` target tuples; the model is "
            "always given both values at the same time."
        ),
    )
    parser.add_argument(
        "--guidance-scales",
        type=str,
        default="1.0",
        help=(
            "Comma-separated guidance scales applied to *both* logP and QED. "
            "Scales other than 1.0 only make sense if the checkpoint was "
            "trained with non-zero logp_dropout_prob / qed_dropout_prob."
        ),
    )
    parser.add_argument(
        "--n-mols",
        type=int,
        default=500,
        help="Approximate molecules per (scale, target tuple) cell.",
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
            "Comma-separated list of seeds.  When set, each cell is run "
            "once per seed and aggregated."
        ),
    )
    parser.add_argument(
        "--overrides",
        nargs="*",
        default=["data=qm9", "model=semla", "cfg=logp_qed"],
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
    qed_signal = guidance.get_signal("qed")
    if logp_signal is None or qed_signal is None:
        raise RuntimeError(
            "Loaded model does not expose joint logP+QED CFG conditioning. "
            "Check that --overrides includes `cfg=logp_qed` and that the "
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

    target_pairs = _parse_target_pairs(args.targets)
    guidance_scales = _parse_float_list(args.guidance_scales)
    module.predict_return_traj = True

    # Force every other CFG signal off; only logP + QED are exercised here.
    original_scales = {s.name: s.guidance_scale for s in guidance.signals}
    all_results: dict[tuple[float, float, float], dict] = {}

    try:
        for s in guidance.signals:
            if s.name not in ("logp", "qed"):
                s.guidance_scale = 0.0

        for scale in guidance_scales:
            logp_signal.guidance_scale = float(scale)
            qed_signal.guidance_scale = float(scale)
            print(
                f"\n{'=' * 60}\n"
                f"joint logP+QED guidance scale={scale}\n"
                f"{'=' * 60}"
            )

            for lp_target, q_target in target_pairs:
                module.predict_overrides = {
                    "logp": float(lp_target),
                    "qed": float(q_target),
                }

                per_seed: list[dict] = []
                # Aggregated across seeds (and across batches within a seed).
                agg = {
                    "valid_logps": [], "valid_qeds": [], "valid_natoms": [], "valid_smiles": [],
                    "invalid_logps": [], "invalid_qeds": [], "invalid_natoms": [], "invalid_smiles": [],
                    "valid_trajs": [], "invalid_trajs": [],
                    "valid_rdkit": [], "invalid_rdkit": [],
                }

                for seed in seeds:
                    pl.seed_everything(int(seed))
                    predictions = trainer.predict(module, dataloaders=test_dl)

                    seed_buckets = {k: [] for k in agg}
                    for pred in predictions or []:
                        out = _collect_from_predict_output(pred, module.vocab)
                        for k in agg:
                            seed_buckets[k].extend(out[k])

                    for k in agg:
                        agg[k].extend(seed_buckets[k])

                    n_valid = len(seed_buckets["valid_logps"])
                    n_total = n_valid + len(seed_buckets["invalid_logps"])
                    per_seed.append(
                        {
                            "seed": int(seed),
                            "n_total": n_total,
                            "n_valid": n_valid,
                            "n_invalid": n_total - n_valid,
                            "validity_rate": (n_valid / n_total) if n_total else 0.0,
                            "valid_stats": _pair_stats(
                                seed_buckets["valid_logps"],
                                seed_buckets["valid_qeds"],
                                lp_target,
                                q_target,
                            ),
                            "all_stats": _pair_stats(
                                seed_buckets["valid_logps"]
                                + seed_buckets["invalid_logps"],
                                seed_buckets["valid_qeds"]
                                + seed_buckets["invalid_qeds"],
                                lp_target,
                                q_target,
                            ),
                            "valid_logps": list(seed_buckets["valid_logps"]),
                            "valid_qeds": list(seed_buckets["valid_qeds"]),
                            "valid_natoms": list(seed_buckets["valid_natoms"]),
                            "invalid_logps": list(seed_buckets["invalid_logps"]),
                            "invalid_qeds": list(seed_buckets["invalid_qeds"]),
                            "invalid_natoms": list(seed_buckets["invalid_natoms"]),
                        }
                    )

                # Cap at args.n_mols (paired prefixes — keep the index-wise
                # alignment between logp / qed / natoms / smiles intact).
                if (
                    len(agg["valid_logps"]) + len(agg["invalid_logps"])
                    > args.n_mols
                ):
                    keep_valid = min(len(agg["valid_logps"]), args.n_mols)
                    for k in (
                        "valid_logps", "valid_qeds", "valid_natoms",
                        "valid_smiles",
                    ):
                        agg[k] = agg[k][:keep_valid]
                    remaining = args.n_mols - keep_valid
                    for k in (
                        "invalid_logps", "invalid_qeds", "invalid_natoms",
                        "invalid_smiles",
                    ):
                        agg[k] = agg[k][:remaining]

                # ``trajs`` / ``rdkit`` are full-coverage so they may exceed
                # the paired-list lengths.  Cap them by their own size.
                if (
                    len(agg["valid_trajs"]) + len(agg["invalid_trajs"])
                    > args.n_mols
                ):
                    keep_valid = min(len(agg["valid_trajs"]), args.n_mols)
                    for k in ("valid_trajs", "valid_rdkit"):
                        agg[k] = agg[k][:keep_valid]
                    remaining = args.n_mols - keep_valid
                    for k in ("invalid_trajs", "invalid_rdkit"):
                        agg[k] = agg[k][:remaining]

                all_logps = agg["valid_logps"] + agg["invalid_logps"]
                all_qeds = agg["valid_qeds"] + agg["invalid_qeds"]
                all_natoms = agg["valid_natoms"] + agg["invalid_natoms"]
                n_total = len(all_logps)
                n_valid = len(agg["valid_logps"])
                summary = {
                    "target_logp": float(lp_target),
                    "target_qed": float(q_target),
                    "guidance_scale": float(scale),
                    "n_total": n_total,
                    "n_valid": n_valid,
                    "n_invalid": n_total - n_valid,
                    "validity_rate": (n_valid / n_total) if n_total else 0.0,
                    "valid_stats": _pair_stats(
                        agg["valid_logps"], agg["valid_qeds"],
                        lp_target, q_target,
                    ),
                    "all_stats": _pair_stats(
                        all_logps, all_qeds, lp_target, q_target,
                    ),
                    "valid_logps": list(agg["valid_logps"]),
                    "valid_qeds": list(agg["valid_qeds"]),
                    "valid_natoms": list(agg["valid_natoms"]),
                    "valid_smiles": list(agg["valid_smiles"]),
                    "invalid_logps": list(agg["invalid_logps"]),
                    "invalid_qeds": list(agg["invalid_qeds"]),
                    "invalid_natoms": list(agg["invalid_natoms"]),
                    "invalid_smiles": list(agg["invalid_smiles"]),
                    "all_logps": all_logps,
                    "all_qeds": all_qeds,
                    "all_natoms": all_natoms,
                    "seeds": [int(s) for s in seeds],
                    "per_seed": per_seed,
                }

                training_metrics = _compute_all_metrics(
                    rdkit_mols=agg["valid_rdkit"] + agg["invalid_rdkit"],
                    metrics=metrics,
                    stability_metrics=stability_metrics,
                    distribution_metrics=distribution_metrics,
                )
                summary["training_metrics"] = training_metrics
                all_results[(float(scale), float(lp_target), float(q_target))] = summary

                if summary["n_total"] == 0:
                    print(f"target=(lp={lp_target}, qed={q_target}): no molecules")
                else:
                    sv = summary["valid_stats"]
                    tm = training_metrics
                    print(
                        f"target=(lp={lp_target:g}, qed={q_target:g}): "
                        f"validity={summary['validity_rate']:.1%} "
                        f"({summary['n_valid']}/{summary['n_total']}), "
                        f"novelty={tm.get('novelty', float('nan')):.1%}, "
                        f"unique={tm.get('uniqueness', float('nan')):.1%}, "
                        f"mol-stab={tm.get('rdkit-molecule-stability', float('nan')):.1%}, "
                        f"mae(lp,valid)={sv['logp_mae']:.2f}, "
                        f"mae(qed,valid)={sv['qed_mae']:.3f}, "
                        f"mean(lp,valid)={sv['logp_mean']:.2f}, "
                        f"mean(qed,valid)={sv['qed_mean']:.3f}"
                    )

                if not args.no_save_trajectories:
                    traj_path = (
                        trajectories_dir
                        / f"scale={scale}_logp={lp_target}_qed={q_target}.pt"
                    )
                    torch.save(
                        {
                            "guidance_scale": float(scale),
                            "target_logp": float(lp_target),
                            "target_qed": float(q_target),
                            "valid_trajectories": agg["valid_trajs"],
                            "invalid_trajectories": agg["invalid_trajs"],
                            "valid_logps": list(agg["valid_logps"]),
                            "valid_qeds": list(agg["valid_qeds"]),
                            "valid_natoms": list(agg["valid_natoms"]),
                            "invalid_logps": list(agg["invalid_logps"]),
                            "invalid_qeds": list(agg["invalid_qeds"]),
                            "invalid_natoms": list(agg["invalid_natoms"]),
                            "valid_smiles": list(agg["valid_smiles"]),
                            "invalid_smiles": list(agg["invalid_smiles"]),
                        },
                        traj_path,
                    )
    finally:
        for s in guidance.signals:
            s.guidance_scale = original_scales[s.name]
        module.predict_overrides = None

    pt_path = args.output_dir / "logp_qed_cfg_results.pt"
    json_path = args.output_dir / "logp_qed_cfg_results.json"
    plot_logp_path = args.output_dir / "logp_qed_cfg_distributions_logp.png"
    plot_qed_path = args.output_dir / "logp_qed_cfg_distributions_qed.png"
    scatter_path = args.output_dir / "logp_qed_scatter.png"
    csv_path = args.output_dir / "logp_qed_natoms_data.csv"

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
    _plot_marginal_grid(
        all_results,
        plot_logp_path,
        value_key="all_logps",
        target_key="target_logp",
        axis_label="RDKit Crippen logP",
        title="Generated logP distributions (all generations)",
        fallback_range=(-4.0, 8.0),
    )
    _plot_marginal_grid(
        all_results,
        plot_qed_path,
        value_key="all_qeds",
        target_key="target_qed",
        axis_label="RDKit QED",
        title="Generated QED distributions (all generations)",
        fallback_range=(0.0, 1.0),
    )
    _plot_logp_qed_scatter(
        all_results,
        scatter_path,
        title="Generated logP vs QED (rows=target tuple, cols=guidance scale)",
    )
    n_csv_rows = _save_per_molecule_csv(all_results, csv_path)

    print(f"\nSaved tensor results to:        {pt_path}")
    print(f"Saved JSON summary to:          {json_path}")
    print(f"Saved logP distribution plot:   {plot_logp_path}")
    print(f"Saved QED distribution plot:    {plot_qed_path}")
    print(f"Saved logP×QED scatter to:     {scatter_path}")
    print(f"Saved per-molecule CSV to:      {csv_path} ({n_csv_rows} rows)")
    if not args.no_save_trajectories:
        print(f"Saved trajectories under:       {trajectories_dir}")


if __name__ == "__main__":
    main()
