#!/usr/bin/env python3
"""Representation-aware full evaluation of a ChemFlow checkpoint.

This is a close refactor of ``eval_full_metrics.py`` for the representation
split. It keeps the old multi-seed/evaluate/aggregate flow, but mirrors the
current ``run.py`` setup:

* validates ``cfg.representation`` against the dataset capabilities,
* projects token priors for pointcloud / charged-pointcloud runs,
* passes ``representation`` and tensor ``batch_metrics`` into the module,
* computes RDKit metrics only when topology is part of the representation,
* saves trajectories in a notebook-friendly format.

Examples
--------
    python eval_scripts/eval_full_metrics_new.py \
        --run-dir outputs/qm9/uncond/2026-05-13/zbnp6nvq \
        --output-dir eval_outputs/zbnp6nvq \
        --n-mols 256 \
        --predict-batch-size 64

    python eval_scripts/eval_full_metrics_new.py \
        --checkpoint outputs/qm9/uncond/2026-05-13/q0vvoie3/every_n_epochs/epoch=91-step=8924.ckpt \
        --overrides data=qm9 model=semla cfg=uncond representation=molecule
"""

from __future__ import annotations

import argparse
import json
import math
import os
from copy import deepcopy
from pathlib import Path

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from rdkit import RDLogger

from chemflow.dataset.molecule_data import MoleculeBatch
from chemflow.dataset.representation import (
    Representation,
    project_distributions_to_representation,
    validate_representation,
)
from chemflow.dataset.vocab import setup_token_weights
from chemflow.utils.metrics import (
    calc_atom_stabilities,
    calc_posebusters_metrics,
    init_metrics,
)
from chemflow.utils.utils import init_uniform_prior


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"
os.environ.setdefault("PROJECT_ROOT", str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Hydra / config plumbing
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
    """Instantiate eval components using the representation-aware ``run.py`` path."""
    OmegaConf.set_struct(cfg, False)
    cfg.data.datamodule.batch_size.test = int(predict_batch_size)

    representation = Representation(cfg.representation)
    dataset_cls = hydra.utils.get_class(cfg.data.datamodule.datasets.train._target_)
    validate_representation(dataset_cls.CAPABILITIES, representation)

    preprocessing = hydra.utils.instantiate(cfg.data.preprocessing)
    vocab = preprocessing.vocab
    distributions = preprocessing.distributions

    loss_weight_distributions = deepcopy(distributions)
    token_prior_distribution = init_uniform_prior(distributions)
    token_prior_distribution = project_distributions_to_representation(
        token_prior_distribution, vocab, representation
    )

    cfg.data.vocab = vocab

    datamodule = hydra.utils.instantiate(
        cfg.data.datamodule,
        _recursive_=False,
        vocab=vocab,
        distributions=token_prior_distribution,
    )
    datamodule.setup()

    allow_charged = bool(cfg.data.get("allow_charged", False))
    base_dataset = datamodule.train_dataset.base_dataset
    train_smiles = (
        base_dataset.get_all_smiles()
        if representation.requires_topology and hasattr(base_dataset, "get_all_smiles")
        else None
    )

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
        batch_metrics=batch_metrics,
        allow_charged=allow_charged,
        representation=representation.value,
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
        batch_metrics,
        representation,
    )


def _load_checkpoint(module, checkpoint_path: Path) -> None:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    clean_state_dict = {k.replace("._orig_mod", ""): v for k, v in state_dict.items()}
    module.load_state_dict(clean_state_dict)


def _find_checkpoint(run_dir: Path) -> Path:
    ckpts = sorted(run_dir.rglob("*.ckpt"), key=lambda p: p.stat().st_mtime)
    if not ckpts:
        raise FileNotFoundError(f"No .ckpt files found under {run_dir}")
    if len(ckpts) > 1:
        print(f"[warn] Found {len(ckpts)} checkpoints; using newest: {ckpts[-1]}")
    return ckpts[-1]


def _load_run_overrides(run_dir: Path) -> list[str]:
    overrides_path = run_dir / ".hydra" / "overrides.yaml"
    if not overrides_path.exists():
        raise FileNotFoundError(f"Missing Hydra overrides file: {overrides_path}")
    raw = OmegaConf.load(overrides_path)
    return [str(x) for x in raw]


# ---------------------------------------------------------------------------
# Per-batch n_atoms sampler, for checkpoints with an n_atoms CFG signal
# ---------------------------------------------------------------------------


class EmpiricalNAtomsSampler(pl.Callback):
    """Sample target n_atoms from the empirical distribution for each batch."""

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
        mol_t = batch[0] if isinstance(batch, (tuple, list)) else batch
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
        bs = self._batch_size(batch)
        idx = torch.multinomial(
            self.probs, num_samples=bs, replacement=True, generator=self.generator
        )
        pl_module.predict_overrides = {
            **(getattr(pl_module, "predict_overrides", None) or {}),
            "n_atoms": idx.to(dtype=torch.long, device=pl_module.device),
        }
        self.sampled_targets.extend(int(x) for x in idx.tolist())


# ---------------------------------------------------------------------------
# Trajectory / mol helpers
# ---------------------------------------------------------------------------


def _final_mol(traj):
    if isinstance(traj, list) and traj:
        return traj[-1]
    return traj


def _mol_n_atoms(mol) -> int | None:
    if mol is None:
        return None
    n = getattr(mol, "num_nodes", None)
    if n is not None:
        try:
            return int(n)
        except Exception:
            pass
    a = getattr(mol, "a", None)
    if a is not None:
        try:
            return int(a.shape[0])
        except Exception:
            pass
    return None


def _natoms_trajectories(trajectories) -> list[list[int]]:
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


def _trajectories_to_rdkit(trajectories, vocab) -> list:
    rdkit_mols: list = []
    for traj in trajectories:
        final = _final_mol(traj)
        if final is None:
            rdkit_mols.append(None)
            continue
        try:
            rdkit_mols.append(
                final.to_rdkit_mol(
                    vocab.atom_tokens, vocab.edge_tokens, vocab.charge_tokens
                )
            )
        except Exception:
            rdkit_mols.append(None)
    return rdkit_mols


def _collect_predictions(predictions) -> tuple[list, list[bool], list, list]:
    """Flatten predict outputs into all, validity mask, valid, invalid."""
    all_trajs: list = []
    is_valid: list[bool] = []
    valid_trajs: list = []
    invalid_trajs: list = []
    for pred in predictions or []:
        if not isinstance(pred, dict):
            continue
        valid = pred.get("valid_mols", []) or []
        invalid = pred.get("invalid_mols", []) or []
        valid_trajs.extend(valid)
        invalid_trajs.extend(invalid)
        all_trajs.extend(valid)
        is_valid.extend([True] * len(valid))
        all_trajs.extend(invalid)
        is_valid.extend([False] * len(invalid))
    return all_trajs, is_valid, valid_trajs, invalid_trajs


def _trim_outputs(
    all_trajs: list,
    is_valid_mask: list[bool],
    valid_trajs: list,
    invalid_trajs: list,
    n_mols: int,
) -> tuple[list, list[bool], list, list]:
    if len(all_trajs) <= n_mols:
        return all_trajs, is_valid_mask, valid_trajs, invalid_trajs

    all_trajs = all_trajs[:n_mols]
    is_valid_mask = is_valid_mask[:n_mols]
    valid_trajs = [t for t, v in zip(all_trajs, is_valid_mask) if v]
    invalid_trajs = [t for t, v in zip(all_trajs, is_valid_mask) if not v]
    return all_trajs, is_valid_mask, valid_trajs, invalid_trajs


def _save_natoms_trajectories(
    trajectories: list,
    is_valid_mask: list[bool],
    seed: int,
    natoms_traj_path: Path,
) -> None:
    natoms_per_traj = _natoms_trajectories(trajectories)
    lengths = {len(seq) for seq in natoms_per_traj}
    if len(lengths) == 1 and natoms_per_traj:
        payload = {
            "seed": int(seed),
            "n_atoms_trajectories": torch.tensor(natoms_per_traj, dtype=torch.long),
            "is_valid_final": torch.tensor(is_valid_mask, dtype=torch.bool),
            "shape_note": "rows = generations, cols = integration steps",
        }
    else:
        payload = {
            "seed": int(seed),
            "n_atoms_trajectories": natoms_per_traj,
            "is_valid_final": list(is_valid_mask),
            "shape_note": "ragged; one inner list per generation",
        }
    torch.save(payload, natoms_traj_path)


def _save_trajectories(
    path: Path,
    *,
    seed: int,
    all_trajs: list,
    is_valid_mask: list[bool],
    valid_trajs: list,
    invalid_trajs: list,
    representation: Representation,
) -> None:
    torch.save(
        {
            "seed": int(seed),
            "representation": representation.value,
            "all_trajectories": all_trajs,
            "is_valid_final": torch.tensor(is_valid_mask, dtype=torch.bool),
            "valid_trajectories": valid_trajs,
            "invalid_trajectories": invalid_trajs,
        },
        path,
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _compute_rdkit_metrics(
    rdkit_mols: list,
    metrics,
    stability_metrics,
    distribution_metrics,
    *,
    run_posebusters: bool,
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

    if run_posebusters:
        try:
            pb = calc_posebusters_metrics(rdkit_mols)
            for k, v in pb.items():
                results[f"posebusters/{k}"] = float(v)
        except Exception as e:
            print(f"PoseBusters failed: {e}")

    return {
        k: float(v) if isinstance(v, (int, float)) else v for k, v in results.items()
    }


def _compute_batch_metrics(trajectories: list, batch_metrics) -> dict:
    if batch_metrics is None or len(batch_metrics) == 0 or not trajectories:
        return {}

    final_mols = [_final_mol(t) for t in trajectories if _final_mol(t) is not None]
    if not final_mols:
        return {}

    batch = MoleculeBatch.from_data_list(final_mols)
    batch_metrics.reset()
    batch_metrics.update(batch)
    out = {}
    for k, v in batch_metrics.compute().items():
        out[f"batch/{k}"] = float(v.item() if isinstance(v, torch.Tensor) else v)
    batch_metrics.reset()
    return out


# ---------------------------------------------------------------------------
# Per-seed generation + metric computation
# ---------------------------------------------------------------------------


def _run_one_seed(
    *,
    seed: int,
    cfg,
    module,
    test_dl,
    distributions,
    vocab,
    metrics,
    stability_metrics,
    distribution_metrics,
    batch_metrics,
    representation: Representation,
    n_mols: int,
    predict_batch_size: int,
    precision: str | None,
    run_posebusters: bool,
    output_dir: Path,
    save_trajectories: bool,
    save_legacy_mol_files: bool,
    save_generic_legacy_mol_files: bool,
    natoms_traj_path: Path | None = None,
) -> tuple[dict, int, int]:
    pl.seed_everything(seed, workers=True)

    n_predict_batches = max(1, math.ceil(n_mols / max(1, predict_batch_size)))
    trainer_kwargs = dict(cfg.trainer.trainer)
    trainer_kwargs["strategy"] = "auto"
    trainer_kwargs["limit_predict_batches"] = n_predict_batches
    if precision is not None:
        trainer_kwargs["precision"] = precision

    callbacks: list = []
    if getattr(module, "cfg_guidance", None) is not None and module.cfg_guidance.has_signal(
        "n_atoms"
    ):
        callbacks.append(
            EmpiricalNAtomsSampler(
                n_atoms_distribution=distributions.n_atoms_distribution, seed=seed
            )
        )

    trainer = pl.Trainer(
        logger=False,
        callbacks=callbacks,
        enable_checkpointing=False,
        **trainer_kwargs,
    )

    module.predict_return_traj = True
    module.predict_overrides = None

    print(
        f"\n[seed={seed}] generating ~{n_mols} samples "
        f"({n_predict_batches} batches x {predict_batch_size})..."
    )
    predictions = trainer.predict(module, dataloaders=test_dl)

    all_trajs, is_valid_mask, valid_trajs, invalid_trajs = _collect_predictions(
        predictions
    )
    all_trajs, is_valid_mask, valid_trajs, invalid_trajs = _trim_outputs(
        all_trajs, is_valid_mask, valid_trajs, invalid_trajs, n_mols
    )

    if save_trajectories:
        traj_path = output_dir / f"trajectories_seed{seed}.pt"
        _save_trajectories(
            traj_path,
            seed=seed,
            all_trajs=all_trajs,
            is_valid_mask=is_valid_mask,
            valid_trajs=valid_trajs,
            invalid_trajs=invalid_trajs,
            representation=representation,
        )
        print(f"[seed={seed}] saved trajectories to: {traj_path}")

    if save_legacy_mol_files:
        torch.save(valid_trajs, output_dir / f"valid_mols_seed{seed}.pt")
        torch.save(invalid_trajs, output_dir / f"invalid_mols_seed{seed}.pt")
        if save_generic_legacy_mol_files:
            torch.save(valid_trajs, output_dir / "valid_mols.pt")
            torch.save(invalid_trajs, output_dir / "invalid_mols.pt")

    if natoms_traj_path is not None:
        _save_natoms_trajectories(all_trajs, is_valid_mask, seed, natoms_traj_path)
        print(f"[seed={seed}] saved n_atoms trajectories to: {natoms_traj_path}")

    seed_metrics: dict[str, float] = {}
    seed_metrics.update(_compute_batch_metrics(all_trajs, batch_metrics))

    if representation.requires_topology:
        rdkit_mols = _trajectories_to_rdkit(all_trajs, vocab)
        n_total = len(rdkit_mols)
        n_parseable = sum(1 for m in rdkit_mols if m is not None)

        print(f"[seed={seed}] computing RDKit metrics on {n_total} samples...")
        seed_metrics.update(
            _compute_rdkit_metrics(
                rdkit_mols=rdkit_mols,
                metrics=metrics,
                stability_metrics=stability_metrics,
                distribution_metrics=distribution_metrics,
                run_posebusters=run_posebusters,
            )
        )
        del rdkit_mols
    else:
        n_total = len(all_trajs)
        n_parseable = 0
        print(f"[seed={seed}] skipped RDKit metrics for {representation.value}.")

    seed_metrics["n_total"] = float(n_total)
    seed_metrics["n_valid_final"] = float(sum(1 for v in is_valid_mask if v))
    seed_metrics["n_rdkit_parseable"] = float(n_parseable)

    del all_trajs, valid_trajs, invalid_trajs, predictions
    return seed_metrics, n_total, n_parseable


# ---------------------------------------------------------------------------
# Aggregation: mean + sample std across seeds
# ---------------------------------------------------------------------------


def _aggregate_metrics(per_seed: list[dict]) -> dict[str, dict]:
    keys = sorted({k for d in per_seed for k in d.keys()})

    aggregated: dict[str, dict] = {}
    for key in keys:
        values = [d.get(key, float("nan")) for d in per_seed]
        arr = np.array(
            [v for v in values if isinstance(v, (int, float)) and not math.isnan(v)],
            dtype=np.float64,
        )
        if arr.size == 0:
            aggregated[key] = {
                "mean": float("nan"),
                "std": float("nan"),
                "per_seed": values,
                "n_seeds_used": 0,
            }
            continue

        mean = float(arr.mean())
        std = float(arr.std(ddof=1)) if arr.size > 1 else float("nan")

        aggregated[key] = {
            "mean": mean,
            "std": std,
            "per_seed": values,
            "n_seeds_used": int(arr.size),
        }
    return aggregated


def _print_summary_table(per_seed: list[dict], aggregated: dict[str, dict]) -> None:
    keys = sorted(aggregated.keys())
    seed_count = len(per_seed)
    header_seeds = "  ".join(f"seed{i + 1:>2}".rjust(12) for i in range(seed_count))
    print("\n" + "=" * 120)
    print("FINAL METRICS (per seed + mean + std across seeds)")
    print("=" * 120)
    print(f"{'metric':40s}  {header_seeds}  {'mean':>12s}  {'std':>12s}")
    print("-" * 120)
    for key in keys:
        agg = aggregated[key]
        per_seed_strs = []
        for d in per_seed:
            v = d.get(key, float("nan"))
            if isinstance(v, (int, float)) and not math.isnan(v):
                per_seed_strs.append(f"{v:12.4f}")
            else:
                per_seed_strs.append(f"{'n/a':>12s}")
        per_seed_part = "  ".join(per_seed_strs)
        mean = agg["mean"]
        std = agg["std"]
        mean_str = f"{mean:12.4f}" if not math.isnan(mean) else f"{'n/a':>12s}"
        std_str = f"{std:12.4f}" if not math.isnan(std) else f"{'n/a':>12s}"
        print(f"{key:40s}  {per_seed_part}  {mean_str}  {std_str}")
    print("=" * 120)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_seed_list(raw: str | None, n_seeds: int) -> list[int]:
    if raw:
        seeds = [int(x.strip()) for x in raw.split(",") if x.strip()]
        if len(seeds) != n_seeds:
            print(
                f"[warn] --seeds had {len(seeds)} entries but --n-seeds={n_seeds}; "
                f"using the provided list of {len(seeds)} seeds."
            )
        return seeds
    return [42 + 1000 * i for i in range(n_seeds)]


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run representation-aware ChemFlow generation metrics over one or "
            "more seeds, saving metrics and notebook-friendly trajectory files."
        )
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help=(
            "Hydra run directory containing .hydra/overrides.yaml and one or "
            "more checkpoints. If provided, --checkpoint and --overrides are "
            "inferred unless explicitly supplied."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to model checkpoint (.ckpt). Optional when --run-dir is set.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("eval_outputs/full_metrics_new"),
        help="Directory to save evaluation artifacts.",
    )
    parser.add_argument(
        "--n-mols",
        type=int,
        default=10_000,
        help="Approximate samples to generate per seed (default: 10000).",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=3,
        help="Number of seeds to repeat the evaluation over (default: 3).",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help=(
            "Comma-separated list of explicit seeds. If omitted, uses "
            "[42, 1042, 2042, ...] up to --n-seeds entries."
        ),
    )
    parser.add_argument(
        "--predict-batch-size",
        type=int,
        default=128,
        help="Predict dataloader batch size (default: 128).",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default=None,
        help=(
            "Optional Lightning precision override for prediction. If omitted, "
            "uses the precision from the composed trainer config."
        ),
    )
    parser.add_argument(
        "--no-posebusters",
        action="store_true",
        help="Disable PoseBusters checks. Always skipped for non-topology modes.",
    )
    parser.add_argument(
        "--no-save-trajectories",
        action="store_true",
        help="Skip saving full generated trajectories.",
    )
    parser.add_argument(
        "--no-save-legacy-mol-files",
        action="store_true",
        help=(
            "Skip valid_mols*.pt / invalid_mols*.pt files used by older notebooks."
        ),
    )
    parser.add_argument(
        "--no-save-natoms-trajectories",
        action="store_true",
        help=(
            "Skip saving per-step atom-count trajectories. By default, one .pt "
            "per seed is written to <output-dir>/natoms_traj_seed{seed}.pt."
        ),
    )
    parser.add_argument(
        "--overrides",
        nargs="*",
        default=None,
        help=(
            "Hydra overrides used to compose config. If omitted with --run-dir, "
            "loads .hydra/overrides.yaml. Otherwise defaults to "
            "data=qm9 model=semla cfg=uncond."
        ),
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    _register_resolvers()
    torch.set_float32_matmul_precision("medium")
    RDLogger.DisableLog("rdApp.*")

    checkpoint = args.checkpoint
    overrides = args.overrides
    if args.run_dir is not None:
        if checkpoint is None:
            checkpoint = _find_checkpoint(args.run_dir)
        if overrides is None:
            overrides = _load_run_overrides(args.run_dir)
    if checkpoint is None:
        raise ValueError("Provide --checkpoint or --run-dir.")
    if overrides is None:
        overrides = ["data=qm9", "model=semla", "cfg=uncond"]

    seeds = _parse_seed_list(args.seeds, args.n_seeds)
    print(f"Running evaluation with seeds: {seeds}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Overrides: {overrides}")

    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.1"):
        cfg = compose(config_name="default", overrides=overrides)

    (
        module,
        test_dl,
        distributions,
        vocab,
        metrics,
        stability_metrics,
        distribution_metrics,
        batch_metrics,
        representation,
    ) = _setup_eval_components(cfg, args.predict_batch_size)
    _load_checkpoint(module, checkpoint)

    per_seed: list[dict] = []
    per_seed_counts: list[tuple[int, int]] = []
    for seed in seeds:
        natoms_traj_path = (
            None
            if args.no_save_natoms_trajectories
            else args.output_dir / f"natoms_traj_seed{seed}.pt"
        )
        seed_metrics, n_total, n_parseable = _run_one_seed(
            seed=seed,
            cfg=cfg,
            module=module,
            test_dl=test_dl,
            distributions=distributions,
            vocab=vocab,
            metrics=metrics,
            stability_metrics=stability_metrics,
            distribution_metrics=distribution_metrics,
            batch_metrics=batch_metrics,
            representation=representation,
            n_mols=args.n_mols,
            predict_batch_size=args.predict_batch_size,
            precision=args.precision,
            run_posebusters=(
                (not args.no_posebusters) and representation.requires_topology
            ),
            output_dir=args.output_dir,
            save_trajectories=not args.no_save_trajectories,
            save_legacy_mol_files=not args.no_save_legacy_mol_files,
            save_generic_legacy_mol_files=seed == seeds[0],
            natoms_traj_path=natoms_traj_path,
        )
        per_seed.append(seed_metrics)
        per_seed_counts.append((n_total, n_parseable))

        seed_path = args.output_dir / f"metrics_seed{seed}.json"
        with seed_path.open("w", encoding="utf-8") as f:
            json.dump(seed_metrics, f, indent=2)

    aggregated = _aggregate_metrics(per_seed)
    _print_summary_table(per_seed, aggregated)

    summary = {
        "checkpoint": str(checkpoint),
        "run_dir": str(args.run_dir) if args.run_dir is not None else None,
        "representation": representation.value,
        "n_mols_requested": args.n_mols,
        "predict_batch_size": args.predict_batch_size,
        "precision": args.precision,
        "seeds": seeds,
        "n_total_per_seed": [t for t, _ in per_seed_counts],
        "n_rdkit_parseable_per_seed": [v for _, v in per_seed_counts],
        "per_seed_metrics": {str(s): m for s, m in zip(seeds, per_seed)},
        "aggregated": aggregated,
        "overrides": list(overrides),
    }

    json_path = args.output_dir / "results.json"
    pt_path = args.output_dir / "results.pt"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    torch.save(summary, pt_path)

    print(f"\nSaved per-seed + aggregated JSON to: {json_path}")
    print(f"Saved per-seed + aggregated tensor file to: {pt_path}")


if __name__ == "__main__":
    main()
