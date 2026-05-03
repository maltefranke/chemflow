#!/usr/bin/env python3
"""Multi-seed evaluation of a pretrained ChemFlow checkpoint.

For each of ``--n-seeds`` random seeds (default 3), this script

1. (Re)seeds Lightning / NumPy / PyTorch deterministically,
2. Generates ``--n-mols`` molecules (default 10000) from the loaded
   checkpoint,
3. Computes the full training-time metric stack
   (validity, uniqueness, novelty, energy / strain, optimisation RMSD,
   atom / molecule stability, n_atoms / atom-type / edge-type / charge KL),
   plus PoseBusters checks.

It then aggregates the per-seed metrics and reports

* the **per-seed** scores (one row per seed),
* the **mean across seeds**,
* the **sample standard deviation** across seeds (``ddof=1``; undefined when
  only one seed contributes a finite value for a metric).

It deliberately does **not** invoke ``chemflow.utils.diagnostics`` (the
self-correction trajectory analysis): only the standard generative
metrics are produced.

Works for both unconditional models and n-atoms-conditional models. When
the loaded checkpoint exposes the n_atoms CFG branch, the per-batch
target atom counts are sampled from the empirical training n_atoms
distribution (mirroring ``eval_natoms_general.py``); otherwise the model
falls back to its built-in prior.

Example
-------
    python eval_scripts/eval_full_metrics.py \
        --checkpoint path/to/model.ckpt \
        --output-dir eval_outputs/full_metrics \
        --n-mols 10000 \
        --n-seeds 3 \
        --seeds 42,123,2026 \
        --predict-batch-size 128
"""

from __future__ import annotations

import argparse
import json
import math
from copy import deepcopy
from pathlib import Path

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from rdkit import RDLogger

from chemflow.dataset.vocab import setup_token_weights
from chemflow.utils.metrics import (
    calc_atom_stabilities,
    calc_posebusters_metrics,
    init_metrics,
)
from chemflow.utils.utils import init_uniform_prior


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"


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
    clean_state_dict = {}
    for k, v in state_dict.items():
        k = k.replace("._orig_mod", "")
        # Older checkpoints stored KL dist metrics under "metrics.*" instead of "distribution_metrics.*"
        if k.startswith("metrics.") and "_dist_kl" in k:
            k = "distribution_metrics." + k[len("metrics."):]
        clean_state_dict[k] = v
    module.load_state_dict(clean_state_dict, strict=False)


# ---------------------------------------------------------------------------
# Per-batch n_atoms sampler (used when the model has natoms CFG)
# ---------------------------------------------------------------------------


class EmpiricalNAtomsSampler(pl.Callback):
    """Sample target n_atoms from an empirical distribution for each predict batch."""

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
        pl_module.predict_target_n_atoms_override = idx.to(
            dtype=torch.long, device=pl_module.device
        )
        pl_module.predict_target_mw_override = None
        self.sampled_targets.extend(int(x) for x in idx.tolist())


# ---------------------------------------------------------------------------
# Trajectory / mol helpers
# ---------------------------------------------------------------------------


def _final_mol(traj):
    if isinstance(traj, list) and traj:
        return traj[-1]
    return traj


def _mol_n_atoms(mol) -> int | None:
    """Pull atom count off a single ``MoleculeData``-like object."""
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
    """For every generation, return the per-step atom count.

    Returns a ``list[list[int]]`` of length ``len(trajectories)``. Each inner
    list is the atom count at every integration step; ``-1`` is used for
    frames whose atom count cannot be inferred (should not happen in
    practice but kept defensive).
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


def _collect_predictions(predictions) -> tuple[list, list[bool]]:
    """Flatten ``predict_step`` outputs into ``(trajectories, is_valid_mask)``.

    The two lists are aligned: ``is_valid_mask[i]`` is whether the final
    frame of ``trajectories[i]`` was deemed a valid molecule by the
    LightningModule's predict_step (which uses ``mol_is_valid``).
    """
    all_trajs: list = []
    is_valid: list[bool] = []
    for pred in predictions or []:
        if not isinstance(pred, dict):
            continue
        valid = pred.get("valid_mols", []) or []
        invalid = pred.get("invalid_mols", []) or []
        all_trajs.extend(valid)
        is_valid.extend([True] * len(valid))
        all_trajs.extend(invalid)
        is_valid.extend([False] * len(invalid))
    return all_trajs, is_valid


def _compute_all_metrics(
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
    n_mols: int,
    predict_batch_size: int,
    run_posebusters: bool,
    natoms_traj_path: Path | None = None,
    traj_dir: Path | None = None,
    traj_n_max: int | None = None,
) -> tuple[dict, int, int]:
    """Generate molecules with ``seed`` and return (metrics, n_total, n_valid).

    If ``natoms_traj_path`` is provided, also saves the per-step atom-count
    trajectory of every generation to that path (``.pt``).
    """
    pl.seed_everything(seed, workers=True)

    n_predict_batches = max(1, math.ceil(n_mols / max(1, predict_batch_size)))
    trainer_kwargs = dict(cfg.trainer.trainer)
    trainer_kwargs["strategy"] = "auto"
    trainer_kwargs["limit_predict_batches"] = n_predict_batches

    callbacks: list = []
    adapter = module.cfg_adapter
    if adapter._has_natoms_cfg:
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
    module.predict_target_n_atoms_override = None
    module.predict_target_mw_override = None

    print(
        f"\n[seed={seed}] generating ~{n_mols} molecules "
        f"({n_predict_batches} batches × {predict_batch_size})..."
    )
    predictions = trainer.predict(module, dataloaders=test_dl)

    all_trajs, is_valid_mask = _collect_predictions(predictions)
    if len(all_trajs) > n_mols:
        all_trajs = all_trajs[:n_mols]
        is_valid_mask = is_valid_mask[:n_mols]

    if natoms_traj_path is not None:
        natoms_per_traj = _natoms_trajectories(all_trajs)
        # If every trajectory has the same length we can stack into a tensor;
        # otherwise we fall back to a list-of-lists.
        lengths = {len(seq) for seq in natoms_per_traj}
        if len(lengths) == 1 and natoms_per_traj:
            natoms_tensor = torch.tensor(natoms_per_traj, dtype=torch.long)
            natoms_payload: dict = {
                "seed": int(seed),
                "n_atoms_trajectories": natoms_tensor,  # [n_mols, n_steps]
                "is_valid_final": torch.tensor(is_valid_mask, dtype=torch.bool),
                "shape_note": "rows = generations, cols = integration steps",
            }
        else:
            natoms_payload = {
                "seed": int(seed),
                "n_atoms_trajectories": natoms_per_traj,  # ragged list[list[int]]
                "is_valid_final": list(is_valid_mask),
                "shape_note": "ragged; one inner list per generation",
            }
        torch.save(natoms_payload, natoms_traj_path)
        print(
            f"[seed={seed}] saved n_atoms trajectories "
            f"({len(natoms_per_traj)} generations) to: {natoms_traj_path}"
        )

    if traj_dir is not None:
        trajs_to_save = all_trajs if traj_n_max is None else all_trajs[:traj_n_max]
        mask = is_valid_mask[: len(trajs_to_save)]
        valid_trajs = [t for t, v in zip(trajs_to_save, mask) if v]
        invalid_trajs = [t for t, v in zip(trajs_to_save, mask) if not v]
        valid_path = traj_dir / f"trajectories_valid_seed{seed}.pt"
        invalid_path = traj_dir / f"trajectories_invalid_seed{seed}.pt"
        torch.save(valid_trajs, valid_path)
        torch.save(invalid_trajs, invalid_path)
        print(
            f"[seed={seed}] saved {len(valid_trajs)} valid + "
            f"{len(invalid_trajs)} invalid trajectories to: {traj_dir}"
        )

    rdkit_mols = _trajectories_to_rdkit(all_trajs, vocab)
    n_total = len(rdkit_mols)
    n_valid = sum(1 for m in rdkit_mols if m is not None)

    print(f"[seed={seed}] computing metrics on {n_total} molecules...")
    seed_metrics = _compute_all_metrics(
        rdkit_mols=rdkit_mols,
        metrics=metrics,
        stability_metrics=stability_metrics,
        distribution_metrics=distribution_metrics,
        run_posebusters=run_posebusters,
    )
    seed_metrics["n_total"] = float(n_total)
    seed_metrics["n_rdkit_parseable"] = float(n_valid)

    # Free memory before the next seed.
    del all_trajs, rdkit_mols, predictions
    return seed_metrics, n_total, n_valid


# ---------------------------------------------------------------------------
# Aggregation: mean + sample std across seeds
# ---------------------------------------------------------------------------


def _aggregate_metrics(per_seed: list[dict]) -> dict[str, dict]:
    """Aggregate per-seed metric dicts into mean and sample std per key."""
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
                f"[warn] --seeds had {len(seeds)} entries but --n-seeds={n_seeds};"
                f" using the provided list of {len(seeds)} seeds."
            )
        return seeds
    return [42 + 1000 * i for i in range(n_seeds)]


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run all standard generative metrics on 10k generated molecules "
            "for multiple random seeds, then report mean and sample standard "
            "deviation across seeds. Does NOT run trajectory diagnostics."
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
        default=Path("eval_outputs/full_metrics"),
        help="Directory to save evaluation artifacts.",
    )
    parser.add_argument(
        "--n-mols",
        type=int,
        default=10_000,
        help="Approximate molecules to generate per seed (default: 10000).",
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
        "--no-posebusters",
        action="store_true",
        help="Disable PoseBusters checks (faster, less RAM).",
    )
    parser.add_argument(
        "--save-trajectories",
        action="store_true",
        help=(
            "Save full per-step MoleculeData trajectories, split into valid and "
            "invalid, to <output-dir>/trajectories_{valid,invalid}_seed{seed}.pt. "
            "Off by default (can be several hundred MB per seed for 10k mols)."
        ),
    )
    parser.add_argument(
        "--save-trajectories-n-max",
        type=int,
        default=None,
        help=(
            "When --save-trajectories is set, cap the total number of saved "
            "trajectories per seed (default: all). Useful to limit file size."
        ),
    )
    parser.add_argument(
        "--no-save-natoms-trajectories",
        action="store_true",
        help=(
            "Skip saving the per-step atom-count trajectory of every "
            "generation. By default, one .pt per seed is written to "
            "<output-dir>/natoms_traj_seed{seed}.pt."
        ),
    )
    parser.add_argument(
        "--overrides",
        nargs="*",
        default=["data=qm9", "model=semla", "cfg=uncond"],
        help=(
            "Hydra overrides used to compose config. Must match the "
            "training-time overrides for the loaded checkpoint."
        ),
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    _register_resolvers()
    torch.set_float32_matmul_precision("medium")
    RDLogger.DisableLog("rdApp.*")

    seeds = _parse_seed_list(args.seeds, args.n_seeds)
    print(f"Running evaluation with seeds: {seeds}")

    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.1"):
        cfg = compose(config_name="default", overrides=args.overrides)

    print("Overrides:", args.overrides)
    print("Integrator config:\n" + OmegaConf.to_yaml(cfg.integrator))

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

    per_seed: list[dict] = []
    per_seed_counts: list[tuple[int, int]] = []
    for seed in seeds:
        natoms_traj_path = (
            None
            if args.no_save_natoms_trajectories
            else args.output_dir / f"natoms_traj_seed{seed}.pt"
        )
        traj_dir = args.output_dir if args.save_trajectories else None
        seed_metrics, n_total, n_valid = _run_one_seed(
            seed=seed,
            cfg=cfg,
            module=module,
            test_dl=test_dl,
            distributions=distributions,
            vocab=vocab,
            metrics=metrics,
            stability_metrics=stability_metrics,
            distribution_metrics=distribution_metrics,
            n_mols=args.n_mols,
            predict_batch_size=args.predict_batch_size,
            run_posebusters=not args.no_posebusters,
            natoms_traj_path=natoms_traj_path,
            traj_dir=traj_dir,
            traj_n_max=args.save_trajectories_n_max,
        )
        per_seed.append(seed_metrics)
        per_seed_counts.append((n_total, n_valid))

        # Persist per-seed snapshot immediately so partial results survive crashes.
        seed_path = args.output_dir / f"metrics_seed{seed}.json"
        with seed_path.open("w", encoding="utf-8") as f:
            json.dump(seed_metrics, f, indent=2)

    aggregated = _aggregate_metrics(per_seed)
    _print_summary_table(per_seed, aggregated)

    summary = {
        "checkpoint": str(args.checkpoint),
        "n_mols_requested": args.n_mols,
        "predict_batch_size": args.predict_batch_size,
        "seeds": seeds,
        "n_total_per_seed": [t for t, _ in per_seed_counts],
        "n_rdkit_parseable_per_seed": [v for _, v in per_seed_counts],
        "per_seed_metrics": {
            str(s): m for s, m in zip(seeds, per_seed)
        },
        "aggregated": aggregated,
        "overrides": list(args.overrides),
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
