"""Evaluate classifier-free guidance steering for n_atoms and MW conditioning.

Sweeps over target atom counts / molecular weights and guidance scales,
generating molecules for each combination and reporting how accurately
the model hits the requested target.

Usage:
    python eval_cfg.py
"""

import hydra
import matplotlib.pyplot as plt
import omegaconf
import torch
from copy import deepcopy
import math
import contextlib
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from rdkit import RDLogger

from chemflow.utils.utils import init_uniform_prior
from chemflow.dataset.vocab import setup_token_weights


OmegaConf.register_new_resolver("oc.eval", eval)
OmegaConf.register_new_resolver("len", lambda x: len(x))
OmegaConf.register_new_resolver("if", lambda cond, t, f: t if cond else f)
OmegaConf.register_new_resolver("eq", lambda x, y: x == y)

torch.set_float32_matmul_precision("medium")
RDLogger.DisableLog("rdApp.*")
pl.seed_everything(42)

TARGET_N_ATOMS = [10, 18, 28]
GUIDANCE_SCALES = [1.0, 5.0, 10.0]

TARGET_MWS = [100.0, 110.0, 125.0]
MW_GUIDANCE_SCALES = [0.75, 0.9, 1.1, 1.25]

PROPERTY_GUIDANCE_SCALES = [0.0, 0.5, 1.0, 2.0, 5.0]
PROP_COLLECT_N_BATCHES = 5   # batches used to derive target-value percentiles
PROP_EVAL_N_BATCHES = 4      # batches per (target_val, scale) cell

PLOT_N_MOLS = 500
PREDICT_BATCH_SIZE = 128


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------


def _infer_mol_n_atoms(mol) -> int | None:
    """Infer atom count from either MoleculeData or batched molecule objects."""
    if mol is None:
        return None

    num_nodes = getattr(mol, "num_nodes", None)
    if num_nodes is not None:
        try:
            return int(num_nodes)
        except Exception:
            pass

    atom_tokens = getattr(mol, "a", None)
    if atom_tokens is not None:
        try:
            return int(atom_tokens.shape[0])
        except Exception:
            pass

    batch = getattr(mol, "batch", None)
    batch_size = getattr(mol, "batch_size", None)
    if batch is not None and batch_size is not None:
        batch_n_atoms = torch.bincount(batch, minlength=batch_size).to(dtype=torch.long)
        if batch_n_atoms.numel() == 1:
            return int(batch_n_atoms.item())

    return None


def _infer_mol_mw(mol, atom_tokens: list[str]) -> float | None:
    """Infer molecular weight from a single molecule."""
    if mol is None or atom_tokens is None:
        return None
    from chemflow.model.cfg import compute_molecular_weight

    a = getattr(mol, "a", None)
    if a is None:
        return None
    mw = compute_molecular_weight(a, atom_tokens)
    return float(mw.item())


def _run_property_guidance_eval(
    module,
    test_dl,
    target_val: float,
    property_indices: list[int],
    n_batches: int,
    device,
) -> tuple[int, int]:
    """Custom predict loop that overrides ``mol_t.y`` with a fixed target
    property value and returns (n_valid, n_total)."""
    from chemflow.utils import rdkit as chemflowRD

    n_valid_total = 0
    n_total = 0

    module.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_dl):
            if batch_idx >= n_batches:
                break
            mol_t, mol_1 = batch
            mol_t = mol_t.to(device)
            mol_1 = mol_1.to(device)

            # Override the conditioned property columns with the target value.
            if mol_t.y is not None and mol_t.y.ndim >= 2:
                new_y = mol_t.y.clone().float()
                for idx in property_indices:
                    if idx < new_y.shape[1]:
                        new_y[:, idx] = target_val
                mol_t.y = new_y

            gen_mols = module.sample(
                (mol_t, mol_1),
                batch_idx,
                return_traj=True,
            )

            for traj in gen_mols:
                mol_final = traj[-1] if isinstance(traj, list) and traj else traj
                rdkit_mol = mol_final.to_rdkit_mol(
                    module.vocab.atom_tokens,
                    module.vocab.edge_tokens,
                    module.vocab.charge_tokens,
                )
                n_total += 1
                if rdkit_mol is not None:
                    try:
                        if chemflowRD.mol_is_valid(rdkit_mol):
                            n_valid_total += 1
                    except Exception:
                        pass

    return n_valid_total, n_total


def _collect_n_atoms_from_predict_output(pred) -> list[int]:
    """Collect per-molecule atom counts from predict_step outputs."""
    all_n_atoms: list[int] = []

    if pred is None:
        return all_n_atoms

    if isinstance(pred, dict):
        molecules = []
        for key in ("valid_mols", "invalid_mols"):
            values = pred.get(key, [])
            if values:
                molecules.extend(values)

        for mol in molecules:
            mol_final = mol[-1] if isinstance(mol, list) and len(mol) > 0 else mol
            n_atoms = _infer_mol_n_atoms(mol_final)
            if n_atoms is not None:
                all_n_atoms.append(n_atoms)
        return all_n_atoms

    batch = getattr(pred, "batch", None)
    batch_size = getattr(pred, "batch_size", None)
    if batch is not None and batch_size is not None:
        batch_n_atoms = torch.bincount(batch, minlength=batch_size).to(dtype=torch.long)
        all_n_atoms.extend(batch_n_atoms.cpu().tolist())
        return all_n_atoms

    n_atoms = _infer_mol_n_atoms(pred)
    if n_atoms is not None:
        all_n_atoms.append(n_atoms)

    return all_n_atoms


def _collect_mw_from_predict_output(
    pred,
    atom_tokens: list[str],
) -> list[float]:
    """Collect per-molecule MW from predict_step outputs."""
    all_mw: list[float] = []
    if pred is None:
        return all_mw

    if isinstance(pred, dict):
        molecules = []
        for key in ("valid_mols", "invalid_mols"):
            values = pred.get(key, [])
            if values:
                molecules.extend(values)

        for mol in molecules:
            mol_final = mol[-1] if isinstance(mol, list) and len(mol) > 0 else mol
            mw = _infer_mol_mw(mol_final, atom_tokens)
            if mw is not None:
                all_mw.append(mw)
        return all_mw

    mw = _infer_mol_mw(pred, atom_tokens)
    if mw is not None:
        all_mw.append(mw)
    return all_mw


# ---------------------------------------------------------------------------
#  Summarisation
# ---------------------------------------------------------------------------


def summarize_property_steering(
    n_valid: int,
    n_total: int,
    target_val: float,
    property_name: str,
    guidance_scale: float,
) -> dict:
    return {
        "property_name": property_name,
        "target_value": round(target_val, 6),
        "guidance_scale": guidance_scale,
        "validity_rate": n_valid / max(n_total, 1),
        "n_valid": n_valid,
        "n_total": n_total,
    }


def summarize_natoms_steering(predictions, target_n_atoms: int) -> dict:
    all_n_atoms: list[int] = []

    for pred in predictions:
        all_n_atoms.extend(_collect_n_atoms_from_predict_output(pred))

    if len(all_n_atoms) > PLOT_N_MOLS:
        all_n_atoms = all_n_atoms[:PLOT_N_MOLS]

    if not all_n_atoms:
        return {"target_n_atoms": target_n_atoms, "n_molecules": 0}

    counts = torch.tensor(all_n_atoms, dtype=torch.float)
    exact = (counts == target_n_atoms).float().mean().item()
    within_1 = ((counts - target_n_atoms).abs() <= 1).float().mean().item()
    within_2 = ((counts - target_n_atoms).abs() <= 2).float().mean().item()

    return {
        "target_n_atoms": target_n_atoms,
        "mean_n_atoms": counts.mean().item(),
        "std_n_atoms": counts.std().item(),
        "median_n_atoms": counts.median().item(),
        "exact_match_rate": exact,
        "within_1_rate": within_1,
        "within_2_rate": within_2,
        "n_molecules": len(all_n_atoms),
        "all_n_atoms": all_n_atoms,
    }


def summarize_mw_steering(
    predictions,
    target_mw: float,
    atom_tokens: list[str],
) -> dict:
    all_mw: list[float] = []

    for pred in predictions:
        all_mw.extend(_collect_mw_from_predict_output(pred, atom_tokens))

    if len(all_mw) > PLOT_N_MOLS:
        all_mw = all_mw[:PLOT_N_MOLS]

    if not all_mw:
        return {"target_mw": target_mw, "n_molecules": 0}

    vals = torch.tensor(all_mw, dtype=torch.float)
    within_10 = ((vals - target_mw).abs() <= 10.0).float().mean().item()
    within_25 = ((vals - target_mw).abs() <= 25.0).float().mean().item()

    return {
        "target_mw": target_mw,
        "mean_mw": vals.mean().item(),
        "std_mw": vals.std().item(),
        "median_mw": vals.median().item(),
        "within_10_rate": within_10,
        "within_25_rate": within_25,
        "n_molecules": len(all_mw),
        "all_mw": [round(v, 2) for v in all_mw],
    }


# ---------------------------------------------------------------------------
#  Plotting
# ---------------------------------------------------------------------------


def plot_natoms_distributions_grid(
    all_results: dict[tuple[float, int], dict],
    natoms_distributions: torch.Tensor,
    output_path: str = "cfg_steering_distributions.png",
):
    """Save a grid of n_atoms histograms (rows=targets, cols=scales)."""
    if not all_results:
        print("No results to plot.")
        return

    guidance_scales = sorted({key[0] for key in all_results})
    target_n_atoms = sorted({key[1] for key in all_results})

    n_rows = len(target_n_atoms)
    n_cols = len(guidance_scales)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.2 * max(1, n_cols), 2.6 * max(1, n_rows)),
        squeeze=False,
    )

    for row_idx, target in enumerate(target_n_atoms):
        for col_idx, scale in enumerate(guidance_scales):
            ax = axes[row_idx][col_idx]
            result = all_results.get((scale, target), {})
            values = result.get("all_n_atoms", [])

            if values:
                x_min = int(min(values))
                x_max = int(max(values))
                bins = list(range(x_min, x_max + 2))
                ax.hist(
                    values,
                    bins=bins,
                    align="left",
                    rwidth=0.85,
                    color="#4C72B0",
                    alpha=0.85,
                    edgecolor="black",
                    linewidth=0.5,
                )
                ax.set_xlim(x_min - 0.5, x_max + 0.5)
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

            if row_idx == 0:
                ax.set_title(f"scale={scale}", fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(
                    f"target={target}\ncount",
                    fontsize=9,
                )
            if row_idx == n_rows - 1:
                ax.set_xlabel("generated n_atoms", fontsize=9)

            ax.set_xlim(0, natoms_distributions.numel())
            ax.tick_params(labelsize=8)

    fig.suptitle(
        "Generated n_atoms distributions",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    print(f"Distribution figure saved to {output_path}")


def plot_mw_distributions_grid(
    all_results: dict[tuple[float, float], dict],
    output_path: str = "cfg_mw_steering_distributions.png",
):
    """Save a grid of MW histograms (rows=targets, cols=scales)."""
    if not all_results:
        print("No MW results to plot.")
        return

    guidance_scales = sorted({key[0] for key in all_results})
    target_mws = sorted({key[1] for key in all_results})

    n_rows = len(target_mws)
    n_cols = len(guidance_scales)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.2 * max(1, n_cols), 2.6 * max(1, n_rows)),
        squeeze=False,
    )

    for row_idx, target in enumerate(target_mws):
        for col_idx, scale in enumerate(guidance_scales):
            ax = axes[row_idx][col_idx]
            result = all_results.get((scale, target), {})
            values = result.get("all_mw", [])

            if values:
                ax.hist(
                    values,
                    bins=30,
                    color="#4C72B0",
                    alpha=0.85,
                    edgecolor="black",
                    linewidth=0.5,
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

            if row_idx == 0:
                ax.set_title(f"scale={scale}", fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(
                    f"target={target:.0f}\ncount",
                    fontsize=9,
                )
            if row_idx == n_rows - 1:
                ax.set_xlabel("generated MW (Da)", fontsize=9)

            ax.tick_params(labelsize=8)

    fig.suptitle(
        "Generated MW distributions",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    print(f"MW distribution figure saved to {output_path}")


def plot_property_guidance_grid(
    all_results: dict[tuple[float, float], dict],
    property_name: str,
    output_path: str = "cfg_property_steering_distributions.png",
):
    """Save a grid of validity-rate bars (rows=targets, cols=guidance scales).

    Note: measuring actual QM9 property accuracy of generated molecules requires
    an external DFT/surrogate predictor. Validity rate is the proxy metric here.
    """
    if not all_results:
        print("No property results to plot.")
        return

    guidance_scales = sorted({key[0] for key in all_results})
    target_vals = sorted({key[1] for key in all_results})

    n_rows = len(target_vals)
    n_cols = len(guidance_scales)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(2.8 * max(1, n_cols), 2.4 * max(1, n_rows)),
        squeeze=False,
    )

    for row_idx, target in enumerate(target_vals):
        for col_idx, scale in enumerate(guidance_scales):
            ax = axes[row_idx][col_idx]
            result = all_results.get((scale, target), {})
            validity = result.get("validity_rate", 0.0)
            n_total = result.get("n_total", 0)

            bar_color = "#4C72B0" if scale > 0.0 else "#7f7f7f"
            ax.bar(
                [0],
                [validity],
                color=bar_color,
                alpha=0.85,
                edgecolor="black",
                linewidth=0.8,
                width=0.6,
            )
            ax.set_ylim(0, 1.15)
            ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
            ax.text(
                0,
                min(validity + 0.06, 1.08),
                f"{validity:.0%}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
            ax.set_xticks([])

            if row_idx == 0:
                ax.set_title(f"scale={scale}", fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(
                    f"target={target:.3g}\nvalidity",
                    fontsize=9,
                )
            if row_idx == n_rows - 1:
                ax.set_xlabel(f"n={n_total}", fontsize=8)

            ax.tick_params(labelsize=8)

    fig.suptitle(
        f"Property CFG validity ({property_name})",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    print(f"Property guidance figure saved to {output_path}")


# ---------------------------------------------------------------------------
#  Diagnostics
# ---------------------------------------------------------------------------


def _print_cfg_diagnostics(module, state_dict, test_dl, device):
    """Print CFG diagnostic info for the loaded model."""
    print("\n--- CFG Diagnostic ---")
    print(f"  n_atoms_strategy: {module.n_atoms_strategy}")
    print(f"  use_ema_for_eval: {module.use_ema_for_eval}")

    adapter = module.cfg_adapter
    has_natoms = adapter._has_natoms_cfg
    has_mw = adapter._has_mw_cfg
    has_props = adapter._has_property_conditioning
    print(f"  _has_property_conditioning: {has_props}")
    print(f"  _has_natoms_cfg:            {has_natoms}")
    print(f"  _has_mw_cfg:                {has_mw}")
    print(f"  natoms_cfg_guidance_scale:   {adapter.natoms_cfg_guidance_scale}")
    print(f"  mw_cfg_guidance_scale:       {adapter.mw_cfg_guidance_scale}")

    eval_model = module._get_model()
    cfg_emb = getattr(
        eval_model.embedding_backbone,
        "cfg_embedding",
        None,
    )
    if cfg_emb is not None:
        n_params = sum(p.numel() for p in cfg_emb.parameters())
        print(f"  cfg_embedding total params:  {n_params}")
        """if hasattr(cfg_emb, "_natoms_null"):
            print(
                f"  _natoms_null norm:           "
                f"{cfg_emb._natoms_null.norm().item():.4f}"
            )"""
        if hasattr(cfg_emb, "_mw_null"):
            print(
                f"  _mw_null norm:               {cfg_emb._mw_null.norm().item():.4f}"
            )

        ckpt_cfg_keys = [
            k for k in state_dict if "model_ema" in k and "cfg_embedding" in k
        ]
        print(f"  checkpoint EMA cfg_embedding keys ({len(ckpt_cfg_keys)}):")
        for k in ckpt_cfg_keys[:8]:
            print(f"    {k}: {state_dict[k].shape}")
    else:
        print("  WARNING: cfg_embedding is None!")

    if not has_natoms:
        print("  Skipping forward-pass diagnostic (no natoms).")
        print("--- End Diagnostic ---\n")
        return

    test_batch = next(iter(test_dl))
    mol_t, mol_1 = test_batch
    mol_t = mol_t.to(device)
    mol_1 = mol_1.to(device)
    eval_model = module._get_model()
    eval_model.set_inference()

    with torch.no_grad():
        bs = mol_t.batch_size
        t_diag = torch.zeros(bs, device=device)
        preds_uncond = eval_model(
            mol_t,
            t_diag.view(-1, 1),
            cfg_inputs={},
        )
        target_10 = torch.full(
            (bs,),
            10,
            dtype=torch.long,
            device=device,
        )
        preds_cond_10 = eval_model(
            mol_t,
            t_diag.view(-1, 1),
            cfg_inputs={"target_n_atoms": target_10},
        )
        target_28 = torch.full(
            (bs,),
            28,
            dtype=torch.long,
            device=device,
        )
        preds_cond_28 = eval_model(
            mol_t,
            t_diag.view(-1, 1),
            cfg_inputs={"target_n_atoms": target_28},
        )

    print("\n  Comparing uncond vs cond(10) vs cond(28):")
    for key in [
        "do_ins_head",
        "do_del_head",
        "ins_rate_head",
        "atom_type_head",
        "pos_head",
    ]:
        u = preds_uncond[key]
        c10 = preds_cond_10[key]
        c28 = preds_cond_28[key]
        d10 = (c10 - u).abs().mean().item()
        d28 = (c28 - u).abs().mean().item()
        d_1028 = (c28 - c10).abs().mean().item()
        print(
            f"    {key:20s}  "
            f"|u-c10|={d10:.6f}  "
            f"|u-c28|={d28:.6f}  "
            f"|c10-c28|={d_1028:.6f}"
        )

    adapter.natoms_cfg_guidance_scale = 5.0
    print(
        f"\n  should_use_natoms_cfg(t=10, s=5.0): "
        f"{adapter.should_use_natoms_cfg(target_10)}"
    )
    adapter.natoms_cfg_guidance_scale = 0.0
    print(
        f"  should_use_natoms_cfg(t=10, s=0.0): "
        f"{adapter.should_use_natoms_cfg(target_10)}"
    )
    print("--- End Diagnostic ---\n")


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------


def eval_cfg(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)

    with contextlib.suppress(Exception):
        cfg.data.datamodule.batch_size.test = PREDICT_BATCH_SIZE
    n_predict_batches = math.ceil(PLOT_N_MOLS / PREDICT_BATCH_SIZE)

    # Fix prior atom count to the median so every molecule starts
    # at the same size — removes a confounding variable from the
    # guidance evaluation.
    cfg.data.n_atoms_strategy = "median"

    # ── data setup (mirrors run.py) ──
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

    n_dist = distributions.n_atoms_distribution
    median_n = int((torch.cumsum(n_dist, 0) >= 0.5).nonzero(as_tuple=True)[0][0].item())
    print(f"Prior n_atoms fixed to median = {median_n}")

    # ── token weights (mirrors run.py) ──
    tw = cfg.model.token_weighting
    atom_type_weights, edge_token_weights, charge_token_weights = setup_token_weights(
        vocab=vocab,
        distributions=loss_weight_distributions,
        weight_alpha=tw.weight_alpha,
        type_loss_token_weights=tw.type_loss_token_weights,
    )

    # ── model setup ──
    module = hydra.utils.instantiate(
        cfg.model.module,
        _recursive_=False,
        distributions=token_prior_distribution,
        loss_weight_distributions=loss_weight_distributions,
        atom_type_weights=atom_type_weights,
        edge_token_weights=edge_token_weights,
        charge_token_weights=charge_token_weights,
    )

    ckpt_path = "/capstor/store/cscs/swissai/a131/frankem/chemflow/logs/wandb/poisson_cfg/chemflow/fy8p6jss/checkpoints/epoch=499-step=9500.ckpt"
    ckpt = torch.load(
        ckpt_path,
        map_location="cpu",
        weights_only=False,
    )

    state_dict = ckpt["state_dict"]
    state_dict = {k.replace("._orig_mod", ""): v for k, v in state_dict.items()}
    module.load_state_dict(state_dict)

    test_dataloaders = datamodule.test_dataloader()
    test_dl = (
        test_dataloaders[0] if isinstance(test_dataloaders, list) else test_dataloaders
    )
    trainer_kwargs = dict(cfg.trainer.trainer)
    trainer_kwargs.setdefault(
        "limit_predict_batches",
        n_predict_batches,
    )
    trainer = pl.Trainer(
        logger=False,
        callbacks=[],
        enable_checkpointing=False,
        **trainer_kwargs,
    )

    # ── diagnostics ──
    module.eval()
    device = next(module.parameters()).device
    _print_cfg_diagnostics(module, state_dict, test_dl, device)

    adapter = module.cfg_adapter
    atom_tokens = list(vocab.atom_tokens)
    module.predict_return_traj = True

    # ── sweep n_atoms guidance ──
    if adapter._has_natoms_cfg:
        all_natoms_results: dict[tuple[float, int], dict] = {}
        original_mw_scale = adapter.mw_cfg_guidance_scale
        adapter.mw_cfg_guidance_scale = 0.0

        for scale in GUIDANCE_SCALES:
            adapter.natoms_cfg_guidance_scale = scale
            print(f"\n{'=' * 60}")
            print(f"n_atoms guidance scale = {scale}")
            print(f"{'=' * 60}")

            for n_atoms in TARGET_N_ATOMS:
                pl.seed_everything(42)
                print(
                    f"  target_n_atoms={n_atoms} ... ",
                    end="",
                    flush=True,
                )
                module.predict_target_n_atoms_override = n_atoms
                module.predict_target_mw_override = None
                predictions = trainer.predict(
                    module,
                    dataloaders=test_dl,
                )
                result = summarize_natoms_steering(
                    predictions,
                    target_n_atoms=n_atoms,
                )
                result["guidance_scale"] = scale
                all_natoms_results[(scale, n_atoms)] = result

                if result["n_molecules"] == 0:
                    print("no molecules (n=0)")
                else:
                    print(
                        f"mean={result['mean_n_atoms']:.2f}  "
                        f"std={result['std_n_atoms']:.2f}  "
                        f"exact={result['exact_match_rate']:.1%}  "
                        f"±1={result['within_1_rate']:.1%}  "
                        f"±2={result['within_2_rate']:.1%}  "
                        f"(n={result['n_molecules']})"
                    )

        adapter.mw_cfg_guidance_scale = original_mw_scale
        torch.save(
            all_natoms_results,
            "cfg_natoms_steering_results.pt",
        )
        print("\nResults saved to cfg_natoms_steering_results.pt")
        plot_natoms_distributions_grid(
            all_natoms_results,
            natoms_distributions=distributions.n_atoms_distribution,
            output_path="cfg_natoms_steering_distributions.png",
        )
    else:
        print("Skipping n_atoms sweep (natoms_encoder disabled)")

    # ── sweep MW guidance ──
    if adapter._has_mw_cfg:
        all_mw_results: dict[tuple[float, float], dict] = {}
        original_natoms_scale = adapter.natoms_cfg_guidance_scale
        adapter.natoms_cfg_guidance_scale = 0.0

        for scale in MW_GUIDANCE_SCALES:
            adapter.mw_cfg_guidance_scale = scale
            print(f"\n{'=' * 60}")
            print(f"MW guidance scale = {scale}")
            print(f"{'=' * 60}")

            for mw in TARGET_MWS:
                pl.seed_everything(42)
                print(
                    f"  target_mw={mw:.0f} Da ... ",
                    end="",
                    flush=True,
                )
                module.predict_target_n_atoms_override = None
                module.predict_target_mw_override = mw
                predictions = trainer.predict(
                    module,
                    dataloaders=test_dl,
                )
                result = summarize_mw_steering(
                    predictions,
                    target_mw=mw,
                    atom_tokens=atom_tokens,
                )
                result["guidance_scale"] = scale
                all_mw_results[(scale, mw)] = result

                if result["n_molecules"] == 0:
                    print("no molecules (n=0)")
                else:
                    print(
                        f"mean={result['mean_mw']:.1f}  "
                        f"std={result['std_mw']:.1f}  "
                        f"±10={result['within_10_rate']:.1%}  "
                        f"±25={result['within_25_rate']:.1%}  "
                        f"(n={result['n_molecules']})"
                    )

        adapter.natoms_cfg_guidance_scale = original_natoms_scale
        torch.save(
            all_mw_results,
            "cfg_mw_steering_results.pt",
        )
        print("\nResults saved to cfg_mw_steering_results.pt")
        plot_mw_distributions_grid(
            all_mw_results,
            output_path="cfg_mw_steering_distributions.png",
        )
    else:
        print("Skipping MW sweep (mw_encoder disabled)")

    # ── sweep property guidance ──
    if adapter._has_property_conditioning:
        # Derive property name(s) from the hydra config.
        try:
            property_names_list = list(cfg.cfg.cfg.property_names)
        except Exception:
            property_names_list = ["gap"]
        primary_prop = property_names_list[0] if property_names_list else "gap"
        prop_indices: list[int] = adapter.property_indices or [4]  # 4 = gap fallback

        # Collect target values from the test-data distribution.
        all_prop_vals: list[float] = []
        for _batch_idx, _batch in enumerate(test_dl):
            if _batch_idx >= PROP_COLLECT_N_BATCHES:
                break
            _mol_t, _ = _batch
            if hasattr(_mol_t, "y") and _mol_t.y is not None and _mol_t.y.ndim >= 2:
                col = prop_indices[0]
                if col < _mol_t.y.shape[1]:
                    all_prop_vals.extend(_mol_t.y[:, col].float().cpu().tolist())

        if all_prop_vals:
            _pv = torch.tensor(all_prop_vals)
            target_prop_values = [
                float(_pv.quantile(0.10).item()),
                float(_pv.quantile(0.50).item()),
                float(_pv.quantile(0.90).item()),
            ]
            print(
                f"\nProperty '{primary_prop}' targets "
                f"(10th/50th/90th pct): "
                + ", ".join(f"{v:.4g}" for v in target_prop_values)
            )
        else:
            target_prop_values = [0.05, 0.14, 0.25]
            print(
                f"\nNo test-data properties found; "
                f"using defaults for '{primary_prop}': "
                + ", ".join(str(v) for v in target_prop_values)
            )

        original_prop_scale = adapter.cfg_guidance_scale
        all_prop_results: dict[tuple[float, float], dict] = {}

        for scale in PROPERTY_GUIDANCE_SCALES:
            adapter.cfg_guidance_scale = scale
            print(f"\n{'=' * 60}")
            print(f"Property ({primary_prop}) guidance scale = {scale}")
            print(f"{'=' * 60}")
            prop_device = next(module.parameters()).device

            for target_val in target_prop_values:
                pl.seed_everything(42)
                print(
                    f"  {primary_prop}={target_val:.4g} ... ",
                    end="",
                    flush=True,
                )
                n_valid, n_total = _run_property_guidance_eval(
                    module=module,
                    test_dl=test_dl,
                    target_val=target_val,
                    property_indices=prop_indices,
                    n_batches=PROP_EVAL_N_BATCHES,
                    device=prop_device,
                )
                result = summarize_property_steering(
                    n_valid=n_valid,
                    n_total=n_total,
                    target_val=target_val,
                    property_name=primary_prop,
                    guidance_scale=scale,
                )
                all_prop_results[(scale, round(target_val, 6))] = result

                if n_total == 0:
                    print("no molecules (n=0)")
                else:
                    print(
                        f"validity={result['validity_rate']:.1%}  "
                        f"(n={n_valid}/{n_total})"
                    )

        adapter.cfg_guidance_scale = original_prop_scale
        torch.save(all_prop_results, "cfg_property_steering_results.pt")
        print("\nResults saved to cfg_property_steering_results.pt")
        plot_property_guidance_grid(
            all_prop_results,
            property_name=primary_prop,
            output_path="cfg_property_steering_distributions.png",
        )
    else:
        print("Skipping property sweep (property conditioning disabled)")


@hydra.main(
    config_path="configs",
    config_name="default",
    version_base="1.1",
)
def main(cfg: omegaconf.DictConfig):
    eval_cfg(cfg)


if __name__ == "__main__":
    main()
