"""Evaluate classifier-free guidance steering for n_atoms conditioning.

Sweeps over target atom counts and guidance scales, generating molecules
for each combination and reporting how accurately the model hits the
requested size.

Usage:
    python eval_cfg.py
"""

import hydra
import omegaconf
import torch
from copy import deepcopy
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from rdkit import RDLogger


OmegaConf.register_new_resolver("oc.eval", eval)
OmegaConf.register_new_resolver("len", lambda x: len(x))
OmegaConf.register_new_resolver("if", lambda cond, t, f: t if cond else f)
OmegaConf.register_new_resolver("eq", lambda x, y: x == y)

torch.set_float32_matmul_precision("medium")
RDLogger.DisableLog("rdApp.*")
pl.seed_everything(42)

# TARGET_N_ATOMS = [10, 18, 28]
# GUIDANCE_SCALES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0]
TARGET_N_ATOMS = [10]
GUIDANCE_SCALES = [7.0, 9.0, 11.0, 15.0, 20.0]


def summarize_natoms_steering(predictions, target_n_atoms: int) -> dict:
    all_n_atoms: list[int] = []

    for pred in predictions:
        if pred is None:
            continue
        batch_n_atoms = torch.bincount(pred.batch, minlength=pred.batch_size).to(
            dtype=torch.long
        )
        all_n_atoms.extend(batch_n_atoms.cpu().tolist())

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


def eval_cfg(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)

    # ── data setup (mirrors run.py) ──────────────────────────────────
    preprocessing = hydra.utils.instantiate(cfg.data.preprocessing)
    vocab = preprocessing.vocab
    distributions = preprocessing.distributions
    loss_weight_distributions = deepcopy(distributions)

    # Sampling prior uses uniform categorical distributions.
    distributions.atom_type_distribution = torch.ones_like(
        distributions.atom_type_distribution
    )
    distributions.atom_type_distribution /= distributions.atom_type_distribution.sum()
    distributions.edge_type_distribution = torch.ones_like(
        distributions.edge_type_distribution
    )
    distributions.edge_type_distribution /= distributions.edge_type_distribution.sum()
    distributions.charge_type_distribution = torch.ones_like(
        distributions.charge_type_distribution
    )
    distributions.charge_type_distribution /= (
        distributions.charge_type_distribution.sum()
    )

    cfg.data.vocab = vocab

    datamodule = hydra.utils.instantiate(
        cfg.data.datamodule,
        _recursive_=False,
        vocab=vocab,
        distributions=distributions,
    )
    datamodule.setup()

    # ── model setup ──────────────────────────────────────────────────
    module = hydra.utils.instantiate(
        cfg.model.module,
        _recursive_=False,
        distributions=distributions,
        loss_weight_distributions=loss_weight_distributions,
    )

    ckpt_path = "/cluster/project/krause/frankem/chemflow/outputs/2026-03-01/23-38-13/logs/chemflow/qrk9nog4/checkpoints/epoch=1379-step=31240.ckpt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Checkpoints saved from a torch.compile()-d model have keys prefixed
    # with "_orig_mod." (e.g. "model._orig_mod.backbone...").  Strip this
    # so the weights load into the uncompiled module.
    state_dict = ckpt["state_dict"]
    state_dict = {k.replace("._orig_mod", ""): v for k, v in state_dict.items()}
    module.load_state_dict(state_dict)

    test_dataloaders = datamodule.test_dataloader()
    test_dl = (
        test_dataloaders[0] if isinstance(test_dataloaders, list) else test_dataloaders
    )
    trainer = pl.Trainer(
        logger=False,
        callbacks=[],
        enable_checkpointing=False,
        **cfg.trainer.trainer,
    )

    # ── diagnostic: verify CFG is actually modifying predictions ─────
    module.eval()
    device = next(module.parameters()).device
    print("\n--- CFG Diagnostic ---")
    print(f"  n_atoms_strategy: {module.n_atoms_strategy}")
    print(f"  use_ema_for_eval: {module.use_ema_for_eval}")
    has_natoms = module.cfg_adapter._has_natoms_cfg
    print(f"  _has_natoms_cfg:  {has_natoms}")
    print(
        f"  cfg_adapter.model is module.model: "
        f"{module.cfg_adapter.model is module.model}"
    )
    print(
        f"  cfg_adapter.natoms_cfg_guidance_scale: "
        f"{module.cfg_adapter.natoms_cfg_guidance_scale}"
    )

    eval_model = module._get_model()
    natoms_emb = eval_model.embedding_backbone.natoms_cfg_embedding
    if natoms_emb is not None:
        n_params = sum(p.numel() for p in natoms_emb.parameters())
        null_norm = natoms_emb.null_embedding.norm().item()
        print(
            f"  natoms_cfg_embedding params: {n_params}  null_emb norm: {null_norm:.4f}"
        )

        ckpt_ema_natoms = [
            k for k in state_dict if "model_ema" in k and "natoms_cfg" in k
        ]
        print(f"  checkpoint EMA natoms keys ({len(ckpt_ema_natoms)}):")
        for k in ckpt_ema_natoms[:5]:
            print(f"    {k}: {state_dict[k].shape}")
    else:
        print("  WARNING: natoms_cfg_embedding is None!")

    test_batch = next(iter(test_dl))
    mol_t, mol_1 = test_batch
    mol_t = mol_t.to(device)
    mol_1 = mol_1.to(device)
    eval_model = module._get_model()
    eval_model.set_inference()

    with torch.no_grad():
        t_diag = torch.zeros(mol_t.batch_size, device=device)
        preds_uncond = eval_model(
            mol_t,
            t_diag.view(-1, 1),
            properties=None,
            target_n_atoms=None,
        )
        target_10 = torch.full((mol_t.batch_size,), 10, dtype=torch.long, device=device)
        preds_cond_10 = eval_model(
            mol_t,
            t_diag.view(-1, 1),
            properties=None,
            target_n_atoms=target_10,
        )
        target_28 = torch.full((mol_t.batch_size,), 28, dtype=torch.long, device=device)
        preds_cond_28 = eval_model(
            mol_t,
            t_diag.view(-1, 1),
            properties=None,
            target_n_atoms=target_28,
        )

    print(f"\n  Comparing uncond vs cond(10) vs cond(28) predictions:")
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
        diff_10 = (c10 - u).abs().mean().item()
        diff_28 = (c28 - u).abs().mean().item()
        diff_10_28 = (c28 - c10).abs().mean().item()
        print(
            f"    {key:20s}  |uncond-cond10|={diff_10:.6f}  "
            f"|uncond-cond28|={diff_28:.6f}  "
            f"|cond10-cond28|={diff_10_28:.6f}"
        )

    module.cfg_adapter.natoms_cfg_guidance_scale = 5.0
    print(
        f"\n  should_use_natoms_cfg(target_10) with scale=5.0: "
        f"{module.cfg_adapter.should_use_natoms_cfg(target_10)}"
    )
    module.cfg_adapter.natoms_cfg_guidance_scale = 0.0
    print(
        f"  should_use_natoms_cfg(target_10) with scale=0.0: "
        f"{module.cfg_adapter.should_use_natoms_cfg(target_10)}"
    )
    print("--- End Diagnostic ---\n")

    # ── sweep over guidance scales x target n_atoms ──────────────────
    all_results: dict[tuple[float, int], dict] = {}
    module.predict_return_traj = False

    for scale in GUIDANCE_SCALES:
        module.cfg_adapter.natoms_cfg_guidance_scale = scale
        print(f"\n{'=' * 60}")
        print(f"Guidance scale = {scale}")
        print(f"{'=' * 60}")

        for n_atoms in TARGET_N_ATOMS:
            pl.seed_everything(42)
            print(f"  target_n_atoms={n_atoms} ... ", end="", flush=True)
            module.predict_target_n_atoms_override = n_atoms
            predictions = trainer.predict(
                module,
                dataloaders=test_dl,
            )
            result = summarize_natoms_steering(predictions, target_n_atoms=n_atoms)
            result["guidance_scale"] = scale
            all_results[(scale, n_atoms)] = result

            print(
                f"mean={result['mean_n_atoms']:.2f}  "
                f"std={result['std_n_atoms']:.2f}  "
                f"exact={result['exact_match_rate']:.1%}  "
                f"±1={result['within_1_rate']:.1%}  "
                f"±2={result['within_2_rate']:.1%}  "
                f"(n={result['n_molecules']})"
            )

    torch.save(all_results, "cfg_steering_results.pt")
    print(f"\nResults saved to cfg_steering_results.pt")


@hydra.main(
    config_path="configs",
    config_name="default",
    version_base="1.1",
)
def main(cfg: omegaconf.DictConfig):
    eval_cfg(cfg)


if __name__ == "__main__":
    main()
