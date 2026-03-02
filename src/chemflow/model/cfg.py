from __future__ import annotations

import torch
import torch.nn.functional as F


class CFGAdapter:
    """Classifier-free guidance adapter for both property and n_atoms conditioning.

    Encapsulates all CFG parameters, masking logic, and the type-aware
    guidance interpolation used during training and inference.
    """

    # Head categories for type-aware classifier-free guidance.
    # Linear extrapolation (positions, logits before softmax/sigmoid)
    _cfg_linear_keys = {
        "atom_type_head",
        "charge_head",
        "edge_type_head",
        "do_ins_head",
        "do_del_head",
        "do_sub_a_head",
        "do_sub_e_head",
    }
    # Log-space CFG for strictly-positive rate outputs (post-softplus)
    _cfg_rate_keys = {"ins_rate_head"}
    # Use conditional predictions directly
    _cfg_use_cond_keys = {"pos_head"}

    def __init__(
        self,
        model: torch.nn.Module,
        cfg_dropout_prob: float = 0.0,
        cfg_guidance_scale: float = 0.0,
        natoms_cfg_dropout_prob: float = 0.15,
        natoms_cfg_guidance_scale: float = 2.5,
    ):
        self.model = model
        self.cfg_dropout_prob = float(cfg_dropout_prob)
        self.cfg_guidance_scale = float(cfg_guidance_scale)
        self.natoms_cfg_dropout_prob = float(natoms_cfg_dropout_prob)
        self.natoms_cfg_guidance_scale = float(natoms_cfg_guidance_scale)

    @property
    def _has_property_conditioning(self) -> bool:
        return self.model.embedding_backbone.property_embedding is not None

    @property
    def _has_natoms_cfg(self) -> bool:
        return self.model.embedding_backbone.natoms_cfg_embedding is not None

    def extract_properties(self, mols_t) -> torch.Tensor | None:
        """Return graph-level properties from the batch, or None."""
        if not self._has_property_conditioning:
            return None
        if hasattr(mols_t, "y") and mols_t.y is not None:
            return mols_t.y.float()
        return None

    def extract_target_n_atoms(self, mols) -> torch.Tensor | None:
        """Return per-graph target atom counts, or None if n_atoms CFG is off."""
        if not self._has_natoms_cfg:
            return None
        return torch.bincount(mols.batch)

    def sample_cfg_drop_mask(
        self, batch_size: int, device: torch.device, training: bool
    ) -> torch.Tensor | None:
        """During training, randomly mask out properties for CFG."""
        if self.cfg_dropout_prob <= 0.0 or not training:
            return None
        return torch.rand(batch_size, device=device) < self.cfg_dropout_prob

    def sample_natoms_cfg_drop_mask(
        self, batch_size: int, device: torch.device, training: bool
    ) -> torch.Tensor | None:
        """During training, randomly mask out target n_atoms for CFG."""
        if self.natoms_cfg_dropout_prob <= 0.0 or not training:
            return None
        return torch.rand(batch_size, device=device) < self.natoms_cfg_dropout_prob

    def apply_cfg(
        self,
        preds_cond: dict,
        preds_uncond: dict,
        guidance_scale: float,
    ) -> dict:
        """Type-aware classifier-free guidance."""
        guided = {}
        w = guidance_scale

        for key, v_cond in preds_cond.items():
            v_uncond = preds_uncond[key]

            if key in self._cfg_linear_keys:
                guided[key] = v_uncond + w * (v_cond - v_uncond)

            elif key in self._cfg_use_cond_keys:
                guided[key] = v_cond

            elif key in self._cfg_rate_keys:
                log_cond = torch.log(v_cond.clamp_min(1e-12))
                log_uncond = torch.log(v_uncond.clamp_min(1e-12))
                guided[key] = torch.exp(log_uncond + w * (log_cond - log_uncond))

            elif key == "gmm_head" and isinstance(v_cond, dict):
                guided_gmm = {}
                for k, gv_cond in v_cond.items():
                    gv_uncond = v_uncond[k]
                    if k in {"pi", "a_probs", "c_probs"}:
                        cond_logits = torch.log(gv_cond.clamp_min(1e-12))
                        uncond_logits = torch.log(gv_uncond.clamp_min(1e-12))
                        guided_logits = uncond_logits + w * (
                            cond_logits - uncond_logits
                        )
                        guided_gmm[k] = F.softmax(guided_logits, dim=-1)
                    else:
                        guided_gmm[k] = gv_cond
                guided[key] = guided_gmm

            else:
                guided[key] = v_cond

        return guided

    def get_training_inputs(
        self, mols_t, mols_1, device: torch.device, training: bool
    ) -> dict:
        """Prepare all CFG-related inputs for a training forward pass.

        Returns a dict with keys: properties, cfg_drop_mask,
        target_n_atoms, natoms_drop_mask.
        """
        properties = self.extract_properties(mols_t)
        cfg_drop_mask = self.sample_cfg_drop_mask(
            mols_t.num_graphs, device, training
        )
        target_n_atoms = self.extract_target_n_atoms(mols_1)
        natoms_drop_mask = self.sample_natoms_cfg_drop_mask(
            mols_t.num_graphs, device, training
        )
        return dict(
            properties=properties,
            cfg_drop_mask=cfg_drop_mask,
            target_n_atoms=target_n_atoms,
            natoms_drop_mask=natoms_drop_mask,
        )

    def should_use_property_cfg(self, properties) -> bool:
        return (
            self.cfg_guidance_scale > 0.0
            and properties is not None
            and self._has_property_conditioning
        )

    def should_use_natoms_cfg(self, target_n_atoms) -> bool:
        return (
            self.natoms_cfg_guidance_scale > 0.0
            and target_n_atoms is not None
            and self._has_natoms_cfg
        )

    def guided_predict(
        self,
        model,
        mol_t,
        t: torch.Tensor,
        prev_preds,
        properties: torch.Tensor | None,
        target_n_atoms: torch.Tensor | None,
    ) -> dict:
        """Run the full CFG inference: unconditional pass + optional guided passes.

        Returns the (possibly guided) prediction dict.
        """
        preds = model(
            mol_t,
            t.view(-1, 1),
            prev_outs=prev_preds,
            properties=None,
            target_n_atoms=None,
        )

        if self.should_use_natoms_cfg(target_n_atoms):
            preds_cond_natoms = model(
                mol_t,
                t.view(-1, 1),
                prev_outs=prev_preds,
                properties=None,
                target_n_atoms=target_n_atoms,
            )
            preds = self.apply_cfg(
                preds_cond_natoms, preds, self.natoms_cfg_guidance_scale
            )

        if self.should_use_property_cfg(properties):
            preds_cond = model(
                mol_t,
                t.view(-1, 1),
                prev_outs=prev_preds,
                properties=properties,
                target_n_atoms=target_n_atoms,
            )
            preds = self.apply_cfg(preds_cond, preds, self.cfg_guidance_scale)

        return preds


@torch.no_grad()
def evaluate_cfg_steering(
    module,
    dataloader,
    target_n_atoms: int,
    num_batches: int | None = None,
) -> dict:
    """Evaluate how well n_atoms CFG steers generation toward a target size.

    Samples molecules from ``dataloader`` with ``target_n_atoms`` overridden
    and reports statistics about the actually generated atom counts.

    Args:
        module: The lightning module (must expose ``.sample()``).
        dataloader: A predict/test dataloader yielding ``(mol_t, mol_1)`` batches.
        target_n_atoms: The desired atom count to condition on.
        num_batches: If set, only process this many batches (useful for quick checks).

    Returns:
        Dictionary with keys:
            target_n_atoms, mean_n_atoms, std_n_atoms,
            median_n_atoms, exact_match_rate, within_1_rate,
            within_2_rate, n_molecules, all_n_atoms (list).
    """
    all_n_atoms: list[int] = []

    for batch_idx, batch in enumerate(dataloader):
        if num_batches is not None and batch_idx >= num_batches:
            break

        mol_t, mol_1 = batch
        # In standalone eval (outside Lightning Trainer predict), dataloader
        # batches stay on CPU; move them to the same device as the module.
        mol_t = mol_t.to(module.device)
        mol_1 = mol_1.to(module.device)
        batch_size = mol_t.batch_size
        override = torch.full(
            (batch_size,), target_n_atoms, dtype=torch.long, device=module.device
        )

        generated = module.sample(
            batch,
            batch_idx,
            return_traj=False,
            target_n_atoms_override=override,
        )

        # Fast path: avoid expensive Python-side to_data_list() conversion.
        # Per-graph atom counts are directly available from the batch vector.
        batch_n_atoms = torch.bincount(
            generated.batch, minlength=generated.batch_size
        ).to(dtype=torch.long)
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
