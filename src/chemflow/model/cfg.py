from __future__ import annotations

import torch
import torch.nn.functional as F

from chemflow.model.embedding import compute_molecular_weight


class CFGAdapter:
    """Classifier-free guidance adapter.

    Manages per-signal dropout, guidance scales, and multi-pass CFG
    inference for property, n_atoms, and molecular-weight conditioning.
    """

    # Head categories for type-aware classifier-free guidance.
    _cfg_linear_keys = {
        "do_ins_head",
        "do_del_head",
        "do_sub_a_head",
        "do_sub_e_head",
    }
    _cfg_rate_keys: set[str] = set()
    _cfg_use_cond_keys = {
        "pos_head",
        "atom_type_head",
        "charge_head",
        "edge_type_head",
        "ins_rate_head",
    }

    def __init__(
        self,
        model: torch.nn.Module,
        cfg_dropout_prob: float = 0.0,
        cfg_guidance_scale: float = 0.0,
        natoms_cfg_dropout_prob: float = 0.15,
        natoms_cfg_guidance_scale: float = 2.5,
        mw_cfg_dropout_prob: float = 0.15,
        mw_cfg_guidance_scale: float = 0.0,
        atom_tokens: list[str] | None = None,
    ):
        self.model = model
        self.cfg_dropout_prob = float(cfg_dropout_prob)
        self.cfg_guidance_scale = float(cfg_guidance_scale)
        self.natoms_cfg_dropout_prob = float(natoms_cfg_dropout_prob)
        self.natoms_cfg_guidance_scale = float(natoms_cfg_guidance_scale)
        self.mw_cfg_dropout_prob = float(mw_cfg_dropout_prob)
        self.mw_cfg_guidance_scale = float(mw_cfg_guidance_scale)
        self.atom_tokens = atom_tokens

    @property
    def _cfg_embedding(self):
        return getattr(
            self.model.embedding_backbone, "cfg_embedding", None
        )

    @property
    def _has_property_conditioning(self) -> bool:
        e = self._cfg_embedding
        return e is not None and getattr(e, "property_encoder", None) is not None

    @property
    def _has_natoms_cfg(self) -> bool:
        e = self._cfg_embedding
        return e is not None and getattr(e, "natoms_encoder", None) is not None

    @property
    def _has_mw_cfg(self) -> bool:
        e = self._cfg_embedding
        return e is not None and getattr(e, "mw_encoder", None) is not None

    def extract_properties(self, mols_t) -> torch.Tensor | None:
        if not self._has_property_conditioning:
            return None
        if hasattr(mols_t, "y") and mols_t.y is not None:
            return mols_t.y.float()
        return None

    def extract_target_n_atoms(self, mols) -> torch.Tensor | None:
        if not self._has_natoms_cfg:
            return None
        return torch.bincount(mols.batch)

    def extract_target_mw(
        self, mols, batch: torch.Tensor | None = None
    ) -> torch.Tensor | None:
        if not self._has_mw_cfg or self.atom_tokens is None:
            return None
        b = batch if batch is not None else mols.batch
        return compute_molecular_weight(
            mols.a, self.atom_tokens, b, mols.num_graphs
        )

    def _sample_drop_mask(
        self, prob: float, batch_size: int,
        device: torch.device, training: bool,
    ) -> torch.Tensor | None:
        if prob <= 0.0 or not training:
            return None
        return torch.rand(batch_size, device=device) < prob

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

    def should_use_mw_cfg(self, target_mw) -> bool:
        return (
            self.mw_cfg_guidance_scale > 0.0
            and target_mw is not None
            and self._has_mw_cfg
        )

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
        """Build the ``cfg_inputs`` dict for a training forward pass."""
        bs = mols_t.num_graphs
        return {
            "properties": self.extract_properties(mols_t),
            "property_drop_mask": self._sample_drop_mask(
                self.cfg_dropout_prob, bs, device, training,
            ),
            "target_n_atoms": self.extract_target_n_atoms(mols_1),
            "natoms_drop_mask": self._sample_drop_mask(
                self.natoms_cfg_dropout_prob, bs, device, training,
            ),
            "target_mw": self.extract_target_mw(mols_1),
            "mw_drop_mask": self._sample_drop_mask(
                self.mw_cfg_dropout_prob, bs, device, training,
            ),
        }

    def _uncond_cfg_inputs(self) -> dict:
        """Return a cfg_inputs dict with every signal set to None."""
        return {
            "properties": None,
            "property_drop_mask": None,
            "target_n_atoms": None,
            "natoms_drop_mask": None,
            "target_mw": None,
            "mw_drop_mask": None,
        }

    def guided_predict(
        self,
        model,
        mol_t,
        t: torch.Tensor,
        prev_preds,
        cfg_inputs: dict,
    ) -> dict:
        """Run CFG inference: unconditional + per-signal guided passes."""
        uncond = self._uncond_cfg_inputs()

        preds = model(
            mol_t, t.view(-1, 1),
            prev_outs=prev_preds,
            cfg_inputs=uncond,
        )

        target_n_atoms = cfg_inputs.get("target_n_atoms")
        target_mw = cfg_inputs.get("target_mw")
        properties = cfg_inputs.get("properties")

        if self.should_use_natoms_cfg(target_n_atoms):
            cond = {**uncond, "target_n_atoms": target_n_atoms}
            preds_cond = model(
                mol_t, t.view(-1, 1),
                prev_outs=prev_preds,
                cfg_inputs=cond,
            )
            preds = self.apply_cfg(
                preds_cond, preds, self.natoms_cfg_guidance_scale,
            )

        if self.should_use_mw_cfg(target_mw):
            cond = {**uncond, "target_mw": target_mw}
            preds_cond = model(
                mol_t, t.view(-1, 1),
                prev_outs=prev_preds,
                cfg_inputs=cond,
            )
            preds = self.apply_cfg(
                preds_cond, preds, self.mw_cfg_guidance_scale,
            )

        if self.should_use_property_cfg(properties):
            preds_cond = model(
                mol_t, t.view(-1, 1),
                prev_outs=prev_preds,
                cfg_inputs=cfg_inputs,
            )
            preds = self.apply_cfg(
                preds_cond, preds, self.cfg_guidance_scale,
            )

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
