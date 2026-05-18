"""Lightning modules specialised for fine-tuning from a pretrained checkpoint.

Two subclasses of :class:`LightningModuleRates`:

* :class:`LightningModuleRatesFinetune` — loads weights from a pretrained
  unconditional DiT checkpoint into ``self.model`` after the regular
  instantiation step. When the backbone is a CrossAttnDiT, the loader
  retargets keys so the DiT weights land under ``self.model.backbone.dit.*``
  and cross-attention blocks keep their zero-gated init. Optionally
  freezes the pretrained backbone.

* :class:`LightningModuleRatesScaffoldFinetune` — fine-tune variant for
  scaffold-decoration / molecule-optimization. Implements the two scaffold
  hooks from the base (``_node_loss_exclusion_mask`` and
  ``_apply_inference_edit_masks``) so scaffold atoms are excluded from the
  relevant training losses (configurable) and their edit channels are
  hard-zeroed at inference time. Insertions seeded *from* scaffold atoms
  remain allowed so decorations can grow.

Use via Hydra by setting ``model.module._target_`` to
``chemflow.model.lightning_module_finetune.LightningModuleRatesScaffoldFinetune``
and providing ``pretrained_ckpt: <path>``.
"""

from __future__ import annotations

import torch

from chemflow.model.lightning_module_transformer import LightningModuleRatesTransformer
from chemflow.model.pretrained import (
    freeze_backbone,
    load_pretrained_into_lightning_model,
)


class LightningModuleRatesFinetune(LightningModuleRatesTransformer):
    """Fine-tuning variant that loads pretrained DiT weights at init.

    Inherits from :class:`LightningModuleRatesTransformer` so the
    Muon-with-AuxAdam optimizer (configured in ``configs/model/dit.yaml``)
    is available; the cross-attn / scaffold-encoder parameters fall into
    the high-lr non-backbone Adam group automatically.
    """

    def __init__(
        self,
        *args,
        pretrained_ckpt: str | None = None,
        pretrained_prefer_ema: bool = True,
        freeze_pretrained_backbone: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if pretrained_ckpt:
            load_pretrained_into_lightning_model(
                self.model,
                pretrained_ckpt,
                prefer_ema=pretrained_prefer_ema,
            )
            # Re-sync the EMA copy so it starts from the loaded weights,
            # not from the freshly instantiated (random) ones.
            self.model_ema.load_state_dict(self.model.state_dict())

            if freeze_pretrained_backbone:
                freeze_backbone(self.model)
                # Mirror the freeze on the EMA copy for parameter-count parity.
                freeze_backbone(self.model_ema)


class LightningModuleRatesScaffoldFinetune(LightningModuleRatesFinetune):
    """Scaffold-aware fine-tuning.

    On top of pretrained loading from :class:`LightningModuleRatesFinetune`,
    implements the two scaffold hooks defined on the base
    :class:`LightningModuleRates`:

    * ``_node_loss_exclusion_mask`` returns ``mols_t.scaffold_mask`` (or a
      ``None`` mask when the input batch doesn't carry one) so scaffold
      atoms are dropped from substitution / non-deletion / position losses
      when the corresponding flag is set.
    * ``_apply_inference_edit_masks`` zeroes ``do_del`` / ``do_sub_a`` for
      scaffold atoms and ``do_sub_e`` for scaffold-scaffold edges before
      they reach the integrator. ``num_ins_pred`` is left untouched so the
      model can still spawn new atoms from scaffold seeds (decoration).

    The flags ``exclude_scaffold_from_sub_loss`` and
    ``exclude_scaffold_from_x_loss`` default to ``True`` — the typical
    fine-tuning setting where scaffold atoms are frozen at inference, so
    training them adds no useful signal and only contributes noise.
    """

    def __init__(
        self,
        *args,
        exclude_scaffold_from_sub_loss: bool = True,
        exclude_scaffold_from_x_loss: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._exclude_scaffold_from_sub_loss = exclude_scaffold_from_sub_loss
        self._exclude_scaffold_from_x_loss = exclude_scaffold_from_x_loss

    def _node_loss_exclusion_mask(self, mols_t) -> torch.Tensor | None:
        # Either flag enables the exclusion path; the base ANDs the mask
        # into ``non_del_mask`` (covering x and sub losses together). If
        # only one of the two channels should be excluded the user can
        # subclass further; for the common case where both are excluded
        # in lockstep this returns the scaffold mask whenever any flag is on.
        if not (
            self._exclude_scaffold_from_sub_loss or self._exclude_scaffold_from_x_loss
        ):
            return None
        sc_mask = getattr(mols_t, "scaffold_mask", None)
        if sc_mask is None:
            return None
        return sc_mask

    def _apply_inference_edit_masks(
        self,
        mol_t,
        do_sub_a_probs,
        do_sub_e_probs,
        do_del_probs,
        num_ins_pred,
    ):
        sc_mask = getattr(mol_t, "scaffold_mask", None)
        if sc_mask is None:
            return do_sub_a_probs, do_sub_e_probs, do_del_probs, num_ins_pred
        sc_node = sc_mask.bool()
        do_sub_a_probs = do_sub_a_probs.clone()
        do_del_probs = do_del_probs.clone()
        do_sub_a_probs[sc_node] = 0.0
        do_del_probs[sc_node] = 0.0
        sc_edge = sc_node[mol_t.edge_index[0]] & sc_node[mol_t.edge_index[1]]
        do_sub_e_probs = do_sub_e_probs.clone()
        do_sub_e_probs[sc_edge] = 0.0
        # num_ins_pred intentionally untouched: scaffold atoms are valid
        # insertion seeds for decoration growth.
        return do_sub_a_probs, do_sub_e_probs, do_del_probs, num_ins_pred
