"""Load pretrained checkpoints for fine-tuning, including DiT→CrossAttnDiT.

The unconditional DiT checkpoints store backbone weights at
``state_dict["model.backbone.<param>"]``. When fine-tuning with a
``CrossAttnDiTBackboneWithHeads`` (which composes a ``DiTBackbone`` as
``self.backbone.dit``), the matching destination keys are
``backbone.dit.<param>``. This helper rewrites keys accordingly and uses
``load_state_dict(strict=False)`` so the cross-attn blocks (zero-gated at init)
and any other new modules remain at their initial values.

Use via :func:`load_pretrained_into_lightning_model` from
:class:`LightningModuleRates.__init__` when ``pretrained_ckpt`` is set.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import torch

LOGGER = logging.getLogger(__name__)


def _extract_lightning_state_dict(
    ckpt: dict, prefer_ema: bool = False
) -> dict[str, torch.Tensor]:
    """Pull the model state dict out of a raw Lightning checkpoint.

    The Lightning module wraps the actual model under ``model.<...>`` and an
    EMA copy under ``model_ema.<...>``. We return just the keys under
    ``model.`` (or ``model_ema.`` if ``prefer_ema``), with that prefix
    stripped so they line up with ``self.model.load_state_dict`` callers.
    """
    sd = ckpt.get("state_dict", ckpt)
    src_prefix = "model_ema." if prefer_ema else "model."
    fallback = "model." if prefer_ema else "model_ema."

    matches = {k: v for k, v in sd.items() if k.startswith(src_prefix)}
    if not matches:
        matches = {k: v for k, v in sd.items() if k.startswith(fallback)}
        if not matches:
            raise ValueError(
                "No keys starting with 'model.' or 'model_ema.' in checkpoint."
            )
        src_prefix = fallback

    # Strip the outer wrapper and any torch.compile() ``_orig_mod.`` prefix.
    stripped: dict[str, torch.Tensor] = {}
    for k, v in matches.items():
        name = k[len(src_prefix) :]
        if name.startswith("_orig_mod."):
            name = name[len("_orig_mod.") :]
        stripped[name] = v
    LOGGER.info(
        "Extracted %d params from checkpoint under '%s'.", len(stripped), src_prefix
    )
    return stripped


def _retarget_backbone_keys_for_cross_attn(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Rewrite ``backbone.<param>`` → ``backbone.dit.<param>``.

    Used when loading a vanilla DiT checkpoint into a CrossAttnDiT
    backbone, which holds the DiT under ``self.backbone.dit``.
    """
    pattern = re.compile(r"^backbone\.(?!dit\.)(.+)")
    retargeted = {}
    for key, value in state_dict.items():
        match = pattern.match(key)
        if match:
            retargeted[f"backbone.dit.{match.group(1)}"] = value
        else:
            retargeted[key] = value
    return retargeted


def load_pretrained_into_lightning_model(
    model: torch.nn.Module,
    ckpt_path: str | Path,
    *,
    prefer_ema: bool = False,
    strict: bool = False,
) -> tuple[list[str], list[str]]:
    """Load a pretrained checkpoint into a Lightning model's ``self.model``.

    Automatically retargets vanilla-DiT keys to the CrossAttnDiT layout when
    necessary. Returns ``(missing, unexpected)`` from ``load_state_dict``.

    Args:
        model: the inner model (``LightningModuleRates.model``), NOT the
            outer Lightning module.
        ckpt_path: Path to the Lightning checkpoint (``.ckpt`` file).
        prefer_ema: If True, load from ``model_ema.*`` keys (typically what
            you want for fine-tuning since EMA weights are smoother).
        strict: Forwarded to ``load_state_dict``. Default ``False`` because
            new modules (cross-attn blocks, edge_cond_proj) won't exist in
            the source checkpoint.
    """
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {ckpt_path}")

    LOGGER.info("Loading pretrained weights from %s (prefer_ema=%s)", ckpt_path, prefer_ema)
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = _extract_lightning_state_dict(raw, prefer_ema=prefer_ema)

    # Detect whether the destination model uses a CrossAttnDiT backbone
    # (which expects backbone weights under ``backbone.dit.*``).
    needs_retarget = any(
        name.startswith("backbone.dit.") for name, _ in model.named_parameters()
    )
    if needs_retarget:
        sd = _retarget_backbone_keys_for_cross_attn(sd)

    missing, unexpected = model.load_state_dict(sd, strict=strict)
    if missing:
        LOGGER.info(
            "Pretrained load: %d missing keys (expected for new modules like "
            "cross_attn_blocks). First few: %s",
            len(missing),
            missing[:5],
        )
    if unexpected:
        LOGGER.warning(
            "Pretrained load: %d unexpected keys (may indicate arch mismatch). "
            "First few: %s",
            len(unexpected),
            unexpected[:5],
        )
    return missing, unexpected


def freeze_backbone(model: torch.nn.Module) -> int:
    """Freeze the pretrained backbone, keep cross-attn / new modules trainable.

    For a ``CrossAttnDiTBackboneWithHeads``: freezes ``backbone.dit.*``,
    leaves ``backbone.cross_attn_blocks.*``, ``backbone.edge_cond_proj.*``,
    ``backbone.edge_cond_gate``, ``backbone.prop_proj.*``, ``backbone.prop_gate``,
    ``embedding_backbone.*``, and ``heads.*`` trainable. For other
    architectures: freezes everything under ``backbone.*``. Returns the number
    of frozen parameters.
    """
    trainable_under_backbone = (
        "backbone.cross_attn_blocks.",
        "backbone.edge_cond_proj.",
        "backbone.edge_cond_gate",
        "backbone.prop_proj.",
        "backbone.prop_gate",
    )
    frozen = 0
    for name, param in model.named_parameters():
        if name.startswith("backbone.dit.") or (
            name.startswith("backbone.")
            and not any(name.startswith(prefix) for prefix in trainable_under_backbone)
        ):
            param.requires_grad = False
            frozen += param.numel()
    LOGGER.info("Froze %d backbone parameters for fine-tuning.", frozen)
    return frozen
