"""Conditioning-signal abstraction.

A `ConditioningSignal` bundles everything that defines one CFG signal:

  * `name`      — string key used in `overrides` / `drop_masks` dicts
  * `out_dim`   — embedding width
  * `cfg_mode`  — how `apply_cfg` blends cond/uncond head outputs
  * `dropout_prob` / `guidance_scale` — training/inference hparams
  * `null_emb`  — learnable replacement for "this signal is unavailable"
  * `encoder`   — `nn.Module` mapping the raw value to `[B, out_dim]`
  * `extract()` — pulls the raw value out of a `MoleculeBatch`

Adding a new signal = subclassing this ABC. The Hydra config instantiates a
list of these directly; the embedding wraps them in a `nn.ModuleList`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig

from chemflow.dataset.qm9 import QM9_PROPERTY_NAMES
from chemflow.model.cfg.extractors import (
    compute_logp,
    compute_molecular_weight,
    compute_qed,
)
from chemflow.model.embedding import SinusoidalEncoding


CFGMode = Literal["linear", "rate", "cond_only"]


class ConditioningSignal(nn.Module, ABC):
    """Base class for one CFG signal.

    Subclasses override `extract()` and supply an `encoder` in `__init__`.
    Encoders must take `[B]` or `[B, *]` tensors and return `[B, out_dim]`,
    and must expose an `out_dim` attribute (or be sized via `out_dim` arg).
    """

    name: str

    def __init__(
        self,
        out_dim: int,
        dropout_prob: float = 0.0,
        guidance_scale: float = 0.0,
        cfg_mode: CFGMode = "linear",
    ):
        super().__init__()
        self.out_dim = int(out_dim)
        self.dropout_prob = float(dropout_prob)
        self.guidance_scale = float(guidance_scale)
        self.cfg_mode = cfg_mode
        self.null_emb = nn.Parameter(torch.randn(self.out_dim) * 0.02)

    @abstractmethod
    def extract(self, mols, ctx: dict[str, Any]) -> torch.Tensor | None:
        """Pull the raw conditioning value out of `mols`.

        `ctx` carries cross-cutting extras (`atom_tokens`, `ins_targets`, ...)
        without bloating the signature. Return `None` if the value is
        unavailable so the caller substitutes the null embedding.
        """

    def encode(
        self,
        value: torch.Tensor | None,
        drop_mask: torch.Tensor | None,
        batch_size: int,
    ) -> torch.Tensor:
        """Encode `value` to `[B, out_dim]`; null + DDP shim handled here."""
        if value is None:
            return self.null_emb.unsqueeze(0).expand(batch_size, -1)

        if value.ndim == 1 and value.dtype.is_floating_point:
            value = value.unsqueeze(-1)
        emb = self.encoder(value)

        if drop_mask is not None:
            null = self.null_emb.unsqueeze(0).expand_as(emb)
            emb = torch.where(drop_mask.unsqueeze(-1), null, emb)
        else:
            # When dropout_prob=0 the null branch never trains. Add a
            # zero-weight reference so DDP keeps null_emb in the graph.
            emb = emb + self.null_emb.sum() * 0
        return emb


# ---------------------------------------------------------------------------
# Concrete signals
# ---------------------------------------------------------------------------


class PropertySignal(ConditioningSignal):
    """Property-vector conditioning, sliced from `mols.y` by name list."""

    name = "properties"

    def __init__(
        self,
        property_names: list[str],
        hidden_dim: int = 128,
        dropout_prob: float = 0.0,
        guidance_scale: float = 0.0,
        cfg_mode: CFGMode = "linear",
    ):
        super().__init__(
            out_dim=hidden_dim,
            dropout_prob=dropout_prob,
            guidance_scale=guidance_scale,
            cfg_mode=cfg_mode,
        )
        unknown = [n for n in property_names if n not in QM9_PROPERTY_NAMES]
        if unknown:
            raise ValueError(
                f"Unknown property name(s): {unknown}. "
                f"Valid: {sorted(QM9_PROPERTY_NAMES)}"
            )
        self.property_names = list(property_names)
        self.property_indices = [QM9_PROPERTY_NAMES[n] for n in property_names]
        num = len(property_names)
        self.encoder = nn.Sequential(
            nn.Linear(num, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        _xavier_init(self)

    def extract(self, mols, ctx):
        y = getattr(mols, "y", None)
        if y is None:
            return None
        return y.float()[:, self.property_indices]


class NAtomsSignal(ConditioningSignal):
    """Per-graph target atom count.

    During training this needs to anticipate atoms that *will* be inserted —
    the wrapper passes the OT-aligned `target_state` plus `ins_targets` via
    `ctx['ins_targets']` and the signal computes
    `(non-aux atoms in target_state) + (atoms staged for insertion)`.

    At inference the target is just `bincount(mols.batch)`.
    """

    name = "n_atoms"

    def __init__(
        self,
        encoder_args: DictConfig | nn.Module,
        dropout_prob: float = 0.0,
        guidance_scale: float = 0.0,
        cfg_mode: CFGMode = "linear",
    ):
        encoder = _resolve_encoder(encoder_args)
        if not hasattr(encoder, "out_dim"):
            raise AttributeError(
                f"NAtomsSignal encoder {type(encoder).__name__} must expose "
                "`out_dim` so the null embedding can be sized."
            )
        super().__init__(
            out_dim=int(encoder.out_dim),
            dropout_prob=dropout_prob,
            guidance_scale=guidance_scale,
            cfg_mode=cfg_mode,
        )
        self.encoder = encoder

    def extract(self, mols, ctx):
        device = mols.batch.device
        num_graphs = int(mols.num_graphs)
        ins_targets = ctx.get("ins_targets")
        if ins_targets is not None and hasattr(mols, "is_auxiliary"):
            is_real = (~mols.is_auxiliary).view(-1).to(torch.long)
            real_per_graph = torch.zeros(num_graphs, dtype=torch.long, device=device)
            real_per_graph.scatter_add_(0, mols.batch, is_real)
            ins_batch = getattr(ins_targets, "batch", None)
            if ins_batch is not None and ins_batch.numel() > 0:
                ins_per_graph = torch.bincount(ins_batch, minlength=num_graphs)
            else:
                ins_per_graph = torch.zeros(num_graphs, dtype=torch.long, device=device)
            return real_per_graph + ins_per_graph
        return torch.bincount(mols.batch, minlength=num_graphs)


class _SinusoidalScalarSignal(ConditioningSignal):
    """Shared base for scalar signals using `Sinusoidal -> MLP` (legacy path).

    If `encoder_args` is provided the external module is used end-to-end
    (e.g. `ExtrapolatableScalarEmbedding`); else the legacy two-stage encode
    is built from `sinusoidal_dim` / `max_period`.
    """

    def __init__(
        self,
        sinusoidal_dim: int,
        max_period: float,
        dropout_prob: float,
        guidance_scale: float,
        cfg_mode: CFGMode,
        encoder_args: DictConfig | nn.Module | None = None,
    ):
        if encoder_args is not None:
            external = _resolve_encoder(encoder_args)
            if not hasattr(external, "out_dim"):
                raise AttributeError(
                    f"{type(self).__name__} encoder {type(external).__name__} "
                    "must expose `out_dim`."
                )
            out_dim = int(external.out_dim)
        else:
            external = None
            out_dim = sinusoidal_dim

        super().__init__(
            out_dim=out_dim,
            dropout_prob=dropout_prob,
            guidance_scale=guidance_scale,
            cfg_mode=cfg_mode,
        )

        if external is not None:
            self.encoder = external
        else:
            self.encoder = nn.Sequential(
                SinusoidalEncoding(sinusoidal_dim, max_period=max_period),
                nn.Linear(sinusoidal_dim, sinusoidal_dim),
                nn.SiLU(),
                nn.Linear(sinusoidal_dim, sinusoidal_dim),
            )
        _xavier_init(self)


class MWSignal(_SinusoidalScalarSignal):
    """Molecular weight (Daltons), computed from atom indices + atom_tokens."""

    name = "mw"

    def __init__(
        self,
        sinusoidal_dim: int = 64,
        max_period: float = 1000.0,
        dropout_prob: float = 0.0,
        guidance_scale: float = 0.0,
        cfg_mode: CFGMode = "linear",
        encoder_args: DictConfig | None = None,
    ):
        super().__init__(
            sinusoidal_dim=sinusoidal_dim,
            max_period=max_period,
            dropout_prob=dropout_prob,
            guidance_scale=guidance_scale,
            cfg_mode=cfg_mode,
            encoder_args=encoder_args,
        )

    def extract(self, mols, ctx):
        atom_tokens = ctx.get("atom_tokens")
        if atom_tokens is None:
            return None
        return compute_molecular_weight(
            mols.a, atom_tokens, mols.batch, int(mols.num_graphs)
        )


class LogPSignal(_SinusoidalScalarSignal):
    """RDKit Crippen logP; reads cached `mols.logp` if present, else SMILES."""

    name = "logp"

    def __init__(
        self,
        sinusoidal_dim: int = 64,
        max_period: float = 20.0,
        dropout_prob: float = 0.0,
        guidance_scale: float = 0.0,
        cfg_mode: CFGMode = "linear",
        encoder_args: DictConfig | None = None,
    ):
        super().__init__(
            sinusoidal_dim=sinusoidal_dim,
            max_period=max_period,
            dropout_prob=dropout_prob,
            guidance_scale=guidance_scale,
            cfg_mode=cfg_mode,
            encoder_args=encoder_args,
        )

    def extract(self, mols, ctx):
        cached = getattr(mols, "logp", None)
        if cached is not None:
            return cached.float().to(_device_of(mols))
        smiles = getattr(mols, "smiles", None)
        if smiles is None:
            return None
        return compute_logp(smiles, device=_device_of(mols))


class QEDSignal(_SinusoidalScalarSignal):
    """RDKit QED in [0, 1]; reads cached `mols.qed` if present, else SMILES."""

    name = "qed"

    def __init__(
        self,
        sinusoidal_dim: int = 64,
        max_period: float = 4.0,
        dropout_prob: float = 0.0,
        guidance_scale: float = 0.0,
        cfg_mode: CFGMode = "linear",
        encoder_args: DictConfig | None = None,
    ):
        super().__init__(
            sinusoidal_dim=sinusoidal_dim,
            max_period=max_period,
            dropout_prob=dropout_prob,
            guidance_scale=guidance_scale,
            cfg_mode=cfg_mode,
            encoder_args=encoder_args,
        )

    def extract(self, mols, ctx):
        cached = getattr(mols, "qed", None)
        if cached is not None:
            return cached.float().to(_device_of(mols))
        smiles = getattr(mols, "smiles", None)
        if smiles is None:
            return None
        return compute_qed(smiles, device=_device_of(mols))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_encoder(encoder_args) -> nn.Module:
    """Accept either an already-instantiated module or a Hydra DictConfig."""
    if isinstance(encoder_args, nn.Module):
        return encoder_args
    return hydra.utils.instantiate(encoder_args)


def _xavier_init(module: nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def _device_of(mols) -> torch.device | None:
    x = getattr(mols, "x", None)
    if x is not None:
        return x.device
    batch = getattr(mols, "batch", None)
    if batch is not None:
        return batch.device
    return None
