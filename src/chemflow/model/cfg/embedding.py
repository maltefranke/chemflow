"""Unified CFG embedding: a list of `ConditioningSignal`s + a final projection.

The embedding is purely value-consuming: it does **not** call `signal.extract()`.
Callers (lightning module via `CFGGuidance.build_overrides`, inference via
`CFGGuidance.guided_predict`) pass already-extracted values in `overrides`.
This keeps the model forward independent of "target vs. current state"
ambiguity — the model only sees `mols_t` (current) and a dict of values
extracted from `mols_1` (target).
"""

from __future__ import annotations

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig

from chemflow.model.cfg.signals import ConditioningSignal


class CFGEmbedding(nn.Module):
    """Encode a configurable set of conditioning signals into one graph-level vector.

    Forward inputs:
      * ``batch_size`` — number of graphs in the current batch.
      * ``overrides`` — ``{signal_name: tensor[B, *]}``.  A missing entry
        means "this signal is unavailable" → the signal's null embedding is
        used.
      * ``drop_masks`` — ``{signal_name: bool tensor[B]}`` for per-signal
        classifier-free dropout (``True`` = replace with null).
    """

    def __init__(
        self,
        out_dim: int,
        signals: list[DictConfig | ConditioningSignal] | None = None,
    ):
        super().__init__()
        self.out_dim = int(out_dim)

        modules: list[ConditioningSignal] = []
        for s in signals or []:
            modules.append(
                s if isinstance(s, ConditioningSignal) else hydra.utils.instantiate(s)
            )

        names = [m.name for m in modules]
        if len(set(names)) != len(names):
            raise ValueError(f"Duplicate signal names in CFGEmbedding: {names}")
        self.signals = nn.ModuleList(modules)

        internal_dim = sum(m.out_dim for m in modules)
        if internal_dim > 0:
            self.projection = nn.Sequential(
                nn.Linear(internal_dim, out_dim),
                nn.SiLU(),
                nn.Linear(out_dim, out_dim),
            )
            for m in self.projection.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        else:
            self.projection = None

    @property
    def signal_names(self) -> list[str]:
        return [s.name for s in self.signals]

    def get_signal(self, name: str) -> ConditioningSignal | None:
        for s in self.signals:
            if s.name == name:
                return s
        return None

    def forward(
        self,
        batch_size: int,
        overrides: dict[str, torch.Tensor | None] | None = None,
        drop_masks: dict[str, torch.Tensor | None] | None = None,
    ) -> torch.Tensor:
        if not self.signals:
            device = next(self.parameters()).device if any(self.parameters()) else torch.device("cpu")
            return torch.zeros(batch_size, self.out_dim, device=device)

        overrides = overrides or {}
        drop_masks = drop_masks or {}

        parts = [
            s.encode(overrides.get(s.name), drop_masks.get(s.name), batch_size)
            for s in self.signals
        ]
        return self.projection(torch.cat(parts, dim=-1))
