from __future__ import annotations

import math

import torch
import torch.nn as nn

from chemflow.model.learnable_loss import UnifiedWeightedLoss

class LossAccumulator:
    """Per-step helper that collects raw losses, applies weights, and builds
    a uniform log dict with raw components, weighted group totals, and stats.

    Usage::

        acc = LossAccumulator(weight_module, groups, device)
        acc.set_losses({"x": x_loss, "c": c_loss, ...})
        acc.add_stat("n_ins", n_inserts)

        total = acc.total_loss()       # weighted scalar
        entries = acc.log_dict()       # ready for pl.log_dict

    Args:
        weight_module: ``UnifiedWeightedLoss`` instance that owns the weights.
        groups: Mapping from group name to list of component keys.
            Each group produces a single ``loss/{group}_weighted`` entry.
        device: Device for fallback zero tensors.
    """

    def __init__(
        self,
        weight_module: UnifiedWeightedLoss,
        groups: dict[str, list[str]],
        device: torch.device,
        time_weight_modules: dict[str, callable] | None = None,
    ):
        self._weight_module = weight_module
        self._groups = groups
        self._device = device
        self._weights = weight_module.get_weight_tensors(device)
        self._losses: dict[str, torch.Tensor] = {}
        self._stats: dict[str, torch.Tensor | float] = {}

        # Store the time weighting modules (defaulting to empty dict)
        self._time_weight_modules = time_weight_modules or {}

    # ── Population ───────────────────────────────────────────────────

    def set_batch_losses(
        self, 
        batch_losses: dict[str, torch.Tensor], 
        t: torch.Tensor | None = None
    ):
        """Applies specific time-weighting to unreduced batch losses and averages them.
        
        Args:
            batch_losses: Dict of unreduced loss tensors (e.g., shape [Batch, ...])
            t: Optional 1D tensor of shape [Batch] containing the time steps.
        """
        self._losses.clear()
        
        for key, loss_b in batch_losses.items():
            if t is not None:
                # 1. Fetch the specific time-weighting module for this loss key.
                # If one wasn't registered, fallback to a safe constant 1.0
                if key in self._time_weight_modules:
                    time_weights = self._time_weight_modules[key](t)
                else:
                    time_weights = torch.ones_like(t)
                
                # 2. Broadcast time_weights to match loss_b dimensions
                # e.g., if loss_b is [Batch, NumNodes, Dim], time_weights becomes [Batch, 1, 1]
                target_shape = [-1] + [1] * (loss_b.ndim - 1)
                tw_broadcast = time_weights.view(*target_shape)
                
                # 3. Apply the time weight
                loss_b = loss_b * tw_broadcast
                
            # 4. Reduce to scalar after weighting
            self._losses[key] = loss_b.mean()


    def set_losses(self, losses: dict[str, torch.Tensor]):
        """
        Set all loss components at once (replaces any previous).
        Assumes losses are already reduced to scalars.
        """
        self._losses = losses

    def add(self, key: str, value: torch.Tensor):
        """Add or overwrite a single loss component."""
        self._losses[key] = value

    def add_stat(self, key: str, value):
        """Add a single statistic for logging."""
        self._stats[key] = value

    def add_stats(self, stats: dict):
        """Merge multiple statistics for logging."""
        self._stats.update(stats)

    # ── Outputs ──────────────────────────────────────────────────────

    def total_loss(self) -> torch.Tensor:
        """Compute total weighted loss via the weight module."""
        return self._weight_module(self._losses)

    def log_dict(self) -> dict[str, torch.Tensor | float]:
        """Build a uniform log dict.

        Sections emitted:
        * ``loss/{component}``            — raw (unweighted) per-component value
        * ``loss_weighted/{component}``   — weighted per-component value
        * ``loss_weighted/{group}``       — sum of weighted components per group
        * ``loss/total``                   — sum of all raw components
        * ``loss_weighted/total``         — sum of all weighted components
        * ``stats/{key}``                 — additional statistics
        * ``weight/{component}``          — current weight values (learnable only)
        """
        entries: dict[str, torch.Tensor | float] = {}

        total_unweighted = 0.0
        total_weighted = 0.0
        for key, val in self._losses.items():
            w_val = self._weights[key] * val
            entries[f"loss/{key}"] = val
            entries[f"loss_weighted/{key}"] = w_val
            total_unweighted = total_unweighted + val
            total_weighted = total_weighted + w_val

        for group, keys in self._groups.items():
            terms = [
                self._weights[k] * self._losses[k]
                for k in keys
                if k in self._losses
            ]
            if terms:
                entries[f"loss_weighted/{group}"] = sum(terms)

        entries["loss/total"] = total_unweighted
        entries["loss_weighted/total"] = total_weighted

        for key, val in self._stats.items():
            entries[f"stats/{key}"] = val

        if self._weight_module.use_learnable:
            for k, w in self._weights.items():
                entries[f"weight/{k}"] = w

        return entries