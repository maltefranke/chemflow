from __future__ import annotations

import math

import torch
import torch.nn as nn


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
    ):
        self._weight_module = weight_module
        self._groups = groups
        self._device = device
        self._weights = weight_module.get_weight_tensors(device)
        self._losses: dict[str, torch.Tensor] = {}
        self._stats: dict[str, torch.Tensor | float] = {}

    # ── Population ───────────────────────────────────────────────────

    def set_losses(self, losses: dict[str, torch.Tensor]):
        """Set all loss components at once (replaces any previous)."""
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


class LearnableWeightedLoss(nn.Module):
    def __init__(self, initial_weights, component_keys):
        """
        initial_weights: dict of heuristic weights (e.g., {"loss1": 0.001, "loss2": 1.0})
        component_keys: ordered list of keys matching the order of losses in forward
        """
        super().__init__()
        self.component_keys = component_keys

        # Convert heuristic weights W to log-variance s
        # Formula: s = -ln(W)
        # We use a Parameter so the optimizer can update it.
        s_values = [-math.log(initial_weights[key]) for key in component_keys]
        self.s = nn.Parameter(torch.tensor(s_values, dtype=torch.float32))

    def forward(self, losses_dict):
        """
        losses_dict: dict of individual loss tensors
        """
        total_loss = 0

        for i, key in enumerate(self.component_keys):
            if key in losses_dict:
                loss = losses_dict[key]
                # Retrieve the learnable parameter s for this loss component
                s_i = self.s[i]

                # Apply the Kendall & Gal formula:
                # L_total = sum( exp(-s_i) * L_i + 0.5 * s_i )
                # Note: The factor 0.5 is technically for MSE/Gaussian,
                # but standard implementations often just use s_i or 0.5*s_i for the regularization term.
                # Using 0.5 * s_i corresponds strictly to Gaussian likelihoods.

                weight = torch.exp(-s_i)
                total_loss += (weight * loss) + (0.5 * s_i)

        return total_loss


class UnifiedWeightedLoss(nn.Module):
    """
    Unified wrapper that applies weights to loss components.
    Supports both learnable weights (Kendall & Gal) and manual weights.
    """

    def __init__(self, manual_weights, component_keys, use_learnable=False):
        """
        manual_weights: dict of manual weight values (used when use_learnable=False)
        component_keys: ordered list of keys for loss components
        use_learnable: if True, uses LearnableWeightedLoss; if False, uses manual weights
        """
        super().__init__()
        self.use_learnable = use_learnable
        self.component_keys = component_keys

        if use_learnable:
            # Initialize learnable weights from manual weights
            eps = 1e-6
            initial_weights = {
                key: max(float(manual_weights[key]), eps)
                if manual_weights[key] > 0
                else eps
                for key in component_keys
            }
            self.learnable_wrapper = LearnableWeightedLoss(
                initial_weights, component_keys
            )
        else:
            # Store manual weights as buffers
            manual_weights_tensor = torch.tensor(
                [manual_weights[key] for key in component_keys], dtype=torch.float32
            )
            self.register_buffer("manual_weights", manual_weights_tensor)
            self.learnable_wrapper = None

    def forward(self, losses_dict):
        """
        losses_dict: dict of individual loss tensors
        Returns: weighted total loss
        """
        if self.use_learnable and self.learnable_wrapper is not None:
            # Use learnable weights (Kendall & Gal formula)
            return self.learnable_wrapper(losses_dict)
        else:
            # Use manual weights (simple multiplication)
            total_loss = 0
            for i, key in enumerate(self.component_keys):
                if key in losses_dict:
                    total_loss += self.manual_weights[i] * losses_dict[key]
            return total_loss

    def get_weights(self):
        """
        Returns current weights (learnable or manual) as a dict.
        """
        if self.use_learnable and self.learnable_wrapper is not None:
            with torch.no_grad():
                weights_list = [
                    torch.exp(-s_i).item() for s_i in self.learnable_wrapper.s
                ]
                return dict(zip(self.component_keys, weights_list, strict=True))
        else:
            weights_list = self.manual_weights.cpu().tolist()
            return dict(zip(self.component_keys, weights_list, strict=True))

    def get_weight_tensors(self, device):
        """
        Returns current weights as a dict of tensors on the specified device.
        """
        if self.use_learnable and self.learnable_wrapper is not None:
            with torch.no_grad():
                weights_list = [
                    torch.exp(-s_i).to(device) for s_i in self.learnable_wrapper.s
                ]
                return dict(zip(self.component_keys, weights_list, strict=True))
        else:
            weights_list = [w.to(device) for w in self.manual_weights]
            return dict(zip(self.component_keys, weights_list, strict=True))
