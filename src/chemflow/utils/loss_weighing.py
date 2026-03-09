from __future__ import annotations

import math

import torch
import torch.nn as nn


class ConstantTimeLossWeighting:
    """Time weighting with a constant multiplier."""

    def __init__(self, constant: float = 1.0):
        self.constant = float(constant)

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        return torch.full_like(t, self.constant, dtype=torch.float32)


class InverseTimeLossWeighting:
    """Time weighting with a constant multiplier."""

    def __init__(self, clamp_max: float = 10.0):
        self.clamp_max = float(clamp_max)

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        w = 1.0 / (1.0 - t + 1e-6)
        return torch.clamp(w, max=self.clamp_max)


class InverseSquaredTimeLossWeighting:
    """Time weighting with a constant multiplier."""

    def __init__(self, clamp_max: float = 10.0):
        self.clamp_max = float(clamp_max)

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        w = 1.0 / ((1.0 - t + 1e-6) ** 2)
        return torch.clamp(w, max=self.clamp_max)

class ShiftedParabolaTimeLossWeighting:
    """Time weighting with a shifted parabola."""

    def __init__(self, shift: float = 2.0, coeff: float = 10.0):
        self.shift = float(shift)
        self.coeff = float(coeff)

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        return 1.0 + self.coeff * t * (1.0 - t) * torch.exp(-self.shift * t)