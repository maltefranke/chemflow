from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import beta


class KappaSchedule(nn.Module, ABC):
    """
    Abstract base class for flow-matching time schedules (kappa schedules).

    A schedule maps time t in [0, 1] to kappa(t) (CDF) and provides the rate
    kappa'(t) / (1 - kappa(t)) used in rate-based flow matching.
    """

    @abstractmethod
    def kappa_t(self, t: torch.Tensor) -> torch.Tensor:
        """Maps t to kappa(t) (CDF)."""
        ...

    @abstractmethod
    def kappa_t_dot(self, t: torch.Tensor) -> torch.Tensor:
        """Derivative of kappa w.r.t. t (PDF)."""
        ...

    @abstractmethod
    def rate(self, t: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
        """Rate: kappa_t_dot(t) / (1 - kappa_t(t))."""
        ...


class BetaSchedule(KappaSchedule):
    def __init__(self, k_alpha: float, k_beta: float):
        """
        Beta schedule using scipy.stats for guaranteed stability.
        """
        super().__init__()
        self.k_alpha = k_alpha
        self.k_beta = k_beta

    def _to_numpy(self, t: torch.Tensor):
        """Helper to convert tensor to numpy array safely."""
        # Detach from graph, move to CPU, convert to numpy
        return t.detach().cpu().numpy()

    def _to_tensor(self, x: np.ndarray, t_template: torch.Tensor):
        """Helper to convert numpy result back to tensor on correct device."""
        return torch.tensor(x, device=t_template.device, dtype=t_template.dtype)

    def kappa_t(self, t: torch.Tensor) -> torch.Tensor:
        """
        Calculates kappa_t (CDF).
        """
        t_np = self._to_numpy(t)

        # scipy.stats.beta.cdf handles the math
        cdf_val = beta.cdf(t_np, self.k_alpha, self.k_beta)

        return self._to_tensor(cdf_val, t)

    def kappa_t_dot(self, t: torch.Tensor) -> torch.Tensor:
        """
        Calculates kappa_t_dot (PDF).
        """
        t_np = self._to_numpy(t)

        # scipy.stats.beta.pdf handles the math
        pdf_val = beta.pdf(t_np, self.k_alpha, self.k_beta)

        return self._to_tensor(pdf_val, t)

    def rate(self, t: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
        """
        Calculates Rate: kappa_t_dot / (1 - kappa_t)
        """
        # We calculate rate in PyTorch to ensure gradients could theoretically
        # flow through the combination, though usually t is constant.
        k_t = self.kappa_t(t)
        k_t_dot = self.kappa_t_dot(t)

        # Prevent division by zero near t=1
        denominator = torch.clamp(1 - k_t, min=epsilon)

        return k_t_dot / denominator


class CubicSchedule(KappaSchedule):
    def __init__(self):
        super().__init__()

    def kappa_t(self, t: torch.Tensor) -> torch.Tensor:
        return t**3

    def kappa_t_dot(self, t: torch.Tensor) -> torch.Tensor:
        return 3 * t**2

    def rate(self, t: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
        denom = torch.clamp(1 - t**3, min=epsilon)
        return 3 * t**2 / denom


class LinearSchedule(KappaSchedule):
    def __init__(self):
        super().__init__()

    def kappa_t(self, t: torch.Tensor) -> torch.Tensor:
        return t

    def kappa_t_dot(self, t: torch.Tensor) -> torch.Tensor:
        return 1

    def rate(self, t: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
        denom = torch.clamp(1 - t, min=epsilon)
        return 1 / denom


class FastPowerSchedule(KappaSchedule):
    """
    Optimized Beta Schedule specifically for Alpha=1.0
    """

    def __init__(self, beta: float):  # e.g. 1.5
        super().__init__()
        self.register_buffer("beta", torch.tensor(float(beta)))

    def kappa_t(self, t: torch.Tensor) -> torch.Tensor:
        # Analytical CDF for Alpha=1: 1 - (1-t)^beta
        return 1 - torch.pow(1 - t, self.beta)

    def kappa_t_dot(self, t: torch.Tensor) -> torch.Tensor:
        # Analytical PDF for Alpha=1: beta * (1-t)^(beta-1)
        return self.beta * torch.pow(1 - t, self.beta - 1)

    def rate(self, t: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
        # Analytical Rate for Alpha=1: beta / (1-t)
        denom = torch.clamp(1 - t, min=epsilon)
        return self.beta / denom


class SmoothstepSchedule(KappaSchedule):
    """
    A simple polynomial schedule that concentrates edits in a bell shape.
    - shift = 1.0 : Edits peak perfectly in the middle.
    - shift < 1.0 : Edits happen earlier (pushes the peak towards t=0).
    - shift > 1.0 : Edits happen later (pushes the peak towards t=1).
    - scale       : Modulates the overall height/intensity of the edits.
    """

    def __init__(self, shift: float = 1.0):
        super().__init__()
        self.shift = shift

    def kappa_t(self, t: torch.Tensor) -> torch.Tensor:
        # Standard smoothstep is 3t^2 - 2t^3.
        # By raising t to the power of 'shift' (p), we warp time to move the peak.
        p = self.shift
        return 3 * t ** (2 * p) - 2 * t ** (3 * p)

    def kappa_t_dot(self, t: torch.Tensor) -> torch.Tensor:
        p = self.shift
        # We clamp t slightly above 0 to prevent NaN errors when shift < 0.5
        t_safe = torch.clamp(t, min=1e-7)

        # The exact mathematical derivative of kappa_t
        return 6 * p * (t_safe ** (2 * p - 1) - t_safe ** (3 * p - 1))

    def rate(self, t: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
        # Standard flow-matching rate formulation
        k_t = self.kappa_t(t)
        k_dot = self.kappa_t_dot(t)
        denom = torch.clamp(1.0 - k_t, min=epsilon)
        return k_dot / denom
