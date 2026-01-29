import torch
import torch.nn as nn
import numpy as np
from scipy.stats import beta


class BetaSchedule(nn.Module):
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


class CubicSchedule(nn.Module):
    def __init__(self):
        super().__init__()

    def kappa_t(self, t: torch.Tensor) -> torch.Tensor:
        return t**3

    def kappa_t_dot(self, t: torch.Tensor) -> torch.Tensor:
        return 3 * t**2

    def rate(self, t: torch.Tensor) -> torch.Tensor:
        return 3 * t**2 / (1 - t**3)


class LinearSchedule(nn.Module):
    def __init__(self):
        super().__init__()

    def kappa_t(self, t: torch.Tensor) -> torch.Tensor:
        return t

    def kappa_t_dot(self, t: torch.Tensor) -> torch.Tensor:
        return 1

    def rate(self, t: torch.Tensor) -> torch.Tensor:
        return 1 / (1 - t)


class FastPowerSchedule(nn.Module):
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
