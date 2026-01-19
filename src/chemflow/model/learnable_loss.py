import torch
import torch.nn as nn
import math


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
