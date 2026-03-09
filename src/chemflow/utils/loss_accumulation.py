import torch

class LossAccumulator:
    """Per-step helper that collects raw losses, applies time and component weights, 
    and builds a uniform log dict separating pure losses from fully weighted losses.

    Args:
        weight_module: ``UnifiedWeightedLoss`` instance that owns the component weights.
        groups: Mapping from group name to list of component keys.
        device: Device for fallback zero tensors.
        time_weight_modules: Mapping from GROUP NAME to a time-weighting callable.
    """

    def __init__(
        self,
        weight_module,
        groups: dict[str, list[str]],
        device: torch.device,
        time_weight_modules: dict[str, callable] | None = None,
    ):
        self._weight_module = weight_module
        self._groups = groups
        self._device = device
        self._weights = weight_module.get_weight_tensors(device)
        self._time_weight_modules = time_weight_modules or {}
        
        # Reverse mapping: component_key -> group_name for fast lookups
        self._key_to_group = {
            key: group for group, keys in groups.items() for key in keys
        }

        # Track pure and time-weighted losses separately
        self._raw_losses: dict[str, torch.Tensor] = {}
        self._tw_losses: dict[str, torch.Tensor] = {}
        
        self._stats: dict[str, torch.Tensor | float] = {}
        self._current_time_weights: dict[str, float] = {}

    # ── Population ───────────────────────────────────────────────────

    def set_batch_losses(
        self, 
        batch_losses: dict[str, torch.Tensor], 
        t: torch.Tensor | None = None,
        masks: dict[str, torch.Tensor] | None = None
    ):
        """Processes unreduced batch losses, applies time-weights, applies masks, and reduces.
        
        Args:
            batch_losses: Dict of loss tensors, shape [num_graphs, ...]
            t: Time steps, shape [num_graphs]
            masks: Optional dict mapping loss keys to boolean tensors of shape [num_graphs].
                   True means the graph should be included in the mean.
        """
        self._raw_losses.clear()
        self._tw_losses.clear()
        self._current_time_weights.clear()
        masks = masks or {}
        
        # Precompute time-weight tensors
        group_tw_tensors = {}
        if t is not None:
            for group, tw_module in self._time_weight_modules.items():
                tw = tw_module(t)
                group_tw_tensors[group] = tw
                self._current_time_weights[group] = tw.mean().item()

        for key, loss_b in batch_losses.items():
            # 1. Apply time-weights to create a weighted copy
            if t is not None:
                group = self._key_to_group.get(key)
                if group is not None and group in group_tw_tensors:
                    tw = group_tw_tensors[group]
                    target_shape = [-1] + [1] * (loss_b.ndim - 1)
                    loss_tw_b = loss_b * tw.view(*target_shape)
                else:
                    loss_tw_b = loss_b
            else:
                loss_tw_b = loss_b

            # 2. Apply masking if provided
            mask = masks.get(key)
            if mask is not None:
                # Filter both the pure and time-weighted losses
                valid_raw = loss_b[mask]
                valid_tw = loss_tw_b[mask]
                
                # Prevent NaN if mask is entirely False
                if valid_raw.numel() > 0:
                    self._raw_losses[key] = valid_raw.mean()
                    self._tw_losses[key] = valid_tw.mean()
                else:
                    # Dummy zero tensor that still requires grad (prevents DDP crashing)
                    zero_loss = (loss_b * 0.0).sum()
                    self._raw_losses[key] = zero_loss
                    self._tw_losses[key] = zero_loss
            else:
                self._raw_losses[key] = loss_b.mean()
                self._tw_losses[key] = loss_tw_b.mean()

    def set_losses(self, losses: dict[str, torch.Tensor]):
        """Fallback for pre-reduced scalar losses."""
        self._raw_losses = losses
        self._tw_losses = losses.copy()
        self._current_time_weights.clear()

    def add_stat(self, key: str, value):
        self._stats[key] = value

    def add_stats(self, stats: dict):
        self._stats.update(stats)

    # ── Outputs ──────────────────────────────────────────────────────

    def total_loss(self) -> torch.Tensor:
        """Compute final loss for backprop: applies component weights to the time-weighted losses."""
        return self._weight_module(self._tw_losses)

    def log_dict(self) -> dict[str, torch.Tensor | float]:
        """Builds a uniform log dict separating pure and fully weighted losses."""
        entries: dict[str, torch.Tensor | float] = {}

        total_pure = 0.0
        total_fully_weighted = 0.0
        
        # 1. Log individual components
        for key in self._raw_losses.keys():
            pure_val = self._raw_losses[key]
            # Fully weighted = time-weighted value * learnable/manual component weight
            fully_weighted_val = self._weights[key] * self._tw_losses[key]
            
            entries[f"loss/{key}"] = pure_val
            entries[f"loss_weighted/{key}"] = fully_weighted_val
            
            total_pure += pure_val
            total_fully_weighted += fully_weighted_val

        # 2. Log group sums
        for group, keys in self._groups.items():
            pure_terms = [self._raw_losses[k] for k in keys if k in self._raw_losses]
            weighted_terms = [self._weights[k] * self._tw_losses[k] for k in keys if k in self._tw_losses]
            
            if pure_terms:
                entries[f"loss/{group}"] = sum(pure_terms)
            if weighted_terms:
                entries[f"loss_weighted/{group}"] = sum(weighted_terms)

        # 3. Log totals
        entries["loss/total"] = total_pure
        entries["loss_weighted/total"] = total_fully_weighted

        # 4. Log stats, weights, and time-weights
        for key, val in self._stats.items():
            entries[f"stats/{key}"] = val

        if getattr(self._weight_module, "use_learnable", False):
            for k, w in self._weights.items():
                entries[f"weight/{k}"] = w

        for group, tw_val in self._current_time_weights.items():
            entries[f"time_weight/{group}"] = tw_val

        return entries