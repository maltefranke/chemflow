import math
from torch.optim.lr_scheduler import _LRScheduler


class LinearOneCycleLR(_LRScheduler):
    """
    Linearly increase LR from initial_lr to max_lr over warmup_steps,
    then keep LR constant at max_lr.
    """

    def __init__(self, optimizer, max_lr, warmup_steps, last_epoch=-1):
        self.max_lr = max_lr
        self.warmup_steps = warmup_steps

        # Save initial LRs of param groups
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # Step-based schedule
        if self.last_epoch >= self.warmup_steps:
            # Constant LR phase
            return [self.max_lr for _ in self.optimizer.param_groups]

        # Linear warmup phase
        warmup_ratio = (self.last_epoch + 1) / self.warmup_steps
        return [
            base_lr + warmup_ratio * (self.max_lr - base_lr)
            for base_lr in self.base_lrs
        ]


class CosineWarmupLR(_LRScheduler):
    """
    Linearly increases LR from initial_lr to max_lr over warmup_steps,
    then decays to min_lr using a cosine schedule over the remaining steps.
    """

    def __init__(
        self, optimizer, max_lr, warmup_steps, total_steps, min_lr=1e-6, last_epoch=-1
    ):
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

        # Save initial LRs of param groups
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # 1. Linear Warmup Phase
        if self.last_epoch < self.warmup_steps:
            alpha = self.last_epoch / self.warmup_steps
            return [
                base_lr + alpha * (self.max_lr - base_lr) for base_lr in self.base_lrs
            ]

        # 2. Cosine Decay Phase
        # Calculate progress from 0.0 to 1.0 within the decay phase
        progress = (self.last_epoch - self.warmup_steps) / (
            self.total_steps - self.warmup_steps
        )
        progress = min(progress, 1.0)  # Clip to ensure we don't go past end

        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))

        return [
            self.min_lr + (self.max_lr - self.min_lr) * cosine_decay
            for _ in self.base_lrs
        ]


class MultiGroupCosineWarmupLR(_LRScheduler):
    """
    Linearly increases LR from 0 to each group's base_lr over warmup_steps,
    then decays to a fraction of the base_lr using a cosine schedule.
    """

    def __init__(
        self, optimizer, warmup_steps, total_steps, min_lr_fraction=0.1, last_epoch=-1
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_fraction = (
            min_lr_fraction  # e.g., 0.1 means decay to 10% of base_lr
        )
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # 1. Linear Warmup Phase
        if self.last_epoch < self.warmup_steps:
            alpha = (self.last_epoch + 1) / max(1, self.warmup_steps)
            return [base_lr * alpha for base_lr in self.base_lrs]

        # 2. Cosine Decay Phase
        progress = (self.last_epoch - self.warmup_steps) / max(
            1, self.total_steps - self.warmup_steps
        )
        progress = min(progress, 1.0)  # Clip to ensure we don't go past end

        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))

        # Calculate the multiplier scale (between min_lr_fraction and 1.0)
        multiplier = self.min_lr_fraction + (1.0 - self.min_lr_fraction) * cosine_decay

        return [base_lr * multiplier for base_lr in self.base_lrs]


class EMADecayScheduler:
    """EMA decay scheduler with hold + fast ramp + slow ramp phases."""

    def __init__(
        self,
        initial_decay: float,
        mid_decay: float,
        final_decay: float,
        warmup_epochs: int,
        fast_ramp_epochs: int,
        total_epochs: int,
    ):
        self.initial_decay = float(initial_decay)
        self.mid_decay = float(mid_decay)
        self.final_decay = float(final_decay)
        self.warmup_epochs = max(0, int(warmup_epochs))
        self.fast_ramp_epochs = max(0, int(fast_ramp_epochs))
        self.total_epochs = int(total_epochs)

    def get_decay(self, current_epoch: int) -> float:
        """Return EMA decay for the given epoch within total training epochs."""
        if self.total_epochs <= 1:
            return float(self.final_decay)

        current_epoch = min(max(0, int(current_epoch)), self.total_epochs - 1)
        last_epoch = float(self.total_epochs - 1)
        warmup = float(min(self.warmup_epochs, self.total_epochs - 1))
        fast_end = float(
            min(self.warmup_epochs + self.fast_ramp_epochs, self.total_epochs - 1)
        )

        if current_epoch < warmup:
            return float(self.initial_decay)

        if current_epoch < fast_end:
            progress = (float(current_epoch) - warmup) / max(fast_end - warmup, 1e-8)
            return float(
                self.initial_decay + progress * (self.mid_decay - self.initial_decay)
            )

        progress = (float(current_epoch) - fast_end) / max(last_epoch - fast_end, 1e-8)
        return float(self.mid_decay + progress * (self.final_decay - self.mid_decay))
