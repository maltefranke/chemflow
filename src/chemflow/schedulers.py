from torch.optim.lr_scheduler import _LRScheduler
import math


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
