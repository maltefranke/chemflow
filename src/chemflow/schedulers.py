
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
