"""Reward / wrapper specs for GRPO.

Each reward declares the set of dataset representations it can run against.
``build_reward`` (in ``run_grpo``) intersects this set with every active
wrapper's set and rejects mismatches before training starts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from chemflow.dataset.representation import Representation


@dataclass(frozen=True)
class RewardSpec:
    """A reward function plus the representations it is meaningful under."""

    fn: Callable
    supported_representations: frozenset[Representation]


@dataclass(frozen=True)
class WrapperSpec:
    """A reward-wrapping factory plus the representations it is meaningful under.

    ``make(base_fn, **kwargs)`` returns a wrapped reward fn with the same
    ``(module, trajectory) -> (Tensor, dict)`` contract as the base.
    """

    make: Callable[..., Callable]
    supported_representations: frozenset[Representation]
