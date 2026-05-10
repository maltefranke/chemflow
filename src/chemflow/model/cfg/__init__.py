"""Classifier-free guidance: signal abstraction, embedding, and inference orchestrator."""

from chemflow.model.cfg.embedding import CFGEmbedding
from chemflow.model.cfg.extractors import (
    compute_logp,
    compute_molecular_weight,
    compute_qed,
)
from chemflow.model.cfg.guidance import CFGGuidance
from chemflow.model.cfg.signals import (
    CFGMode,
    ConditioningSignal,
    LogPSignal,
    MWSignal,
    NAtomsSignal,
    PropertySignal,
    QEDSignal,
)

__all__ = [
    "CFGEmbedding",
    "CFGGuidance",
    "CFGMode",
    "ConditioningSignal",
    "LogPSignal",
    "MWSignal",
    "NAtomsSignal",
    "PropertySignal",
    "QEDSignal",
    "compute_logp",
    "compute_molecular_weight",
    "compute_qed",
]
