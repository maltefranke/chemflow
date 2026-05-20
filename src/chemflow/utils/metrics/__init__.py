"""Code adapted from SemlaFlow.

Package layout (split by dependency):
- ``tensor``      : geometric tensor metrics + bin edges/accumulators (no RDKit)
- ``distribution``: KL distribution metrics, batch-fed (no RDKit)
- ``chemistry``   : RDKit-mol metrics + shared pool + SEMLA + PoseBusters
- ``plotting``    : GT-vs-generated marginal plots

Public surface re-exported here so ``from chemflow.utils.metrics import X``
keeps working unchanged.
"""

import torch
from torchmetrics import MetricCollection

from chemflow.utils.metrics.tensor import (
    BatchMetric,
    MinDistanceViolation,
    PairwiseDistanceL1,
    RangeOverflow,
    RoGL1,
    build_batch_metrics,
)
from chemflow.utils.metrics.distribution import (
    AtomCountKL,
    AtomTypeKL,
    ChargeTypeKL,
    EdgeTypeKL,
)
from chemflow.utils.metrics.chemistry import (
    AtomStability,
    AverageEnergy,
    AverageOptRmsd,
    AverageStrainEnergy,
    EnergyValidity,
    GenerativeMetric,
    MoleculeStability,
    Novelty,
    SEMLAAtomStability,
    SEMLAMoleculeStability,
    SEMLAValidity,
    Uniqueness,
    Validity,
    calc_atom_stabilities,
    calc_posebusters_metrics,
)
from chemflow.utils.metrics.plotting import (
    build_batch_marginal_plots,
    build_marginal_plots,
    plot_marginal_comparison,
)


def init_metrics(
    train_smiles: list[str] | None = None,
    target_n_atoms_distribution: torch.Tensor | None = None,
    atom_type_distribution: torch.Tensor | None = None,
    edge_type_distribution: torch.Tensor | None = None,
    charge_type_distribution: torch.Tensor | None = None,
    atom_tokens: list[str] | None = None,
    edge_tokens: list[str] | None = None,
    charge_tokens: list[str] | None = None,
    allow_charged: bool = False,
    distributions=None,
    representation=None,
):

    metrics = {
        "validity": Validity(allow_charged=allow_charged),
        "semla-validity": SEMLAValidity(),
        "semla-atom-stability": SEMLAAtomStability(),
        "semla-molecule-stability": SEMLAMoleculeStability(),
        "uniqueness": Uniqueness(),
        **({"novelty": Novelty(train_smiles)} if train_smiles is not None else {}),
        "energy-validity": EnergyValidity(),
        "opt-energy-validity": EnergyValidity(optimise=True),
        "energy": AverageEnergy(),
        "energy-per-atom": AverageEnergy(per_atom=True),
        "strain": AverageStrainEnergy(),
        "strain-per-atom": AverageStrainEnergy(per_atom=True),
        "opt-rmsd": AverageOptRmsd(),
    }

    # Distribution metrics: tensor-native KLs computed straight from the batch,
    # so they run in every representation. Kept in their own collection so they
    # accumulate across an entire validation epoch (rather than per batch) and
    # drive the ground-truth-vs-generated marginal plots. Edge/charge KLs are
    # only registered when the representation actually carries that field —
    # otherwise they are degenerate (all <NO_BOND> / neutral by projection) and
    # would log a meaningless 0.
    want_edges = representation is None or representation.requires_topology
    want_charges = representation is None or representation.requires_charges
    distribution_metrics: dict = {}
    if target_n_atoms_distribution is not None:
        distribution_metrics["atom_count_dist_kl"] = AtomCountKL(
            target_n_atoms_distribution
        )
    if atom_type_distribution is not None and atom_tokens is not None:
        distribution_metrics["atom_type_dist_kl"] = AtomTypeKL(
            atom_type_distribution, labels=atom_tokens
        )
    if edge_type_distribution is not None and edge_tokens is not None and want_edges:
        distribution_metrics["edge_type_dist_kl"] = EdgeTypeKL(
            edge_type_distribution, labels=edge_tokens
        )
    if charge_type_distribution is not None and charge_tokens is not None and want_charges:
        distribution_metrics["charge_type_dist_kl"] = ChargeTypeKL(
            charge_type_distribution, labels=charge_tokens
        )

    stability_metrics = {
        "rdkit-atom-stability": AtomStability(),
        "rdkit-molecule-stability": MoleculeStability(),
    }

    metrics = MetricCollection(metrics, compute_groups=False)
    stability_metrics = MetricCollection(stability_metrics, compute_groups=False)
    distribution_metrics = MetricCollection(distribution_metrics, compute_groups=False)

    # Geometric batch metrics — built from training-set target stats stored in
    # Distributions. Run in every representation. The collection may be empty if
    # Distributions lacks the geometric target stats.
    batch_metrics_dict: dict = {}
    if distributions is not None:
        batch_metrics_dict = build_batch_metrics(distributions)
    batch_metrics = MetricCollection(batch_metrics_dict, compute_groups=False)

    return metrics, stability_metrics, distribution_metrics, batch_metrics


def calc_metrics_(rdkit_mols, metrics) -> dict:
    """Run an RDKit-mol metric collection over a list of mols and return the
    scalar results. Stability lives in its own collection with a different
    input contract (precomputed bools) — drive it separately. Atom-count KL
    runs tensor-side as ``AtomCountKL``."""
    metrics.reset()
    metrics.update(rdkit_mols)
    return {
        k: v.item() if isinstance(v, torch.Tensor) else v
        for k, v in metrics.compute().items()
    }
