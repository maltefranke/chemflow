"""Ground-truth vs generated marginal plots for wandb logging."""

import torch
from torchmetrics import MetricCollection

from chemflow.utils.metrics.tensor import RG_RANGE


def plot_marginal_comparison(
    gen_hist: torch.Tensor,
    target_hist: torch.Tensor,
    labels: list[str] | None,
    title: str,
    xlabel: str,
    eps: float = 1e-8,
):
    """Return a matplotlib Figure comparing ground-truth vs generated marginal densities.

    Both inputs are unnormalized histograms; they are normalized to densities here.
    """
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    import numpy as np

    gen = gen_hist.detach().to("cpu", dtype=torch.float32)
    target = target_hist.detach().to("cpu", dtype=torch.float32)

    n = max(gen.numel(), target.numel())
    if gen.numel() < n:
        gen = torch.cat([gen, torch.zeros(n - gen.numel())])
    if target.numel() < n:
        target = torch.cat([target, torch.zeros(n - target.numel())])

    gen_sum = float(gen.sum())
    target_sum = float(target.sum())
    gen = gen / max(gen_sum, eps)
    target = target / max(target_sum, eps)

    if labels is None or len(labels) != n:
        labels = [str(i) for i in range(n)]

    x = np.arange(n)
    width = 0.4

    fig, ax = plt.subplots(figsize=(max(6.0, n * 0.35), 4.0))
    ax.bar(
        x - width / 2.0,
        target.numpy(),
        width,
        label="ground truth",
        color="steelblue",
        alpha=0.85,
    )
    ax.bar(
        x + width / 2.0,
        gen.numpy(),
        width,
        label="generated",
        color="darkorange",
        alpha=0.85,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45 if n > 10 else 0, ha="right")
    ax.set_ylabel("probability")
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


# Per-distribution-metric plot captions, keyed by the collection key set in
# ``init_metrics``. Anything not listed is skipped (no caption == not plotted).
_DIST_PLOT_META = {
    "atom_count_dist_kl": ("Number of atoms: ground truth vs generated", "n_atoms"),
    "atom_type_dist_kl": ("Atom types: ground truth vs generated", "atom type"),
    "edge_type_dist_kl": (
        "Edge types (upper-tri pairs incl. NO_BOND): ground truth vs generated",
        "edge type",
    ),
    "charge_type_dist_kl": ("Formal charges: ground truth vs generated", "formal charge"),
}


def build_marginal_plots(distribution_metrics: MetricCollection) -> dict:
    """One Figure per distribution metric with samples, keyed by a short name.

    Generic over the shared ``gen_hist`` / ``target`` / ``labels`` attribute
    contract of the tensor distribution metrics. Metrics that have not seen any
    samples are skipped. Caller closes the figures after logging.
    """
    plots: dict = {}
    for key, (title, xlabel) in _DIST_PLOT_META.items():
        if key not in distribution_metrics:
            continue
        m = distribution_metrics[key]
        if float(m.gen_hist.sum()) <= 0.0:
            continue
        plots[key.replace("_dist_kl", "")] = plot_marginal_comparison(
            gen_hist=m.gen_hist,
            target_hist=m.target,
            labels=m.labels,
            title=title,
            xlabel=xlabel,
        )
    return plots


def build_batch_marginal_plots(metrics, atom_tokens: list[str] | None = None):
    """GT-vs-generated plot for the geometric batch metrics that carry a
    histogram. Returns ``{wandb_key: matplotlib.Figure}``.

    Currently just radius of gyration — the 1D atom-count / atom-type marginals
    moved to the distribution collection (see ``build_marginal_plots``).
    """
    figs: dict = {}
    by_name = {k: m for k, m in metrics.items()} if hasattr(metrics, "items") else {}
    m = by_name.get("rog_l1")
    if m is not None and getattr(m, "gen_hist", None) is not None:
        figs["val/batch/plots/rog"] = plot_marginal_comparison(
            m.gen_hist,
            m.target_hist,
            None,
            "Radius of gyration",
            f"RoG bin (Å in [{RG_RANGE[0]}, {RG_RANGE[1]}])",
        )
    return figs
