"""Tensor-based generative metrics for pointcloud-mode evaluation.

The metrics in ``metrics.py`` consume RDKit molecules and therefore can't operate
in pointcloud modes (runtime edges labeled ``<NO_BOND>`` are not chemical bonds). These mirror the
KL / distribution-distance ideas but work directly on ``MoleculeBatch`` tensors,
so no bond perception is required.

Target stats (per-element-pair distance histogram, RoG histogram) are computed
once at preprocessing time on training-set coords in Angstroms and stored in
``Distributions``. Generated coords arrive normalized (divided by
``coordinate_std`` in the dataset's ``__getitem__``); each metric undoes that
scaling internally so it can compare like-for-like.
"""

import torch
from torchmetrics import Metric

from chemflow.dataset.molecule_data import MoleculeBatch


# Shared bin edges (Angstroms). Used both at preprocessing time to compute
# target histograms and at metric time to bin generated samples — they must
# match exactly. Defaults are generous enough for QM9 / GEOM / TMQM; values
# above the upper bound are counted in an overflow accumulator (see the binning
# helpers below) so silent clamping is visible at validation time.
DIST_RANGE = (0.0, 15.0)
DIST_N_BINS = 150
RG_RANGE = (0.0, 8.0)
RG_N_BINS = 80


def dist_edges() -> torch.Tensor:
    return torch.linspace(DIST_RANGE[0], DIST_RANGE[1], DIST_N_BINS + 1)


def rg_edges() -> torch.Tensor:
    return torch.linspace(RG_RANGE[0], RG_RANGE[1], RG_N_BINS + 1)


def _kl(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    p = p / p.sum().clamp(min=eps)
    q = q / q.sum().clamp(min=eps)
    return (p * (p.clamp(min=eps).log() - q.clamp(min=eps).log())).sum()


def accumulate_pairwise_distance_hist(
    hist: torch.Tensor,
    coord: torch.Tensor,
    atom_idx: torch.Tensor,
    edges: torch.Tensor,
) -> tuple[int, int]:
    """Add upper-triangular pair distances to ``hist`` (A, A, B), in place.

    The pair key is ``(min(a_i, a_j), max(a_i, a_j))`` so the histogram is
    naturally invariant to atom ordering.

    Returns ``(n_overflow, n_total)`` for the pairs in this molecule, so callers
    can track the fraction of distances that fell beyond the histogram range
    (silent clamping otherwise hides "the cloud is way too large").
    """
    n = coord.shape[0]
    if n < 2:
        return 0, 0
    i_idx, j_idx = torch.triu_indices(n, n, offset=1, device=coord.device)
    dists = (coord[i_idx] - coord[j_idx]).norm(dim=-1)
    lo = torch.minimum(atom_idx[i_idx], atom_idx[j_idx])
    hi = torch.maximum(atom_idx[i_idx], atom_idx[j_idx])
    edges_d = edges.to(coord.device)
    n_overflow = int((dists >= edges_d[-1]).sum().item())
    bin_idx = torch.bucketize(dists, edges_d).sub_(1).clamp_(0, hist.shape[-1] - 1)
    a = hist.shape[0]
    flat = (lo.long() * a + hi.long()) * hist.shape[-1] + bin_idx
    hist.view(-1).index_add_(
        0, flat, torch.ones_like(flat, dtype=hist.dtype, device=coord.device)
    )
    return n_overflow, int(dists.numel())


def accumulate_rog_hist(
    hist: torch.Tensor, coord: torch.Tensor, edges: torch.Tensor
) -> tuple[int, int]:
    """Add one molecule's radius of gyration into ``hist`` (B,), in place.

    Returns ``(n_overflow, n_total)`` (0 or 1 each) for the same reason as
    ``accumulate_pairwise_distance_hist``.
    """
    if coord.shape[0] == 0:
        return 0, 0
    centered = coord - coord.mean(dim=0)
    rog = (centered.pow(2).sum(dim=-1).mean()).sqrt()
    edges_d = edges.to(coord.device)
    overflow = int((rog >= edges_d[-1]).item())
    bin_idx = (
        torch.bucketize(rog.unsqueeze(0), edges_d).sub_(1).clamp_(0, hist.shape[0] - 1)
    )
    hist.index_add_(0, bin_idx, torch.ones(1, dtype=hist.dtype, device=coord.device))
    return overflow, 1


class PointCloudMetric(Metric):
    """Base class for tensor-based pointcloud metrics."""

    def update(self, batch: MoleculeBatch, coord_scale: float = 1.0) -> None:
        raise NotImplementedError

    def compute(self) -> torch.Tensor:
        raise NotImplementedError


class PCAtomCountKL(PointCloudMetric):
    """KL(gen || target) over atom counts per molecule."""

    def __init__(self, target_distribution: torch.Tensor, eps: float = 1e-8, **kwargs):
        super().__init__(**kwargs)
        target = target_distribution.detach().to(torch.float32)
        self.eps = eps
        self.K = int(target.numel())
        self.register_buffer("target", target / target.sum().clamp(min=eps))
        self.add_state(
            "gen_hist",
            default=torch.zeros(self.K, dtype=torch.float32),
            dist_reduce_fx="sum",
        )

    def update(self, batch: MoleculeBatch, coord_scale: float = 1.0) -> None:
        counts = torch.bincount(batch.batch, minlength=batch.num_graphs).clamp(
            0, self.K - 1
        )
        self.gen_hist += torch.bincount(counts, minlength=self.K).to(torch.float32)

    def compute(self) -> torch.Tensor:
        if self.gen_hist.sum() <= 0:
            return torch.tensor(0.0, device=self.gen_hist.device)
        return _kl(self.gen_hist, self.target, self.eps)


class PCAtomTypeKL(PointCloudMetric):
    """KL(gen || target) over atom-type histogram."""

    def __init__(self, target_distribution: torch.Tensor, eps: float = 1e-8, **kwargs):
        super().__init__(**kwargs)
        target = target_distribution.detach().to(torch.float32)
        self.eps = eps
        self.K = int(target.numel())
        self.register_buffer("target", target / target.sum().clamp(min=eps))
        self.add_state(
            "gen_hist",
            default=torch.zeros(self.K, dtype=torch.float32),
            dist_reduce_fx="sum",
        )

    def update(self, batch: MoleculeBatch, coord_scale: float = 1.0) -> None:
        a = batch.a.clamp(0, self.K - 1)
        self.gen_hist += torch.bincount(a, minlength=self.K).to(torch.float32)

    def compute(self) -> torch.Tensor:
        if self.gen_hist.sum() <= 0:
            return torch.tensor(0.0, device=self.gen_hist.device)
        return _kl(self.gen_hist, self.target, self.eps)


class PCPairwiseDistanceL1(PointCloudMetric):
    """Mean per-element-pair L1 distance between gen and target distance histograms.

    For every ordered (lo, hi) atom-type pair the *target* has meaningful mass
    for, we normalize gen and target slices independently and report
    ``|gen - target|.sum()``. The mean over those pairs is a 0–2 metric where 0
    means perfect distributional match.

    Note: gen-empty pairs are *not* excluded. If the target has C–N distances
    but the model never generates a C–N pair, the L1 against zeros is 1.0 and
    that pair counts in the average — which is what we want.
    """

    def __init__(self, target_hist: torch.Tensor, min_pair_count: float = 10.0, **kwargs):
        super().__init__(**kwargs)
        self.register_buffer("target_hist", target_hist.detach().to(torch.float32))
        self.register_buffer("edges", dist_edges())
        self.min_pair_count = min_pair_count
        self.add_state(
            "gen_hist",
            default=torch.zeros_like(self.target_hist),
            dist_reduce_fx="sum",
        )

    def update(self, batch: MoleculeBatch, coord_scale: float = 1.0) -> None:
        x = batch.x * coord_scale
        for g in batch.batch.unique():
            mask = batch.batch == g
            accumulate_pairwise_distance_hist(
                self.gen_hist, x[mask], batch.a[mask], self.edges
            )

    def compute(self) -> torch.Tensor:
        gen = self.gen_hist  # (A, A, B)
        tgt = self.target_hist
        pair_t = tgt.sum(dim=-1)
        # Target-populated only: a pair the training set knows about counts in
        # the metric even if the model generates zero of them (gen_n = 0 →
        # L1 == tgt_n.sum() == 1, which is maximal pain).
        populated = pair_t >= self.min_pair_count
        if not populated.any():
            return torch.tensor(0.0, device=gen.device)
        gen_n = gen / gen.sum(dim=-1).unsqueeze(-1).clamp(min=1.0)
        tgt_n = tgt / pair_t.unsqueeze(-1).clamp(min=1.0)
        l1 = (gen_n - tgt_n).abs().sum(dim=-1)  # (A, A)
        return l1[populated].mean()


class PCMinDistanceViolation(PointCloudMetric):
    """Fraction of generated molecules whose smallest pairwise distance is below
    a physical threshold (default 0.5 Å). Catches collapse and overlap."""

    def __init__(self, threshold: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.threshold = float(threshold)
        self.add_state(
            "violations", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, batch: MoleculeBatch, coord_scale: float = 1.0) -> None:
        x = batch.x * coord_scale
        for g in batch.batch.unique():
            mask = batch.batch == g
            coord = x[mask]
            n = coord.shape[0]
            if n < 2:
                continue
            i_idx, j_idx = torch.triu_indices(n, n, offset=1, device=coord.device)
            dmin = (coord[i_idx] - coord[j_idx]).norm(dim=-1).min()
            if dmin.item() < self.threshold:
                self.violations += 1.0
            self.total += 1.0

    def compute(self) -> torch.Tensor:
        if self.total <= 0:
            return torch.tensor(0.0, device=self.violations.device)
        return self.violations / self.total


class PCRangeOverflow(PointCloudMetric):
    """Fraction of generated values that landed outside the histogram range.

    If this is non-trivial (say >1%), the fixed range in this module is too
    small for the dataset and we're silently clamping into the last bin —
    revisit ``DIST_RANGE`` / ``RG_RANGE`` or switch to target-derived edges.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.register_buffer("dist_edges_buf", dist_edges())
        self.register_buffer("rog_edges_buf", rg_edges())
        self.add_state("dist_over", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("dist_total", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("rog_over", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("rog_total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, batch: MoleculeBatch, coord_scale: float = 1.0) -> None:
        x = batch.x * coord_scale
        d_max = self.dist_edges_buf[-1]
        r_max = self.rog_edges_buf[-1]
        for g in batch.batch.unique():
            mask = batch.batch == g
            coord = x[mask]
            n = coord.shape[0]
            if n >= 2:
                i_idx, j_idx = torch.triu_indices(n, n, offset=1, device=coord.device)
                dists = (coord[i_idx] - coord[j_idx]).norm(dim=-1)
                self.dist_over += (dists >= d_max).sum().float()
                self.dist_total += float(dists.numel())
            if n >= 1:
                rog = (coord - coord.mean(dim=0)).pow(2).sum(dim=-1).mean().sqrt()
                self.rog_over += (rog >= r_max).float()
                self.rog_total += 1.0

    def compute(self) -> torch.Tensor:
        d = self.dist_over / self.dist_total.clamp(min=1.0)
        r = self.rog_over / self.rog_total.clamp(min=1.0)
        return torch.maximum(d, r)


class PCRoGL1(PointCloudMetric):
    """L1 distance between gen and target normalized radius-of-gyration histograms."""

    def __init__(self, target_hist: torch.Tensor, **kwargs):
        super().__init__(**kwargs)
        self.register_buffer("target_hist", target_hist.detach().to(torch.float32))
        self.register_buffer("edges", rg_edges())
        self.add_state(
            "gen_hist",
            default=torch.zeros_like(self.target_hist),
            dist_reduce_fx="sum",
        )

    def update(self, batch: MoleculeBatch, coord_scale: float = 1.0) -> None:
        x = batch.x * coord_scale
        for g in batch.batch.unique():
            mask = batch.batch == g
            accumulate_rog_hist(self.gen_hist, x[mask], self.edges)

    def compute(self) -> torch.Tensor:
        gen = self.gen_hist
        tgt = self.target_hist
        if gen.sum() <= 0 or tgt.sum() <= 0:
            return torch.tensor(0.0, device=gen.device)
        gen_n = gen / gen.sum().clamp(min=1.0)
        tgt_n = tgt / tgt.sum().clamp(min=1.0)
        return (gen_n - tgt_n).abs().sum()


def build_pointcloud_marginal_plots(metrics, atom_tokens: list[str] | None = None):
    """Produce a small set of GT-vs-generated marginal plots from a populated
    ``PointCloudMetric`` collection. Returns ``{wandb_key: matplotlib.Figure}``.

    v1 covers the three 1D marginals — atom count, atom type, radius of gyration.
    Per-pair distance plots are deferred; with ``A`` atom types you'd get up to
    ``A*(A+1)/2`` figures and it's better to pick the few that matter once we
    know which pairs we care about.
    """
    from chemflow.utils.metrics import plot_marginal_comparison

    figs: dict = {}
    by_name = {k: m for k, m in metrics.items()} if hasattr(metrics, "items") else {}

    def _maybe_plot(name, gen_attr, target_attr, labels, title, xlabel, key):
        m = by_name.get(name)
        if m is None:
            return
        gen = getattr(m, gen_attr, None)
        tgt = getattr(m, target_attr, None)
        if gen is None or tgt is None:
            return
        figs[key] = plot_marginal_comparison(gen, tgt, labels, title, xlabel)

    _maybe_plot(
        "atom_count_kl", "gen_hist", "target",
        labels=None, title="Atom count", xlabel="num atoms",
        key="val/pc/plots/atom_count",
    )
    _maybe_plot(
        "atom_type_kl", "gen_hist", "target",
        labels=list(atom_tokens) if atom_tokens is not None else None,
        title="Atom type", xlabel="token",
        key="val/pc/plots/atom_type",
    )
    _maybe_plot(
        "rog_l1", "gen_hist", "target_hist",
        labels=None, title="Radius of gyration",
        xlabel=f"RoG bin (Å in [{RG_RANGE[0]}, {RG_RANGE[1]}])",
        key="val/pc/plots/rog",
    )
    return figs


def build_pointcloud_metrics(distributions, num_atom_types: int) -> dict[str, PointCloudMetric]:
    """Construct the v1 pointcloud metric set from training-side target stats.

    Returns an empty dict if Distributions lacks either pointcloud target stat
    (e.g. preprocessing was run in geometric_graph mode and skipped them). Both
    are checked independently in case one is added without the other later.
    """
    if (
        distributions.pairwise_distance_histogram is None
        or distributions.radius_of_gyration_histogram is None
    ):
        return {}
    return {
        "min_dist_violation": PCMinDistanceViolation(threshold=0.5),
        "range_overflow": PCRangeOverflow(),
        "atom_count_kl": PCAtomCountKL(distributions.n_atoms_distribution),
        "atom_type_kl": PCAtomTypeKL(distributions.atom_type_distribution),
        "pair_dist_l1": PCPairwiseDistanceL1(distributions.pairwise_distance_histogram),
        "rog_l1": PCRoGL1(distributions.radius_of_gyration_histogram),
    }
