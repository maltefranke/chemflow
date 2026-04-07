import os
import random

import math
import os
import random

import torch
import numpy as np
import hydra
from torch.utils.data import Dataset

from chemflow.dataset.data_utils import (
    compute_scaffold_decoration_counts,
    select_scaffold_pairs_by_neighbor_count,
)
from chemflow.dataset.molecule_data import MoleculeBatch
from chemflow.flow_matching.sampling import sample_prior_graph


class FlowMatchingDatasetWrapper(Dataset):
    """Wraps a molecule dataset to move prior sampling and interpolation into
    ``__getitem__``, enabling parallel processing across DataLoader workers.

    For **training**, each call to ``__getitem__`` returns a fully-processed
    ``(mol_t, mol_1, ins_targets, t)`` tuple ready for batching.

    For **validation / test**, it returns ``(sample, target)`` with the prior
    sample already drawn.

    Annealing strategy (``n_atoms_strategy="annealing"``)
    -------------------------------------------------------
    The prior atom count M is sampled from a temperature-controlled
    distribution that interpolates between *fixed* and *empirical* sampling:

        log P(M | N) = log P_emp(M) - β_epoch · (M - N)²

    where N = target atom count and P_emp is the dataset n-atoms distribution.

    * Large β  → distribution collapses to a delta at M = N  (fixed phase).
    * β = 0    → recovers the empirical distribution P_emp  (flexible phase).
    * Intermediate β → a soft neighbourhood around N weighted by P_emp.

    β is annealed from ``anneal_beta_max`` to 0 via an S-shaped (sigmoid)
    schedule parameterised by ``anneal_start_epoch``, ``anneal_end_epoch``
    and ``anneal_steepness``.

    Call :py:meth:`set_epoch` each epoch (done automatically by
    :class:`~chemflow.model.lightning_module.LightningModuleRates`).
    """

    def __init__(
        self,
        base_dataset,
        distributions,
        interpolator,
        n_atoms_strategy="flexible",
        time_dist=None,
        stage="train",
        # Annealing curriculum parameters (used only when n_atoms_strategy="annealing")
        anneal_start_epoch: int = 0,
        anneal_end_epoch: int = 150,
        anneal_steepness: float = 6.0,
        anneal_beta_max: float = 5.0,
    ):
        self.base_dataset = base_dataset
        self.distributions = distributions
        self.interpolator = interpolator
        self.n_atoms_strategy = n_atoms_strategy
        self.time_dist = time_dist
        self.stage = stage

        self.anneal_start_epoch = anneal_start_epoch
        self.anneal_end_epoch = anneal_end_epoch
        self.anneal_steepness = anneal_steepness
        self.anneal_beta_max = anneal_beta_max
        self.current_epoch = 0

        if self.n_atoms_strategy == "median":
            n_atoms_dist = self.distributions.n_atoms_distribution
            n_atoms_cumsum = torch.cumsum(n_atoms_dist, dim=0)
            self._median_n_atoms = int(
                (n_atoms_cumsum >= 0.5).nonzero(as_tuple=True)[0][0].item()
            )

        hydra.utils.log.info(
            "[%s] stage=%s, n_molecules=%d",
            self.__class__.__name__,
            self.stage,
            len(self),
        )
        if self.n_atoms_strategy == "annealing":
            n_dist = self.distributions.n_atoms_distribution.float()
            # log P_emp(M) for each index M; -inf where P_emp = 0
            self._log_p_emp = torch.log(n_dist.clamp(min=1e-10))
            self._n_atoms_range = torch.arange(len(self._log_p_emp), dtype=torch.float)

    # ------------------------------------------------------------------
    # Epoch tracking
    # ------------------------------------------------------------------

    def set_epoch(self, epoch: int) -> None:
        """Update the current training epoch for the annealing schedule.

        Called automatically by :class:`LightningModuleRates` at the start
        of every training epoch.
        """
        self.current_epoch = epoch

    # ------------------------------------------------------------------
    # Annealing helpers
    # ------------------------------------------------------------------

    def _annealing_alpha(self) -> float:
        """S-shaped progress alpha in [0, 1] over training epochs.

        alpha = 0  →  start of curriculum (large β, effectively fixed)
        alpha = 1  →  end of curriculum   (β = 0, fully empirical)
        """
        epoch = self.current_epoch
        if epoch <= self.anneal_start_epoch:
            return 0.0
        if epoch >= self.anneal_end_epoch:
            return 1.0
        t = (epoch - self.anneal_start_epoch) / (
            self.anneal_end_epoch - self.anneal_start_epoch
        )
        k = self.anneal_steepness
        # Sigmoid centred at t = 0.5, normalised so alpha(0) = 0, alpha(1) = 1.
        s = 1.0 / (1.0 + math.exp(-k * (t - 0.5)))
        s0 = 1.0 / (1.0 + math.exp(k * 0.5))  # value at t = 0
        s1 = 1.0 / (1.0 + math.exp(-k * 0.5))  # value at t = 1
        return (s - s0) / (s1 - s0)

    def _get_n_atoms_annealed(self, target) -> int | None:
        """Sample M from log P(M|N) = log P_emp(M) - β · (M - N)².

        Returns ``None`` once β reaches 0, delegating to the internal
        empirical sampler in :func:`sample_prior_graph`.
        """
        alpha = self._annealing_alpha()
        beta = self.anneal_beta_max * (1.0 - alpha)

        if beta < 1e-6:
            return None  # fully empirical — let sample_prior_graph handle it

        n_target = float(target.num_nodes)
        logits = self._log_p_emp - beta * (self._n_atoms_range - n_target) ** 2

        # Enforce a minimum molecule size of 3 atoms
        logits = logits.clone()
        logits[:3] = float("-inf")

        probs = torch.softmax(logits, dim=0)
        return int(torch.multinomial(probs, 1).item())

    # ------------------------------------------------------------------

    def _get_n_atoms(self, target):
        if self.n_atoms_strategy == "fixed":
            return target.num_nodes
        elif self.n_atoms_strategy == "approx":
            n = target.num_nodes + int((torch.randn(1) * 2).round().item())
            return max(3, n)
        elif self.n_atoms_strategy == "median":
            return self._median_n_atoms
        elif self.n_atoms_strategy == "annealing":
            return self._get_n_atoms_annealed(target)
        else:
            return None

    def __getitem__(self, index):
        target = self.base_dataset[index]

        n_atoms = self._get_n_atoms(target)
        sample = sample_prior_graph(self.distributions, n_atoms=n_atoms)

        if self.stage == "train":
            t = self.time_dist.sample((1,)).squeeze(0)
            t = torch.clamp(t, min=0.0, max=1 - 1e-8)

            mol_t, mol_1, ins_targets = self.interpolator.interpolate_single(
                sample, target, t.item()
            )

            if hasattr(target, "y") and target.y is not None:
                mol_t.y = target.y

            return mol_t, mol_1, ins_targets, t

        return sample, target

    def __len__(self):
        return len(self.base_dataset)


class FlowMatchingDatasetWrapperScaffoldDecoration(FlowMatchingDatasetWrapper):
    """Wrapper that uses same-scaffold molecules as flow-matching sources.

    For each target molecule, the source is sampled from the pool of training
    molecules that share the same Bemis-Murcko scaffold.  Molecules with no
    scaffold (acyclic) or a disconnected scaffold are excluded and never appear
    as targets or sources.

    The scaffold group mapping is precomputed by :class:`Preprocessing` and
    passed via ``scaffold_groups`` (a dict with keys ``"mol_to_group"`` and
    ``"groups"``) or loaded lazily from ``scaffold_groups_path``.

    Drop-in replacement for :class:`FlowMatchingDatasetWrapper` — select via
    ``data.datamodule.wrapper._target_`` in the Hydra config.
    """

    def __init__(
        self,
        base_dataset,
        distributions,
        interpolator,
        n_atoms_strategy="flexible",
        time_dist=None,
        stage="train",
        scaffold_groups_dir=None,
        assignment_method: str = "mcs_constrained",
    ):
        super().__init__(
            base_dataset, distributions, interpolator, n_atoms_strategy, time_dist, stage
        )
        self._assignment_method = assignment_method

        path = os.path.join(scaffold_groups_dir, f"scaffold_groups_{stage}.pt")

        if os.path.exists(path):
            scaffold_groups = torch.load(path, weights_only=False)
        else:
            from chemflow.dataset.data_utils import compute_scaffold_groups
            mol_to_group, groups = compute_scaffold_groups(base_dataset)
            scaffold_groups = {"mol_to_group": mol_to_group, "groups": groups}
            os.makedirs(scaffold_groups_dir, exist_ok=True)
            torch.save(scaffold_groups, path)

        # Rebuild if missing or if stored in old single-match format (list[tuple] vs list[list[tuple]])
        if "scaffold_atom_indices" not in scaffold_groups or not isinstance(
            scaffold_groups["scaffold_atom_indices"][0], list
        ):
            from chemflow.dataset.data_utils import compute_scaffold_atom_indices
            scaffold_groups["scaffold_atom_indices"] = compute_scaffold_atom_indices(base_dataset)
            torch.save(scaffold_groups, path)

        if "scaffold_decoration_counts" not in scaffold_groups:
            scaffold_groups["scaffold_decoration_counts"] = (
                compute_scaffold_decoration_counts(
                    base_dataset, scaffold_groups["scaffold_atom_indices"]
                )
            )
            torch.save(scaffold_groups, path)

        if "scaffold_substituents" not in scaffold_groups:
            from chemflow.dataset.data_utils import compute_scaffold_substituents
            scaffold_groups["scaffold_substituents"] = compute_scaffold_substituents(
                base_dataset, scaffold_groups["scaffold_atom_indices"]
            )
            torch.save(scaffold_groups, path)

        mol_to_group = scaffold_groups["mol_to_group"]          # torch.Tensor[N]
        groups = scaffold_groups["groups"]                      # list[list[int]]
        self._scaffold_atom_indices = scaffold_groups["scaffold_atom_indices"]  # list[tuple[int,...]]
        self._scaffold_decoration_counts = scaffold_groups["scaffold_decoration_counts"]
        self._scaffold_substituents = scaffold_groups["scaffold_substituents"]

        # Keep only molecules in a group with ≥2 members (need ≥1 other source)
        self._filtered_indices = [
            i
            for i in range(len(self.base_dataset))
            if mol_to_group[i] >= 0 and len(groups[mol_to_group[i]]) >= 2
        ]
        self._mol_to_group = mol_to_group
        self._groups = groups

    def __len__(self):
        if hasattr(self, "_filtered_indices"):
            return len(self._filtered_indices)
        return len(self.base_dataset)

    def __getitem__(self, index):
        base_idx = self._filtered_indices[index]
        target = self.base_dataset[base_idx]

        gid = int(self._mol_to_group[base_idx])
        candidates = [i for i in self._groups[gid] if i != base_idx]
        source_idx = random.choice(candidates)
        source = self.base_dataset[source_idx]

        if index % 10000 == 0:
            hydra.utils.log.info(
                "[scaffold] __getitem__ index=%d  source: %s  →  target: %s",
                index,
                self.base_dataset.get(source_idx).smiles,
                self.base_dataset.get(base_idx).smiles,
            )

        if self.stage == "train":
            t = self.time_dist.sample((1,)).squeeze(0)
            t = torch.clamp(t, min=0.0, max=1 - 1e-8)

            src_matches = self._scaffold_atom_indices[source_idx]  # list[tuple]
            tgt_matches = self._scaffold_atom_indices[base_idx]    # list[tuple]

            if src_matches and tgt_matches:
                src_dec = self._scaffold_decoration_counts[source_idx]
                tgt_dec = self._scaffold_decoration_counts[base_idx]
                scaffold_pairs = select_scaffold_pairs_by_neighbor_count(
                    src_dec, src_matches, tgt_dec, tgt_matches
                )
            else:
                scaffold_pairs = []

            # Extract substituents if using substituent-based assignment
            if self._assignment_method == "substituent" and scaffold_pairs and self._scaffold_substituents is not None:
                src_subs = self._scaffold_substituents[source_idx]
                tgt_subs = self._scaffold_substituents[base_idx]
                scaffold_substituents = (src_subs, tgt_subs)
            else:
                scaffold_substituents = None

            mol_t, mol_1, ins_targets = self.interpolator.interpolate_single(
                source, target, t.item(),
                scaffold_pairs=scaffold_pairs,
                scaffold_substituents=scaffold_substituents,
            )

            n_scaffold = len(scaffold_pairs) if scaffold_pairs else 0
            scaffold_mask = torch.zeros(mol_t.num_nodes, dtype=torch.long)
            scaffold_mask[:n_scaffold] = 1
            mol_t.scaffold_mask = scaffold_mask

            if hasattr(target, "y") and target.y is not None:
                mol_t.y = target.y

            return mol_t, mol_1, ins_targets, t

        src_matches = self._scaffold_atom_indices[source_idx]
        if src_matches:
            scaffold_indices = list(src_matches[0])
            scaffold_mask = torch.zeros(source.num_nodes, dtype=torch.long)
            scaffold_mask[scaffold_indices] = 1
            source.scaffold_mask = scaffold_mask

        return source, target


def train_collate_fn(batch):
    """Collate pre-processed training samples into batched tensors.

    Offsets the local insertion metadata on ``ins_targets`` so indices become
    valid in the batched tensors.

    Downstream contract for ``ins_targets``:
    - ``spawn_node_idx`` indexes batched ``mol_t`` (for insertion-rate/GMM loss).
    - ``ins_edge_spawn_idx`` indexes batched ``mol_t`` (query node).
    - ``ins_edge_existing_idx`` indexes batched ``mol_t`` (existing endpoint).
    - ``ins_edge_ins_local_idx`` indexes batched ``ins_targets`` (inserted node).
    - ``ins_edge_types`` stores edge class targets.
    """
    mol_t_list = []
    mol_1_list = []
    ins_targets_list = []
    t_list = []

    offset = 0
    ins_offset = 0
    for mol_t, mol_1, ins_targets, t in batch:
        if (
            hasattr(ins_targets, "spawn_node_idx")
            and ins_targets.spawn_node_idx.numel() > 0
        ):
            ins_targets.spawn_node_idx.add_(offset)
        if (
            hasattr(ins_targets, "ins_edge_spawn_idx")
            and ins_targets.ins_edge_spawn_idx.numel() > 0
        ):
            ins_targets.ins_edge_spawn_idx.add_(offset)
        if (
            hasattr(ins_targets, "ins_edge_existing_idx")
            and ins_targets.ins_edge_existing_idx.numel() > 0
        ):
            ins_targets.ins_edge_existing_idx.add_(offset)
        if (
            hasattr(ins_targets, "ins_edge_ins_local_idx")
            and ins_targets.ins_edge_ins_local_idx.numel() > 0
        ):
            ins_targets.ins_edge_ins_local_idx.add_(ins_offset)
        if (
            hasattr(ins_targets, "ins_to_ins_edge_src_local_idx")
            and ins_targets.ins_to_ins_edge_src_local_idx.numel() > 0
        ):
            ins_targets.ins_to_ins_edge_src_local_idx.add_(ins_offset)
        if (
            hasattr(ins_targets, "ins_to_ins_edge_dst_local_idx")
            and ins_targets.ins_to_ins_edge_dst_local_idx.numel() > 0
        ):
            ins_targets.ins_to_ins_edge_dst_local_idx.add_(ins_offset)
        if (
            hasattr(ins_targets, "ins_to_ins_edge_spawn_src_idx")
            and ins_targets.ins_to_ins_edge_spawn_src_idx.numel() > 0
        ):
            ins_targets.ins_to_ins_edge_spawn_src_idx.add_(offset)
        if (
            hasattr(ins_targets, "ins_to_ins_edge_spawn_dst_idx")
            and ins_targets.ins_to_ins_edge_spawn_dst_idx.numel() > 0
        ):
            ins_targets.ins_to_ins_edge_spawn_dst_idx.add_(offset)

        mol_t_list.append(mol_t)
        mol_1_list.append(mol_1)
        ins_targets_list.append(ins_targets)
        t_list.append(t)

        offset += mol_t.num_nodes
        ins_offset += ins_targets.num_nodes

    mol_t_batch = MoleculeBatch.from_data_list(mol_t_list)
    mol_1_batch = MoleculeBatch.from_data_list(mol_1_list)
    ins_targets_batch = MoleculeBatch.from_data_list(ins_targets_list)
    t_batch = torch.stack(t_list)

    return mol_t_batch, mol_1_batch, ins_targets_batch, t_batch


def eval_collate_fn(batch):
    """Collate evaluation samples (prior, target) into batched tensors."""
    samples, targets = zip(*batch)
    samples_batched = MoleculeBatch.from_data_list(list(samples))
    targets_batched = MoleculeBatch.from_data_list(list(targets))
    return samples_batched, targets_batched


def worker_init_fn(worker_id):
    """Seed numpy and Python random per worker so randomness diverges."""
    info = torch.utils.data.get_worker_info()
    seed = info.seed % (2**32)
    np.random.seed(seed)
    random.seed(seed)
