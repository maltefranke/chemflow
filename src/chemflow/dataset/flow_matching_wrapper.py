import torch
import numpy as np
from torch.utils.data import Dataset

from chemflow.flow_matching.sampling import sample_prior_graph
from chemflow.dataset.molecule_data import MoleculeBatch


class FlowMatchingDatasetWrapper(Dataset):
    """Wraps a molecule dataset to move prior sampling and interpolation into
    ``__getitem__``, enabling parallel processing across DataLoader workers.

    For **training**, each call to ``__getitem__`` returns a fully-processed
    ``(mol_t, mol_1, ins_targets, t)`` tuple ready for batching.

    For **validation / test**, it returns ``(sample, target)`` with the prior
    sample already drawn.
    """

    def __init__(
        self,
        base_dataset,
        distributions,
        interpolator,
        n_atoms_strategy="flexible",
        time_dist=None,
        stage="train",
    ):
        self.base_dataset = base_dataset
        self.distributions = distributions
        self.interpolator = interpolator
        self.n_atoms_strategy = n_atoms_strategy
        self.time_dist = time_dist
        self.stage = stage

        if self.n_atoms_strategy == "median":
            n_atoms_dist = self.distributions.n_atoms_distribution
            n_atoms_cumsum = torch.cumsum(n_atoms_dist, dim=0)
            self._median_n_atoms = int(
                (n_atoms_cumsum >= 0.5).nonzero(as_tuple=True)[0][0].item()
            )

    def _get_n_atoms(self, target):
        if self.n_atoms_strategy == "fixed":
            return target.num_nodes
        elif self.n_atoms_strategy == "approx":
            n = target.num_nodes + int((torch.randn(1) * 2).round().item())
            return max(3, n)
        elif self.n_atoms_strategy == "median":
            return self._median_n_atoms
        else:
            return None

    def __getitem__(self, index):
        target = self.base_dataset[index]

        n_atoms = self._get_n_atoms(target)
        sample = sample_prior_graph(self.distributions, n_atoms=n_atoms)

        if self.stage == "train":
            t = self.time_dist.sample((1,)).squeeze(0)
            t = torch.clamp(t, min=0.0, max=1 - 1e-6)

            mol_t, mol_1, ins_targets = self.interpolator.interpolate_single(
                sample, target, t.item()
            )

            if hasattr(target, "y") and target.y is not None:
                mol_t.y = target.y

            return mol_t, mol_1, ins_targets, t

        return sample, target

    def __len__(self):
        return len(self.base_dataset)


def train_collate_fn(batch):
    """Collate pre-processed training samples into batched tensors.

    Offsets the local ``spawn_node_idx`` / ``ins_edge_*_idx`` attributes on
    ``ins_targets`` so they become valid global indices into the batched
    ``mol_t``.
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
    """Seed numpy per worker so scipy/numpy randomness diverges across workers."""
    info = torch.utils.data.get_worker_info()
    seed = info.seed % (2**32)
    np.random.seed(seed)
