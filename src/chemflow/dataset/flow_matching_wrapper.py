import math
import torch
import numpy as np
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset

from chemflow.flow_matching.sampling import sample_prior_graph
from chemflow.dataset.molecule_data import MoleculeBatch
from chemflow.dataset.representation import (
    Representation,
    project_molecule_to_representation,
)
from chemflow.dataset.vocab import Vocab


class FlowMatchingDatasetWrapper(Dataset):
    """Wraps a molecule dataset to move prior sampling and interpolation into
    ``__getitem__``, enabling parallel processing across DataLoader workers.

    For **training**, each call to ``__getitem__`` returns a fully-processed
    ``(mol_t, mol_1, ins_targets, t)`` tuple ready for batching. When
    ``n_augmentations > 1`` (training only), ``__getitem__`` instead returns a
    list of ``n_augmentations`` such tuples — same molecule / prior / time /
    interpolation noise, but each with an independent random rotation applied
    to the coordinates. The DataLoader's ``batch_size`` should therefore be
    divided by ``n_augmentations`` so that the effective collated batch size
    stays constant.

    For **validation / test**, it returns ``(sample, target)`` with the prior
    sample already drawn.
    """

    def __init__(
        self,
        base_dataset,
        distributions,
        interpolator,
        vocab: Vocab,
        representation: str | Representation = Representation.GEOMETRIC_GRAPH,
        n_atoms_strategy="flexible",
        time_dist=None,
        stage="train",
        rotate: bool = False,
        n_augmentations: int = 1,
    ):
        self.base_dataset = base_dataset
        self.distributions = distributions
        self.interpolator = interpolator
        self.vocab = vocab
        self.representation = Representation(representation)
        self.n_atoms_strategy = n_atoms_strategy
        self.time_dist = time_dist
        self.stage = stage
        self.rotate = rotate
        self.n_augmentations = max(1, int(n_augmentations))

        if self.n_atoms_strategy == "median":
            n_atoms_dist = self.distributions.n_atoms_distribution
            n_atoms_cumsum = torch.cumsum(n_atoms_dist, dim=0)
            self._median_n_atoms = int(
                (n_atoms_cumsum >= 0.5).nonzero(as_tuple=True)[0][0].item()
            )

        if self.n_atoms_strategy == "uniform":
            n_atoms_dist = self.distributions.n_atoms_distribution
            nonzero_idx = (n_atoms_dist > 0).nonzero(as_tuple=True)[0]
            self._n_atoms_min = max(int(nonzero_idx[0].item()) - 5, 3)
            self._n_atoms_max = int(nonzero_idx[-1].item()) + 5

    def _get_n_atoms(self, target):
        if self.n_atoms_strategy == "fixed":
            return target.num_nodes
        elif self.n_atoms_strategy == "approx":
            n = target.num_nodes + int((torch.randn(1) * 2).round().item())
            return max(3, n)
        elif self.n_atoms_strategy == "median":
            return self._median_n_atoms
        elif self.n_atoms_strategy == "uniform":
            return int(
                torch.randint(self._n_atoms_min, self._n_atoms_max + 1, (1,)).item()
            )
        else:
            return None

    @staticmethod
    def _apply_rotation_inplace(mol, R):
        """Rotate ``mol.x`` in place. ``R`` is a (3, 3) tensor matching ``mol.x.dtype``."""
        if mol.x is not None and mol.x.numel() > 0:
            mol.x = mol.x @ R

    def _augment(self, mol_t, mol_1, ins_targets, t):
        """Produce one augmented copy with a fresh random 3D rotation applied
        consistently to ``mol_t.x``, ``mol_1.x`` and ``ins_targets.x``."""
        aug_mol_t = mol_t.clone()
        aug_mol_1 = mol_1.clone()
        aug_ins_targets = ins_targets.clone()

        if self.rotate:
            R = torch.from_numpy(Rotation.random(1).as_matrix()[0]).to(
                dtype=aug_mol_t.x.dtype
            )
            self._apply_rotation_inplace(aug_mol_t, R)
            self._apply_rotation_inplace(aug_mol_1, R)
            self._apply_rotation_inplace(aug_ins_targets, R)

        return aug_mol_t, aug_mol_1, aug_ins_targets, t.clone()

    def __getitem__(self, index):
        target = self.base_dataset[index]
        target = project_molecule_to_representation(
            target, self.vocab, self.representation
        )

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

            # The interpolator builds mol_1 from filter_nodes(target) and drops
            # non-tensor attributes; re-attach the SMILES so downstream CFG
            # signals (e.g. logP) can be computed from mol_1.
            if hasattr(target, "smiles"):
                mol_1.smiles = target.smiles
                mol_t.smiles = target.smiles

            if self.n_augmentations <= 1:
                if self.rotate:
                    return self._augment(mol_t, mol_1, ins_targets, t)
                return mol_t, mol_1, ins_targets, t

            # Same processed sample, N independent rotations.
            return [
                self._augment(mol_t, mol_1, ins_targets, t)
                for _ in range(self.n_augmentations)
            ]

        return sample, target

    def __len__(self):
        return len(self.base_dataset)


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
    # When ``n_augmentations > 1`` the wrapper returns a list of tuples per
    # __getitem__ call (same molecule, different rotations). Flatten so the
    # rest of this function can iterate uniformly.
    flat_batch = []
    for item in batch:
        if isinstance(item, list):
            flat_batch.extend(item)
        else:
            flat_batch.append(item)
    batch = flat_batch

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
    """Seed numpy per worker so scipy/numpy randomness diverges across workers."""
    info = torch.utils.data.get_worker_info()
    seed = info.seed % (2**32)
    np.random.seed(seed)
