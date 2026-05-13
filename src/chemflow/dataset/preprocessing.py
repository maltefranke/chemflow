"""Preprocessing class for calculating token distributions from datasets."""

import importlib
import os

import torch
from torch_geometric.data import Dataset
from torch_geometric.utils import to_dense_adj

from chemflow.dataset.vocab import Distributions, Vocab
from chemflow.dataset.representation import neutral_charge_index
from chemflow.utils.pointcloud_metrics import (
    DIST_N_BINS,
    RG_N_BINS,
    accumulate_pairwise_distance_hist,
    accumulate_rog_hist,
    dist_edges,
    rg_edges,
)
from chemflow.utils.utils import (
    token_to_index,
    z_to_atom_types,
)


class Preprocessing:
    """
    Preprocessing class that computes token distributions from a training dataset.

    This class should be instantiated before the datamodule to compute distributions
    from the training dataset only. The computed tokens and distributions are then
    passed to the datamodule and model.

    Tokens are discovered from the data and saved to a file. On subsequent runs,
    tokens are loaded from the file if it exists.
    """

    # Special tokens that must always be present
    NO_BOND_TOKEN = "<NO_BOND>"

    def __init__(
        self,
        root: str,
        train_dataset: Dataset | dict | None = None,
        atom_tokens_path: str = None,
        edge_tokens_path: str = None,
        charge_tokens_path: str = None,
        distributions_path: str = None,
    ):
        """
        Initialize preprocessing.

        Args:
            root: Root directory path for the QM9 dataset
            train_dataset: Training dataset instance or Hydra config for lazy
                instantiation. It is only instantiated when preprocessing files are
                missing.
            atom_tokens_path: Path to save/load atom tokens.
                If None, uses root/atom_tokens.txt
            edge_tokens_path: Path to save/load edge tokens.
                If None, uses root/edge_tokens.txt
            charge_tokens_path: Path to save/load charge tokens.
                If None, uses root/charge_tokens.txt
            distributions_path: Path to save/load distributions.
                If None, uses root/distributions.pt
        """
        self.root = root
        self._train_dataset_cfg = train_dataset
        self.train_dataset = None

        # Set default tokens path if not provided
        if atom_tokens_path is None:
            atom_tokens_path = os.path.join(self.root, "atom_tokens.txt")

        # Set default edge tokens path if not provided
        if edge_tokens_path is None:
            edge_tokens_path = os.path.join(self.root, "edge_tokens.txt")

        # Set default charge tokens path if not provided
        if charge_tokens_path is None:
            charge_tokens_path = os.path.join(self.root, "charge_tokens.txt")

        # Set default distributions path if not provided
        if distributions_path is None:
            distributions_path = os.path.join(self.root, "distributions.pt")

        self.atom_tokens_path = atom_tokens_path
        self.edge_tokens_path = edge_tokens_path
        self.charge_tokens_path = charge_tokens_path
        self.distributions_path = distributions_path

        has_all_files = (
            os.path.exists(self.atom_tokens_path)
            and os.path.exists(self.edge_tokens_path)
            and os.path.exists(self.charge_tokens_path)
            and os.path.exists(self.distributions_path)
        )

        if not has_all_files:
            self._ensure_train_dataset()

        # Load or compute tokens (both computed together if either is missing)
        self.vocab = self._load_or_compute_tokens()
        self.distributions = self._load_or_compute_distributions()

        # remove train dataset to free memory
        self.train_dataset = None

    def _ensure_train_dataset(self):
        """Instantiate train dataset lazily only when preprocessing needs it."""
        if self.train_dataset is not None:
            return

        if self._train_dataset_cfg is None:
            raise ValueError(
                "train_dataset is required when preprocessing artifacts are missing."
            )

        if isinstance(self._train_dataset_cfg, Dataset):
            self.train_dataset = self._train_dataset_cfg
            return

        try:
            hydra = importlib.import_module("hydra")
        except ImportError as e:
            raise ImportError(
                "Hydra is required to instantiate train_dataset config lazily."
            ) from e

        self.train_dataset = hydra.utils.instantiate(self._train_dataset_cfg)

    def _load_or_compute_tokens(self) -> Vocab:
        """
        Load tokens from files if they exist,
        otherwise compute both from data in a single loop and save.

        Returns:
            Vocab object
        """
        # Try to load both from files
        atom_tokens_exist = os.path.exists(self.atom_tokens_path)
        edge_tokens_exist = os.path.exists(self.edge_tokens_path)
        charge_tokens_exist = os.path.exists(self.charge_tokens_path)

        if atom_tokens_exist and edge_tokens_exist and charge_tokens_exist:
            atom_tokens = self._load_tokens(self.atom_tokens_path)
            edge_tokens = self._load_tokens(self.edge_tokens_path)
            charge_tokens = self._load_tokens(self.charge_tokens_path)
        else:
            # Compute both from data in a single loop
            atom_tokens, edge_tokens, charge_tokens = self._compute_tokens_from_data()

            # Save both to files
            if not atom_tokens_exist:
                self._save_tokens(self.atom_tokens_path, atom_tokens)
            if not edge_tokens_exist:
                self._save_tokens(self.edge_tokens_path, edge_tokens)
            if not charge_tokens_exist:
                self._save_tokens(self.charge_tokens_path, charge_tokens)

        token_dict = {
            "atom_tokens": atom_tokens,
            "edge_tokens": edge_tokens,
            "charge_tokens": charge_tokens,
        }

        vocab = Vocab(**token_dict)
        return vocab

    def _compute_tokens_from_data(
        self,
    ) -> tuple[list[str], list[str], list[str]]:
        """
        Compute both tokens and edge tokens by extracting unique types
        from training data in a single loop.

        Returns:
            Tuple of (tokens, edge_tokens, charge_tokens)
        """
        # Load dataset to extract unique atom and edge type

        # Extract all atom types and edge types from the dataset in one loop
        all_atom_types = set()
        all_edge_type_indices = set()
        all_charge_tokens = set()

        for i in range(len(self.train_dataset)):
            data = self.train_dataset.get(i)

            # Extract atom types
            atom_types = z_to_atom_types(data.z.tolist())
            all_atom_types.update(atom_types)

            # Extract edge types — only when the source provides bonds.
            # Datasets without topology (e.g. TMQM) won't have `edge_attr` at all,
            # in which case the canonical edge vocab falls back to ["<NO_BOND>"].
            if (
                hasattr(data, "edge_attr")
                and data.edge_attr is not None
                and data.edge_attr.numel() > 0
            ):
                all_edge_type_indices.update(data.edge_attr.tolist())

            if hasattr(data, "charges") and data.charges is not None:
                all_charge_tokens.update(data.charges.tolist())

        # Convert to sorted lists for deterministic ordering
        atom_types_sorted = sorted(all_atom_types)
        edge_type_indices_sorted = sorted(all_edge_type_indices)
        charge_tokens_sorted = sorted(all_charge_tokens)

        # Atom tokens: discovered atom types only (no special tokens)
        tokens = atom_types_sorted

        # Canonical edge tokens: [NO_BOND] + discovered bond types. Fallback to
        # [NO_BOND] alone when the source has no bonds (e.g. TMQM) so that
        # `${len:...}` and head/embedding shapes stay well-defined downstream.
        bond_type_tokens = [str(int(idx)) for idx in edge_type_indices_sorted]
        edge_tokens = [self.NO_BOND_TOKEN, *bond_type_tokens]

        # Canonical charge tokens: discovered as sorted ints. Fallback to ["0"]
        # when the source has no charges, for the same reason.
        charge_tokens = [str(int(idx)) for idx in charge_tokens_sorted]
        if not charge_tokens:
            charge_tokens = ["0"]

        return tokens, edge_tokens, charge_tokens

    def _save_tokens(self, path: str, tokens: list[str]):
        """Save tokens to file.

        Args:
            path: Path to the tokens file
            tokens: List of tokens to save
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            for token in tokens:
                f.write(f"{token}\n")

    def _load_tokens(self, path: str) -> list[str]:
        """Load tokens from file.

        Args:
            path: Path to the tokens file

        Returns:
            List of tokens loaded from file
        """
        with open(path, "r") as f:
            tokens = [line.strip() for line in f.readlines()]
        return tokens

    def _load_or_compute_distributions(self) -> Distributions:
        """
        Load distributions from file if exists, otherwise compute from data and save.

        Returns:
            Distributions object
        """
        # Try to load from file
        if os.path.exists(self.distributions_path):
            distributions = self._load_distributions()
        else:
            # Compute distributions from data
            distributions = self._compute_distributions()

            # Save distributions to file
            self._save_distributions(distributions)

        distributions = Distributions(**distributions)
        return distributions

    def _compute_distributions(self) -> dict[str, torch.Tensor]:
        """Compute distributions from the training dataset."""
        # Compute distributions from the provided training dataset only.
        dataset = self.train_dataset

        # Compute edge type and number of atoms distributions
        # Also collect coordinates for std calculation
        all_num_atoms = []
        all_atom_type_indices = []
        all_edge_type_indices = []
        all_coords = []
        all_charges = []

        # Pointcloud-mode evaluation targets — computed canonically so the same
        # cache serves any representation.
        num_atom_types = len(self.vocab.atom_tokens)
        pair_hist = torch.zeros(
            num_atom_types, num_atom_types, DIST_N_BINS, dtype=torch.float32
        )
        rog_hist = torch.zeros(RG_N_BINS, dtype=torch.float32)
        d_edges = dist_edges()
        r_edges = rg_edges()

        for i in range(len(dataset)):
            data = dataset.get(i)
            num_atoms = data.num_nodes

            atom_types = z_to_atom_types(data.z.tolist())
            atom_type_indices_t = torch.tensor(
                [
                    token_to_index(self.vocab.atom_tokens, token)
                    for token in atom_types
                ],
                dtype=torch.long,
            )
            all_atom_type_indices.append(atom_type_indices_t)

            # Edge histogram: built only if the dataset provides bonds. Sources
            # without topology contribute nothing here (vocab is the singleton
            # canonical [NO_BOND] for them anyway).
            if hasattr(data, "edge_attr") and data.edge_attr is not None and data.edge_attr.numel() > 0:
                edge_types = data.edge_attr
                adj_matrix = to_dense_adj(
                    data.edge_index, edge_attr=edge_types, max_num_nodes=num_atoms
                )
                adj_matrix = adj_matrix.squeeze(0)
                triu_indices = torch.triu_indices(
                    row=num_atoms, col=num_atoms, offset=1
                )
                dense_edge_types = adj_matrix[triu_indices[0], triu_indices[1]]
                all_edge_type_indices.append(dense_edge_types.long())

            coord = data.pos - data.pos.mean(dim=0)
            all_coords.append(coord)

            accumulate_pairwise_distance_hist(
                pair_hist, coord, atom_type_indices_t, d_edges
            )
            accumulate_rog_hist(rog_hist, coord, r_edges)

            if hasattr(data, "charges") and data.charges is not None:
                charges = data.charges.tolist()
                charge_type_indices = [
                    token_to_index(self.vocab.charge_tokens, str(token))
                    for token in charges
                ]
                all_charges.append(
                    torch.tensor(charge_type_indices, dtype=torch.long)
                )

            all_num_atoms.append(num_atoms)

        all_num_atoms = torch.tensor(all_num_atoms, dtype=torch.long)
        n_atoms_distribution = all_num_atoms.bincount()
        n_atoms_distribution = n_atoms_distribution / n_atoms_distribution.sum()

        all_atom_type_indices = torch.cat(all_atom_type_indices, dim=0)
        atom_type_distribution = all_atom_type_indices.bincount(
            minlength=len(self.vocab.atom_tokens)
        ).float()
        atom_type_distribution = atom_type_distribution / atom_type_distribution.sum()

        # Canonical edge distribution. Sources without bonds emit no edge entries;
        # we fall back to a one-hot at NO_BOND (token 0) so the singleton-canonical
        # vocab maps to a proper one-hot prior.
        num_edge_tokens = len(self.vocab.edge_tokens)
        if all_edge_type_indices:
            all_edge_type_indices = torch.cat(all_edge_type_indices, dim=0)
            edge_type_distribution = all_edge_type_indices.bincount(
                minlength=num_edge_tokens
            ).float()
            edge_type_distribution = (
                edge_type_distribution / edge_type_distribution.sum().clamp(min=1.0)
            )
        else:
            edge_type_distribution = torch.zeros(num_edge_tokens)
            edge_type_distribution[0] = 1.0

        # Canonical charge distribution. Sources without charges emit no entries;
        # fall back to one-hot at the canonical "0" index when available.
        num_charge_tokens = len(self.vocab.charge_tokens)
        if all_charges:
            all_charges = torch.cat(all_charges, dim=0)
            charge_type_distribution = all_charges.long().bincount(
                minlength=num_charge_tokens
            ).float()
            charge_type_distribution = (
                charge_type_distribution / charge_type_distribution.sum()
            )
        else:
            charge_type_distribution = torch.zeros(num_charge_tokens)
            charge_type_distribution[neutral_charge_index(self.vocab)] = 1.0

        # Compute coordinate std across all coordinates in the dataset
        all_coords = torch.cat(all_coords, dim=0)  # Shape: (total_atoms, 3)

        # Use overall std (mean of per-dimension stds) or keep per-dimension
        # Using overall std as a scalar for simplicity
        coordinate_std = all_coords.std()  # Shape: (1)
        coordinate_std = torch.tensor(coordinate_std, dtype=torch.float32).item()

        return {
            "atom_type_distribution": atom_type_distribution,
            "edge_type_distribution": edge_type_distribution,
            "charge_type_distribution": charge_type_distribution,
            "n_atoms_distribution": n_atoms_distribution,
            "coordinate_std": coordinate_std,
            "pairwise_distance_histogram": pair_hist,
            "radius_of_gyration_histogram": rog_hist,
        }

    def _save_distributions(self, distributions: dict[str, torch.Tensor]):
        """Save distributions to file."""
        os.makedirs(os.path.dirname(self.distributions_path), exist_ok=True)
        torch.save(distributions, self.distributions_path)

    def _load_distributions(self) -> dict[str, torch.Tensor]:
        """Load distributions from file."""
        return torch.load(self.distributions_path, weights_only=False)
