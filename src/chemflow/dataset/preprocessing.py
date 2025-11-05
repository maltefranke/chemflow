"""Preprocessing class for calculating token distributions from datasets."""

import os
import torch
from torch_geometric.datasets import QM9
from chemflow.utils import (
    edge_types_to_triu_entries,
    z_to_atom_types,
    token_to_index,
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
    SPECIAL_TOKENS = ["<MASK>", "<DEATH>"]

    def __init__(
        self,
        root: str,
        tokens_path: str = None,
        distributions_path: str = None,
    ):
        """
        Initialize preprocessing.

        Args:
            root: Root directory path for the QM9 dataset
            tokens_path: Path to save/load tokens. If None, uses root/tokens.txt
            distributions_path: Path to save/load distributions.
                If None, uses root/distributions.pt
        """
        self.root = root

        # Set default tokens path if not provided
        if tokens_path is None:
            tokens_path = os.path.join(self.root, "tokens.txt")

        # Set default distributions path if not provided
        if distributions_path is None:
            distributions_path = os.path.join(self.root, "distributions.pt")

        self.tokens_path = tokens_path
        self.distributions_path = distributions_path

        # Load or compute tokens
        self.tokens = self._load_or_compute_tokens()

        # Will be computed in compute_distributions
        self.atom_type_distribution = None
        self.edge_type_distribution = None
        self.n_atoms_distribution = None
        self.distributions = None

        # Load or compute distributions
        self._load_or_compute_distributions()

    def _load_or_compute_tokens(self) -> list[str]:
        """
        Load tokens from file if exists, otherwise compute from data and save.

        Returns:
            List of token strings
        """
        # Try to load from file
        if os.path.exists(self.tokens_path):
            tokens = self._load_tokens()
            return tokens

        # Compute tokens from data
        tokens = self._compute_tokens_from_data()

        # Save tokens to file
        self._save_tokens(tokens)

        return tokens

    def _compute_tokens_from_data(self) -> list[str]:
        """Compute tokens by extracting unique atom types from training data."""
        # Load QM9 dataset to extract unique atom types
        dataset = QM9(root=self.root)

        # Extract all atom types from the dataset
        all_atom_types = set()
        for i in range(len(dataset)):
            data = dataset[i]
            atom_types = z_to_atom_types(data.z.tolist())
            all_atom_types.update(atom_types)

        # Convert to sorted list for deterministic ordering
        atom_types_sorted = sorted(all_atom_types)

        # Combine special tokens (always first) with discovered atom types
        tokens = self.SPECIAL_TOKENS + atom_types_sorted

        return tokens

    def _save_tokens(self, tokens: list[str]):
        """Save tokens to file."""
        os.makedirs(os.path.dirname(self.tokens_path), exist_ok=True)
        with open(self.tokens_path, "w") as f:
            for token in tokens:
                f.write(f"{token}\n")

    def _load_tokens(self) -> list[str]:
        """Load tokens from file."""
        with open(self.tokens_path, "r") as f:
            tokens = [line.strip() for line in f.readlines()]
        return tokens

    def _load_or_compute_distributions(self):
        """
        Load distributions from file if exists, otherwise compute from data and save.

        Returns:
            None (sets instance attributes)
        """
        # Try to load from file
        if os.path.exists(self.distributions_path):
            distributions = self._load_distributions()
            self.atom_type_distribution = distributions["atom_type_distribution"]
            self.edge_type_distribution = distributions["edge_type_distribution"]
            self.n_atoms_distribution = distributions["n_atoms_distribution"]
            self.distributions = distributions
            return

        # Compute distributions from data
        distributions = self._compute_distributions()

        # Save distributions to file
        self._save_distributions(distributions)

        # Store as instance attributes
        self.atom_type_distribution = distributions["atom_type_distribution"]
        self.edge_type_distribution = distributions["edge_type_distribution"]
        self.n_atoms_distribution = distributions["n_atoms_distribution"]
        self.distributions = distributions

    def _compute_distributions(self) -> dict[str, torch.Tensor]:
        """Compute distributions from the training dataset."""
        # Load QM9 dataset to compute distributions
        dataset = QM9(root=self.root)

        # Compute atom type distribution
        atom_types = z_to_atom_types(dataset.z.tolist())
        atom_type_indices = [token_to_index(self.tokens, token) for token in atom_types]
        atom_type_indices = torch.tensor(atom_type_indices, dtype=torch.long)

        # Create distribution over all tokens
        all_token_indices = torch.arange(len(self.tokens), dtype=torch.long)
        atom_type_distribution = (
            atom_type_indices.unsqueeze(1) == all_token_indices
        ).sum(dim=0)
        atom_type_distribution = atom_type_distribution / atom_type_distribution.sum()

        # Compute edge type and number of atoms distributions
        all_num_atoms = []
        all_edge_types = []

        for i in range(len(dataset)):
            data = dataset[i]
            num_atoms = data.num_nodes

            triu_edge_types = edge_types_to_triu_entries(
                data.edge_index, data.edge_attr, num_atoms
            )

            all_num_atoms.append(num_atoms)
            all_edge_types.append(triu_edge_types)

        all_num_atoms = torch.tensor(all_num_atoms, dtype=torch.long)
        n_atoms_distribution = all_num_atoms.bincount()
        n_atoms_distribution = n_atoms_distribution / n_atoms_distribution.sum()

        all_edge_types = torch.cat(all_edge_types, dim=0)
        edge_type_distribution = all_edge_types.bincount()
        edge_type_distribution = edge_type_distribution / edge_type_distribution.sum()

        return {
            "atom_type_distribution": atom_type_distribution,
            "edge_type_distribution": edge_type_distribution,
            "n_atoms_distribution": n_atoms_distribution,
        }

    def _save_distributions(self, distributions: dict[str, torch.Tensor]):
        """Save distributions to file."""
        os.makedirs(os.path.dirname(self.distributions_path), exist_ok=True)
        torch.save(distributions, self.distributions_path)

    def _load_distributions(self) -> dict[str, torch.Tensor]:
        """Load distributions from file."""
        return torch.load(self.distributions_path, weights_only=False)
