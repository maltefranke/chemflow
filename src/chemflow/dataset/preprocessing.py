"""Preprocessing class for calculating token distributions from datasets."""

import os

import torch
from torch_geometric.utils import to_dense_adj

from chemflow.dataset.qm9 import QM9Charges
from chemflow.utils import (
    token_to_index,
    z_to_atom_types,
)
from chemflow.dataset.vocab import Vocab, Distributions


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
    MASK_TOKEN = "<MASK>"
    NO_BOND_TOKEN = "<NO_BOND>"
    SPECIAL_TOKENS = [MASK_TOKEN]
    EDGE_SPECIAL_TOKENS = [MASK_TOKEN, NO_BOND_TOKEN]

    def __init__(
        self,
        root: str,
        atom_tokens_path: str = None,
        edge_tokens_path: str = None,
        charge_tokens_path: str = None,
        distributions_path: str = None,
    ):
        """
        Initialize preprocessing.

        Args:
            root: Root directory path for the QM9 dataset
            tokens_path: Path to save/load tokens. If None, uses root/tokens.txt
            edge_tokens_path: Path to save/load edge tokens.
                If None, uses root/edge_tokens.txt
            charge_tokens_path: Path to save/load charge tokens.
                If None, uses root/charge_tokens.txt
            distributions_path: Path to save/load distributions.
                If None, uses root/distributions.pt
        """
        self.root = root

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

        # Load or compute tokens (both computed together if either is missing)
        self.vocab = self._load_or_compute_tokens()
        self.distributions = self._load_or_compute_distributions()

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
        # Load QM9 dataset to extract unique atom and edge types
        dataset = QM9Charges(root=self.root)

        # Extract all atom types and edge types from the dataset in one loop
        all_atom_types = set()
        all_edge_type_indices = set()
        all_charge_tokens = set()

        for i in range(len(dataset)):
            data = dataset[i]

            # Extract atom types
            atom_types = z_to_atom_types(data.z.tolist())
            all_atom_types.update(atom_types)

            # Extract edge types
            # QM9 edge_attr is one-hot encoded, so we need to convert to indices
            if data.edge_attr.numel() > 0:
                # Convert one-hot to indices: argmax gives 0=single, 1=double,
                # 2=triple, 3=aromatic. Then add 1 to get: 1=single, 2=double,
                # 3=triple, 4=aromatic
                edge_type_indices = data.edge_attr.argmax(dim=-1) + 1
                all_edge_type_indices.update(edge_type_indices.tolist())

            if hasattr(data, "charges") and data.charges is not None:
                all_charge_tokens.update(data.charges.tolist())

        # Convert to sorted lists for deterministic ordering
        atom_types_sorted = sorted(all_atom_types)
        edge_type_indices_sorted = sorted(all_edge_type_indices)
        charge_tokens_sorted = sorted(all_charge_tokens)

        # Combine special tokens (always first) with discovered atom types
        tokens = self.SPECIAL_TOKENS + atom_types_sorted

        # Create bond type tokens (as strings)
        bond_type_tokens = [str(int(idx)) for idx in edge_type_indices_sorted]

        # Edge tokens order: [NO_BOND_TOKEN] + [bond types] + [MASK_TOKEN]
        # This way: 0 maps to NO_BOND (index 0), 1-4 map to bond types,
        # MASK is last
        edge_tokens = [self.NO_BOND_TOKEN, *bond_type_tokens, self.MASK_TOKEN]

        charge_tokens = [str(int(idx)) for idx in charge_tokens_sorted]

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
        # Load QM9 dataset to compute distributions
        dataset = QM9Charges(root=self.root)

        # Compute atom type distribution
        atom_types = z_to_atom_types(dataset.z.tolist())
        atom_type_indices = [
            token_to_index(self.vocab.atom_tokens, token) for token in atom_types
        ]
        atom_type_indices = torch.tensor(atom_type_indices, dtype=torch.long)

        # Create distribution over all tokens
        all_token_indices = torch.arange(len(self.vocab.atom_tokens), dtype=torch.long)
        atom_type_distribution = (
            atom_type_indices.unsqueeze(1) == all_token_indices
        ).sum(dim=0)
        atom_type_distribution = atom_type_distribution / atom_type_distribution.sum()

        # Compute edge type and number of atoms distributions
        # Also collect coordinates for std calculation
        all_num_atoms = []
        all_edge_type_indices = []
        all_coords = []
        all_charges = []

        for i in range(len(dataset)):
            data = dataset[i]
            num_atoms = data.num_nodes

            # Convert edge_attr (one-hot) to dense adjacency matrix
            # This gives: 0=no bond, 1=single, 2=double, 3=triple, 4=aromatic
            edge_types = data.edge_attr.argmax(dim=-1) + 1
            adj_matrix = to_dense_adj(
                data.edge_index, edge_attr=edge_types, max_num_nodes=num_atoms
            )
            adj_matrix = adj_matrix.squeeze(0)  # Remove batch dimension

            # Get all entries from the dense adjacency matrix (including no-bonds)
            # Flatten the upper triangle (excluding diagonal) to match triu format
            triu_indices = torch.triu_indices(row=num_atoms, col=num_atoms, offset=1)
            dense_edge_types = adj_matrix[triu_indices[0], triu_indices[1]]

            # Map edge type values (0-4) to edge token indices
            # 0 -> NO_BOND_TOKEN (index 0)
            # 1-4 -> bond type tokens (indices 1-4)
            # We'll handle MASK separately if needed
            # 0-4 directly map to indices 0-4
            edge_token_indices = dense_edge_types.long()

            # Remove center of mass for each molecule
            # (same as in FlowMatchingQM9Dataset)
            coord = data.pos - data.pos.mean(dim=0)
            all_coords.append(coord)

            if hasattr(data, "charges") and data.charges is not None:
                charges = data.charges.tolist()
                charge_type_indices = [
                    token_to_index(self.vocab.charge_tokens, str(token))
                    for token in charges
                ]
                charge_type_indices = torch.tensor(
                    charge_type_indices, dtype=torch.long
                )
                all_charges.append(charge_type_indices)

            all_num_atoms.append(num_atoms)
            all_edge_type_indices.append(edge_token_indices)

        all_num_atoms = torch.tensor(all_num_atoms, dtype=torch.long)
        n_atoms_distribution = all_num_atoms.bincount()
        n_atoms_distribution = n_atoms_distribution / n_atoms_distribution.sum()

        # Concatenate all edge type indices and compute distribution
        all_edge_type_indices = torch.cat(all_edge_type_indices, dim=0)

        # Create distribution over all edge tokens
        # Edge tokens are: [NO_BOND (0), bond types (1-4), MASK (5)]
        num_edge_tokens = len(self.vocab.edge_tokens)
        edge_type_distribution = all_edge_type_indices.bincount(
            minlength=num_edge_tokens
        )
        # Normalize (MASK will have 0 count, which is fine)
        edge_type_distribution = edge_type_distribution.float()
        if edge_type_distribution.sum() > 0:
            edge_type_distribution = (
                edge_type_distribution / edge_type_distribution.sum()
            )
        else:
            edge_type_distribution = torch.ones(num_edge_tokens) / num_edge_tokens

        # Compute charge type distribution
        all_charges = torch.cat(all_charges, dim=0)
        charge_type_indices = all_charges.long()
        charge_type_distribution = charge_type_indices.bincount(
            minlength=len(self.vocab.charge_tokens)
        )
        charge_type_distribution = (
            charge_type_distribution / charge_type_distribution.sum()
        )

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
        }

    def _save_distributions(self, distributions: dict[str, torch.Tensor]):
        """Save distributions to file."""
        os.makedirs(os.path.dirname(self.distributions_path), exist_ok=True)
        torch.save(distributions, self.distributions_path)

    def _load_distributions(self) -> dict[str, torch.Tensor]:
        """Load distributions from file."""
        return torch.load(self.distributions_path, weights_only=False)
