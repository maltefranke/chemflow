from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class Vocab:
    atom_tokens: list[str]
    edge_tokens: list[str]
    charge_tokens: list[str]


@dataclass
class Distributions:
    atom_type_distribution: torch.Tensor
    edge_type_distribution: torch.Tensor
    charge_type_distribution: torch.Tensor
    n_atoms_distribution: torch.Tensor
    coordinate_std: torch.Tensor

    def _to_primitive_types(self):
        return self(
            atom_type_distribution=self.atom_type_distribution.tolist(),
            edge_type_distribution=self.edge_type_distribution.tolist(),
            charge_type_distribution=self.charge_type_distribution.tolist(),
            n_atoms_distribution=self.n_atoms_distribution.tolist(),
            coordinate_std=self.coordinate_std.item(),
        )

    def _to_tensors(self):
        return self(
            atom_type_distribution=torch.tensor(self.atom_type_distribution),
            edge_type_distribution=torch.tensor(self.edge_type_distribution),
            charge_type_distribution=torch.tensor(self.charge_type_distribution),
            n_atoms_distribution=torch.tensor(self.n_atoms_distribution),
            coordinate_std=torch.tensor(self.coordinate_std),
        )
