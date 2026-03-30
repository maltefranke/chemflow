from dataclasses import dataclass
from typing import Any

import torch

from chemflow.utils.utils import compute_token_weights


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


def setup_token_weights(
    vocab: Vocab,
    distributions: Distributions,
    weight_alpha: float,
    type_loss_token_weights: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute atom, edge, and charge token weights from training distributions.

    Returns:
        (atom_type_weights, edge_token_weights, charge_token_weights)
    """
    atom_type_weights = compute_token_weights(
        token_list=vocab.atom_tokens,
        distribution=distributions.atom_type_distribution,
        special_token_names=[],
        weight_alpha=weight_alpha,
        type_loss_token_weights=type_loss_token_weights,
    )
    edge_token_weights = compute_token_weights(
        token_list=vocab.edge_tokens,
        distribution=distributions.edge_type_distribution,
        special_token_names=["<NO_BOND>"],
        weight_alpha=weight_alpha,
        type_loss_token_weights=type_loss_token_weights,
    )
    charge_token_weights = compute_token_weights(
        token_list=vocab.charge_tokens,
        distribution=distributions.charge_type_distribution,
        special_token_names=[],
        weight_alpha=weight_alpha,
        type_loss_token_weights=type_loss_token_weights,
    )
    return atom_type_weights, edge_token_weights, charge_token_weights
