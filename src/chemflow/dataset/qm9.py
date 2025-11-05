from torch_geometric.datasets import QM9
import torch

from chemflow.utils import (
    edge_types_to_triu_entries,
    z_to_atom_types,
    token_to_index,
)


class FlowMatchingQM9Dataset(QM9):
    def __init__(
        self,
        root,
        tokens: list[str],
        transform=None,
        pre_transform=None,
    ):
        super().__init__(root, transform, pre_transform)

        self.tokens = tokens

    def __getitem__(self, index):
        data = super().__getitem__(index)

        # remove center of mass
        coord = data.pos - data.pos.mean(dim=0)

        atom_types = data.z
        atom_types = z_to_atom_types(atom_types.tolist())
        atom_types = [token_to_index(self.tokens, token) for token in atom_types]
        atom_types = torch.tensor(atom_types, dtype=torch.long)

        edge_types = edge_types_to_triu_entries(
            data.edge_index, data.edge_attr, data.num_nodes
        )
        target = {
            "coord": coord,
            "atom_types": atom_types,
            "edge_types": edge_types,
        }
        return target


if __name__ == "__main__":
    qm9 = FlowMatchingQM9Dataset(
        root="/cluster/project/krause/frankem/chemflow/data/qm9"
    )
    print(qm9[0])
