from torch_geometric.datasets import QM9
import os
import torch
from torch_geometric.utils import to_dense_adj
from chemflow.flow_matching.sampling import sample_prior_graph
from chemflow.utils import edge_types_to_triu_entries


class FlowMatchingQM9Dataset(QM9):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)

        distributions = self.get_distributions()
        self.atom_type_distribution = distributions["atom_type_distribution"]
        self.edge_type_distribution = distributions["edge_type_distribution"]
        self.n_atoms_distribution = distributions["n_atoms_distribution"]

    def get_distributions(self):
        distributions_file = os.path.join(self.processed_dir, "distributions.pt")
        if os.path.exists(distributions_file):
            return torch.load(distributions_file)
        else:
            distributions = {}
            atom_types = self.z.unique().sort()[0]
            # add a NONE-ATOM at 0
            atom_types = torch.cat([torch.tensor([0]), atom_types])
            atom_type_distribution = (self.z.unsqueeze(1) == atom_types).sum(dim=0)
            atom_type_distribution = (
                atom_type_distribution / atom_type_distribution.sum()
            )
            distributions["atom_type_distribution"] = atom_type_distribution

            all_num_atoms = []
            all_edge_types = []

            for i in range(len(self)):
                data = super(FlowMatchingQM9Dataset, self).__getitem__(i)

                num_atoms = data.num_nodes

                triu_edge_types = edge_types_to_triu_entries(
                    data.edge_index, data.edge_attr, num_atoms
                )

                all_num_atoms.append(num_atoms)
                all_edge_types.append(triu_edge_types)

            all_num_atoms = torch.tensor(all_num_atoms, dtype=torch.long)
            n_atoms_distribution = all_num_atoms.bincount()
            n_atoms_distribution = n_atoms_distribution / n_atoms_distribution.sum()
            distributions["n_atoms_distribution"] = n_atoms_distribution

            all_edge_types = torch.cat(all_edge_types, dim=0)
            edge_type_distribution = all_edge_types.bincount()
            edge_type_distribution = (
                edge_type_distribution / edge_type_distribution.sum()
            )
            distributions["edge_type_distribution"] = edge_type_distribution

            torch.save(distributions, distributions_file)
            return distributions

    def __getitem__(self, index):
        sample = sample_prior_graph(
            self.atom_type_distribution,
            self.edge_type_distribution,
            self.n_atoms_distribution,
        )

        data = super().__getitem__(index)

        # remove center of mass
        coord = data.pos - data.pos.mean(dim=1)
        atom_types = data.z
        edge_types = edge_types_to_triu_entries(
            data.edge_index, data.edge_attr, data.num_nodes
        )
        target = {
            "coord": coord,
            "atom_types": atom_types,
            "edge_types": edge_types,
        }
        return sample, target


if __name__ == "__main__":
    qm9 = FlowMatchingQM9Dataset(
        root="/cluster/project/krause/frankem/chemflow/data/qm9"
    )
    print(qm9[0])
