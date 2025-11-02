import torch
from torch.utils.data import Dataset
import os
import pandas as pd
from chemflow.flow_matching.sampling import sample_prior_graph


def custom_preprocess(path, preprocess_workers):
    # TODO place your custom preprocess logic here

    ############## EXAMPLE CODE ##############
    # Create multiple graphs with different sizes
    num_graphs = 5
    data_list = []

    for graph_idx in range(num_graphs):
        # Random number of nodes for each graph
        num_nodes = torch.randint(10, 50, (1,)).item()

        # node feats (required)
        atom_feats = torch.randn(num_nodes, 10)

        # positions (required)
        coord = torch.randn(num_nodes, 3)

        # edge index (required) - create some random edges
        num_edges = torch.randint(num_nodes, num_nodes * 2, (1,)).item()
        edge_index = torch.randint(0, num_nodes, (2, num_edges))

        # Create data dictionary with required fields
        data_dict = {
            "atom_feats": atom_feats,
            "coord": coord,
            "edge_index": edge_index,
        }

        # Add optional edge attributes (set to None to simulate missing data)
        edge_attr = torch.randn(num_edges, 5)
        data_dict["edge_attr"] = edge_attr
        # else: edge_attr is None (not included in dict)

        # Add optional graph attributes (set to None to simulate missing data)
        graph_attr = torch.randn(8)
        data_dict["graph_attr"] = graph_attr
        # else: graph_attr is None (not included in dict)

        data_list.append(data_dict)

    return data_list


class CustomGraphDataset(Dataset):
    def __init__(self, path, save_path, preprocess_workers):
        self.path = path
        self.save_path = save_path
        self.preprocess_workers = preprocess_workers

        # will be set in preprocess
        self.cached_data = None
        self.atom_type_distribution = None
        self.edge_type_distribution = None
        self.n_atoms_distribution = None

        self.preprocess(save_path, preprocess_workers)

    def preprocess(self, save_path, preprocess_workers):
        if os.path.exists(save_path):
            data, distributions = torch.load(save_path, weights_only=False)

            self.cached_data = data
            self.atom_type_distribution = distributions["atom_type_distribution"]
            self.edge_type_distribution = distributions["edge_type_distribution"]
            self.n_atoms_distribution = distributions["n_atoms_distribution"]
        else:
            data, distributions = custom_preprocess(
                self.path,
                preprocess_workers,
            )

            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save((data, distributions), save_path)

    def __len__(self):
        return len(self.cached_data)

    def __getitem__(self, index):
        return self.cached_data[index]


class FlowMatchingGraphDataset(Dataset):
    def __getitem__(self, index):
        sample = sample_prior_graph(
            self.atom_type_distribution,
            self.edge_type_distribution,
            self.n_atoms_distribution,
        )
        target = self.cached_data[index]

        return sample, target


if __name__ == "__main__":
    data = custom_preprocess(None, None)
    print(data[0].keys())
