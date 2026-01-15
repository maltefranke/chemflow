import torch
from torch.distributions import Categorical

from chemflow.dataset.molecule_data import MoleculeData
from chemflow.utils import build_fully_connected_edge_index
from chemflow.dataset.vocab import Distributions, Vocab


def sample_prior_graph(
    distributions: Distributions,
    n_atoms=None,
):
    p_atom_types = Categorical(probs=distributions.atom_type_distribution)
    p_edge_types = Categorical(probs=distributions.edge_type_distribution)
    p_charge_types = Categorical(probs=distributions.charge_type_distribution)
    p_n_atoms = Categorical(probs=distributions.n_atoms_distribution)

    # sample number of atoms from train distribution
    if n_atoms:
        N_atoms = n_atoms
    else:
        N_atoms = p_n_atoms.sample()

    # sample atom types from train distribution
    atom_types = p_atom_types.sample(sample_shape=(N_atoms,))
    atom_types = atom_types.to(torch.long)

    # sample charge types from train distribution
    charge_types = p_charge_types.sample(sample_shape=(N_atoms,))
    charge_types = charge_types.to(torch.long)

    # sample coordinates randomly, and make sure to center the coordinates
    coord = torch.randn(N_atoms, 3)
    coord = coord - coord.mean(dim=0)

    # instantiate fully connected
    edge_index = build_fully_connected_edge_index(N_atoms)

    # sample edge types for upper triangle (excluding diagonal) of adjacency matrix
    N_edges = (N_atoms**2 - N_atoms) // 2
    triu_edge_types = p_edge_types.sample(sample_shape=(N_edges,))
    triu_edge_types = triu_edge_types.to(torch.long)

    # edge types to triu_matrix
    triu_indices = torch.triu_indices(N_atoms, N_atoms, offset=1)
    edge_types_matrix = torch.zeros(N_atoms, N_atoms, dtype=torch.long)
    edge_types_matrix[triu_indices[0], triu_indices[1]] = triu_edge_types

    edge_types_symmetric = edge_types_matrix + edge_types_matrix.T

    edge_types = edge_types_symmetric[edge_index[0], edge_index[1]]
    edge_types = edge_types.to(torch.long)

    sampled_graph = MoleculeData(
        x=coord,
        a=atom_types,
        e=edge_types,
        c=charge_types,
        edge_index=edge_index,
    )
    return sampled_graph


class Sampler:
    def __init__(self, vocab: Vocab, distributions: Distributions):
        self.vocab = vocab
        self.distributions = distributions

    def sample_de_novo():
        pass

    def sample_conformer(self, graph):
        pass

    def sample(self, batch):
        all_sampled_graphs = []
        for batch_i in batch:
            task = batch_i.task

            if task == "de_novo":
                sampled_graph = self.sample_de_novo()
            elif task == "conformer":
                sampled_graph = self.sample_conformer(batch_i)
            else:
                raise ValueError(f"Invalid task: {task}")

            all_sampled_graphs.append(sampled_graph)

        return all_sampled_graphs
