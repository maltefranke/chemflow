import torch
from torch.distributions import Categorical, MixtureSameFamily, Normal, Independent
import torch.nn.functional as F

from chemflow.dataset.molecule_data import MoleculeData
from torch_geometric.utils import remove_self_loops
from chemflow.utils import build_fully_connected_edge_index


def sample_prior_graph(
    atom_type_distribution,
    edge_type_distribution,
    charge_type_distribution,
    n_atoms_distribution,
    n_atoms=None,
):
    p_atom_types = Categorical(probs=atom_type_distribution)
    p_edge_types = Categorical(probs=edge_type_distribution)
    p_charge_types = Categorical(probs=charge_type_distribution)
    p_n_atoms = Categorical(probs=n_atoms_distribution)

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


def sample_births(unmatched_mol_1, t, sigma=1.0):
    num_unmatched_atoms = unmatched_mol_1.num_nodes

    # sample birth times
    birth_times = torch.rand(num_unmatched_atoms, device=unmatched_mol_1.x.device)

    # select birth times that are less than t
    birth_times_mask = birth_times < t
    birth_times = birth_times[birth_times_mask]

    born_nodes_1 = unmatched_mol_1.subgraph(birth_times_mask)
    unborn_nodes_1 = unmatched_mol_1.subgraph(~birth_times_mask)

    return birth_times, born_nodes_1, unborn_nodes_1


def sample_deaths(unmatched_x0, unmatched_a0, t):
    """
    unmatched_x0 (N, D)
    unmatched_a0 (N, C)
    t (float)
    """
    num_unmatched_atoms = unmatched_x0.shape[0]

    # sample death times
    death_times = torch.rand(num_unmatched_atoms, device=unmatched_x0.device)
    is_dead = death_times < t
    x0_alive_at_xt = unmatched_x0[~is_dead]
    a0_alive_at_xt = unmatched_a0[~is_dead]
    return death_times, is_dead, x0_alive_at_xt, a0_alive_at_xt
