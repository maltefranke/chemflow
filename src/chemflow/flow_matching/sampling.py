import torch
from torch.distributions import Categorical, MixtureSameFamily, Normal, Independent


def sample_prior_graph(
    atom_type_distribution, edge_type_distribution, n_atoms_distribution
):
    p_atom_types = Categorical(probs=atom_type_distribution)
    p_edge_types = Categorical(probs=edge_type_distribution)
    p_n_atoms = Categorical(probs=n_atoms_distribution)

    # sample number of atoms from train distribution
    N_atoms = p_n_atoms.sample()

    # sample atom types from train distribution
    atom_types = p_atom_types.sample(sample_shape=(N_atoms,))

    # sample coordinates randomly
    coord = torch.randn(N_atoms, 3)

    # sample edge types for upper triangle (excluding diagonal) of adjacency matrix
    N_edges = (N_atoms**2 - N_atoms) // 2
    triu_edge_types = p_edge_types.sample((N_edges,))
    triu_edge_types = triu_edge_types.to(torch.long)

    """# edge types to triu_matrix
    triu_indices = torch.triu_indices(N_atoms, N_atoms, offset=1)
    edge_types_matrix = torch.zeros(N_atoms, N_atoms, dtype=torch.long)
    edge_types_matrix[triu_indices[0], triu_indices[1]] = edge_types"""

    sampled_graph = {
        "atom_types": atom_types,
        "coord": coord,
        "edge_types": triu_edge_types,
    }
    return sampled_graph


def sample_birth_locations(unmatched_x1, t, sigma=1.0):
    num_unmatched_atoms = unmatched_x1.shape[0]

    # sample birth times
    birth_times = torch.rand(num_unmatched_atoms)

    # select birth times that are less than t
    birth_times_mask = birth_times < t
    birth_times = birth_times[birth_times_mask]
    birth_mu = unmatched_x1[birth_times_mask]

    unborn_x1 = unmatched_x1[~birth_times_mask]

    # sigma has shrinking variance with time
    # intuition: we are more certain about the birth location as time goes on
    sigma = sigma * (1 - t)
    birth_location_t_birth = birth_mu + torch.randn_like(birth_mu) * sigma
    return birth_times, birth_location_t_birth, birth_mu, unborn_x1


def sample_deaths(unmatched_x0, t):
    """
    unmatched_x0 (N, D)
    t (float)
    """
    num_unmatched_atoms = unmatched_x0.shape[0]

    # sample death times
    death_times = torch.rand(num_unmatched_atoms)
    is_dead = death_times < t
    x0_alive_at_xt = unmatched_x0[~is_dead]
    return death_times, is_dead, x0_alive_at_xt
