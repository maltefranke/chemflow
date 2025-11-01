import torch
from torch.nn.utils.rnn import pad_sequence


def sample_different_sizes(
    batch_size, N_mu, N_sigma, x_mu, x_sigma
) -> tuple[torch.Tensor, torch.Tensor]:
    N = (
        torch.randn(
            batch_size,
        )
        * N_sigma
        + N_mu
    )
    N = N.int()

    x = [(torch.randn(n, 1) * x_sigma + x_mu) for n in N]
    x = pad_sequence(x, batch_first=True, padding_value=-1e3)
    x_mask = (x != -1e3).bool()
    return x, x_mask


def sample_gmm_different_sizes(
    batch_size, N_mu, N_sigma
) -> tuple[torch.Tensor, torch.Tensor]:
    N = (
        torch.randn(
            batch_size,
        )
        * N_sigma
        + N_mu
    )
    N = N.int()

    def draw_gmm_sample(n):
        output = torch.empty(n, 1)
        rand = torch.rand(n) < 0.5

        output[rand] = torch.randn(rand.to(torch.int).sum(), 1) * 2 + 7
        output[~rand] = torch.randn((n - rand.to(torch.int).sum()), 1) * 2 - 7
        return output

    x = [draw_gmm_sample(n) for n in N]
    x = pad_sequence(x, batch_first=True, padding_value=-1e3)
    x_mask = (x != -1e3).bool()
    return x, x_mask


def sample_prior(
    batch_size,
    prior_N_mu=100,
    prior_N_sigma=7,
    prior_x_mu=0,
    prior_x_sigma=3,
    target_N_mu=100,
    target_N_sigma=1,
    target_x_mu=0,
    target_x_sigma=1,
):
    # N = torch.randint(5, 25, (batch_size,))
    x0, x0_mask = sample_different_sizes(
        batch_size, prior_N_mu, prior_N_sigma, prior_x_mu, prior_x_sigma
    )

    # M = torch.randint(15, 35, (batch_size,))
    """x1, x1_mask = sample_different_sizes(
        batch_size, target_N_mu, target_N_sigma, target_x_mu, target_x_sigma
    )"""

    x1, x1_mask = sample_gmm_different_sizes(batch_size, target_N_mu, target_N_sigma)
    return x0, x0_mask, x1, x1_mask


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
