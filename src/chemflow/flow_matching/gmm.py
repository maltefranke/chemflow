"""
GMM utilities for birth-death flow matching.

This module contains utilities for working with Gaussian Mixture Models (GMMs)
in the context of birth-death flow matching, including parameter extraction,
GMM creation, sampling, and loss computation.
"""

import torch
from torch.distributions import Categorical, MixtureSameFamily, Normal, Independent

import torch.nn.functional as F


def get_typed_gmm_components(gmm_output):
    """
    Create GMM component distributions from the pre-computed equivariant output dictionary.

    Args:
        gmm_output (dict): The dictionary returned by compute_equivariant_gmm containing:
            - 'mu': [B, K, D] (Equivariant Means)
            - 'sigma': [B, K] (Invariant Isotropic Scales)
            - 'pi': [B, K] (Invariant Mixture Weights)
            - 'types': [B, K, N_types] (Invariant Type Probs)
        D (int): Dimension of the spatial data (usually 3).

    Returns:
        tuple:
            - mixture_weights (Categorical): Batch shape [B, 1]. Event shape [].
            - spatial_dist (Independent): Batch shape [B, 1, K]. Event shape [D].
            - type_dist (Categorical): Batch shape [B, 1, K]. Event shape [].
    """

    # 1. Unpack and handle single-batch case (if necessary)
    mu = gmm_output["mu"]  # [B, K, D]
    sigma = gmm_output["sigma"]  # [B, K]
    pi = gmm_output["pi"]  # [B, K]
    a_probs = gmm_output["a_probs"]  # [B, K, T]
    c_probs = gmm_output["c_probs"]  # [B, K, T]

    # If the input was unbatched (K, D), add batch dim to make it (1, K, D)
    if mu.dim() == 2:
        mu = mu.unsqueeze(0)
        sigma = sigma.unsqueeze(0)
        pi = pi.unsqueeze(0)
        a_probs = a_probs.unsqueeze(0)
        c_probs = c_probs.unsqueeze(0)

    # 2. Reshape for Broadcasting: (B, ...) -> (B, 1, ...)
    # This prepares the distributions to be evaluated against N_samples
    # (broadcasting B, 1, K against B, N, K)

    # --- Mixture Weights ---
    # Shape: [B, K] -> [B, 1, K]
    # We use 'probs' because compute_equivariant_gmm already applied softmax
    mixture_weights = Categorical(probs=pi.unsqueeze(1))

    # --- Spatial Distribution ---
    # Means: [B, K, D] -> [B, 1, K, D]
    mu_expanded = mu.unsqueeze(1)

    # Sigmas: [B, K] -> [B, 1, K, 1]
    # Note: We append a trailing 1 so it broadcasts against the D dimension of Mu.
    # Since sigma is isotropic, it applies to x, y, and z equally.
    sigma_expanded = sigma.unsqueeze(1).unsqueeze(-1)

    # Create Normal [B, 1, K, D] -> Wrap in Independent to sum log-probs over D
    x_dist = Independent(Normal(mu_expanded, sigma_expanded), 1)

    # --- Type Distribution ---
    # Shape: [B, K, T] -> [B, 1, K, T]
    a_dist = Categorical(probs=a_probs.unsqueeze(1))
    c_dist = Categorical(probs=c_probs.unsqueeze(1))

    return mixture_weights, x_dist, a_dist, c_dist


def sample_from_typed_gmm(
    gmm_params, num_samples, K=10, D=1, N_a=4, N_c=4, squeeze_output=False
):
    """
    Sample from the joint GMM created from predicted parameters.

    This function manually implements the 3-step sampling process:
    1. Sample component indices m ~ p(m)
    2. Gather parameters for p(x|m) and p(c|m)
    3. Sample types c ~ p(c|m) and locations x ~ p(x|m)

    Args:
        gmm_params (torch.Tensor): Predicted GMM parameters.
            Shape: (K + 2*K*D + K*N_types,) for single batch or
                   (B, K + 2*K*D + K*N_types) for batch.
        num_samples (int): Number of samples to draw.
        K (int): Number of GMM components.
        D (int): Dimension of the data.
        N_a (int): Number of discrete atom types.
        N_c (int): Number of discrete charge types.

    Returns:
        tuple:
            - sampled_locations (torch.Tensor): Shape (num_samples, D) or (B, num_samples, D)
            - sampled_types (torch.Tensor): Shape (num_samples,) or (B, num_samples)
    """
    # --- 1. Get GMM Distributions ---
    # mixture_weights: batch_shape=[B, 1]
    # spatial_dist:    batch_shape=[B, 1, K]
    # type_dist:       batch_shape=[B, 1, K]
    mixture_weights, x_dist, a_dist, c_dist = get_typed_gmm_components(gmm_params)

    # --- 2. Sample Component Indices ---
    # mixture_weights.sample() prepends sample_shape to batch_shape
    # Shape: (num_samples, B, 1)
    component_indices = mixture_weights.sample((num_samples,))

    # Reshape to (B, num_samples) for gathering
    component_indices = component_indices.permute(1, 0, 2).squeeze(-1)

    # --- 3. Gather Parameters for Chosen Components ---

    # Get all component parameters, squeezing out the '1' dim
    # (B, 1, K, D) -> (B, K, D)
    means_all = x_dist.base_dist.loc.squeeze(1)
    scales_all = x_dist.base_dist.scale.squeeze(1)
    # (B, 1, K, N_types) -> (B, K, N_types)
    logits_all_a = a_dist.logits.squeeze(1)
    logits_all_c = c_dist.logits.squeeze(1)

    # Create expanded indices for gathering
    # (B, num_samples) -> (B, num_samples, 1)
    idx = component_indices.unsqueeze(-1)

    # (B, num_samples, 1) -> (B, num_samples, D)
    idx_locs = idx.expand(-1, -1, D)
    # (B, num_samples, 1) -> (B, num_samples, N_types)
    idx_a = idx.expand(-1, -1, N_a)
    idx_c = idx.expand(-1, -1, N_c)

    # Gather the parameters for the sampled components
    # We gather from dim 1 (the K dim)
    # torch.gather(input, dim, index)
    chosen_means = torch.gather(means_all, 1, idx_locs)
    chosen_scales = torch.gather(scales_all, 1, idx_locs)
    chosen_logits_a = torch.gather(logits_all_a, 1, idx_a)
    chosen_logits_c = torch.gather(logits_all_c, 1, idx_c)

    # --- 4. Sample from Chosen Components ---
    # Sample locations: x ~ p(x|m*)
    # Shape: (B, num_samples, D)
    sampled_x = Normal(chosen_means, chosen_scales).sample()

    # Sample types: c ~ p(c|m*)
    # Shape: (B, num_samples)
    sampled_a = Categorical(logits=chosen_logits_a).sample()
    sampled_c = Categorical(logits=chosen_logits_c).sample()

    # --- 5. Handle non-batched output ---
    if squeeze_output:
        sampled_x = sampled_x.squeeze(0)
        sampled_a = sampled_a.squeeze(0)
        sampled_c = sampled_c.squeeze(0)

    return sampled_x, sampled_a, sampled_c
