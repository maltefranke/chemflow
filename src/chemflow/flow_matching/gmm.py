"""
GMM utilities for birth-death flow matching.

This module contains utilities for working with Gaussian Mixture Models (GMMs)
in the context of birth-death flow matching, including parameter extraction,
GMM creation, sampling, and loss computation.
"""

import torch
from torch.distributions import Categorical, MixtureSameFamily, Normal, Independent


def get_gmm(gmm_params, K=10, D=1):
    """
    Create a GMM distribution from predicted parameters.
    ...
    """
    # ... (Handle single batch case) ...
    B = gmm_params.shape[0]

    # Extract GMM parameters
    # Logits: (B, K) -> (B, 1, K)
    logits = gmm_params[:, :K].unsqueeze(1)

    # Means: (B, K*D) -> (B, K, D) -> (B, 1, K, D)
    means = gmm_params[:, K : K + K * D].reshape(B, K, D).unsqueeze(1)

    # Log-Variances: (B, K*D) -> (B, K, D) -> (B, 1, K, D)
    log_vars = gmm_params[:, K + K * D :].reshape(B, K, D).unsqueeze(1)

    # Calculate sigmas (std devs) from log-variances
    # Shape is now (B, 1, K, D)
    sigmas = torch.sqrt(torch.exp(log_vars) + 1e-6)

    # Create the K component distributions
    # Normal's batch_shape=[B, 1, K, D], event_shape=[]
    # Independent's batch_shape=[B, 1, K], event_shape=[D]
    component_dist = Independent(Normal(means, sigmas), 1)

    # Create the mixture weights
    # Categorical's batch_shape=[B, 1]
    mixture_weights = Categorical(logits=logits)

    # Create the final GMM distribution
    # gmm.batch_shape is now [B, 1]
    # gmm.event_shape is [D]
    gmm = MixtureSameFamily(mixture_weights, component_dist)

    # ... (Squeeze output) ...
    return gmm


def interpolate_gmm(means, t, sigma=1.0, num_samples=100):
    """
    Draws samples from a GMM with N components.

    Args:
        means (torch.Tensor): The means of the N Gaussian components. Shape: (N, D)
        t (torch.Tensor): A tensor containing a single time-step value.
        sigma (float): Base sigma value.
        num_samples (int): The number of samples to draw (M).

    Returns:
        torch.Tensor: Samples from the GMM. Shape: (num_samples, D)
    """
    N, D = means.shape

    # Ensure t is a scalar
    t_scalar = t.item() if t.numel() == 1 else t

    # Calculate the sigma for all components
    sigma_val = sigma * (1 - t_scalar)
    sigmas = sigma_val * torch.ones_like(means)

    # Define mixture weights: N components, equally weighted
    mixture_weights = Categorical(probs=torch.ones(N, device=means.device))

    # Define component distributions
    base_distributions = Normal(means, sigmas)
    component_distributions = Independent(base_distributions, 1)

    # Create the GMM
    gmm = MixtureSameFamily(mixture_weights, component_distributions)

    # Draw samples
    samples = gmm.sample((num_samples,))

    return samples


def get_gmm_parameters_shape(K, D):
    """
    Get the expected shape of GMM parameters.

    Args:
        K (int): Number of GMM components
        D (int): Dimension of the data

    Returns:
        int: Total number of parameters (K + 2 * K * D)
    """
    return K + 2 * K * D


def sample_from_gmm(gmm_params, num_samples, K=10, D=1):
    """
    Sample from a GMM created from predicted parameters.

    Args:
        gmm_params (torch.Tensor): Predicted GMM parameters.
            Shape: (K + 2 * K * D,) for single batch or (B, K + 2 * K * D) for batch
        num_samples (int): Number of samples to draw
        K (int): Number of GMM components
        D (int): Dimension of the data

    Returns:
        torch.Tensor: Samples from the GMM. Shape: (num_samples, D) or (B, num_samples, D)
    """
    gmm = get_gmm(gmm_params, K, D)
    samples = gmm.sample((num_samples,))
    samples = samples.reshape(gmm_params.shape[0], num_samples, D)
    return samples


########################################################
# Typed Mixture Model
# Includes a class label for each component


def get_typed_gmm_components(gmm_params, K=10, D=3, N_types=4):
    """
    Create GMM component distributions from predicted parameters for the
    joint distribution p(m)p(x|m)p(c|m).

    This function is based on the provided GMM-only function and adds
    parameter parsing for discrete types.

    It returns three separate distributions, as MixtureSameFamily
    cannot be used for joint distributions of different types (Normal and Categorical).

    Args:
        gmm_params (torch.Tensor): The raw output from the network.
            Shape: (B, K + 2*K*D + K*N_types)
        K (int): Number of GMM components.
        D (int): Dimension of the data.
        N_types (int): Number of discrete types.

    Returns:
        tuple:
            - mixture_weights (Categorical): Dist for p(m). batch_shape=[B, 1].
            - spatial_dist (Independent): Dist for p(x|m). batch_shape=[B, 1, K].
            - type_dist (Categorical): Dist for p(c|m). batch_shape=[B, 1, K].
    """
    # ... (Handle single batch case) ...
    if gmm_params.dim() == 1:
        # Unsqueeze to add batch dimension if not present
        gmm_params = gmm_params.unsqueeze(0)

    B = gmm_params.shape[0]

    # --- Define parameter boundaries ---
    start_logits = 0
    end_logits = K

    start_means = end_logits
    end_means = start_means + K * D

    start_log_vars = end_means
    end_log_vars = start_log_vars + K * D

    start_type_logits = end_log_vars
    end_type_logits = start_type_logits + K * N_types

    # --- Extract GMM parameters ---

    # 1. Mixture Logits: (B, K) -> (B, 1, K)
    logits = gmm_params[:, start_logits:end_logits].unsqueeze(1)

    # 2. Spatial Means: (B, K*D) -> (B, K, D) -> (B, 1, K, D)
    means = gmm_params[:, start_means:end_means].reshape(B, K, D).unsqueeze(1)

    # 3. Spatial Log-Variances: (B, K*D) -> (B, K, D) -> (B, 1, K, D)
    log_vars = gmm_params[:, start_log_vars:end_log_vars].reshape(B, K, D).unsqueeze(1)

    # 4. Type Logits: (B, K*N_types) -> (B, K, N_types) -> (B, 1, K, N_types)
    type_logits = (
        gmm_params[:, start_type_logits:end_type_logits]
        .reshape(B, K, N_types)
        .unsqueeze(1)
    )

    # --- Create component distributions ---

    # Calculate sigmas (std devs) from log-variances
    # Shape is now (B, 1, K, D)
    sigmas = torch.sqrt(torch.exp(log_vars) + 1e-6)

    # 1. Create the K spatial component distributions
    # Normal's batch_shape=[B, 1, K, D], event_shape=[]
    # Independent's batch_shape=[B, 1, K], event_shape=[D]
    spatial_dist = Independent(Normal(means, sigmas), 1)

    # 2. Create the K type component distributions
    # Categorical's batch_shape=[B, 1, K]
    type_dist = Categorical(logits=type_logits)

    # 3. Create the mixture weights distribution
    # Categorical's batch_shape=[B, 1]
    mixture_weights = Categorical(logits=logits)

    # ... (Squeeze output logic from original function would go here if B==1) ...

    return mixture_weights, spatial_dist, type_dist


def interpolate_typed_gmm(
    p_x_1,
    p_c_0,
    p_c_1,
    t,
    num_samples=20,
    sigma=0.2,
):
    """
    Draws interpolated samples (x_t, c_t) given targets (x_1, c_1).

    This function implements the interpolation path p_t(x, c | x_1, c_1)
    by sampling from a mixture model where:
    1. The spatial components are p_t(x | x_1) = N(x | x_1, (sigma*(1-t))^2)
    2. The type components are p_t(c | c_1) = (1-t)*p_0(c) + t*p_1(c)

    Args:
        means (torch.Tensor): The target spatial locations (x_1).
            Shape: (N, D)
        types (torch.Tensor): The target discrete types (c_1).
            Shape: (N,)
        t (torch.Tensor): A tensor containing a single time-step value.
        N_types (int): The total number of discrete types.
        sigma (float): Base sigma value for spatial noise.
        num_samples (int): The number of samples to draw (M).

    Returns:
        tuple:
            - sampled_locations (torch.Tensor): Shape (num_samples, D)
            - sampled_types (torch.Tensor): Shape (num_samples,)
    """
    N, D = p_x_1.shape

    # Ensure t is a scalar
    t_scalar = t.item() if t.numel() == 1 else t

    # --- 1. Define Mixture Weights (Uniform) ---
    # We will pick one of the N target pairs uniformly to sample from.
    mixture_weights = Categorical(probs=torch.ones(N, device=p_x_1.device))

    # --- 2. Define Spatial Components p(x|m) ---
    # Calculate the time-dependent sigma for all components
    # This is the standard deviation sigma(t)
    sigma_val = sigma * (1 - t_scalar) + 1e-6
    sigmas = sigma_val * torch.ones_like(p_x_1)

    # The N spatial distributions are N(x_1, sigma(t)^2)
    spatial_components = Independent(Normal(p_x_1, sigmas), 1)

    # --- 3. Define Type Components p(c|m) ---
    # Interpolate the probabilities: p_t = (1-t)*p0 + t*p1
    # Shapes: (1, N_types) + (N, N_types) -> (N, N_types)
    p_t_probs = (1 - t_scalar) * p_c_0 + t_scalar * p_c_1

    # Renormalize just in case of float precision issues
    p_t_probs = p_t_probs / torch.sum(p_t_probs, dim=-1, keepdim=True)

    # The N type distributions
    type_components = Categorical(probs=p_t_probs)

    # --- 4. Manually Sample from the Joint Mixture ---
    # We must sample manually to link the spatial and type samples.
    # We cannot use MixtureSameFamily here.

    # a. Sample component indices (which target pair to use)
    # Shape: (num_samples,)
    chosen_indices = mixture_weights.sample((num_samples,))

    # b. Sample locations from the chosen spatial components
    # Gather the means and sigmas for the chosen components
    chosen_means = p_x_1[chosen_indices]  # Shape: (num_samples, D)
    chosen_sigmas = sigmas[chosen_indices]  # Shape: (num_samples, D)

    # Sample locations
    sampled_locations = Normal(chosen_means, chosen_sigmas).sample()

    # c. Sample types from the chosen type components
    # Gather the probability distributions for the chosen components
    chosen_type_probs = p_t_probs[chosen_indices]  # Shape: (num_samples, N_types)

    # Sample types
    sampled_types = Categorical(probs=chosen_type_probs).sample()
    sampled_types = sampled_types.view(-1)

    return sampled_locations, sampled_types


def get_typed_gmm_parameters_shape(K, D, N_types):
    """
    Get the expected shape of typed GMM parameters.

    Args:
        K (int): Number of GMM components.
        D (int): Dimension of the data.
        N_types (int): Number of discrete types.

    Returns:
        int: Total number of parameters (K + 2*K*D + K*N_types)
    """
    # K (mixture weights) + K*D (means) + K*D (log_vars) + K*N_types (type_logits)
    return K + 2 * K * D + K * N_types


def sample_from_typed_gmm(gmm_params, num_samples, K=10, D=1, N_types=4):
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
        N_types (int): Number of discrete types.

    Returns:
        tuple:
            - sampled_locations (torch.Tensor): Shape (num_samples, D) or (B, num_samples, D)
            - sampled_types (torch.Tensor): Shape (num_samples,) or (B, num_samples)
    """

    # --- 0. Handle non-batched input ---
    squeeze_output = False
    if gmm_params.dim() == 1:
        gmm_params = gmm_params.unsqueeze(0)
        squeeze_output = True

    B = gmm_params.shape[0]

    # --- 1. Get GMM Distributions ---
    # mixture_weights: batch_shape=[B, 1]
    # spatial_dist:    batch_shape=[B, 1, K]
    # type_dist:       batch_shape=[B, 1, K]
    mixture_weights, spatial_dist, type_dist = get_typed_gmm_components(
        gmm_params, K, D, N_types
    )

    # --- 2. Sample Component Indices ---
    # mixture_weights.sample() prepends sample_shape to batch_shape
    # Shape: (num_samples, B, 1)
    component_indices = mixture_weights.sample((num_samples,))

    # Reshape to (B, num_samples) for gathering
    component_indices = component_indices.permute(1, 0, 2).squeeze(-1)

    # --- 3. Gather Parameters for Chosen Components ---

    # Get all component parameters, squeezing out the '1' dim
    # (B, 1, K, D) -> (B, K, D)
    means_all = spatial_dist.base_dist.loc.squeeze(1)
    scales_all = spatial_dist.base_dist.scale.squeeze(1)
    # (B, 1, K, N_types) -> (B, K, N_types)
    logits_all = type_dist.logits.squeeze(1)

    # Create expanded indices for gathering
    # (B, num_samples) -> (B, num_samples, 1)
    idx = component_indices.unsqueeze(-1)

    # (B, num_samples, 1) -> (B, num_samples, D)
    idx_locs = idx.expand(-1, -1, D)
    # (B, num_samples, 1) -> (B, num_samples, N_types)
    idx_types = idx.expand(-1, -1, N_types)

    # Gather the parameters for the sampled components
    # We gather from dim 1 (the K dim)
    # torch.gather(input, dim, index)
    chosen_means = torch.gather(means_all, 1, idx_locs)
    chosen_scales = torch.gather(scales_all, 1, idx_locs)
    chosen_logits = torch.gather(logits_all, 1, idx_types)

    # --- 4. Sample from Chosen Components ---
    # Sample locations: x ~ p(x|m*)
    # Shape: (B, num_samples, D)
    sampled_locations = Normal(chosen_means, chosen_scales).sample()

    # Sample types: c ~ p(c|m*)
    # Shape: (B, num_samples)
    sampled_types = Categorical(logits=chosen_logits).sample()

    # --- 5. Handle non-batched output ---
    if squeeze_output:
        sampled_locations = sampled_locations.squeeze(0)
        sampled_types = sampled_types.squeeze(0)

    return sampled_locations, sampled_types


if __name__ == "__main__":
    num_samples = 100
    K = 10
    D = 3
    N_types = 4

    batch_size = 10
    predicted_gmm_params = torch.randn(batch_size, K + 2 * K * D + K * N_types)

    mixture_weights, spatial_dist, type_dist = get_typed_gmm_components(
        predicted_gmm_params, K, D, N_types
    )
    print(mixture_weights.probs.shape)
    print(spatial_dist.base_dist.mean.shape)
    print(type_dist.probs.shape)

    means = torch.randn(K, D)
    types = torch.randint(0, N_types, (K,))
    t = torch.tensor(0.5)
    sigma = 1.0
    sampled_locations, sampled_types = interpolate_typed_gmm(
        means, types, t, N_types, sigma, num_samples
    )
    print(sampled_locations.shape)
    print(sampled_types.shape)

    target_locations = torch.randn(batch_size, num_samples, D)
    target_types = torch.randint(0, N_types, (batch_size, num_samples))
    loss = typed_gmm_loss(
        predicted_gmm_params, target_locations, target_types, K, D, N_types
    )
    print(f"Loss: {loss.item()}")

    n_params = get_typed_gmm_parameters_shape(K, D, N_types)
    print(f"Number of parameters: {n_params}")

    sampled_locations, sampled_types = sample_from_typed_gmm(
        predicted_gmm_params, num_samples, K, D, N_types
    )
    print(sampled_locations.shape)
    print(sampled_types.shape)
