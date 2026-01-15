"""
GMM utilities for birth-death flow matching.

This module contains utilities for working with Gaussian Mixture Models (GMMs)
in the context of birth-death flow matching, including parameter extraction,
GMM creation, sampling, and loss computation.
"""

import torch
from torch.distributions import Categorical, MixtureSameFamily, Normal, Independent

from chemflow.utils import segment_softmax
from external_code.egnn import unsorted_segment_mean
import torch.nn.functional as F


def get_gmm(gmm_output, D=3):
    """
    Create a standard GMM distribution from the equivariant output dictionary.

    Args:
        gmm_output (dict): Contains 'mu' [B, K, D], 'sigma' [B, K], 'pi' [B, K]
        D (int): Dimension of data.

    Returns:
        MixtureSameFamily:
            batch_shape=[B, 1]
            event_shape=[D]
    """
    mu = gmm_output["mu"]  # [B, K, D]
    sigma = gmm_output["sigma"]  # [B, K]
    pi = gmm_output["pi"]  # [B, K]

    # Handle single batch case (unbatched inputs)
    if mu.dim() == 2:
        mu = mu.unsqueeze(0)
        sigma = sigma.unsqueeze(0)
        pi = pi.unsqueeze(0)

    # --- 1. Prepare Mixture Weights ---
    # Shape: [B, K] -> [B, 1, K]
    # This defines the probability of selecting component k for the single sample in dim 1
    mix_dist = Categorical(probs=pi.unsqueeze(1))

    # --- 2. Prepare Components ---
    # Means: [B, K, D] -> [B, 1, K, D]
    mu_expanded = mu.unsqueeze(1)

    """# Sigmas: [B, K] -> [B, 1, K] -> [B, 1, K, D] (Isotropic expansion)
    sigma_expanded = sigma.unsqueeze(1).unsqueeze(-1).expand(-1, -1, D)"""
    # Sigmas: [B, K] -> [B, 1, K] -> [B, 1, K, 1] -> [B, 1, K, D] (Isotropic expansion)
    sigma_expanded = sigma.unsqueeze(1).unsqueeze(-1).expand(-1, -1, -1, D)

    # Define Component Distribution
    # Batch Shape: [B, 1, K] | Event Shape: [D]
    comp_dist = Independent(Normal(mu_expanded, sigma_expanded), 1)

    # --- 3. Combine into GMM ---
    # Resulting Batch Shape: [B, 1] | Event Shape: [D]
    gmm = MixtureSameFamily(mix_dist, comp_dist)

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
        gmm_params (dict): Predicted GMM parameters.
            Shape: {
                "mu": [B, K, D],
                "sigma": [B, K],
                "pi": [B, K],
            }
        num_samples (int): Number of samples to draw
        K (int): Number of GMM components
        D (int): Dimension of the data

    Returns:
        torch.Tensor: Samples from the GMM. Shape: (num_samples, D) or (B, num_samples, D)
    """
    gmm = get_gmm(gmm_params, D)
    samples = gmm.sample((num_samples,))
    samples = samples.reshape(gmm_params["mu"].shape[0], num_samples, D)
    return samples


########################################################
# Typed Mixture Model
# Includes a class label for each component


"""def get_typed_gmm_components(gmm_params, K=10, D=3, N_types=4):
    
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

    return mixture_weights, spatial_dist, type_dist"""


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


def compute_equivariant_gmm_old(gmm_pred, xt, batch_id, K, num_tokens=0):
    """
    Computes GMM parameters from EGNN output.

    Args:
        gmm_pred: [N, Output_Dim] Raw output from EGNN head.
        xt:       [N, 3]          Equivariant node coordinates.
        batch_id: [N]             Graph assignment index.
        K:        int             Number of Gaussians.
        D:        int             Spatial dimension (usually 3).
        num_tokens: int (Optional) Number of discrete types.
                                   If 0, types are skipped.
    """
    num_graphs = batch_id.max().item() + 1

    # --- 1. Slice the Raw Predictions ---
    cursor = 0

    # A. Mixture Logits (K)
    raw_mix = gmm_pred[:, cursor : cursor + K]
    cursor += K

    # B. Position Attention Logits (K)
    # Used to calculate the weighted center of mass
    raw_pos_attn = gmm_pred[:, cursor : cursor + K]
    cursor += K

    # C. Variances (K)
    # Isotropic scalar variances
    raw_vars = gmm_pred[:, cursor : cursor + K]
    cursor += K

    # D. Types (K * T) - Conditional
    raw_types = None
    if num_tokens is not None and num_tokens > 0:
        raw_types = gmm_pred[:, cursor : cursor + (K * num_tokens)]
        # cursor += (K * num_tokens) # Update if there were more fields after types

    # --- 2. Compute Invariant Properties (Pool over nodes) ---

    # Global Mixture Weights
    mix_logits = unsorted_segment_mean(raw_mix, batch_id, num_graphs)
    pi = F.softmax(mix_logits, dim=-1)  # Shape: [B, K]

    # Global Variances
    # Softplus ensures positivity.
    vars_pooled = unsorted_segment_mean(raw_vars, batch_id, num_graphs)
    sigma = F.softplus(vars_pooled)  # Shape: [B, K]

    # Global Types (Optional)
    type_probs = None
    if raw_types is not None:
        types_pooled = unsorted_segment_mean(raw_types, batch_id, num_graphs)
        # Reshape to [B, K, T] and softmax over types
        type_probs = F.softmax(types_pooled.view(num_graphs, K, num_tokens), dim=-1)

    # --- 3. Compute Equivariant Means (Weighted Center of Mass) ---

    # Calculate attention weights (Softmax over nodes within each graph)
    # alpha shape: [N, K]
    # Note: Requires the segment_softmax helper defined previously
    alpha = segment_softmax(raw_pos_attn, batch_id, num_graphs)

    # Weighted Sum of Coordinates
    # xt shape: [N, 3] -> [N, 1, 3]
    # alpha shape: [N, K] -> [N, K, 1]
    weighted_pos = xt.unsqueeze(1) * alpha.unsqueeze(-1)  # Shape: [N, K, 3]

    # Sum over nodes belonging to the same graph
    weighted_pos_flat = weighted_pos.view(weighted_pos.size(0), -1)

    means_flat = torch.zeros(num_graphs, K * 3, device=xt.device)
    means_flat.index_add_(0, batch_id, weighted_pos_flat)

    mu = means_flat.view(num_graphs, K, 3)  # Shape: [B, K, 3]

    # --- 4. Return Dictionary ---
    output = {
        "mu": mu,  # Equivariant [B, K, 3]
        "sigma": sigma,  # Invariant [B, K]
        "pi": pi,  # Invariant [B, K]
    }

    if type_probs is not None:
        output["types"] = type_probs  # Invariant [B, K, T]

    return output


def compute_equivariant_gmm(gmm_pred, xt, K, D, N_a, N_c):
    """
    Decodes GMM parameters using the Hybrid (2K) approach.

    Args:
        gmm_pred: [N, Output_Dim] Raw logits from GMMHead.
        xt:       [N, 3]          Equivariant node coordinates.
        K:        int             Number of Gaussians.
        N_a:      int           Number of discrete atom types.
        N_c:      int           Number of discrete charge types.

    Returns:
        dict: {
            "pi":    [B, K],    # Invariant Mixing Weights
            "mu":    [B, K, 3], # Equivariant Means
            "sigma": [B, K],    # Invariant Variances (Isotropic)
            "a_probs": [B, K, N_a],  # (Optional) Invariant Atom Type Probs
            "c_probs": [B, K, N_c],  # (Optional) Invariant Charge Type Probs
        }
    """
    N = xt.shape[0]

    # --- 1. Slice Inputs ---
    # First K: Mixture Logits (K)
    pi = gmm_pred[:, :K]
    pi = F.softmax(pi, dim=-1)

    # Next K: Variances
    var_logits = gmm_pred[:, K : 2 * K]
    # Enforce positivity on raw logits
    sigma_per_node = F.softplus(var_logits)  # [N, K]

    # Next K*D: Handle the spatial mu
    pred_mu = gmm_pred[:, 2 * K : K * (2 + D)]
    pred_mu = pred_mu.view(N, K, D)
    mu_per_node = xt.unsqueeze(1) + pred_mu

    # Handle the types
    # Next K*N_a: Atom types
    raw_a_probs = gmm_pred[:, K * (2 + D) : K * (2 + D + N_a)]
    # Next K*N_c: Charge types
    raw_c_probs = gmm_pred[:, K * (2 + D + N_a) : K * (2 + D + N_a + N_c)]

    # Reshape: [N, K, T]
    a_probs_per_node = raw_a_probs.view(N, K, N_a)
    c_probs_per_node = raw_c_probs.view(N, K, N_c)

    # Softmax over types -> probs per node
    node_a_probs = F.softmax(a_probs_per_node, dim=-1)
    node_c_probs = F.softmax(c_probs_per_node, dim=-1)

    # Return the output
    output = {
        "pi": pi,
        "mu": mu_per_node,
        "sigma": sigma_per_node,
        "a_probs": node_a_probs,
        "c_probs": node_c_probs,
    }

    return output


def interpolate_typed_gmm(
    mu1,
    p_a0,
    p_a1,
    p_c0,
    p_c1,
    t,
    num_samples=20,
    sigma=1.0,
):
    """
    Draws interpolated samples (x_t, c_t) given targets (x_1, c_1).

    This function implements the interpolation path p_t(x, c | x_1, c_1)
    by sampling from a mixture model where:
    1. The spatial components are p_t(x | x_1) = N(x | x_1, (sigma*(1-t))^2)
    2. The type components are p_t(c | c_1) = (1-t)*p_0(c) + t*p_1(c)

    Args:
        mu1 (torch.Tensor): The target spatial locations (x_1).
            Shape: (N, D)
        p_a0 (torch.Tensor): The empirical atom types distribution.
            Shape: (N, N_types)
        p_a1 (torch.Tensor): The target atom types (a_1).
            Shape: (N, N_types)
        p_c0 (torch.Tensor): The empirical charge types distribution.
            Shape: (N, N_types)
        p_c1 (torch.Tensor): The target charge types (c_1).
            Shape: (N, N_types)
        t (torch.Tensor): A tensor containing a single time-step value.
        num_samples (int): The number of samples to draw (M).
        sigma (float): Base sigma value for spatial noise.

    Returns:
        tuple:
            - sampled_locations (torch.Tensor): Shape (num_samples, D)
            - sampled_types (torch.Tensor): Shape (num_samples,)
    """
    N, D = mu1.shape

    # Ensure t is a scalar
    t_scalar = t.item() if t.numel() == 1 else t

    # --- 1. Define Mixture Weights (Uniform) ---
    # We will pick one of the N target pairs uniformly to sample from.
    mixture_weights = Categorical(probs=torch.ones(N, device=mu1.device))

    # --- 2. Define Spatial Components p(x|m) ---
    # Calculate the time-dependent sigma for all components
    # This is the standard deviation sigma(t)
    sigma_val = sigma * (1 - t_scalar) + 1e-6
    sigmas = sigma_val * torch.ones_like(mu1)

    # The N spatial distributions are N(x_1, sigma(t)^2)
    spatial_components = Independent(Normal(mu1, sigmas), 1)

    # --- 3. Define Type Components p(c|m) ---
    # Interpolate the probabilities: p_t = (1-t)*p0 + t*p1
    # Shapes: (1, N_types) + (N, N_types) -> (N, N_types)
    p_at_probs = (1 - t_scalar) * p_a0 + t_scalar * p_a1
    p_ct_probs = (1 - t_scalar) * p_c0 + t_scalar * p_c1

    # Renormalize just in case of float precision issues
    p_at_probs = p_at_probs / torch.sum(p_at_probs, dim=-1, keepdim=True)
    p_ct_probs = p_ct_probs / torch.sum(p_ct_probs, dim=-1, keepdim=True)

    # The N type distributions
    a_type_components = Categorical(probs=p_at_probs)
    c_type_components = Categorical(probs=p_ct_probs)

    # --- 4. Manually Sample from the Joint Mixture ---
    # We must sample manually to link the spatial and type samples.
    # We cannot use MixtureSameFamily here.

    # a. Sample component indices (which target pair to use)
    # Shape: (num_samples,)
    chosen_indices = mixture_weights.sample((num_samples,))

    # b. Sample locations from the chosen spatial components
    # Gather the means and sigmas for the chosen components
    chosen_means = mu1[chosen_indices]  # Shape: (num_samples, D)
    chosen_sigmas = sigmas[chosen_indices]  # Shape: (num_samples, D)

    # Sample locations
    sampled_locations = Normal(chosen_means, chosen_sigmas).sample()

    # c. Sample types from the chosen type components
    # Gather the probability distributions for the chosen components
    chosen_at_probs = p_at_probs[chosen_indices]  # Shape: (num_samples, N_types)
    chosen_ct_probs = p_ct_probs[chosen_indices]  # Shape: (num_samples, N_types)

    # Sample types
    sampled_at = Categorical(probs=chosen_at_probs).sample()
    sampled_ct = Categorical(probs=chosen_ct_probs).sample()

    sampled_at = sampled_at.view(-1)
    sampled_ct = sampled_ct.view(-1)

    return sampled_locations, sampled_at, sampled_ct, chosen_indices


def get_typed_gmm_parameters_shape(K, D, N_a, N_c):
    """
    Get the expected shape of typed GMM parameters.

    Args:
        K (int): Number of GMM components.
        D (int): Dimension of the data.
        N_a (int): Number of discrete atom types.
        N_c (int): Number of discrete charge types.

    Returns:
        int: Total number of parameters (K + 2*K*D + K*N_a + K*N_c)
    """
    # K (mixture weights) + K*D (means) + K*D (log_vars) + K*N_types (type_logits)
    return K + 2 * K * D + K * N_a + K * N_c


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
