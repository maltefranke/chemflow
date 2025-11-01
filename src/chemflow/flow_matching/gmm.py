"""
GMM utilities for birth-death flow matching.

This module contains utilities for working with Gaussian Mixture Models (GMMs)
in the context of birth-death flow matching, including parameter extraction,
GMM creation, sampling, and loss computation.
"""

import torch
from torch.distributions import Categorical, MixtureSameFamily, Normal, Independent


def get_gmm(predicted_gmm_params, K=10, D=1):
    """
    Create a GMM distribution from predicted parameters.
    ...
    """
    # ... (Handle single batch case) ...
    B = predicted_gmm_params.shape[0]

    # Extract GMM parameters
    # Logits: (B, K) -> (B, 1, K)
    logits = predicted_gmm_params[:, :K].unsqueeze(1)

    # Means: (B, K*D) -> (B, K, D) -> (B, 1, K, D)
    means = predicted_gmm_params[:, K : K + K * D].reshape(B, K, D).unsqueeze(1)

    # Log-Variances: (B, K*D) -> (B, K, D) -> (B, 1, K, D)
    log_vars = predicted_gmm_params[:, K + K * D :].reshape(B, K, D).unsqueeze(1)

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


def sample_from_gmm(means, t, sigma=1.0, num_samples=100):
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


def gmm_loss(predicted_gmm_params, target, K=10, D=1):
    """
    Computes the NLL loss for GMM predictions.

    Args:
        predicted_gmm_params (torch.Tensor): The raw output from the network.
            Shape: (B, K + 2 * K * D)
        target (torch.Tensor): The target data.
            Shape: (B, N_samples, D) where N_samples is the number of samples drawn from GMM
        K (int): Number of GMM components
        D (int): Dimension of the data

    Returns:
        torch.Tensor: The negative log-likelihood loss
    """
    B = predicted_gmm_params.shape[0]

    # Handle case where target might be empty or have different shapes
    if target.numel() == 0:
        return torch.tensor(0.0, device=predicted_gmm_params.device)

    # Handle different target shapes
    if target.dim() == 2:
        # If target is (N_samples, D), add batch dimension
        target = target.unsqueeze(0)  # Shape: (1, N_samples, D)
        B = 1
    elif target.dim() == 1:
        # If target is (N_samples,), reshape to (1, N_samples, 1)
        target = target.unsqueeze(0).unsqueeze(-1)  # Shape: (1, N_samples, 1)
        B = 1

    # Ensure we have the right batch size
    if predicted_gmm_params.shape[0] != B:
        # If batch sizes don't match, use the first batch item
        predicted_gmm_params = predicted_gmm_params[:B]

    # Flatten target samples for loss calculation
    # target_flat = target.view(-1, D)  # Shape: (B*N_samples, D)

    # Create GMM from predicted parameters
    gmm = get_gmm(predicted_gmm_params, K, D)

    # Calculate the log-probability of the target data under the GMM
    log_likelihood = gmm.log_prob(target)

    # The loss is the negative log-likelihood, averaged over all samples
    loss = -torch.mean(log_likelihood)

    return loss


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


def sample_from_predicted_gmm(predicted_gmm_params, num_samples, K=10, D=1):
    """
    Sample from a GMM created from predicted parameters.

    Args:
        predicted_gmm_params (torch.Tensor): Predicted GMM parameters.
            Shape: (K + 2 * K * D,) for single batch or (B, K + 2 * K * D) for batch
        num_samples (int): Number of samples to draw
        K (int): Number of GMM components
        D (int): Dimension of the data

    Returns:
        torch.Tensor: Samples from the GMM. Shape: (num_samples, D) or (B, num_samples, D)
    """
    gmm = get_gmm(predicted_gmm_params, K, D)
    samples = gmm.sample((num_samples,))
    return samples
