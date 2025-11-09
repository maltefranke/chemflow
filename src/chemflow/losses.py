import torch
import torch.nn.functional as F
from chemflow.flow_matching.gmm import get_gmm, get_typed_gmm_components


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

    # Create GMM from predicted parameters
    gmm = get_gmm(predicted_gmm_params, K, D)

    # Calculate the log-probability of the target data under the GMM
    log_likelihood = gmm.log_prob(target)

    # The loss is the negative log-likelihood, averaged over all samples
    loss = -torch.mean(log_likelihood)

    return loss


def typed_gmm_loss(
    predicted_gmm_params,
    target_locations,
    target_types,
    target_batch_ids,
    K=10,
    D=3,
    N_types=6,
):
    """
    Computes the NLL loss for joint GMM predictions.

    Calculates: log p(x, c) = LogSumExp_m [ log p(m) + log p(x|m) + log p(c|m) ]

    Args:
        predicted_gmm_params (torch.Tensor): The raw output from the network.
            Shape: (B, K + 2*K*D + K*N_types)
        target_locations (torch.Tensor): The target spatial data.
            Shape: (N_samples, D)
        target_types (torch.Tensor): The target type data.
            Shape: (N_samples, N_classes)
        K (int): Number of GMM components.
        D (int): Dimension of the data.
        N_types (int): Number of discrete types.

    Returns:
        torch.Tensor: The negative log-likelihood loss.
    """

    mask = (target_locations != -1e3).any(dim=-1)

    if mask.sum() == 0:
        # all targets are masked, so no loss
        return torch.tensor(0.0, device=predicted_gmm_params.device)
    else:
        target_locations = target_locations[mask]
        target_types = target_types[mask]

        # pred mask by batch ids
        # TODO is unique robust?
        valid_batch_ids = target_batch_ids[mask].unique()
        predicted_gmm_params = predicted_gmm_params[valid_batch_ids]

    B = predicted_gmm_params.shape[0]

    # Handle case where target might be empty
    if target_locations.numel() == 0 or target_types.numel() == 0:
        return torch.tensor(0.0, device=predicted_gmm_params.device)

    # Handle non-batched target shapes (e.g., from a single graph)
    if target_locations.dim() == 2:
        """# If target is (N_samples, D), add batch dimension
        target_locations = target_locations.unsqueeze(0)  # Shape: (1, N_samples, D)
        target_types = target_types.unsqueeze(0)  # Shape: (1, N_samples)
        B = 1"""
        target_locations = target_locations.reshape(B, -1, D)
        target_types = target_types.reshape(B, -1)

    # Ensure we have the right batch size
    if predicted_gmm_params.shape[0] != B:
        # If batch sizes don't match, use the first batch item
        predicted_gmm_params = predicted_gmm_params[:B]

    # Create GMM component distributions from predicted parameters
    # mixture_weights: batch_shape=[B, 1]
    # spatial_dist:    batch_shape=[B, 1, K]
    # type_dist:       batch_shape=[B, 1, K]
    mixture_weights, spatial_dist, type_dist = get_typed_gmm_components(
        predicted_gmm_params, K, D, N_types
    )

    # --- Calculate the log-probability ---

    # 1. Get log p(m)
    # Use logits_normalized for stable log-probabilities of the mixture weights
    # Shape: (B, 1, K)
    log_mix_weights = mixture_weights.probs

    # 2. Get log p(x|m)
    # We need log_prob of [B, N_samples, D] target under [B, 1, K] dists.
    # Unsqueeze target to [B, N_samples, 1, D] for broadcasting.
    # Broadcasting [B, N_samples, 1] with [B, 1, K] gives [B, N_samples, K]
    # Resulting shape: [B, N_samples, K]
    log_prob_spatial = spatial_dist.log_prob(target_locations.unsqueeze(2))

    # 3. Get log p(c|m)
    # We need log_prob of [B, N_samples] target under [B, 1, K] dists.
    # Unsqueeze target to [B, N_samples, 1] for broadcasting.
    # Broadcasting [B, N_samples, 1] with [B, 1, K] gives [B, N_samples, K]
    # Resulting shape: [B, N_samples, K]
    log_prob_types = type_dist.log_prob(target_types.unsqueeze(-1))

    # 4. Combine: log p(m) + log p(x|m) + log p(c|m)
    # log_mix_weights (B, 1, K) broadcasts to (B, N_samples, K)
    # Resulting shape: (B, N_samples, K)
    log_per_component = log_mix_weights + log_prob_spatial + log_prob_types

    # 5. LogSumExp over the K components (dim=2)
    # This computes log( sum_m( p(m, x, c) ) ) = log p(x, c)
    # Resulting shape: (B, N_samples)
    log_likelihood = torch.logsumexp(log_per_component, dim=-1)

    # The loss is the negative log-likelihood, averaged over all samples
    loss = -torch.mean(log_likelihood)

    return loss


def rate_loss(predicted_rate, target_rate):
    rate_loss = F.poisson_nll_loss(
        predicted_rate,
        target_rate,
        log_input=False,
    )

    # Check for NaN and replace with 0 if needed
    if torch.isnan(rate_loss):
        rate_loss = torch.tensor(0.0, device=predicted_rate.device)

    return rate_loss
