import torch
import torch.nn.functional as F
from chemflow.flow_matching.gmm import get_gmm, get_typed_gmm_components


def gmm_loss(gmm_output, target_locations, target_batch_ids, mask_value=-1e3):
    """
    Computes NLL for non-typed Equivariant GMM on variable sized graphs.

    Args:
        gmm_output (dict): Output from compute_equivariant_gmm (mu, sigma, pi).
        target_locations (Tensor): [N_total, D] Node positions.
        target_batch_ids (Tensor): [N_total] Batch index per node.
    """

    # --- 1. Filter Invalid Targets ---
    valid_mask = (target_locations != mask_value).any(dim=-1)

    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=target_locations.device, requires_grad=True)

    locs = target_locations[valid_mask]  # [N_valid, D]
    batch_idx = target_batch_ids[valid_mask]  # [N_valid]

    # --- 2. Gather Params (B -> N) ---
    # Expand graph-level params to node-level params
    gmm_expanded = {
        "mu": gmm_output["mu"][batch_idx],  # [N, K, D]
        "sigma": gmm_output["sigma"][batch_idx],  # [N, K]
        "pi": gmm_output["pi"][batch_idx],  # [N, K]
    }

    # --- 3. Create Distribution ---
    # The helper sees "Batch Size" as N_valid.
    # Returns distribution with batch_shape=[N_valid, 1], event_shape=[D]
    gmm_dist = get_gmm(gmm_expanded, D=locs.shape[-1])

    # --- 4. Compute Log Prob ---
    # Target: [N, D] -> [N, 1, D] to match dist batch_shape [N, 1]
    log_likelihood = gmm_dist.log_prob(locs.unsqueeze(1))

    # Loss is negative mean log likelihood
    return -torch.mean(log_likelihood)


def typed_gmm_loss(
    gmm_output,
    target_x,
    target_a,
    target_c,
    class_weights_a,
    class_weights_c,
    reduction="mean",
):
    """
    Computes the NLL loss using the get_typed_gmm_components helper.

    Strategy:
    1. Filter invalid nodes.
    2. Expand the global GMM params (B) to node-level params (N) using indices.
    3. Use the helper to create distributions of shape [N, 1, K].
    4. Compute loss.
    """

    # --- 3. Create Distributions using Helper ---
    # The helper treats the first dimension as "Batch".
    # Here, our "Batch" is N_valid.
    # Returns distributions with batch_shape=[N, 1, K]

    mix_dist, x_dist, a_dist, c_dist = get_typed_gmm_components(gmm_output)

    N = mix_dist.logits.shape[0]
    D = target_x.shape[-1]

    target_x = target_x.view(N, 1, 1, -1)
    target_a = target_a.view(N, 1, 1)
    target_c = target_c.view(N, 1, 1)

    # --- 4. Compute Log-Probabilities ---
    eps = 1e-6

    # A. Spatial Log-Prob
    # Target: [N, 3] -> [N, 1, 3] to broadcast against dist [N, 1, K]
    log_prob_x = x_dist.log_prob(target_x) / D  # Result: [N, 1, K]

    # handle shrinking variance --> makes it MSE-like loss
    # log_prob_x = log_prob_x * 2 * (sigma_t.pow(2)).clamp(min=1e-5).reshape(N, 1, 1)

    # B. Type Log-Prob
    # TODO maybe we have to implement a weighted log_prob for the types
    # Target: [N] -> [N, 1] to broadcast against dist [N, 1, K]
    log_prob_a = a_dist.log_prob(target_a)  # Result: [N, 1, K]
    log_prob_c = c_dist.log_prob(target_c)  # Result: [N, 1, K]

    # C. Mixture Log-Prob
    # Categorical log_prob expects value, but here we just want the log-probs of the weights themselves.
    # Since mixture_weights is a Categorical dist, we can access logits directly or evaluate log_prob.
    # However, since we are doing log_sum_exp over the components K, we simply need log(pi).
    # The helper returns a Categorical, so we can just grab the logits (normalized).
    log_prob_mix = mix_dist.logits  # Result: [N, 1, K]

    # --- 5. Combine and Loss ---
    # Sum log-probs: [N, 1, K]
    log_joint = log_prob_mix + log_prob_x + log_prob_a + log_prob_c

    # LogSumExp over components (dim=-1): [N, 1]
    log_likelihood = torch.logsumexp(log_joint, dim=-1)

    a_weight = class_weights_a[target_a].view(-1, 1)
    c_weight = class_weights_c[target_c].view(-1, 1)

    # Instead of multiplying the weights, we add them up to prevent exploding weight
    weights = (a_weight + c_weight) / 2
    log_likelihood = log_likelihood * weights

    # Average over nodes
    if reduction == "mean":
        nll = -torch.mean(log_likelihood)
    elif reduction == "sum":
        nll = -torch.sum(log_likelihood)
    elif reduction == "none":
        nll = -log_likelihood
    else:
        raise ValueError(f"Invalid reduction: {reduction}")

    return nll, weights
