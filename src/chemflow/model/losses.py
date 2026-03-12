import torch
import torch.nn.functional as F

from chemflow.model.gmm import get_typed_gmm_components


def _type_log_prob_from_ce(target, dist, class_weights, name, N):
    """Compute per-component type log-prob using PyTorch CE."""
    logits = dist.logits  # [N, 1, K, T]
    K = logits.shape[2]
    num_classes = logits.shape[-1]
    logits_flat = logits.reshape(-1, num_classes)

    if target.dim() == 1:
        if target.shape[0] != N:
            raise ValueError(
                f"{name} hard targets must have shape [N], got {target.shape} for N={N}"
            )
        target_idx = target.long()
        target_idx_flat = target_idx.view(N, 1, 1).expand(N, 1, K).reshape(-1)
        ce = F.cross_entropy(
            logits_flat,
            target_idx_flat,
            weight=class_weights,
            reduction="none",
        )
        return -ce.view(N, 1, K)

    if target.dim() == 2:
        if target.shape[0] != N:
            raise ValueError(
                f"{name} soft targets must have shape [N, T], "
                f"got {target.shape} for N={N}"
            )
        num_classes = dist.logits.shape[-1]
        if target.shape[1] != num_classes:
            raise ValueError(
                f"{name} soft targets class dim mismatch: "
                f"expected {num_classes}, got {target.shape[1]}"
            )

        target_probs = target.to(dtype=dist.logits.dtype)
        target_probs = target_probs / target_probs.sum(dim=-1, keepdim=True).clamp_min(
            1e-8
        )
        target_probs_flat = (
            target_probs.view(N, 1, 1, num_classes)
            .expand(N, 1, K, num_classes)
            .reshape(-1, num_classes)
        )
        ce = F.cross_entropy(
            logits_flat,
            target_probs_flat,
            weight=class_weights,
            reduction="none",
        )
        return -ce.view(N, 1, K)

    raise ValueError(
        f"{name} targets must be rank-1 indices or rank-2 "
        f"probabilities, got rank {target.dim()}"
    )


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

    # --- 4. Compute Log-Probabilities ---

    # A. Spatial Log-Prob
    # Target: [N, 3] -> [N, 1, 3] to broadcast against dist [N, 1, K]
    log_prob_x = x_dist.log_prob(target_x) / D  # Result: [N, 1, K]

    # handle shrinking variance --> makes it MSE-like loss
    # log_prob_x = log_prob_x * 2 * (sigma_t.pow(2)).clamp(min=1e-5).reshape(N, 1, 1)

    # B. Type Log-Prob
    # TODO maybe we have to implement a weighted log_prob for the types
    # Target: [N] -> [N, 1] to broadcast against dist [N, 1, K]
    log_prob_a = _type_log_prob_from_ce(
        target_a, a_dist, class_weights_a, "target_a", N
    )
    log_prob_c = _type_log_prob_from_ce(
        target_c, c_dist, class_weights_c, "target_c", N
    )

    # C. Mixture Log-Prob
    # Categorical log_prob expects values, but here we only need log(pi)
    # for each mixture component before log-sum-exp over K.
    # The helper returns a Categorical, so we can just grab the logits (normalized).
    log_prob_mix = mix_dist.logits  # Result: [N, 1, K]

    # --- 5. Combine spatial + type + mixture terms ---
    log_joint = log_prob_mix + log_prob_x + log_prob_a + log_prob_c

    # LogSumExp over components (dim=-1): [N, 1]
    log_likelihood = torch.logsumexp(log_joint, dim=-1)

    # Average over nodes
    if reduction == "mean":
        nll = -torch.mean(log_likelihood)
    elif reduction == "sum":
        nll = -torch.sum(log_likelihood)
    elif reduction == "none":
        nll = -log_likelihood
    else:
        raise ValueError(f"Invalid reduction: {reduction}")

    ones = torch.ones((N, 1, 1), device=target_x.device, dtype=target_x.dtype)
    return nll, (ones, ones)
