from typing import Any
import hydra
from omegaconf import DictConfig
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
)
import torch
from torch_geometric.utils import to_dense_adj
from rdkit import Chem
from torch_geometric.utils import remove_self_loops, sort_edge_index


def build_callbacks(cfg: DictConfig) -> list[Callback]:
    callbacks: list[Callback] = []

    if "early_stopping" in cfg.callbacks:
        hydra.utils.log.info("Adding callback <EarlyStopping>")
        callbacks.append(
            EarlyStopping(
                monitor=cfg.callbacks.monitor_metric,
                mode=cfg.callbacks.monitor_metric_mode,
                patience=cfg.callbacks.early_stopping.patience,
                verbose=cfg.callbacks.early_stopping.verbose,
            )
        )

    if "model_checkpoints" in cfg.callbacks:
        hydra.utils.log.info("Adding callback <ModelCheckpoint>")
        callbacks.append(
            ModelCheckpoint(
                monitor=cfg.callbacks.monitor_metric,
                mode=cfg.callbacks.monitor_metric_mode,
                save_top_k=cfg.callbacks.model_checkpoints.save_top_k,
                verbose=cfg.callbacks.model_checkpoints.verbose,
                save_last=cfg.callbacks.model_checkpoints.save_last,
            )
        )

    if "every_n_epochs_checkpoint" in cfg.callbacks:
        hydra.utils.log.info(
            f"Adding callback <ModelCheckpoint> for every {cfg.callbacks.every_n_epochs_checkpoint.every_n_epochs} epochs"
        )
        callbacks.append(
            ModelCheckpoint(
                dirpath="every_n_epochs",
                every_n_epochs=cfg.callbacks.every_n_epochs_checkpoint.every_n_epochs,
                save_top_k=cfg.callbacks.every_n_epochs_checkpoint.save_top_k,
                verbose=cfg.callbacks.every_n_epochs_checkpoint.verbose,
                save_last=cfg.callbacks.every_n_epochs_checkpoint.save_last,
            )
        )

    return callbacks


def edge_types_to_triu_entries(edge_index, edge_types_one_hot, num_atoms):
    # By default, 0 is a single bond, 1 is a double bond etc.
    # When creating the adj_matrix we need to add a NONE-BOND at 0
    # Therefore, we add 1 to the edge types
    # 0: no bond, 1: single, 2: double, 3: triple, 4: aromatic
    edge_types = edge_types_one_hot.argmax(dim=-1) + 1

    adj_matrix = to_dense_adj(edge_index, edge_attr=edge_types, max_num_nodes=num_atoms)
    adj_matrix = adj_matrix.squeeze()

    # only keep the upper triangle (excluding diagonal) of the adj matrix
    triu_indices = torch.triu_indices(row=num_atoms, col=num_atoms, offset=1)
    triu_edge_types = adj_matrix[triu_indices[0], triu_indices[1]]

    return triu_edge_types


def edge_types_to_symmetric(edge_index, edge_types, num_atoms):
    """
    Convert edge types to a symmetric adjacency matrix.
    edge_index: Shape (2, E) - edge indices
    edge_types: Shape (E) - edge types
    num_atoms: int - number of atoms in the graph
    Returns:
        adj_matrix: Shape (N, N) - symmetric adjacency matrix
    """
    adj_matrix = to_dense_adj(edge_index, edge_attr=edge_types, max_num_nodes=num_atoms)
    adj_matrix = adj_matrix.squeeze()
    return adj_matrix


def z_to_atom_types(z):
    """Convert the atomic numbers to atom symbols with rdkit."""

    atom_symbols = []
    for z_i in z:
        atom_symbols.append(Chem.GetPeriodicTable().GetElementSymbol(z_i))
    return atom_symbols


def token_to_index(token_list, token: str):
    return token_list.index(token)


def index_to_token(token_list, index: int):
    return token_list[index]


def rigid_alignment(x_0, x_1, pre_centered=False):
    """
    See: https://en.wikipedia.org/wiki/Kabsch_algorithm
    Alignment of two point clouds using the Kabsch algorithm.
    Based on: https://gist.github.com/bougui505/e392a371f5bab095a3673ea6f4976cc8
    """
    d = x_0.shape[1]
    assert x_0.shape == x_1.shape, "x_0 and x_1 must have the same shape"

    # remove COM from data and record initial COM
    if pre_centered:
        x_0_mean = torch.zeros(1, d)
        x_1_mean = torch.zeros(1, d)
        x_0_c = x_0
        x_1_c = x_1
    else:
        x_0_mean = x_0.mean(dim=0, keepdim=True)
        x_1_mean = x_1.mean(dim=0, keepdim=True)
        x_0_c = x_0 - x_0_mean
        x_1_c = x_1 - x_1_mean

    # Covariance matrix
    H = x_0_c.T.mm(x_1_c)
    U, S, V = torch.svd(H)
    # Rotation matrix
    R = V.mm(U.T)
    # Translation vector
    if pre_centered:
        t = torch.zeros(1, d)
    else:
        t = x_1_mean - R.mm(x_0_mean.T).T  # has shape (1, D)

    """# apply rotation to x_0_c
    x_0_aligned = x_0_c.mm(R.T)

    # move x_0_aligned to its original frame
    x_0_aligned = x_0_aligned + x_0_mean

    # apply the translation
    x_0_aligned = x_0_aligned + t

    return x_0_aligned"""

    return R, t


def segment_softmax(logits, segment_ids, num_segments):
    """
    Numerically stable softmax over segments (graphs).
    """
    # 1. Find max per segment for stability
    # shape: (num_segments, K)
    m = torch.zeros((num_segments, logits.size(1)), device=logits.device).fill_(
        -float("inf")
    )
    m = m.index_reduce(0, segment_ids, logits, reduce="amax", include_self=False)

    # 2. Subtract max (broadcast back to nodes)
    # shape: (N, K)
    logits_stable = logits - m[segment_ids]

    # 3. Exponentiate
    exp_logits = torch.exp(logits_stable)

    # 4. Sum exp per segment
    # shape: (num_segments, K)
    exp_sum = torch.zeros_like(m)
    exp_sum.index_add_(0, segment_ids, exp_logits)

    # 5. Divide (broadcast back to nodes)
    # shape: (N, K)
    probs = exp_logits / (exp_sum[segment_ids] + 1e-6)
    return probs


def build_fully_connected_edge_index(N_atoms, device="cpu"):
    edge_index = torch.cartesian_prod(
        torch.arange(N_atoms, device=device),
        torch.arange(N_atoms, device=device),
    )
    edge_index = edge_index.T
    edge_index = remove_self_loops(edge_index)[0]
    return edge_index


def compute_token_weights(
    token_list: list[str],
    distribution: torch.Tensor,
    special_token_names: list[str],
    weight_alpha: float = 1.0,
    type_loss_token_weights: str = "training",
) -> torch.Tensor:
    """
    Compute token weights with special handling for special tokens.

    This function can be used for both node tokens (atom types) and edge tokens.
    It computes inverse frequency weights, normalizes regular tokens, and assigns
    appropriate weights to special tokens.

    Args:
        token_list: List of all tokens (e.g., atom types or edge types)
        distribution: Distribution of token types from training data.
            MUST have the same length as token_list and be aligned by index.
        special_token_names: List of special token names to handle specially
            (e.g., ["<MASK>", "<DEATH>"] for nodes or ["<MASK>", "<NO_BOND>"] for edges)
        weight_alpha: Alpha parameter for weight scaling (default: 1.0)
        type_loss_token_weights: "uniform" or "training" - if "uniform", returns
            uniform weights (all 1.0) except for <MASK> which stays 0.0

    Returns:
        Weights tensor with same shape as distribution
    """
    # Validate inputs
    if len(distribution) != len(token_list):
        raise ValueError(
            f"Distribution length ({len(distribution)}) must match "
            f"token_list length ({len(token_list)})"
        )

    # Get indices for special tokens (with validation)
    special_token_indices = set()
    mask_token_idx = None
    for token_name in special_token_names:
        if token_name not in token_list:
            print(f"Warning: Special token '{token_name}' not in token_list, skipping")
            continue
        idx = token_to_index(token_list, token_name)
        special_token_indices.add(idx)
        if token_name == "<MASK>":
            mask_token_idx = idx

    # --- Handle uniform weights case early ---
    if type_loss_token_weights == "uniform":
        final_weights = torch.ones_like(distribution)
        # <MASK> should always have 0 weight (ignore in loss)
        if mask_token_idx is not None:
            final_weights[mask_token_idx] = 0.0
        return final_weights

    # --- 1. Initial Weight Calculation (Inverse Frequency) ---
    epsilon = 1e-8
    weights = 1.0 / (distribution + epsilon)
    weights = weights**weight_alpha  # Apply alpha scaling

    # --- 2. Isolate & Normalize REGULAR Tokens ---
    all_indices = set(range(len(token_list)))
    regular_token_indices = list(all_indices - special_token_indices)

    if not regular_token_indices:
        # Fallback if no regular tokens (unlikely, but good to handle)
        final_weights = torch.ones_like(weights)
        if mask_token_idx is not None:
            final_weights[mask_token_idx] = 0.0
        return final_weights

    # Get weights for only the regular tokens
    regular_weights = weights[regular_token_indices]

    # Normalize *only* the regular weights to have a mean of 1.0
    mean_regular_weight = regular_weights.mean()
    if mean_regular_weight > 0:
        regular_weights = regular_weights / mean_regular_weight

    # --- 3. Build Final Weights Tensor ---
    # Start with the full weights (including special tokens computed from distribution)
    final_weights = weights.clone()

    # Assign normalized regular weights to their correct positions
    final_weights[regular_token_indices] = regular_weights

    # Handle special tokens
    for token_name in special_token_names:
        if token_name not in token_list:
            continue
        token_idx = token_to_index(token_list, token_name)

        if token_name == "<MASK>":
            # <MASK> should never contribute to loss
            final_weights[token_idx] = 0.0
        else:
            # For other special tokens (e.g., <NO_BOND>), use their actual
            # inverse frequency weight, normalized the same way as regular tokens
            # This ensures common special tokens (like <NO_BOND>) get low weight
            special_weight = weights[token_idx]
            if mean_regular_weight > 0:
                special_weight = special_weight / mean_regular_weight
            final_weights[token_idx] = special_weight

    # set the minimum weight to 1.0
    final_weights = final_weights / final_weights.min()

    return final_weights


def symmetrize_upper_triangle(edge_index, edge_attr):
    """
    Reconstructs the full symmetric edge_index and edge_attr from an
    upper triangular representation (row <= col).

    Mirroring Logic:
    - Off-diagonal edges (row < col) are duplicated and swapped (col, row).
    - Diagonal edges (row == col) are kept as-is (appearing once).
    """
    row, col = edge_index

    # 1. Identify strictly off-diagonal edges to mirror
    #    (Self-loops should not be duplicated)
    mask = row < col

    # 2. Create the mirrored part (Lower Triangle)
    #    Flip row/col and select attributes
    mirror_index = torch.stack([col[mask], row[mask]], dim=0)
    mirror_attr = edge_attr[mask]

    # 3. Concatenate Original (Triu + Diag) with Mirrored (Tril)
    full_edge_index = torch.cat([edge_index, mirror_index], dim=1)
    full_edge_attr = torch.cat([edge_attr, mirror_attr], dim=0)

    # 4. Sort to ensure canonical PyG order (row-major)
    return sort_edge_index(full_edge_index, full_edge_attr)


def remove_token_from_distribution(token_list, distribution, token="<MASK>"):
    token_index = token_to_index(token_list, token)
    token_list.remove(token)
    distribution = torch.cat(
        [distribution[:token_index], distribution[token_index + 1 :]]
    )
    return token_list, distribution


def validate_no_cross_batch_edges(
    edge_index: torch.Tensor,
    batch: torch.Tensor,
    location: str = "",
) -> bool:
    """
    Validates that all edges are within the same batch (no cross-batch edges).

    Args:
        edge_index: Shape (2, E) - edge indices
        batch: Shape (N,) - batch assignment for each node
        location: String identifier for where this check is called (for debugging)

    Returns:
        True if valid (no cross-batch edges), False otherwise
    """
    if edge_index is None or edge_index.numel() == 0:
        return True

    edge_batches_src = batch[edge_index[0]]
    edge_batches_tgt = batch[edge_index[1]]
    cross_batch_mask = edge_batches_src != edge_batches_tgt

    if cross_batch_mask.any():
        n_cross = cross_batch_mask.sum().item()
        print(f"WARNING [{location}]: Found {n_cross} cross-batch edges!")
        print(f"  Cross-batch edge indices: {edge_index[:, cross_batch_mask]}")
        print(f"  Source batches: {edge_batches_src[cross_batch_mask]}")
        print(f"  Target batches: {edge_batches_tgt[cross_batch_mask]}")
        return False

    return True


class EdgeAligner:
    """
    Helper class to handle edge indices and attributes.
    Can be used to align edge indices and attributes between two groups of edges.
    """

    def __init__(self):
        pass

    def get_canonical_upper_triangle_with_index(
        self, edge_index, edge_attr, include_diag=False
    ):
        """
        Returns (edge_index, edge_attr) for the upper triangle.
        """
        # 1. Sort first to ensure canonical order
        edge_index, edge_attr = sort_edge_index(edge_index, edge_attr)
        row, col = edge_index

        # 2. Filter
        if include_diag:
            mask = row <= col
        else:
            mask = row < col

        # Return BOTH the filtered index and attributes
        return edge_index[:, mask], edge_attr[mask]

    def _process_edge_group(self, edge_index, attrs):
        """
        Helper: Takes an edge_index and a LIST of attributes.
        Returns the canonical upper-tri edge_index and the list of processed attributes.
        """
        if not attrs:
            return None, []

        processed_attrs = []
        canonical_index = None

        for attr in attrs:
            # We call your existing helper for each attribute
            curr_index, curr_attr_triu = self.get_canonical_upper_triangle_with_index(
                edge_index, attr
            )

            # Store the index from the first attribute to use for validation
            if canonical_index is None:
                canonical_index = curr_index

            processed_attrs.append(curr_attr_triu)

        return canonical_index, processed_attrs

    def align_edges(self, source_group, target_group=None):
        """
        Flexible handler for aligning edge attributes.

        Args:
            source_group: Tuple of (edge_index, [attr1, attr2, ...])
            target_group: (Optional) Tuple of (edge_index, [attr1, attr2, ...])

        Returns:
            Flat tuple of all processed attributes from source then target.
        """
        src_index, src_attrs = source_group
        tgt_index, tgt_attrs = (None, None) if target_group is None else target_group

        # 1. Process Source Group
        src_triu_index, src_triu_attrs = self._process_edge_group(src_index, src_attrs)

        # 2. Process Target Group (if exists)
        tgt_triu_attrs = []
        if target_group:
            tgt_triu_index, tgt_triu_attrs = self._process_edge_group(
                tgt_index, tgt_attrs
            )

            # 3. Assert Alignment
            assert torch.all(src_triu_index == tgt_triu_index), (
                "The edge indices between source and target groups must be the same."
            )

            # Return all attributes flattened
            out = {
                "edge_index": (src_triu_index, tgt_triu_index),
                "edge_attr": tuple(src_triu_attrs + tgt_triu_attrs),
            }

        else:
            out = {
                "edge_index": src_triu_index,
                "edge_attr": src_triu_attrs,
            }
        return out

    def symmetrize_edges(
        self, edge_index_triu: torch.Tensor, attrs: list[torch.Tensor]
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """
        Reverse: Upper Triangle -> Full Symmetric
        Handles a list of attributes efficiently by calculating the sort permutation once.
        """
        row, col = edge_index_triu

        # 1. Identify off-diagonal edges to mirror
        mask = row < col

        # 2. Create the mirrored index
        mirror_index = torch.stack([col[mask], row[mask]], dim=0)
        full_edge_index = torch.cat([edge_index_triu, mirror_index], dim=1)

        # 3. Compute Sort Permutation ONCE
        # We cannot use sort_edge_index repeatedly or we risk attributes getting out of sync
        # if the sort isn't stable. We compute indices explicitly.
        # Simple Lexical Sort: row * max_cols + col
        # (Or use PyTorch's stable sort on rows then cols)
        num_nodes = full_edge_index.max().item() + 1
        perm = (full_edge_index[0] * num_nodes + full_edge_index[1]).argsort()

        # Apply sort to index
        full_edge_index_sorted = full_edge_index[:, perm]

        # 4. Process all attributes
        full_attrs_sorted = []
        for attr in attrs:
            # Mirror the attributes corresponding to the mask
            mirror_attr = attr[mask]
            full_attr = torch.cat([attr, mirror_attr], dim=0)

            # Apply the SAME permutation
            full_attrs_sorted.append(full_attr[perm])

        return full_edge_index_sorted, tuple(full_attrs_sorted)
