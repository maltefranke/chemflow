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
        x_0_mean = torch.zeros(1, d, device=x_0.device)
        x_1_mean = torch.zeros(1, d, device=x_1.device)
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
        t = torch.zeros(1, d, device=x_0.device)
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
