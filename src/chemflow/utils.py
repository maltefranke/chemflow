import hydra
from omegaconf import DictConfig
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
)
import torch
from torch_geometric.utils import to_dense_adj


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
