import hydra
from omegaconf import DictConfig
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
)


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
