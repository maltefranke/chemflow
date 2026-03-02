import torch.distributed as dist

from chemflow.model.lightning_module import LightningModuleRates
from external_code.muon import MuonWithAuxAdam, SingleDeviceMuonWithAuxAdam
from omegaconf import OmegaConf
import hydra


class LightningModuleRatesTransformer(LightningModuleRates):
    def configure_optimizers(self):
        # Identify backbone parameters by id for Muon routing
        backbone_param_ids = {id(p) for p in self.model.backbone.parameters()}

        # Backbone 2D weights                      -> Muon
        # All other 2D+ params (embs, heads, etc.) -> Adam high LR (emb_optimizer)
        # All 1D params everywhere (biases, norms)  -> Adam low LR (aux_optimizer)
        muon_params = []
        high_lr_params = []
        adam_params = []
        for p in self.model.parameters():
            if id(p) in backbone_param_ids and p.ndim >= 2:
                muon_params.append(p)
            elif id(p) not in backbone_param_ids and p.ndim >= 2:
                high_lr_params.append(p)
            else:
                adam_params.append(p)

        # Learnable loss weights are scalars — low-LR aux group
        if (
            self.loss_weight_wrapper.use_learnable
            and self.loss_weight_wrapper.learnable_wrapper is not None
        ):
            adam_params.extend(
                list(self.loss_weight_wrapper.learnable_wrapper.parameters())
            )

        # Read hyperparameters from config
        opt_cfg = self.optimizer_config.optimizer
        aux_cfg = self.optimizer_config.aux_optimizer
        emb_cfg = self.optimizer_config.emb_optimizer

        param_groups = [
            dict(
                params=adam_params,
                lr=aux_cfg.lr,
                betas=tuple(aux_cfg.betas),
                eps=aux_cfg.get("eps", 1e-10),
                weight_decay=aux_cfg.weight_decay,
                use_muon=False,
            ),
            dict(
                params=high_lr_params,
                lr=emb_cfg.lr,
                betas=tuple(aux_cfg.betas),
                eps=aux_cfg.get("eps", 1e-10),
                weight_decay=aux_cfg.weight_decay,
                use_muon=False,
            ),
            dict(
                params=muon_params,
                lr=opt_cfg.lr,
                momentum=opt_cfg.momentum,
                weight_decay=opt_cfg.weight_decay,
                use_muon=True,
            ),
        ]

        if dist.is_initialized():
            optimizer = MuonWithAuxAdam(param_groups)
        else:
            optimizer = SingleDeviceMuonWithAuxAdam(param_groups)

        # Instantiate scheduler from config
        scheduler_cfg = dict(
            OmegaConf.to_container(self.optimizer_config.scheduler, resolve=True)
        )
        scheduler_cfg["optimizer"] = optimizer
        scheduler = hydra.utils.instantiate(scheduler_cfg)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": self.optimizer_config.interval,
                "monitor": self.optimizer_config.monitor,
            },
            "monitor": self.optimizer_config.monitor,
        }
