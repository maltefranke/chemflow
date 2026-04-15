import torch.distributed as dist

from chemflow.model.lightning_module import LightningModuleRates
from external_code.muon import MuonWithAuxAdam, SingleDeviceMuonWithAuxAdam
from omegaconf import OmegaConf
import hydra


class LightningModuleRatesTransformer(LightningModuleRates):
    def configure_optimizers(self):
        opt_cfg = self.optimizer_config.optimizer

        # Collect all trainable parameters
        params = list(self.model.parameters())
        if (
            self.loss_weight_wrapper.use_learnable
            and self.loss_weight_wrapper.learnable_wrapper is not None
        ):
            params.extend(list(self.loss_weight_wrapper.learnable_wrapper.parameters()))

        if "_target_" in opt_cfg:
            # Standard optimizer — instantiate directly via Hydra (e.g. torch.optim.AdamW)
            optimizer = hydra.utils.instantiate(opt_cfg, params=params)
        else:
            # Muon path: three param groups routed by parameter shape and origin
            backbone_param_ids = {id(p) for p in self.model.backbone.parameters()}
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

            print(f"Muon params: {len(muon_params)}, High LR params: {len(high_lr_params)}, Adam params: {len(adam_params)}")
            
            if (
                self.loss_weight_wrapper.use_learnable
                and self.loss_weight_wrapper.learnable_wrapper is not None
            ):
                adam_params.extend(
                    list(self.loss_weight_wrapper.learnable_wrapper.parameters())
                )
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
