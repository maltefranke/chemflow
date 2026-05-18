import copy
from contextlib import ExitStack

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
import hydra
from chemflow.model.losses import (
    typed_gmm_loss,
    do_action_loss,
    rate_loss,
    class_loss,
)

from chemflow.utils.utils import EDGE_ALIGNER
from external_code.egnn import unsorted_segment_mean

from chemflow.dataset.molecule_data import MoleculeBatch
from chemflow.dataset.vocab import Vocab, Distributions
from chemflow.utils.metrics import (
    calc_metrics_,
    MetricCollection,
    calc_posebusters_metrics,
    init_metrics,
    build_marginal_plots,
)
from lightning.pytorch.utilities import grad_norm
from chemflow.utils.loss_accumulation import LossAccumulator
from chemflow.model.learnable_loss import UnifiedWeightedLoss
from chemflow.utils.loss_weighing import (
    InverseSquaredTimeLossWeighting,
)
from chemflow.utils import rdkit_utils as chemflowRD
from chemflow.utils.lr_schedulers import EMADecayScheduler


class LightningModuleRates(pl.LightningModule):
    """
    This model implements an EditFlow model with adjustments for stability inspired by OneFlow.
    Crucially, we do not use the rate k_t_dot / (1 - k_t) directly in the formulation.

    Specifically, we have the following adjustments:
    - Insertion:
            Binary predictor (do insert or not?)
            Rate predictor (how many insertions per node?) for nodes requiring insertions.
                This can either be a classifier (CE loss) or a Poisson regression (Poisson NLL loss).
            GMM predictor (what atom type, charge and where)
            Edge predictor (what edge type and where) depending on GMM predictions.
                Is realized via Gumbel-Softmax trick to differentiate through sampling
    - Deletion:
            Binary predictor (do delete or not?) for nodes that require deletions.
    - Substitution:
            Binary predictor (do substitute or not?) for nodes that require substitutions.
            Atom type and edge type predictor (what atom type, charge and edge type and where)

    - Charge prediction
    - Position prediction

    For regularization:
    - Number of deletions predicted globally for each graph.
    - Number of insertions predicted globally for each graph.
    """

    LOSS_GROUPS: dict[str, list[str]] = {
        "sub": ["do_sub_a", "sub_a_class", "do_sub_e", "sub_e_class"],
        "del": ["do_del"],
        "ins": ["ins_rate", "ins_gmm", "ins_e", "ins_e_ii"],
        "x": ["x"],
        "c": ["c"],
    }

    def __init__(
        self,
        model: DictConfig,
        integrator: DictConfig,
        loss_weights: DictConfig,
        optimizer_config: DictConfig,
        gmm_params: DictConfig,
        vocab: Vocab,
        distributions: Distributions,
        loss_weight_distributions: Distributions,
        atom_type_weights: torch.Tensor,
        edge_token_weights: torch.Tensor,
        charge_token_weights: torch.Tensor,
        cfg_guidance: DictConfig,
        time_dist: DictConfig,
        metrics: MetricCollection,
        stability_metrics: MetricCollection,
        distribution_metrics: MetricCollection | None = None,
        n_atoms_strategy: str = "fixed",
        ins_noise_scale: float = 0.5,
        use_learnable_loss_weights: bool = False,
        ema_decay: float = 0.999,
        ema_decay_scheduler: EMADecayScheduler | DictConfig | None = None,
        use_ema_for_eval: bool = True,
        use_time_weights: bool = False,
        allow_charged: bool = False,
        log_grad_norms_every_n_steps: int = 100,
        grad_norms_granularity: str = "component",
    ):
        super().__init__()

        self.model = hydra.utils.instantiate(model)

        # vocab and distributions
        self.vocab = vocab
        self.distributions = distributions
        self.loss_weight_distributions = loss_weight_distributions

        self.ins_noise_scale = ins_noise_scale

        # Whether charged species are considered chemically valid for the dataset
        # (e.g. QM9 contains only neutral molecules; GEOM contains charged ones).
        self.allow_charged = allow_charged

        # setup ema scheduler
        self.ema_decay = float(ema_decay)
        if isinstance(ema_decay_scheduler, DictConfig):
            ema_decay_scheduler = hydra.utils.instantiate(ema_decay_scheduler)
        self.ema_decay_scheduler = ema_decay_scheduler
        self.use_ema_for_eval = use_ema_for_eval

        self.gmm_params = gmm_params
        self.n_atoms_strategy = n_atoms_strategy
        self.time_dist = hydra.utils.instantiate(time_dist)

        self.integrator = hydra.utils.instantiate(
            integrator, distributions=self.distributions
        )

        # weights for class losses
        self.register_buffer("atom_type_weights", atom_type_weights)
        self.register_buffer("edge_token_weights", edge_token_weights)
        self.register_buffer("charge_token_weights", charge_token_weights)

        # metrics tracking for validation
        self.metrics = metrics
        self.stability_metrics = stability_metrics
        # Distribution metrics accumulate over the entire validation epoch so we
        # can log a single pooled KL and render marginal plots at epoch end.
        self.distribution_metrics = distribution_metrics

        self.cfg_guidance = hydra.utils.instantiate(
            cfg_guidance,
            model=self.model,
            atom_tokens=list(self.vocab.atom_tokens),
        )

        # Exponential moving average of model parameters for stable inference
        self.model_ema = copy.deepcopy(self.model)
        for p in self.model_ema.parameters():
            p.requires_grad = False
        self.model_ema.eval()

        # handling of edge utilities, especially for upper-triangular handling
        self.edge_aligner = EDGE_ALIGNER

        # set up loss weighting for individual loss components
        loss_weight_values = {k: float(v) for k, v in loss_weights.items()}
        self.loss_weight_wrapper = UnifiedWeightedLoss(
            manual_weights=loss_weight_values,
            component_keys=list(loss_weight_values.keys()),
            use_learnable=use_learnable_loss_weights,
        )

        # optionally weigh by interpolation time
        if use_time_weights:
            time_weights = {
                "x": InverseSquaredTimeLossWeighting(clamp_max=100.0),
                "c": lambda t: self.integrator.sub_schedule.rate(t).clamp(max=100.0),
                "ins": lambda t: self.integrator.ins_schedule.rate(t).clamp(max=100.0),
                "del": lambda t: self.integrator.del_schedule.rate(t).clamp(max=100.0),
                "sub": lambda t: self.integrator.sub_schedule.rate(t).clamp(max=100.0),
            }
        else:
            # always use time-weighting for x and c
            time_weights = {"x": InverseSquaredTimeLossWeighting(clamp_max=100.0), "c": lambda t: self.integrator.sub_schedule.rate(t).clamp(max=100.0)}

        # object to tie loss computation, weighting, and logging together
        self.loss_accumulator = LossAccumulator(
            self.loss_weight_wrapper, self.LOSS_GROUPS, self.device, time_weights
        )

        self.optimizer_config = optimizer_config

        # Per-loss gradient-norm logging. Each invocation runs one extra
        # backward per reported term, so this is rate-limited via a step
        # interval and is off by default.
        self.log_grad_norms_every_n_steps = int(log_grad_norms_every_n_steps)
        if grad_norms_granularity not in {"component", "group", "both"}:
            raise ValueError(
                "grad_norms_granularity must be one of component/group/both, "
                f"got {grad_norms_granularity!r}"
            )
        self.grad_norms_granularity = grad_norms_granularity

        self.is_compiled = False

        # lastly, save hyperparameters
        # ``cfg_guidance`` wraps ``self.model``; excluding it avoids capturing
        # a duplicate (cyclic) reference to the backbone in the hparams
        # snapshot and keeps checkpoint size down.
        self.save_hyperparameters(
            ignore=[
                "ema_decay_scheduler",
                "metrics",
                "stability_metrics",
                "distribution_metrics",
                "cfg_guidance",
            ]
        )

    def compile(self):
        """Compile the model using torch.compile."""
        if self.is_compiled:
            return
        print("Compiling model...")
        # Per-loss grad-norm logging calls torch.autograd.grad(..., retain_graph=True)
        # before the main backward. torch.compile's donated-buffer optimization frees
        # activations after the first backward and rejects any subsequent one, so
        # disable it when grad-norm tracking is active.
        if self.log_grad_norms_every_n_steps > 0:
            import torch._functorch.config as _functorch_config

            _functorch_config.donated_buffer = False
        self.model = torch.compile(self.model, dynamic=True)
        self.is_compiled = True

    def _get_model(self):
        """Return model or model_ema for inference based on use_ema_for_eval."""
        if self.use_ema_for_eval and self.model_ema is not None:
            return self.model_ema
        return self.model

    def _update_ema(self):
        """Update EMA parameters: ema = decay * ema + (1 - decay) * model."""
        ema_decay = float(self.ema_decay)
        if self.ema_decay_scheduler is not None:
            ema_decay = self.ema_decay_scheduler.get_decay(
                current_epoch=self.current_epoch,
            )
        with torch.no_grad():
            for p_ema, p in zip(
                self.model_ema.parameters(),
                self.model.parameters(),
            ):
                p_ema.mul_(ema_decay).add_(p.data, alpha=1.0 - ema_decay)

        self.log(
            "ema/decay",
            ema_decay,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
        )

    def forward(self, x):
        pass

    # ------------------------------------------------------------------
    # Hooks for fine-tuning subclasses (e.g. scaffold-decoration).
    # Default implementations are no-ops so the base behaviour is unchanged.
    # ------------------------------------------------------------------

    def _node_loss_exclusion_mask(self, mols_t) -> torch.Tensor | None:
        """Optional per-node bool mask: True = drop this node from
        non-deletion / substitution / position losses.

        Used by scaffold fine-tuning to exclude scaffold atoms from losses
        that they can't meaningfully contribute to (since they are frozen
        during inference). Default ``None`` keeps all nodes.
        """
        return None

    def _apply_inference_edit_masks(
        self,
        mol_t,
        do_sub_a_probs: torch.Tensor,
        do_sub_e_probs: torch.Tensor,
        do_del_probs: torch.Tensor,
        num_ins_pred: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Optional hook to zero out edit probabilities at certain positions
        before they reach the integrator. Default returns inputs unchanged.

        Subclasses override to hard-zero del / sub-a / sub-e (scaffold-scaffold)
        probabilities at scaffold positions while leaving insertions intact so
        decorations can still grow.
        """
        return do_sub_a_probs, do_sub_e_probs, do_del_probs, num_ins_pred

    def shared_step(self, batch, batch_idx):
        self.model.set_training()
        # Define the training step logic here
        if self.vocab.atom_tokens is None:
            raise ValueError(
                "vocab.atom_tokens must be set before training. Call set_vocab() first."
            )

        mols_t, mols_1, ins_targets, t = batch

        # randomized self-conditioning with p = 0.5 during training
        is_random_self_conditioning = (torch.rand(1) > 0.5).item()

        ctx = self.cfg_guidance.build_ctx(ins_targets=ins_targets)
        overrides = self.cfg_guidance.build_overrides(mols_1, ctx)
        drop_masks = self.cfg_guidance.sample_drop_masks(
            batch_size=mols_t.num_graphs,
            device=self.device,
            training=self.training,
        )

        preds = self.model(
            mols_t,
            t.view(-1, 1),
            is_random_self_conditioning=is_random_self_conditioning,
            overrides=overrides,
            drop_masks=drop_masks,
        )
        # NOTE: we separate this from the forward pass because CFG acts on the logits before activations.
        self.model.apply_activations(preds)

        a_pred = preds["atom_type_head"]
        x_pred = preds["pos_head"]
        e_pred = preds["edge_type_head"]
        c_pred = preds["charge_head"]

        ins_rate_pred = preds["ins_rate_head"]
        gmm_pred_dict = preds["gmm_head"]

        do_del_head = preds["do_del_head"]
        do_sub_a_head = preds["do_sub_a_head"]
        do_sub_e_head = preds["do_sub_e_head"]

        # Mask for nodes that are scheduled for deletion — reused for x, c, and a losses.
        # Deletion nodes have their target type/charge frozen to the prior sample, so
        # supervising against them would bias predictions towards the prior distribution.
        to_delete_mask = mols_t.lambda_del > 0.0
        non_del_mask = ~to_delete_mask

        # Optional per-node exclusion (e.g. scaffold atoms in fine-tuning).
        # ``_excl_mask`` is True at nodes that should NOT contribute to non-del /
        # substitution / position losses; defaults to ``None`` (no exclusion).
        _excl_mask = self._node_loss_exclusion_mask(mols_t)
        if _excl_mask is not None:
            _excl_mask = _excl_mask.bool()
            non_del_mask = non_del_mask & ~_excl_mask

        # NOTE take loss on all non-deletion nodes
        c_loss, c_batch_mask = class_loss(
            c_pred,
            mols_1.c,
            self.charge_token_weights,
            non_del_mask,
            mols_t.batch,
            mols_t.num_graphs,
            reduction="none",
        )

        # 1. Handle substitutions

        #### Handle atom type substitutions
        # NOTE take loss on all nodes
        do_sub_a_loss = do_action_loss(
            do_sub_a_head,
            mols_t.lambda_a_sub,
            mols_t.batch,
            mols_t.num_graphs,
        )
        # NOTE take loss on all non-deletion & to-be-substituted nodes
        # del nodes have lambda_a_sub = 0 anyway, this is just for clarity
        to_sub_mask = (non_del_mask) & (mols_t.lambda_a_sub > 0.0)
        sub_a_class_loss, sub_a_batch_mask = class_loss(
            a_pred,
            mols_1.a,
            self.atom_type_weights,
            to_sub_mask,
            mols_t.batch,
            mols_t.num_graphs,
            reduction="none",
        )
        #### Handle edge type substitutions
        edge_infos = self.edge_aligner.align_edges(
            source_group=(mols_t.edge_index, [e_pred, do_sub_e_head]),
            target_group=(mols_1.edge_index, [mols_1.e, mols_t.lambda_e_sub]),
        )
        e_batch_triu = mols_t.batch[edge_infos["edge_index"][0][0]]
        e_pred_triu, do_sub_e_head_triu = edge_infos["edge_attr"][:2]
        e_target_triu, do_sub_e_target_triu = edge_infos["edge_attr"][2:]

        # Supervise e_pred on all non-deletion edges (same fix as sub_a_class_loss).
        # Edges where either endpoint is scheduled for deletion are excluded:
        # their targets are frozen to the source value, so supervising e_pred on them
        # would train it to predict prior edge types — the wrong signal.
        _src_idx = edge_infos["edge_index"][0][0]
        _dst_idx = edge_infos["edge_index"][0][1]
        non_del_edge_mask = ~to_delete_mask[_src_idx] & ~to_delete_mask[_dst_idx]
        if _excl_mask is not None:
            # Exclude edges where BOTH endpoints are excluded (matches inference
            # masking: scaffold-scaffold edges can never change).
            non_del_edge_mask = non_del_edge_mask & ~(
                _excl_mask[_src_idx] & _excl_mask[_dst_idx]
            )

        # NOTE As per EditFlow, only count class loss for edges that need modification
        do_sub_e_loss = do_action_loss(
            do_sub_e_head_triu,
            do_sub_e_target_triu,
            e_batch_triu,
            mols_t.num_graphs,
            reduction="none",
        )

        # NOTE take loss on all non-deletion & to-be-substituted edges
        to_sub_edge_mask = (non_del_edge_mask) & (do_sub_e_target_triu > 0.0)
        sub_e_class_loss, sub_e_batch_mask = class_loss(
            e_pred_triu,
            e_target_triu,
            self.edge_token_weights,
            to_sub_edge_mask,
            e_batch_triu,
            mols_t.num_graphs,
            reduction="none",
        )

        do_del_loss = torch.tensor(0.0, device=self.device)

        if self.n_atoms_strategy != "fixed":
            # Per-graph masks for insertion *edge* losses: only average over graphs that
            # have at least one supervised (non-del-filtered) edge. Using ins_batch_mask
            # from rate loss would include graphs with insertions but no edge targets, or
            # graphs whose ins edge rows are all-zero from unsorted_segment_mean.
            ins_e_batch_mask = torch.zeros(
                mols_t.num_graphs, dtype=torch.bool, device=self.device
            )
            ins_e_ii_batch_mask = torch.zeros(
                mols_t.num_graphs, dtype=torch.bool, device=self.device
            )

            # 2. Handle deletions (no class changes here!)
            do_del_loss = do_action_loss(
                do_del_head,
                mols_t.lambda_del,
                mols_t.batch,
                mols_t.num_graphs,
                reduction="none",
            )

            # 3. Handle insertions
            # Poisson NLL over all nodes (including those with zero insertions).
            ins_loss_gmm = torch.tensor(0.0, device=self.device)
            ins_loss_rate, ins_batch_mask = rate_loss(
                ins_rate_pred,
                mols_t.n_ins,
                mols_t.batch,
                mols_t.num_graphs,
                reduction="none",
            )

            # indices of nodes in mol_t that spawn/predict each insertion
            spawn_node_idx = ins_targets.spawn_node_idx

            if spawn_node_idx.numel() > 0:
                _ins_dtype = ins_rate_pred.dtype
                ins_loss_e = torch.zeros(
                    mols_t.num_graphs, 1, device=self.device, dtype=_ins_dtype
                )
                ins_loss_e_ii = torch.zeros(
                    mols_t.num_graphs, 1, device=self.device, dtype=_ins_dtype
                )

                # we must take the NLL for the closest nodes only
                gmm_dict_pred = {
                    "mu": gmm_pred_dict["mu"][spawn_node_idx],
                    "sigma": gmm_pred_dict["sigma"][spawn_node_idx],
                    "pi": gmm_pred_dict["pi"][spawn_node_idx],
                    "a_probs": gmm_pred_dict["a_probs"][spawn_node_idx],
                    "c_probs": gmm_pred_dict["c_probs"][spawn_node_idx],
                }

                gmm_loss, _ = typed_gmm_loss(
                    gmm_dict_pred,
                    ins_targets.x,
                    ins_targets.a,
                    ins_targets.c,
                    self.atom_type_weights,
                    self.charge_token_weights,
                    reduction="none",
                )

                ins_loss_gmm = gmm_loss.view(-1)

                # reduce the loss to per-graph level
                ins_loss_gmm = unsorted_segment_mean(
                    ins_loss_gmm.view(-1, 1),
                    mols_t.batch[spawn_node_idx],
                    mols_t.num_graphs,
                )

                # Compute insertion edge loss if available
                # Index spaces:
                # - ins_edge_spawn_idx / ins_edge_existing_idx index current-state nodes in mols_t
                # - ins_edge_ins_local_idx indexes inserted nodes in ins_targets (future nodes)
                # - ins_edge_types provides integer edge-class supervision targets
                spawn_idx = ins_targets.ins_edge_spawn_idx
                existing_idx = ins_targets.ins_edge_existing_idx

                if spawn_idx.numel() > 0 and existing_idx.numel() > 0:
                    ins_edge_non_del = (
                        non_del_mask[spawn_idx] & non_del_mask[existing_idx]
                    )
                    if ins_edge_non_del.any():
                        # Get edge predictions using the insertion edge head
                        ins_edge_logits = self.model.predict_insertion_edges(
                            mols_t=mols_t,
                            out_dict=preds,
                            spawn_node_idx=spawn_idx,
                            existing_node_idx=existing_idx,
                            # Use canonical future node tensors and gather per-edge inserted attrs.
                            # NOTE predict edge types for future nodes, not current nodes
                            # NOTE then at inference we will adjust to current noise level
                            ins_x=ins_targets.x[ins_targets.ins_edge_ins_local_idx],
                            ins_a=ins_targets.a[ins_targets.ins_edge_ins_local_idx],
                            ins_c=ins_targets.c[ins_targets.ins_edge_ins_local_idx],
                        )
                        if ins_edge_logits is not None and ins_edge_logits.numel() > 0:
                            # Reuse the shared class loss helper (same masking +
                            # per-graph averaging) to compute insertion edge loss.
                            ins_loss_e, ins_e_batch_mask = class_loss(
                                ins_edge_logits,
                                ins_targets.ins_edge_types,
                                self.edge_token_weights,
                                ins_edge_non_del,
                                mols_t.batch[spawn_idx],
                                mols_t.num_graphs,
                                reduction="none",
                            )

                # Compute ins -> ins edge loss
                spawn_src_ii = ins_targets.ins_to_ins_edge_spawn_src_idx
                spawn_dst_ii = ins_targets.ins_to_ins_edge_spawn_dst_idx
                if spawn_src_ii.numel() > 0:
                    ii_non_del = non_del_mask[spawn_src_ii] & non_del_mask[spawn_dst_ii]
                    if ii_non_del.any():
                        edges_ii_src = ins_targets.ins_to_ins_edge_src_local_idx
                        edges_ii_dst = ins_targets.ins_to_ins_edge_dst_local_idx
                        ins_ii_logits = self.model.predict_insertion_edges_ins_to_ins(
                            mols_t=mols_t,
                            out_dict=preds,
                            spawn_src_idx=spawn_src_ii,
                            ins_x_src=ins_targets.x[edges_ii_src],
                            ins_a_src=ins_targets.a[edges_ii_src],
                            ins_c_src=ins_targets.c[edges_ii_src],
                            ins_a_dst=ins_targets.a[edges_ii_dst],
                            ins_x_dst=ins_targets.x[edges_ii_dst],
                        )
                        if ins_ii_logits is not None and ins_ii_logits.numel() > 0:
                            # Reuse the shared class loss helper (same masking +
                            # per-graph averaging) to compute ins->ins edge loss.
                            ins_loss_e_ii, ins_e_ii_batch_mask = class_loss(
                                ins_ii_logits,
                                ins_targets.ins_to_ins_edge_types,
                                self.edge_token_weights,
                                ii_non_del,
                                mols_t.batch[spawn_src_ii],
                                mols_t.num_graphs,
                                reduction="none",
                            )

            else:
                # No insertion targets in this batch: keep scalar placeholders so the
                # loss accumulator's mask path stays consistent with ndim-0 dummy losses.
                ins_loss_e = torch.tensor(0.0, device=self.device)
                ins_loss_e_ii = torch.tensor(0.0, device=self.device)

        else:
            # NOTE ins_rate_head, edge_head, gmm_head unused, throws an error (unused_params)
            # NOTE therefore we add a dummy loss wrt. edge_head and gmm_head
            ins_rate_head_loss = sum(
                p.sum() for p in self.model.heads.heads["ins_rate_head"].parameters()
            )
            edge_head_loss = sum(p.sum() for p in self.model.ins_edge_head.parameters())
            gmm_head_loss = sum(p.sum() for p in self.model.ins_gmm_head.parameters())

            ins_loss_e = 0.0 * edge_head_loss
            ins_loss_e_ii = 0.0 * edge_head_loss
            ins_loss_gmm = 0.0 * gmm_head_loss
            ins_loss_rate = 0.0 * ins_rate_head_loss

            ins_batch_mask = torch.zeros(
                mols_t.num_graphs, dtype=torch.bool, device=self.device
            )
            ins_e_batch_mask = ins_batch_mask
            ins_e_ii_batch_mask = ins_batch_mask

        # 4. Calculate the flow matching loss
        # Only compute the loss for nodes that are not to be deleted
        # NOTE Edge case all deletes would lead to unused params error, but is highly unlikely
        x_loss = F.mse_loss(
            x_pred[non_del_mask], mols_1.x[non_del_mask], reduction="none"
        )
        x_loss = unsorted_segment_mean(
            x_loss, mols_t.batch[non_del_mask], mols_t.num_graphs
        )
        # Keep only graphs that have at least one node contributing to x_loss
        # (i.e., at least one node that is not marked for deletion).
        x_batch_mask = torch.zeros(
            mols_t.num_graphs, dtype=torch.bool, device=self.device
        )
        x_batch_mask[mols_t.batch[non_del_mask]] = True

        if not x_batch_mask.any():
            x_loss = 0.0 * sum(p.sum() for p in self.model.pos_head.parameters())

        loss_masks = {
            "x": x_batch_mask,
            "c": c_batch_mask,
            "ins_rate": ins_batch_mask,
            "ins_e": ins_e_batch_mask,
            "ins_e_ii": ins_e_ii_batch_mask,
            "ins_gmm": ins_batch_mask,
            "sub_a_class": sub_a_batch_mask,
            "sub_e_class": sub_e_batch_mask,
        }

        # 5. Combine, weight, and log all losses
        self.loss_accumulator.set_batch_losses(
            {
                "do_sub_a": do_sub_a_loss,
                "sub_a_class": sub_a_class_loss,
                "do_sub_e": do_sub_e_loss,
                "sub_e_class": sub_e_class_loss,
                "do_del": do_del_loss,
                "ins_rate": ins_loss_rate,
                "ins_gmm": ins_loss_gmm,
                "ins_e": ins_loss_e,
                "ins_e_ii": ins_loss_e_ii,
                "x": x_loss,
                "c": c_loss,
            },
            t=t,
            masks=loss_masks,
        )

        self.loss_accumulator.add_stats(
            {
                "n_ins": float((mols_t.lambda_ins > 0.0).sum().item()),
                "n_del": float((mols_t.lambda_del > 0.0).sum().item()),
            }
        )

        loss = self.loss_accumulator.total_loss()
        self.log_dict(self.loss_accumulator.log_dict(), prog_bar=False, logger=True)

        # ins_edge_head.unseen_node_embedding is exclusive to forward_ins_to_ins
        # (InsertionEdgeHead.forward never references it). On batches without a
        # single ins-to-ins edge — more likely with smaller effective batches
        # such as n_augmentations > 1 — the param ends up with .grad = None and
        # DDP rejects the step. Add a 0-weighted reference so it always lands
        # in the autograd graph.
        loss = loss + 0.0 * self.model.ins_edge_head.unseen_node_embedding.sum()

        return loss

    def training_step(self, batch, batch_idx):
        self.model_ema.eval()
        loss = self.shared_step(batch, batch_idx)
        self._maybe_log_per_loss_grad_norms()
        self.log("loss/train", loss.detach(), prog_bar=True, logger=True)
        return loss

    def _maybe_log_per_loss_grad_norms(self) -> None:
        """Log per-loss-term gradient norms at the configured cadence.

        Runs *before* Lightning calls ``loss.backward()``, leveraging
        ``retain_graph=True`` inside ``LossAccumulator.compute_grad_norms`` so
        the autograd graph survives for the actual backward pass. The
        ``.grad`` attributes on parameters are not touched, so this is safe to
        interleave with DDP's reducer (which only fires on
        ``loss.backward()``).
        """
        every = self.log_grad_norms_every_n_steps
        if every <= 0 or (self.global_step % every) != 0:
            return

        norms = self.loss_accumulator.compute_grad_norms(
            self.model.parameters(),
            granularity=self.grad_norms_granularity,
            use_weighted=False,
            apply_component_weights=False,
        )
        if not norms:
            return

        component_keys = {k for keys in self.LOSS_GROUPS.values() for k in keys}
        entries: dict[str, torch.Tensor] = {}
        for key, val in norms.items():
            prefix = "grad_norm" if key in component_keys else "grad_norm_group"
            entries[f"{prefix}/{key}"] = val.detach()

        self.log_dict(
            entries,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
        )

    def on_validation_epoch_start(self):
        # Distribution metrics accumulate histograms across the epoch; reset them
        # here so each validation cycle starts from a clean state.
        if self.distribution_metrics is not None:
            self.distribution_metrics.reset()

    def validation_step(self, batch, batch_idx):
        self.model_ema.eval()
        batched_mols = self.sample(batch, batch_idx, return_traj=False)

        # List of MoleculeData objects
        mols = batched_mols.to_data_list()
        del batched_mols

        # List of RDKit molecules or None
        rdkit_mols = []
        for mol in mols:
            try:
                rdkit_mols.append(
                    mol.to_rdkit_mol(
                        self.vocab.atom_tokens,
                        self.vocab.edge_tokens,
                        self.vocab.charge_tokens,
                    )
                )
            except Exception as e:
                print(f"Error converting molecule to RDKit: {e}")
                rdkit_mols.append(None)

        n_mols = len(mols)
        del mols

        # eval_metrics contains Python floats (not GPU tensors)
        eval_metrics = calc_metrics_(rdkit_mols, self.metrics)
        print(eval_metrics)

        self.log_dict(
            {f"val/{key}": value for key, value in eval_metrics.items()},
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            batch_size=n_mols,
        )

        # Accumulate marginal histograms across the epoch without resetting.
        if self.distribution_metrics is not None:
            self.distribution_metrics.update(rdkit_mols)

        try:
            # pb_metrics = calc_posebusters_metrics(rdkit_mols)
            # print(pb_metrics)
            pb_metrics = False
        except Exception as e:
            print(f"Error calculating PoseBusters metrics: {e}")
            pb_metrics = False

        if pb_metrics:
            self.log_dict(
                {f"posebusters/{key}": value for key, value in pb_metrics.items()},
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                batch_size=n_mols,
            )

    def on_validation_epoch_end(self):
        if self.distribution_metrics is None:
            return

        try:
            # Pooled (epoch-level) KL scalars. `compute()` handles the DDP
            # cross-process reduction internally and restores local state on exit.
            dist_results = self.distribution_metrics.compute()
            self.log_dict(
                {
                    f"val/{key}": (v.detach() if isinstance(v, torch.Tensor) else v)
                    for key, v in dist_results.items()
                },
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                sync_dist=False,
            )

            # Marginal plots. `sync_context()` is DDP-collective so *all* ranks
            # must enter the block, but only rank 0 reads the synced histograms
            # and pushes the figures through the logger (which handles step
            # alignment with the scalars logged above).
            with ExitStack() as stack:
                for m in self.distribution_metrics.values():
                    stack.enter_context(m.sync_context())

                if self.trainer.is_global_zero and hasattr(self.logger, "log_image"):
                    figures = build_marginal_plots(self.distribution_metrics)
                    for name, fig in figures.items():
                        self.logger.log_image(key=f"val/marginals/{name}", images=[fig])
                    if figures:
                        import matplotlib.pyplot as plt

                        for fig in figures.values():
                            plt.close(fig)
        except Exception as e:
            print(f"Error in validation-epoch-end distribution logging: {e}")
        finally:
            self.distribution_metrics.reset()

    def predict_step(self, batch, batch_idx):
        return_traj = bool(getattr(self, "predict_return_traj", True))

        # Optional graph-wise property override (e.g. QED delta for the
        # molecule-optimization fine-tune). When set, replaces the
        # dataset-supplied ``target_props`` on the input batch with a
        # user-chosen value broadcast across the batch.
        prop_override = getattr(self, "predict_target_props_override", None)
        if prop_override is not None:
            mol_t, mol_1 = batch
            if isinstance(prop_override, (int, float)):
                prop_override = torch.tensor(
                    [[float(prop_override)]], dtype=torch.float
                )
            elif not torch.is_tensor(prop_override):
                prop_override = torch.tensor(prop_override, dtype=torch.float)
            if prop_override.dim() == 1:
                prop_override = prop_override.unsqueeze(0)
            B = int(mol_t.num_graphs)
            if prop_override.shape[0] == 1 and B > 1:
                prop_override = prop_override.expand(B, -1).contiguous()
            mol_t.target_props = prop_override.to(
                device=mol_t.x.device, dtype=mol_t.x.dtype
            )
            batch = (mol_t, mol_1)

        gen_mols = self.sample(
            batch,
            batch_idx,
            return_traj=return_traj,
            overrides=getattr(self, "predict_overrides", None),
        )

        # do quick validity check of the generated molecules
        # take the last state of the trajectory and check validity
        last_mols_rdkit = [
            i[-1].to_rdkit_mol(
                self.vocab.atom_tokens, self.vocab.edge_tokens, self.vocab.charge_tokens
            )
            for i in gen_mols
        ]
        mol_is_valid = []
        for mol in last_mols_rdkit:
            if mol is None:
                mol_is_valid.append(False)
                continue
            try:
                mol_is_valid.append(
                    chemflowRD.mol_is_valid(mol, allow_charged=self.allow_charged)
                )
            except Exception as e:
                print(f"Error checking validity of molecule: {e}")
                mol_is_valid.append(False)

        valid_mols = [i for i, valid in zip(gen_mols, mol_is_valid) if valid]
        # Keep full trajectories for invalid molecules so failures can be inspected over time.
        invalid_mols = [i for i, valid in zip(gen_mols, mol_is_valid) if not valid]
        invalid_mols_rdkit = [
            i for i, valid in zip(last_mols_rdkit, mol_is_valid) if not valid
        ]

        return {
            "valid_mols": valid_mols,
            "invalid_mols": invalid_mols,
            "invalid_mols_rdkit": invalid_mols_rdkit,
        }

    def _broadcast_override(
        self,
        value: torch.Tensor | float | int | None,
        batch_size: int,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        """Coerce a scalar / tensor user-supplied override to ``[B, ...]``."""
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return torch.full((batch_size,), float(value), dtype=dtype, device=self.device)
        return value.to(self.device)

    def sample(
        self,
        batch,
        batch_idx,
        return_traj: bool = False,
        overrides: dict[str, torch.Tensor | float | int] | None = None,
    ):
        """Inference step for flow matching.

        Args:
            batch: Tuple ``(mol_t, mol_1)`` from the dataloader.
            batch_idx: Batch index.
            return_traj: Return per-molecule trajectories instead of the final state.
            overrides: Optional ``{signal_name: tensor | scalar}`` to inject a
                target value into a CFG signal.  Recognised names are the
                signal names registered on the model's ``cfg_embedding``
                (e.g. ``"n_atoms"``, ``"mw"``, ``"logp"``, ``"qed"``).
                Scalars are broadcast to ``[batch_size]``.

        Returns:
            If ``return_traj`` is False: a ``MoleculeBatch`` with the final state.
            If ``return_traj`` is True: a ``List[List[MoleculeData]]`` (one per molecule).
        """
        model = self._get_model()
        model.set_inference()
        if self.vocab.atom_tokens is None:
            raise ValueError(
                "vocab.atom_tokens must be set before prediction. "
                "Call set_vocab() first."
            )

        # Get batch size from the batch (assuming it's similar to training)
        mol_t, mol_1 = batch

        batch_size = mol_t.batch_size

        _ = mol_t.remove_com()

        # Start at t=0
        t = torch.zeros(batch_size, device=self.device)
        step_sizes = self.integrator.get_time_steps()

        def _scaled_cpu_snapshot(m):
            """Scale coords back to data units and move a detached copy to CPU."""
            snap = m.clone()
            snap.x = snap.x * self.distributions.coordinate_std
            return snap.to("cpu")

        # Trajectory storage (only used when return_traj=True).
        # Frames live on CPU to avoid linear GPU growth over the integration loop.
        mol_traj = [_scaled_cpu_snapshot(mol_t)] if return_traj else None

        # previous outputs for self-conditioning. none at the beginning
        preds = None

        # Build the per-signal override dict once: signal.extract(mols_1)
        # for each signal not already supplied by the user, broadcasted to
        # tensors on the right device.
        ctx = self.cfg_guidance.build_ctx()
        user_overrides: dict[str, torch.Tensor | None] = {}
        if overrides:
            dtype_by_signal = {
                "n_atoms": torch.long,
                "mw": torch.float,
                "logp": torch.float,
                "qed": torch.float,
                "properties": torch.float,
            }
            for name, v in overrides.items():
                user_overrides[name] = self._broadcast_override(
                    v, batch_size, dtype_by_signal.get(name, torch.float)
                )
        signal_overrides = self.cfg_guidance.build_overrides(
            mol_1, ctx, user_overrides=user_overrides
        )

        # Integration loop: integrate from t=0 to t=1
        for step_idx, step_size in enumerate(step_sizes):
            batch_id = mol_t.batch

            # Force contiguity on every step, avoid memory fragmentation.
            mol_t.edge_index = mol_t.edge_index.contiguous()
            if mol_t.batch is not None:
                mol_t.batch = mol_t.batch.contiguous()

            prev_preds = preds

            preds = self.cfg_guidance.guided_predict(
                model,
                mol_t,
                t,
                prev_preds,
                signal_overrides,
            )
            # Release previous step's preds as soon as self-conditioning no longer needs them.
            del prev_preds

            # Extract predictions
            x1_pred = preds["pos_head"]  # (N_total, D)

            # Pass logits directly to Categorical instead of softmax(...) -> probs.
            # The softmax-then-Categorical(probs=) path produces rows that miss the
            # simplex by ~1e-3 under fp16/bf16 (validate_args raises), and is also
            # numerically lossy compared to the internal log-sum-exp path.
            T_a = 1.0
            a_pred = torch.distributions.Categorical(
                logits=(preds["atom_type_head"] / T_a).float()
            ).sample()

            T_c = 1.0
            c_pred = torch.distributions.Categorical(
                logits=(preds["charge_head"] / T_c).float()
            ).sample()

            # NOTE: predictions are for full adj matrix.
            # NOTE: Will take triu and resymmetrize in integration step
            T_e = 1.0
            e_pred = torch.distributions.Categorical(
                logits=(preds["edge_type_head"] / T_e).float()
            ).sample()

            mol_1_pred = MoleculeBatch(
                x=x1_pred,
                a=a_pred,  # one-hot
                c=c_pred,  # one-hot
                e=e_pred,  # one-hot
                edge_index=mol_t.edge_index.clone(),
                batch=batch_id.clone(),
            )

            gmm_dict_pred = preds["gmm_head"]

            # Extract decision heads (scalar logits, will be passed through sigmoid in integration)
            do_sub_a_probs = torch.sigmoid(preds["do_sub_a_head"].view(-1))
            do_sub_e_probs = torch.sigmoid(preds["do_sub_e_head"].view(-1))
            do_del_probs = torch.sigmoid(preds["do_del_head"].view(-1))

            # Extract insertion rate prediction (number of insertions per node)
            num_ins_pred = preds["ins_rate_head"].view(-1)

            # If we fix the number of atoms, we will not use the jump process
            if self.n_atoms_strategy == "fixed":
                num_ins_pred = torch.zeros_like(num_ins_pred)
                do_del_probs = torch.zeros_like(do_del_probs)

            # Subclass hook: scaffold fine-tuning hard-zeros del / sub-a / sub-e
            # (scaffold-scaffold) probs at scaffold positions, leaving insertions
            # alone so decorations can still grow.
            do_sub_a_probs, do_sub_e_probs, do_del_probs, num_ins_pred = (
                self._apply_inference_edit_masks(
                    mol_t, do_sub_a_probs, do_sub_e_probs, do_del_probs, num_ins_pred
                )
            )

            # Get insertion edge head if available
            ins_edge_head = getattr(model, "ins_edge_head", None)
            h_latent = preds["h_latent"]

            # Integrate one step (edge prediction happens inside if head is provided)
            mol_t = self.integrator.integrate_step_gnn(
                mol_t=mol_t.clone(),
                mol_1_pred=mol_1_pred,
                do_sub_a_probs=do_sub_a_probs,
                do_sub_e_probs=do_sub_e_probs,
                do_del_probs=do_del_probs,
                num_ins_pred=num_ins_pred,
                ins_gmm_preds=gmm_dict_pred,
                t=t,
                dt=step_size,
                h_latent=h_latent,
                ins_edge_head=ins_edge_head,
            )

            # Drop per-step intermediates before the next forward pass so the peak
            # live-tensor count doesn't include two steps' worth of activations.
            del (
                mol_1_pred,
                x1_pred,
                a_pred,
                c_pred,
                e_pred,
                gmm_dict_pred,
                do_sub_a_probs,
                do_sub_e_probs,
                do_del_probs,
                num_ins_pred,
                h_latent,
            )

            # remove mean from xt for each batch
            _ = mol_t.remove_com()

            # Update time forward
            # Number of graphs stays constant (batch_size)
            t = t + step_size

            # Save state to trajectory on CPU; non-traj path only keeps the final frame.
            if return_traj:
                mol_traj.append(_scaled_cpu_snapshot(mol_t))

        del preds

        if return_traj:
            # rectify the trajectory such that we get a traj for each molecule
            traj_lists = [mol_traj_i.to_data_list() for mol_traj_i in mol_traj]

            traj_per_mol = [[] for _ in range(batch_size)]
            for i in range(batch_size):
                for mol_traj_i in traj_lists:
                    traj_per_mol[i].append(mol_traj_i[i])

            # Return results
            return traj_per_mol

        else:
            # Scale coordinates once for the final frame.
            mol_last = mol_t.clone()
            mol_last.x = mol_last.x * self.distributions.coordinate_std
            return mol_last

    def configure_optimizers(self):
        # Collect all parameters for the optimizer
        params = list(self.model.parameters())

        # Add learnable loss weight parameters if enabled
        if (
            self.loss_weight_wrapper.use_learnable
            and self.loss_weight_wrapper.learnable_wrapper is not None
        ):
            params.extend(list(self.loss_weight_wrapper.learnable_wrapper.parameters()))

        # Instantiate optimizer from config
        optimizer_cfg = dict(
            OmegaConf.to_container(self.optimizer_config.optimizer, resolve=True)
        )
        optimizer_cfg["params"] = params
        optimizer = hydra.utils.instantiate(optimizer_cfg)

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

    def on_before_optimizer_step(self, optimizer):
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        norms = grad_norm(self, norm_type=2)
        self.log_dict(norms, on_step=True, on_epoch=False)

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_closure,
        on_tpu=False,
        using_native_amp=False,
        using_lbfgs=False,
    ):
        optimizer_closure()
        optimizer.step()
        self._update_ema()


if __name__ == "__main__":
    # Instantiate the LightningModule and run the training loop
    model = LightningModuleRates()
    trainer = pl.Trainer()
    trainer.fit(model)
