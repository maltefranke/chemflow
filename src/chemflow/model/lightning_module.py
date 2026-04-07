import copy
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
import hydra
from chemflow.model.losses import typed_gmm_loss

from chemflow.utils.utils import EdgeAligner
from external_code.egnn import unsorted_segment_mean, unsorted_segment_sum

from chemflow.dataset.molecule_data import MoleculeBatch
from chemflow.dataset.vocab import Vocab, Distributions
from chemflow.utils.metrics import (
    calc_metrics_,
    calc_posebusters_metrics,
    init_metrics,
)
from lightning.pytorch.utilities import grad_norm
from chemflow.utils.loss_accumulation import LossAccumulator
from chemflow.model.learnable_loss import UnifiedWeightedLoss
from chemflow.utils.loss_weighing import (
    InverseSquaredTimeLossWeighting,
    ConstantTimeLossWeighting,
    ShiftedParabolaTimeLossWeighting,
)
from chemflow.utils import rdkit as chemflowRD
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
        cfg_adapter: DictConfig,
        time_dist: DictConfig,
        n_atoms_strategy: str = "fixed",
        ins_noise_scale: float = 0.5,
        use_learnable_loss_weights: bool = False,
        ema_decay: float = 0.999,
        ema_decay_scheduler: EMADecayScheduler | DictConfig | None = None,
        use_ema_for_eval: bool = True,
        use_time_weights: bool = False,
    ):
        super().__init__()

        self.vocab = vocab
        self.distributions = distributions
        self.loss_weight_distributions = loss_weight_distributions
        self.ins_noise_scale = ins_noise_scale
        self.ema_decay = float(ema_decay)
        if isinstance(ema_decay_scheduler, DictConfig):
            ema_decay_scheduler = hydra.utils.instantiate(ema_decay_scheduler)
        self.ema_decay_scheduler = ema_decay_scheduler
        self.use_ema_for_eval = use_ema_for_eval

        self.gmm_params = gmm_params
        self.n_atoms_strategy = n_atoms_strategy
        self.time_dist = hydra.utils.instantiate(time_dist)

        # Strip legacy "l_" prefix from config keys (YAML configs still use it)
        loss_weight_values = {
            k.removeprefix("l_"): float(v) for k, v in loss_weights.items()
        }
        self.optimizer_config = optimizer_config

        self.integrator = hydra.utils.instantiate(
            integrator, distributions=self.distributions
        )

        self.register_buffer("atom_type_weights", atom_type_weights)
        self.register_buffer("edge_token_weights", edge_token_weights)
        self.register_buffer("charge_token_weights", charge_token_weights)

        self.metrics, self.stability_metrics = init_metrics(
            target_n_atoms_distribution=self.loss_weight_distributions.n_atoms_distribution,
            atom_type_distribution=self.loss_weight_distributions.atom_type_distribution,
            edge_type_distribution=self.loss_weight_distributions.edge_type_distribution,
            atom_tokens=list(self.vocab.atom_tokens),
            edge_tokens=list(self.vocab.edge_tokens),
        )

        self.save_hyperparameters(ignore=["ema_decay_scheduler"])
        self.model = hydra.utils.instantiate(model)

        self.cfg_adapter = hydra.utils.instantiate(
            cfg_adapter,
            model=self.model,
            atom_tokens=list(self.vocab.atom_tokens),
        )

        # Exponential moving average of model parameters for stable inference
        self.model_ema = copy.deepcopy(self.model)
        for p in self.model_ema.parameters():
            p.requires_grad = False
        self.model_ema.eval()

        self.edge_aligner = EdgeAligner()

        self.loss_weight_wrapper = UnifiedWeightedLoss(
            manual_weights=loss_weight_values,
            component_keys=list(loss_weight_values.keys()),
            use_learnable=use_learnable_loss_weights,
        )

        if use_time_weights:
            time_weights = {
                "x": InverseSquaredTimeLossWeighting(clamp_max=100.0),
                "c": InverseSquaredTimeLossWeighting(clamp_max=100.0),
                "ins": lambda t: self.integrator.ins_schedule.rate(t).clamp(max=100.0),
                "del": lambda t: self.integrator.del_schedule.rate(t).clamp(max=100.0),
                "sub": lambda t: self.integrator.sub_schedule.rate(t).clamp(max=100.0),
            }
        else:
            time_weights = {}

        self.loss_accumulator = LossAccumulator(
            self.loss_weight_wrapper, self.LOSS_GROUPS, self.device, time_weights
        )

        self.is_compiled = False

    def compile(self):
        """Compile the model using torch.compile."""
        if self.is_compiled:
            return
        print("Compiling model...")
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

    def _reduce_loss(self, loss, reduction: str = "mean"):
        if reduction == "mean":
            return loss.mean()
        elif reduction == "sum":
            return loss.sum()
        elif reduction == "none":
            return loss
        else:
            raise ValueError(f"Invalid reduction: {reduction}")

    # fmt: off
    _LEGACY_BUFFER_KEYS = frozenset({
        # substitution losses
        "l_do_sub_a", "l_sub_a_class", "l_do_sub_e", "l_sub_e_class",
        # deletion and insertion losses
        "l_do_del", "l_do_ins", "l_ins_rate", "l_ins_gmm", "l_ins_e", "l_ins_e_ii",
        # move, charge losses
        "l_x", "l_c",
        # EMA counts for balancing BCE pos_weight in do_action_loss.
        "ema_n_ins", "ema_n_del",
    })
    # fmt: on

    def load_state_dict(self, state_dict, strict=True):
        """Load state dict with backward compatibility for old checkpoints."""
        state_dict = {
            k: v for k, v in state_dict.items() if k not in self._LEGACY_BUFFER_KEYS
        }
        has_ema = any(k.startswith("model_ema.") for k in state_dict)
        load_strict = strict and has_ema
        super().load_state_dict(state_dict, strict=load_strict)
        if not has_ema and self.model_ema is not None:
            self.model_ema.load_state_dict(self.model.state_dict(), strict=True)

    def forward(self, x):
        pass

    def safe_loss(self, loss):
        """Replace NaN or Inf losses with 0.0 to prevent training instability.

        Returns a zero loss that's connected to the computation graph to ensure
        gradients can be computed for gradient clipping.
        """
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Loss is NaN or Inf, skipping")
            # Create a zero loss connected to model parameters to maintain gradient flow
            # Use a small epsilon to avoid disconnecting from the graph
            dummy_param = next(iter(self.model.parameters()))
            return (
                torch.tensor(0.0, device=self.device, dtype=dummy_param.dtype)
                * dummy_param.sum()
                * 1e-8  # Small multiplier to keep graph connected but effectively zero
            )
        return loss

    def do_action_loss(
        self,
        do_action_pred,
        num_actions,
        batch,
        num_graphs,
        reduction: str = "mean",
    ):
        """Calculate the do action loss for the given do action predictions.
        Is used for all actions (substitutions, deletions).

        Args:
            do_action_pred: The predicted do action values.
            num_actions: The number of actions for each node.
            batch: The batch indices for each node.
            num_graphs: The number of graphs in the batch.
        """
        do_action = num_actions > 0.0

        dtype = do_action_pred.dtype
        pos_weight = torch.tensor(1.0, device=self.device, dtype=dtype)

        # NOTE: we use BCEWithLogitsLoss for scalar logit predictions
        # do_action_pred is shape (N,) with logits
        do_action_loss = F.binary_cross_entropy_with_logits(
            do_action_pred.view(-1),
            do_action.float().view(-1),
            pos_weight=pos_weight,
            reduction="none",
        )
        do_action_loss = unsorted_segment_mean(
            do_action_loss.view(-1, 1), batch, num_graphs
        )

        return self._reduce_loss(do_action_loss, reduction)

    def rate_loss(
        self, rate_pred, num_actions, batch, num_graphs, reduction: str = "mean"
    ):
        """
        Calculate the Poisson NLL rate loss for insertion predictions.

        Args:
            rate_pred: The predicted rate values, shape (N,) or (N, 1).
            num_actions: The integer number of actions for each node.
            batch: The batch indices for each node.
            num_graphs: The number of graphs in the batch.
        """

        # Apply Poisson NLL to ALL nodes, including those with zero insertions.
        rate_loss = F.poisson_nll_loss(
            rate_pred.view(-1),
            num_actions.view(-1),
            log_input=False,
            reduction="none",
            full=True,
        )

        # NOTE: first normalize by number of nodes / graphs
        # Otherwise, nodes with more atoms will have more weight by design
        rate_loss = unsorted_segment_mean(rate_loss.view(-1, 1), batch, num_graphs)

        # Every graph contributes since we supervise all nodes.
        batch_has_modified = torch.ones(
            num_graphs, 1, dtype=torch.bool, device=rate_pred.device
        )

        return self._reduce_loss(rate_loss, reduction), batch_has_modified

    def class_loss(
        self,
        class_pred,
        class_target,
        class_weights,
        do_action_mask,
        batch,
        num_graphs,
        reduction: str = "mean",
    ):
        """
        Calculate the class loss. Is not applied to masked nodes / edges.
        Concretely:
            c: class_loss is applied to all non-del nodes.
            a: class_loss is applied to all non-del & to-be-substituted nodes.
            e: class_loss is applied to all edges between non-del nodes that need to be substituted.
            ins_e : class_loss is applied to all edges between ins & non-del nodes.
            ins_e_ii: class_loss is applied to all edges between ins & ins nodes.
        """
        # 1. Safety check for empty actions to prevent NaNs or crashes
        if not do_action_mask.any():
            zero_loss = torch.zeros(num_graphs, 1, device=class_pred.device)
            zero_mask = torch.zeros(
                num_graphs, dtype=torch.bool, device=class_pred.device
            )
            return self._reduce_loss(zero_loss, reduction), zero_mask

        # 2. Compute CE ONLY on the nodes that actually need modification
        masked_loss = F.cross_entropy(
            class_pred[do_action_mask],
            class_target[do_action_mask],
            weight=class_weights,
            reduction="none",
        )

        # 3. Pool the masked losses per graph
        class_loss = unsorted_segment_mean(
            masked_loss.view(-1, 1), batch[do_action_mask], num_graphs
        )

        # 4. Determine which graphs had modifications
        batch_has_modified = (
            unsorted_segment_sum(
                do_action_mask.float().view(-1, 1),
                batch,
                num_graphs,
            )
            > 0
        )

        return self._reduce_loss(class_loss, reduction), batch_has_modified

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

        cfg_inputs = self.cfg_adapter.get_training_inputs(
            mols_t, mols_1, self.device, self.training
        )

        preds = self.model(
            mols_t,
            t.view(-1, 1),
            is_random_self_conditioning=is_random_self_conditioning,
            cfg_inputs=cfg_inputs,
        )

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

        # NOTE take loss on all non-deletion nodes
        c_loss, c_batch_mask = self.class_loss(
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
        do_sub_a_loss = self.do_action_loss(
            do_sub_a_head,
            mols_t.lambda_a_sub,
            mols_t.batch,
            mols_t.num_graphs,
        )
        # NOTE take loss on all non-deletion & to-be-substituted nodes
        # del nodes have lambda_a_sub = 0 anyway, this is just for clarity
        to_sub_mask = (non_del_mask) & (mols_t.lambda_a_sub > 0.0)
        sub_a_class_loss, sub_a_batch_mask = self.class_loss(
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

        # NOTE As per EditFlow, only count class loss for edges that need modification
        do_sub_e_loss = self.do_action_loss(
            do_sub_e_head_triu,
            do_sub_e_target_triu,
            e_batch_triu,
            mols_t.num_graphs,
            reduction="none",
        )

        # NOTE take loss on all non-deletion & to-be-substituted edges
        to_sub_edge_mask = (non_del_edge_mask) & (do_sub_e_target_triu > 0.0)
        sub_e_class_loss, sub_e_batch_mask = self.class_loss(
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
            do_del_loss = self.do_action_loss(
                do_del_head,
                mols_t.lambda_del,
                mols_t.batch,
                mols_t.num_graphs,
                reduction="none",
            )

            # 3. Handle insertions
            # Poisson NLL over all nodes (including those with zero insertions).
            ins_loss_gmm = torch.tensor(0.0, device=self.device)
            ins_loss_rate, ins_batch_mask = self.rate_loss(
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

                ins_loss_gmm = self._reduce_loss(ins_loss_gmm, "none")

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
                            ins_loss_e, ins_e_batch_mask = self.class_loss(
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
                        ins_ii_logits = self.model.predict_insertion_edges_ins_to_ins(
                            mols_t=mols_t,
                            out_dict=preds,
                            spawn_src_idx=spawn_src_ii,
                            ins_x_src=ins_targets.x[
                                ins_targets.ins_to_ins_edge_src_local_idx
                            ],
                            ins_a_src=ins_targets.a[
                                ins_targets.ins_to_ins_edge_src_local_idx
                            ],
                            ins_c_src=ins_targets.c[
                                ins_targets.ins_to_ins_edge_src_local_idx
                            ],
                            ins_a_dst=ins_targets.a[
                                ins_targets.ins_to_ins_edge_dst_local_idx
                            ],
                            ins_x_dst=ins_targets.x[
                                ins_targets.ins_to_ins_edge_dst_local_idx
                            ],
                        )
                        if ins_ii_logits is not None and ins_ii_logits.numel() > 0:
                            # Reuse the shared class loss helper (same masking +
                            # per-graph averaging) to compute ins->ins edge loss.
                            ins_loss_e_ii, ins_e_ii_batch_mask = self.class_loss(
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
                "n_ins": (mols_t.lambda_ins > 0.0).sum().float(),
                "n_del": (mols_t.lambda_del > 0.0).sum().float(),
            }
        )

        loss = self.safe_loss(self.loss_accumulator.total_loss())
        self.log_dict(self.loss_accumulator.log_dict(), prog_bar=False, logger=True)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        self.log("loss/train", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        batched_mols = self.sample(batch, batch_idx, return_traj=False)

        # List of MoleculeData objects
        mols = batched_mols.to_data_list()

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

        eval_metrics = calc_metrics_(rdkit_mols, self.metrics)
        print(eval_metrics)

        # Log validation metrics in val/ group
        val_metrics_dict = {f"val/{key}": value for key, value in eval_metrics.items()}
        self.log_dict(
            val_metrics_dict,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            batch_size=len(mols),
        )

        try:
            # pb_metrics = calc_posebusters_metrics(rdkit_mols)
            pb_metrics = False
        except Exception as e:
            print(f"Error calculating PoseBusters metrics: {e}")
            pb_metrics = False

        if pb_metrics:
            pb_metrics_dict = {
                f"posebusters/{key}": value for key, value in pb_metrics.items()
            }
            self.log_dict(
                pb_metrics_dict,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                batch_size=len(mols),
            )
            eval_metrics.update(pb_metrics)

        return eval_metrics

    def test_step(self, batch, batch_idx):
        # Define the test step logic here
        pass

    def predict_step(self, batch, batch_idx):
        return_traj = bool(getattr(self, "predict_return_traj", True))
        target_override = getattr(self, "predict_target_n_atoms_override", None)
        target_mw_override = getattr(self, "predict_target_mw_override", None)
        gen_mols = self.sample(
            batch,
            batch_idx,
            return_traj=return_traj,
            target_n_atoms_override=target_override,
            target_mw_override=target_mw_override,
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
                mol_is_valid.append(chemflowRD.mol_is_valid(mol))
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

    def _build_inference_cfg_inputs(
        self,
        mol_t,
        mol_1,
        batch_size,
        target_n_atoms_override=None,
        target_mw_override=None,
    ) -> dict:
        """Build the cfg_inputs dict for inference."""
        properties = self.cfg_adapter.extract_properties(mol_t)

        if target_n_atoms_override is not None:
            if isinstance(target_n_atoms_override, int):
                target_n_atoms = torch.full(
                    (batch_size,),
                    target_n_atoms_override,
                    dtype=torch.long,
                    device=self.device,
                )
            else:
                target_n_atoms = target_n_atoms_override.to(
                    self.device,
                )
        else:
            target_n_atoms = self.cfg_adapter.extract_target_n_atoms(
                mol_1,
            )

        if target_mw_override is not None:
            if isinstance(target_mw_override, (int, float)):
                target_mw = torch.full(
                    (batch_size,),
                    float(target_mw_override),
                    dtype=torch.float,
                    device=self.device,
                )
            else:
                target_mw = target_mw_override.to(self.device)
        else:
            target_mw = self.cfg_adapter.extract_target_mw(mol_1)

        return {
            "properties": properties,
            "property_drop_mask": None,
            "target_n_atoms": target_n_atoms,
            "natoms_drop_mask": None,
            "target_mw": target_mw,
            "mw_drop_mask": None,
        }

    def sample(
        self,
        batch,
        batch_idx,
        return_traj: bool = False,
        target_n_atoms_override: torch.Tensor | int | None = None,
        target_mw_override: torch.Tensor | float | None = None,
    ):
        """
        Inference step for flow matching.

        Args:
            batch: Batch of data (for now, we'll use batch size to determine number of graphs)
            batch_idx: Batch index
            return_traj: If True, return trajectory for each molecule. If False, return final state only.
            target_n_atoms_override: If provided, overrides the target n_atoms extracted from data.
                Shape (batch_size,) with integer atom counts.
            target_mw_override: If provided, overrides the target molecular weight.
                Shape (batch_size,) with MW in Daltons, or a scalar float.

        Returns:
            If return_traj is False: MoleculeBatch - final sampled molecules
            If return_traj is True: List[List[MoleculeData]] - trajectory for each molecule
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

        scaffold_mask = getattr(mol_t, 'scaffold_mask', None)

        # Start at t=0
        t = torch.zeros(batch_size, device=self.device)
        step_sizes = self.integrator.get_time_steps()

        # Trajectory storage
        mol_traj = [mol_t.clone()]

        # previous outputs for self-conditioning. none at the beginning
        preds = None

        cfg_inputs = self._build_inference_cfg_inputs(
            mol_t,
            mol_1,
            batch_size,
            target_n_atoms_override,
            target_mw_override,
        )

        # Integration loop: integrate from t=0 to t=1
        for step_size in step_sizes:
            batch_id = mol_t.batch

            prev_preds = preds

            if scaffold_mask is not None:
                mol_t.scaffold_mask = scaffold_mask

            preds = self.cfg_adapter.guided_predict(
                model,
                mol_t,
                t,
                prev_preds,
                cfg_inputs,
            )

            # Extract predictions
            x1_pred = preds["pos_head"]  # (N_total, D)

            a_pred = preds["atom_type_head"]  # (N_total, num_classes)
            T_a = 1.0
            a_pred = F.softmax(a_pred / T_a, dim=-1)
            a_pred = torch.distributions.Categorical(probs=a_pred).sample()

            c_pred = preds["charge_head"]  # (N_total, num_classes)
            T_c = 1.0
            c_pred = F.softmax(c_pred / T_c, dim=-1)
            c_pred = torch.distributions.Categorical(probs=c_pred).sample()

            # NOTE: predictions are for full adj matrix.
            # NOTE: Will take triu and resymmetrize in integration step
            T_e = 1.0
            e_pred = preds["edge_type_head"]
            e_pred = F.softmax(e_pred / T_e, dim=-1)
            e_pred = torch.distributions.Categorical(probs=e_pred).sample()

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
            do_sub_a_logits = preds["do_sub_a_head"]  # Shape (N, 1) or (N,)
            do_sub_e_logits = preds["do_sub_e_head"]  # Shape (E, 1) or (E,)
            do_del_logits = preds["do_del_head"]  # Shape (N, 1) or (N,)

            # Ensure logits are 1D
            if do_sub_a_logits.ndim > 1:
                do_sub_a_logits = do_sub_a_logits.squeeze(-1)
            if do_sub_e_logits.ndim > 1:
                do_sub_e_logits = do_sub_e_logits.squeeze(-1)
            if do_del_logits.ndim > 1:
                do_del_logits = do_del_logits.squeeze(-1)

            # Apply sigmoid to convert logits to probabilities
            do_sub_a_probs = torch.sigmoid(do_sub_a_logits)
            do_sub_e_probs = torch.sigmoid(do_sub_e_logits)
            do_del_probs = torch.sigmoid(do_del_logits)

            # Extract insertion rate prediction (number of insertions per node)
            ins_rate_output = preds["ins_rate_head"]
            num_ins_pred = ins_rate_output  # Shape (N, 1) or (N,)
            if num_ins_pred.ndim > 1:
                num_ins_pred = num_ins_pred.squeeze(-1)

            # if we fix the number of atoms, we will not use the jump process
            if self.n_atoms_strategy == "fixed":
                num_ins_pred = torch.zeros_like(num_ins_pred)
                do_del_probs = torch.zeros_like(do_del_probs)

            # Hard-zero all discrete rates for scaffold atoms so the scaffold
            # identity is guaranteed to be preserved during inference.
            # Positions are left free so the model can learn geometric relaxation.
            if scaffold_mask is not None:
                smask = scaffold_mask.bool()
                do_del_probs[smask] = 0.0
                do_sub_a_probs[smask] = 0.0
                num_ins_pred[smask] = 0.0
                sc_edge_mask = smask[mol_t.edge_index[0]] & smask[mol_t.edge_index[1]]
                do_sub_e_probs[sc_edge_mask] = 0.0

            # Get insertion edge head if available
            ins_edge_head = getattr(model, "ins_edge_head", None)

            # Integrate one step (edge prediction happens inside if head is provided)
            mol_t, keep_mask, n_insertions = self.integrator.integrate_step_gnn(
                mol_t=mol_t.clone(),
                mol_1_pred=mol_1_pred.clone(),
                do_sub_a_probs=do_sub_a_probs,
                do_sub_e_probs=do_sub_e_probs,
                do_del_probs=do_del_probs,
                num_ins_pred=num_ins_pred,
                ins_gmm_preds=gmm_dict_pred,
                t=t,
                dt=step_size,
                h_latent=preds.get("h_latent"),
                ins_edge_head=ins_edge_head,
            )

            if scaffold_mask is not None:
                if keep_mask is not None:
                    scaffold_mask = scaffold_mask[keep_mask]
                if n_insertions > 0:
                    scaffold_mask = torch.cat([
                        scaffold_mask,
                        torch.zeros(n_insertions, dtype=torch.long, device=scaffold_mask.device),
                    ])

            # remove mean from xt for each batch
            _ = mol_t.remove_com()

            # Update time forward
            # Number of graphs stays constant (batch_size)
            t = t + step_size

            # correct the coordinates
            mol_t_cloned = mol_t.clone()
            mol_t_cloned.x = mol_t_cloned.x * self.distributions.coordinate_std

            # Save  state to trajectory
            mol_traj.append(mol_t_cloned)

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
            return mol_traj[-1].clone()

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

    def on_train_epoch_start(self) -> None:
        """Push the current epoch into the train dataset wrapper for annealing."""
        try:
            dl = self.trainer.train_dataloader
            ds = getattr(dl, "dataset", None)
            if ds is not None and hasattr(ds, "set_epoch"):
                ds.set_epoch(self.current_epoch)
        except Exception:
            pass

    def on_before_optimizer_step(self, optimizer):
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        norms = grad_norm(self, norm_type=2)
        self.log_dict(norms)

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
        """
        Override for the optimizer_step hook.

        This function checks for NaN gradients after the backward pass
        (which is called inside optimizer_closure()) and skips the
        optimizer step if any are found.
        """

        # Run the closure.
        # This function is provided by Lightning and will:
        # 1. Clear gradients (optimizer.zero_grad())
        # 2. Compute the loss (call training_step)
        # 3. Run the backward pass (loss.backward())
        optimizer_closure()

        # --- Your custom logic starts here ---

        # Check if any gradients are NaN
        found_nan = False
        for param in self.parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                found_nan = True
                break

        if found_nan:
            # Log or print a warning
            print(
                f"WARNING: Skipping optimizer step at epoch {epoch}, batch {batch_idx} due to NaN gradients."
            )

            # We must manually zero the gradients again
            # because the new gradients from the next batch
            # will be *added* to the existing NaN gradients.
            optimizer.zero_grad(set_to_none=True)
        else:
            # No NaN gradients found, proceed with the optimizer step
            optimizer.step()
            # Update EMA after successful optimizer step
            self._update_ema()


if __name__ == "__main__":
    # Instantiate the LightningModule and run the training loop
    model = LightningModuleRates()
    trainer = pl.Trainer()
    trainer.fit(model)
