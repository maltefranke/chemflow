import copy
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
import hydra
from chemflow.model.losses import typed_gmm_loss

from chemflow.utils.utils import (
    token_to_index,
    compute_token_weights,
    EdgeAligner,
)
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
from chemflow.model.cfg import CFGAdapter
from chemflow.utils.loss_weighing import (
    InverseSquaredTimeLossWeighting,
    ConstantTimeLossWeighting,
    ShiftedParabolaTimeLossWeighting,
)
from chemflow.utils import rdkit as chemflowRD


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
        "ins": ["do_ins", "ins_rate", "ins_gmm", "ins_e", "ins_e_ii"],
        "x": ["x"],
        "c": ["c"],
        "budget": ["global_ins_budget", "global_del_budget"],
    }

    def __init__(
        self,
        model: DictConfig = None,
        integrator: DictConfig = None,
        loss_weights: DictConfig = None,
        optimizer_config: DictConfig = None,
        gmm_params: DictConfig = None,
        n_atoms_strategy: str = "fixed",
        type_loss_token_weights: str = "uniform",  # "uniform" or "training"
        cat_weighting: DictConfig = None,
        time_dist: DictConfig = None,
        vocab: Vocab = None,
        distributions: Distributions = None,
        loss_weight_distributions: Distributions = None,
        ins_noise_scale: float = 0.5,
        use_learnable_loss_weights: bool = False,
        edit_loss_ema_decay: float = 0.99,
        edit_loss_ema_eps: float = 1e-8,
        # Model EMA (exponential moving average of parameters for stable inference)
        ema_decay: float = 0.999,
        use_ema_for_eval: bool = True,
        # Classifier-free guidance parameters (property conditioning)
        cfg_dropout_prob: float = 0.0,
        cfg_guidance_scale: float = 0.0,
        # Classifier-free guidance parameters (target n_atoms conditioning)
        natoms_cfg_dropout_prob: float = 0.15,
        natoms_cfg_guidance_scale: float = 2.5,
    ):
        super().__init__()

        self.vocab = vocab
        self.distributions = distributions
        self.loss_weight_distributions = (
            loss_weight_distributions
            if loss_weight_distributions is not None
            else distributions
        )

        self.cat_weighting = cat_weighting
        self.ins_noise_scale = ins_noise_scale
        self.edit_loss_ema_decay = float(edit_loss_ema_decay)
        self.edit_loss_ema_eps = float(edit_loss_ema_eps)
        self.ema_decay = float(ema_decay)
        self.use_ema_for_eval = use_ema_for_eval

        self.gmm_params = gmm_params
        self.n_atoms_strategy = n_atoms_strategy
        self.type_loss_token_weights = type_loss_token_weights

        if time_dist is None:
            time_dist = DictConfig(
                {
                    "_target_": "torch.distributions.Uniform",
                    "low": 0.0,
                    "high": 1.0,
                }
            )
        self.time_dist = hydra.utils.instantiate(time_dist)

        # Set default loss weights if not provided
        if loss_weights is None:
            loss_weights = DictConfig(
                {
                    "do_sub_a": 1.0,
                    "sub_a_class": 1.0,
                    "do_sub_e": 1.0,
                    "sub_e_class": 1.0,
                    "do_del": 1.0,
                    "do_ins": 1.0,
                    "ins_rate": 1.0,
                    "ins_gmm": 1.0,
                    "ins_e": 1.0,
                    "x": 1.0,
                    "c": 1.0,
                    "global_ins_budget": 1.0,
                    "global_del_budget": 1.0,
                }
            )

        # Strip legacy "l_" prefix from config keys (YAML configs still use it)
        loss_weight_values = {
            k.removeprefix("l_"): float(v) for k, v in loss_weights.items()
        }

        # Set default optimizer config if not provided
        if optimizer_config is None:
            optimizer_config = DictConfig(
                {
                    "optimizer": {
                        "_target_": "torch.optim.Adam",
                        "lr": 1e-3,
                    },
                    "scheduler": {
                        "_target_": "torch.optim.lr_scheduler.ReduceLROnPlateau",
                        "mode": "min",
                        "factor": 0.85,
                        "patience": 10,
                    },
                    "monitor": "val_loss",
                }
            )
        self.optimizer_config = optimizer_config

        self.integrator = hydra.utils.instantiate(
            integrator, distributions=self.distributions
        )

        # Always compute token distribution weights for weighted cross-entropy loss
        atom_special_tokens = []

        atom_type_weights = compute_token_weights(
            token_list=self.vocab.atom_tokens,
            distribution=self.loss_weight_distributions.atom_type_distribution,
            special_token_names=atom_special_tokens,
            weight_alpha=self.cat_weighting.weight_alpha,
            type_loss_token_weights=self.type_loss_token_weights,
        )
        self.register_buffer("atom_type_weights", atom_type_weights)

        # Compute edge token distribution weights for weighted cross-entropy loss
        edge_special_tokens = ["<NO_BOND>"]

        edge_weights = compute_token_weights(
            token_list=self.vocab.edge_tokens,
            distribution=self.loss_weight_distributions.edge_type_distribution,
            special_token_names=edge_special_tokens,
            weight_alpha=self.cat_weighting.weight_alpha,
            type_loss_token_weights=self.type_loss_token_weights,
        )
        self.register_buffer("edge_token_weights", edge_weights)

        charge_weights = compute_token_weights(
            token_list=self.vocab.charge_tokens,
            distribution=self.loss_weight_distributions.charge_type_distribution,
            special_token_names=[],  # no special tokens for charges
            weight_alpha=self.cat_weighting.weight_alpha,
            type_loss_token_weights=self.type_loss_token_weights,
        )
        self.register_buffer("charge_token_weights", charge_weights)

        self.metrics, self.stability_metrics = init_metrics(
            target_n_atoms_distribution=self.distributions.n_atoms_distribution
        )

        # EMA counts for balancing BCE pos_weight in do_action_loss.
        self.register_buffer("ema_do_ins_pos", torch.tensor(1.0))
        self.register_buffer("ema_do_ins_neg", torch.tensor(1.0))
        self.register_buffer("ema_do_del_pos", torch.tensor(1.0))
        self.register_buffer("ema_do_del_neg", torch.tensor(1.0))

        self.save_hyperparameters()
        self.model = hydra.utils.instantiate(model)

        self.cfg_adapter = CFGAdapter(
            model=self.model,
            cfg_dropout_prob=cfg_dropout_prob,
            cfg_guidance_scale=cfg_guidance_scale,
            natoms_cfg_dropout_prob=natoms_cfg_dropout_prob,
            natoms_cfg_guidance_scale=natoms_cfg_guidance_scale,
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

        # TODO make these configurable
        time_weights = {
            "x": InverseSquaredTimeLossWeighting(clamp_max=100.0),
            "c": InverseSquaredTimeLossWeighting(clamp_max=100.0),
            "ins": lambda t: self.integrator.ins_schedule.rate(t).clamp(max=100.0),
            "del": lambda t: self.integrator.del_schedule.rate(t).clamp(max=100.0),
            "sub": lambda t: self.integrator.sub_schedule.rate(t).clamp(max=100.0),
            "budget": lambda t: self.integrator.ins_schedule.rate(t).clamp(max=100.0),
        }

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
        with torch.no_grad():
            for p_ema, p in zip(
                self.model_ema.parameters(),
                self.model.parameters(),
            ):
                p_ema.mul_(self.ema_decay).add_(p.data, alpha=1.0 - self.ema_decay)

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
        "l_do_del", "l_do_ins", "l_ins_rate", "l_ins_gmm", "l_ins_e",
        # move, charge and budget losses
        "l_x", "l_c", "l_global_ins_budget", "l_global_del_budget",
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
        ema_key: str | None = None,
        reduction: str = "mean",
    ):
        """Calculate the do action loss for the given do action predictions.
        Is used for all actions (substitutions, deletions, insertions).

        Args:
            do_action_pred: The predicted do action values.
            num_actions: The number of actions for each node.
            batch: The batch indices for each node.
            num_graphs: The number of graphs in the batch.
            ema_key: Optional key for EMA-balanced BCE `pos_weight`.
                Supported values are "ins" and "del".
        """
        do_action = num_actions > 0.0

        # calculate dynamic weighting for the rate loss
        dtype = do_action_pred.dtype
        num_total_actions = do_action.sum().detach().to(device=self.device, dtype=dtype)
        num_total = torch.tensor(
            float(do_action.shape[0]), device=self.device, dtype=dtype
        )
        num_no_action = (num_total - num_total_actions).clamp(min=0.0)

        if ema_key in {"ins", "del"}:
            pos_name = f"ema_do_{ema_key}_pos"
            neg_name = f"ema_do_{ema_key}_neg"
            ema_pos = getattr(self, pos_name)
            ema_neg = getattr(self, neg_name)

            if self.training:
                decay = self.edit_loss_ema_decay
                one_minus_decay = 1.0 - decay
                ema_pos.mul_(decay).add_(
                    num_total_actions.to(ema_pos.dtype) * one_minus_decay
                )
                ema_neg.mul_(decay).add_(
                    num_no_action.to(ema_neg.dtype) * one_minus_decay
                )

            pos_weight = ema_neg / ema_pos.clamp(min=self.edit_loss_ema_eps)
            pos_weight = pos_weight.to(device=self.device, dtype=dtype)
        else:
            pos_weight = num_no_action / num_total_actions.clamp(
                min=self.edit_loss_ema_eps
            )
            pos_weight = pos_weight.to(device=self.device, dtype=dtype)

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

        do_action = num_actions > 0.0
        do_action = do_action.view(-1)

        rate_loss = F.poisson_nll_loss(
            rate_pred[do_action].view(-1),
            num_actions[do_action].view(-1),
            log_input=False,
            reduction="none",
            full=True,
        )

        # NOTE: first normalize by number of nodes / graphs
        # Otherwise, nodes with more atoms will have more weight by design
        rate_loss = unsorted_segment_mean(
            rate_loss.view(-1, 1), batch[do_action], num_graphs
        )

        batch_has_modified = (
            unsorted_segment_sum(
                do_action.view(-1, 1),
                batch,
                num_graphs,
            )
            > 0
        )

        return self._reduce_loss(rate_loss, reduction), batch_has_modified

    def class_loss(
        self,
        class_pred,
        class_target,
        class_weights,
        num_actions,
        batch,
        num_graphs,
        reduction: str = "mean",
    ):
        """
        Calculate the class loss for the given class predictions and target classes.
        Is only used for atom type, edge type, and charge classes.

        Args:
            class_pred: The predicted class values.
            class_target: The target class values.
            class_weights: The weights for each class.
            num_actions: The number of actions for each node.
            batch: The batch indices for each node.
            num_graphs: The number of graphs in the batch.
        """
        do_action = num_actions > 0.0

        class_loss = F.cross_entropy(
            class_pred,
            class_target,
            weight=class_weights,
            reduction="none",
        )

        # NOTE As per EditFlow, only count class loss for nodes that need modification
        class_loss = unsorted_segment_mean(
            class_loss[do_action].view(-1, 1), batch[do_action], num_graphs
        )

        batch_has_modified = (
            unsorted_segment_sum(
                do_action.view(-1, 1),
                batch,
                num_graphs,
            )
            > 0
        )

        return self._reduce_loss(class_loss, reduction), batch_has_modified

    def global_budget_loss(
        self, graph_rate_pred, target_budget, reduction: str = "mean"
    ):
        """
        Calculate global budget loss from graph-level expected action counts.

        Uses Poisson NLL between predicted expected counts (sum of node-wise rates)
        and the target number of remaining actions per graph.

        Args:
            graph_rate_pred: Shape (num_graphs,) expected count per graph.
            target_budget: Shape (num_graphs,) target count per graph.

        Returns:
            loss: Scalar Poisson NLL loss averaged over graphs.
        """
        eps = 1e-8
        graph_rate_pred = graph_rate_pred.float().clamp(min=eps)
        target_budget = target_budget.float()
        loss = F.poisson_nll_loss(
            graph_rate_pred,
            target_budget,
            log_input=False,
            reduction=reduction,
            full=True,
        )
        return loss

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
            properties=cfg_inputs["properties"],
            property_drop_mask=cfg_inputs["cfg_drop_mask"],
            target_n_atoms=cfg_inputs["target_n_atoms"],
            natoms_drop_mask=cfg_inputs["natoms_drop_mask"],
        )

        a_pred = preds["atom_type_head"]
        x_pred = preds["pos_head"]
        e_pred = preds["edge_type_head"]
        c_pred = preds["charge_head"]

        do_ins_head = preds["do_ins_head"]
        ins_rate_pred = preds["ins_rate_head"]
        gmm_pred_dict = preds["gmm_head"]

        do_del_head = preds["do_del_head"]
        do_sub_a_head = preds["do_sub_a_head"]
        do_sub_e_head = preds["do_sub_e_head"]

        # NOTE charges not in mol_t, so we need to predict them for all atoms
        pred_all_atoms = torch.ones_like(mols_1.c, dtype=torch.bool)
        c_loss, c_batch_mask = self.class_loss(
            c_pred,
            mols_1.c,
            self.charge_token_weights,
            pred_all_atoms,
            mols_t.batch,
            mols_t.num_graphs,
            reduction="none",
        )

        # 1. Handle substitutions

        #### Handle atom type substitutions
        do_sub_a_loss = self.do_action_loss(
            do_sub_a_head, mols_t.lambda_a_sub, mols_t.batch, mols_t.num_graphs
        )
        sub_a_class_loss, sub_a_batch_mask = self.class_loss(
            a_pred,
            mols_1.a,
            self.atom_type_weights,
            mols_t.lambda_a_sub,
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

        # NOTE As per EditFlow, only count class loss for edges that need modification
        do_sub_e_loss = self.do_action_loss(
            do_sub_e_head_triu,
            do_sub_e_target_triu,
            e_batch_triu,
            mols_t.num_graphs,
            reduction="none",
        )
        sub_e_class_loss, sub_e_batch_mask = self.class_loss(
            e_pred_triu,
            e_target_triu,
            self.edge_token_weights,
            do_sub_e_target_triu,
            e_batch_triu,
            mols_t.num_graphs,
            reduction="none",
        )

        do_del_loss = torch.tensor(0.0, device=self.device)
        do_ins_loss = torch.tensor(0.0, device=self.device)

        if self.n_atoms_strategy != "fixed":
            # TODO maybe we should weight the deletion loss and insertion loss by their number of deletions and insertions?
            # TODO e.g. w = n_del / (n_del + n_ins)
            # 2. Handle deletions (no class changes here!)
            do_del_loss = self.do_action_loss(
                do_del_head,
                mols_t.lambda_del,
                mols_t.batch,
                mols_t.num_graphs,
                ema_key="del",
                reduction="none",
            )

            # 3. Handle insertions
            do_ins_loss = self.do_action_loss(
                do_ins_head,
                mols_t.lambda_ins,
                mols_t.batch,
                mols_t.num_graphs,
                ema_key="ins",
                reduction="none",
            )

            # Count number of insertions per node
            # Initialize with zeros for all nodes
            ins_loss_gmm = torch.tensor(0.0, device=self.device)
            ins_loss_e = torch.tensor(0.0, device=self.device)
            ins_loss_rate = torch.tensor(0.0, device=self.device)

            # indices of nodes in mol_t that spawn/predict each insertion
            spawn_node_idx = ins_targets.spawn_node_idx
            ins_batch_mask = torch.zeros(
                mols_t.num_graphs, dtype=torch.bool, device=self.device
            )

            if spawn_node_idx.numel() > 0:
                # Count how many insertions each node spawns
                num_inserts_per_node = mols_t.n_ins

                # mask indicating which nodes are focal for insertion
                ins_loss_rate, ins_batch_mask = self.rate_loss(
                    ins_rate_pred,
                    num_inserts_per_node,
                    mols_t.batch,
                    mols_t.num_graphs,
                    reduction="none",
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
                        ins_loss_e = F.cross_entropy(
                            ins_edge_logits,
                            ins_targets.ins_edge_types,
                            weight=self.edge_token_weights,
                            reduction="none",
                        )

                        # NOTE All inserted edges will have a non-zero target rate
                        # NOTE Therefore no more filtering needed

                        # reduce the loss to per-graph level
                        ins_loss_e = unsorted_segment_mean(
                            ins_loss_e.view(-1, 1),
                            mols_t.batch[spawn_idx],
                            mols_t.num_graphs,
                        )

                        ins_loss_e = self._reduce_loss(ins_loss_e, "none")

                # Compute ins -> ins edge loss
                ins_loss_e_ii = torch.tensor(0.0, device=self.device)
                spawn_src_ii = ins_targets.ins_to_ins_edge_spawn_src_idx
                if spawn_src_ii.numel() > 0:
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
                        ins_loss_e_ii = F.cross_entropy(
                            ins_ii_logits,
                            ins_targets.ins_to_ins_edge_types,
                            weight=self.edge_token_weights,
                            reduction="none",
                        )
                        ins_loss_e_ii = unsorted_segment_mean(
                            ins_loss_e_ii.view(-1, 1),
                            mols_t.batch[spawn_src_ii],
                            mols_t.num_graphs,
                        )
                        ins_loss_e_ii = self._reduce_loss(ins_loss_e_ii, "none")

            else:
                # NOTE ins_rate_head,edge_head, gmm_head unused, throws an error (unused_params)
                # NOTE therefore we add a dummy loss wrt. edge_head and gmm_head
                ins_rate_head_loss = sum(
                    p.sum() for p in self.model.ins_rate_head.parameters()
                )
                edge_head_loss = sum(
                    p.sum() for p in self.model.ins_edge_head.parameters()
                )
                gmm_head_loss = sum(
                    p.sum() for p in self.model.ins_gmm_head.parameters()
                )

                ins_loss_e = 0.0 * edge_head_loss
                ins_loss_e_ii = 0.0 * edge_head_loss
                ins_loss_gmm = 0.0 * gmm_head_loss
                ins_loss_rate = 0.0 * ins_rate_head_loss

        # 4. Calculate the flow matching loss
        # Only compute the loss for nodes that are not to be deleted
        # NOTE Edge case all deletes would lead to unused params error, but is highly unlikely
        to_delete_mask = mols_t.lambda_del > 0.0
        x_loss = F.mse_loss(
            x_pred[~to_delete_mask], mols_1.x[~to_delete_mask], reduction="none"
        )
        x_loss = unsorted_segment_mean(
            x_loss, mols_t.batch[~to_delete_mask], mols_t.num_graphs
        )
        # Keep only graphs that have at least one node contributing to x_loss
        # (i.e., at least one node that is not marked for deletion).
        x_batch_mask = torch.zeros(
            mols_t.num_graphs, dtype=torch.bool, device=self.device
        )
        x_batch_mask[mols_t.batch[~to_delete_mask]] = True

        if not x_batch_mask.any():
            x_loss = 0.0 * sum(p.sum() for p in self.model.pos_head.parameters())

        # 4.5 Compute global budget losses from node-wise rates.
        # Target: n_ins_missing and n_del_missing computed during interpolation.
        global_ins_budget_loss = torch.tensor(0.0, device=self.device)
        global_del_budget_loss = torch.tensor(0.0, device=self.device)

        if self.n_atoms_strategy != "fixed":
            ins_budget_target = mols_t.n_ins_missing
            del_budget_target = mols_t.n_del_missing

            # Insertion node-wise rates: directly use the predicted rates.
            ins_rate_node = ins_rate_pred.view(-1)
            ins_graph_expected = unsorted_segment_sum(
                ins_rate_node.view(-1, 1),
                mols_t.batch,
                mols_t.num_graphs,
            ).view(-1)

            # Deletion node-wise rate: one deletion max per node -> Bernoulli expectation.
            del_node_expected = torch.sigmoid(do_del_head.view(-1))
            del_graph_expected = unsorted_segment_sum(
                del_node_expected.view(-1, 1),
                mols_t.batch,
                mols_t.num_graphs,
            ).view(-1)

            global_ins_budget_loss = self.global_budget_loss(
                ins_graph_expected,
                ins_budget_target,
                reduction="none",
            )
            global_del_budget_loss = self.global_budget_loss(
                del_graph_expected,
                del_budget_target,
                reduction="none",
            )

        else:
            # Keep graph connectivity when the global budget terms are disabled.
            global_ins_budget_loss = 0.0 * (do_ins_head.sum() + ins_rate_pred.sum())
            global_del_budget_loss = 0.0 * do_del_head.sum()

        loss_masks = {
            "x": x_batch_mask,
            "c": c_batch_mask,
            "ins_rate": ins_batch_mask,
            "ins_e": ins_batch_mask,
            "ins_e_ii": ins_batch_mask,
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
                "do_ins": do_ins_loss,
                "ins_rate": ins_loss_rate,
                "ins_gmm": ins_loss_gmm,
                "ins_e": ins_loss_e,
                "ins_e_ii": ins_loss_e_ii,
                "x": x_loss,
                "c": c_loss,
                "global_ins_budget": global_ins_budget_loss,
                "global_del_budget": global_del_budget_loss,
            },
            t=t,
            masks=loss_masks,
        )

        self.loss_accumulator.add_stats(
            {
                "n_ins": (mols_t.lambda_ins > 0.0).sum().float(),
                "n_del": (mols_t.lambda_del > 0.0).sum().float(),
                "ema_do_ins_pos": self.ema_do_ins_pos,
                "ema_do_ins_neg": self.ema_do_ins_neg,
                "ema_do_del_pos": self.ema_do_del_pos,
                "ema_do_del_neg": self.ema_do_del_neg,
            }
        )

        if self.n_atoms_strategy != "fixed":
            self.loss_accumulator.add_stats(
                {
                    "ins_budget_pred_mean": ins_graph_expected.mean(),
                    "del_budget_pred_mean": del_graph_expected.mean(),
                    "ins_budget_target_mean": ins_budget_target.float().mean(),
                    "del_budget_target_mean": del_budget_target.float().mean(),
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
        gen_mols = self.sample(
            batch,
            batch_idx,
            return_traj=return_traj,
            target_n_atoms_override=target_override,
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

    def sample(
        self,
        batch,
        batch_idx,
        return_traj: bool = False,
        target_n_atoms_override: torch.Tensor | int | None = None,
    ):
        """
        Inference step for flow matching.

        Args:
            batch: Batch of data (for now, we'll use batch size to determine number of graphs)
            batch_idx: Batch index
            return_traj: If True, return trajectory for each molecule. If False, return final state only.
            target_n_atoms_override: If provided, overrides the target n_atoms extracted from data.
                Shape (batch_size,) with integer atom counts.

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

        # Start at t=0
        t = torch.zeros(batch_size, device=self.device)
        step_sizes = self.integrator.get_time_steps()

        # Trajectory storage
        mol_traj = [mol_t.clone()]

        # previous outputs for self-conditioning. none at the beginning
        preds = None

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
                target_n_atoms = target_n_atoms_override.to(self.device)
        else:
            target_n_atoms = self.cfg_adapter.extract_target_n_atoms(mol_1)

        # Integration loop: integrate from t=0 to t=1
        for step_size in step_sizes:
            batch_id = mol_t.batch

            prev_preds = preds

            preds = self.cfg_adapter.guided_predict(
                model,
                mol_t,
                t,
                prev_preds,
                properties,
                target_n_atoms,
            )

            # Extract predictions
            x1_pred = preds["pos_head"]  # (N_total, D)

            a_pred = preds["atom_type_head"]  # (N_total, num_classes)
            a_pred = F.softmax(a_pred, dim=-1)
            a_pred = torch.distributions.Categorical(probs=a_pred).sample()

            c_pred = preds["charge_head"]  # (N_total, num_classes)
            c_pred = F.softmax(c_pred, dim=-1)
            c_pred = torch.distributions.Categorical(probs=c_pred).sample()

            # NOTE: predictions are for full adj matrix.
            # NOTE: Will take triu and resymmetrize in integration step
            e_pred = preds["edge_type_head"]
            e_pred = F.softmax(e_pred, dim=-1)
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
            do_ins_logits = preds["do_ins_head"]  # Shape (N, 1) or (N,)

            # Ensure logits are 1D
            if do_sub_a_logits.ndim > 1:
                do_sub_a_logits = do_sub_a_logits.squeeze(-1)
            if do_sub_e_logits.ndim > 1:
                do_sub_e_logits = do_sub_e_logits.squeeze(-1)
            if do_del_logits.ndim > 1:
                do_del_logits = do_del_logits.squeeze(-1)
            if do_ins_logits.ndim > 1:
                do_ins_logits = do_ins_logits.squeeze(-1)

            # Apply sigmoid to convert logits to probabilities
            do_sub_a_probs = torch.sigmoid(do_sub_a_logits)
            do_sub_e_probs = torch.sigmoid(do_sub_e_logits)
            do_del_probs = torch.sigmoid(do_del_logits)
            do_ins_probs = torch.sigmoid(do_ins_logits)

            # Extract insertion rate prediction (number of insertions per node)
            ins_rate_output = preds["ins_rate_head"]
            num_ins_pred = ins_rate_output  # Shape (N, 1) or (N,)
            if num_ins_pred.ndim > 1:
                num_ins_pred = num_ins_pred.squeeze(-1)

            # if we fix the number of atoms, we will not use the jump process
            if self.n_atoms_strategy == "fixed":
                num_ins_pred = torch.zeros_like(num_ins_pred)
                do_del_probs = torch.zeros_like(do_del_probs)

            # Get insertion edge head if available
            ins_edge_head = getattr(model, "ins_edge_head", None)

            # Integrate one step (edge prediction happens inside if head is provided)
            # Note: do_ins is computed inside integrate_step_gnn from num_ins_pred or global_ins_budget
            mol_t = self.integrator.integrate_step_gnn(
                mol_t=mol_t.clone(),
                mol_1_pred=mol_1_pred.clone(),
                do_sub_a_probs=do_sub_a_probs,
                do_sub_e_probs=do_sub_e_probs,
                do_del_probs=do_del_probs,
                do_ins_probs=do_ins_probs,
                num_ins_pred=num_ins_pred,
                ins_gmm_preds=gmm_dict_pred,
                t=t,
                dt=step_size,
                h_latent=preds.get("h_latent"),
                ins_edge_head=ins_edge_head,
            )

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
        
        final_mol = mol_1_pred.clone()
        final_mol.remove_com()
        final_mol.x = final_mol.x * self.distributions.coordinate_std
        mol_traj.append(final_mol)
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
