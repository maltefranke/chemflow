import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
import hydra
from chemflow.losses import typed_gmm_loss
from chemflow.losses import gmm_loss as untyped_gmm_loss

from chemflow.utils import (
    token_to_index,
    compute_token_weights,
    EdgeAligner,
    validate_no_cross_batch_edges,
)
from external_code.egnn import unsorted_segment_mean, unsorted_segment_sum

from chemflow.dataset.molecule_data import MoleculeBatch
from chemflow.dataset.vocab import Vocab, Distributions
from chemflow.metrics import calc_metrics_, init_metrics
from lightning.pytorch.utilities import grad_norm
from chemflow.model.learnable_loss import UnifiedWeightedLoss


class LightningModuleRates(pl.LightningModule):
    def __init__(
        self,
        model: DictConfig = None,
        integrator: DictConfig = None,
        loss_weights: DictConfig = None,
        optimizer_config: DictConfig = None,
        gmm_params: DictConfig = None,
        n_atoms_strategy: str = "fixed",
        ins_rate_strategy: str = "poisson",  # "poisson" or "classification"
        type_loss_token_weights: str = "uniform",  # "uniform" or "training"
        cat_weighting: DictConfig = None,
        time_dist: DictConfig = None,
        vocab: Vocab = None,
        distributions: Distributions = None,
        ins_noise_scale: float = 0.5,
        use_learnable_loss_weights: bool = False,
    ):
        super().__init__()

        self.vocab = vocab
        self.distributions = distributions

        self.cat_weighting = cat_weighting
        self.ins_noise_scale = ins_noise_scale

        self.gmm_params = gmm_params
        self.n_atoms_strategy = n_atoms_strategy
        self.ins_rate_strategy = ins_rate_strategy
        self.type_loss_token_weights = type_loss_token_weights

        if self.cat_weighting.cat_strategy == "mask":
            self.atom_mask_index = token_to_index(self.vocab.atom_tokens, "<MASK>")
            self.edge_mask_index = token_to_index(self.vocab.edge_tokens, "<MASK>")
        else:
            self.atom_mask_index = None
            self.edge_mask_index = None

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
                    "l_sub_a_class": 1.0,
                    # "l_sub_c_rate": 1.0,
                    # "l_sub_c_class": 1.0,
                    "do_sub_e": 1.0,
                    "l_sub_e_class": 1.0,
                    "do_del": 1.0,
                    "do_ins": 1.0,
                    "ins_rate": 1.0,
                    "l_ins_gmm": 1.0,
                    "l_ins_e": 1.0,
                    "l_x": 1.0,
                    "l_c": 1.0,
                }
            )

        # register the loss weights as buffers
        for key, value in loss_weights.items():
            self.register_buffer(key, torch.tensor(value))

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
        if self.cat_weighting.cat_strategy == "mask":
            atom_special_tokens.append("<MASK>")

        atom_type_weights = compute_token_weights(
            token_list=self.vocab.atom_tokens,
            distribution=self.distributions.atom_type_distribution,
            special_token_names=atom_special_tokens,
            weight_alpha=self.cat_weighting.weight_alpha,
            type_loss_token_weights=self.type_loss_token_weights,
        )
        self.register_buffer("atom_type_weights", atom_type_weights)

        # Compute edge token distribution weights for weighted cross-entropy loss
        edge_special_tokens = ["<NO_BOND>"]
        if self.cat_weighting.cat_strategy == "mask":
            edge_special_tokens.append("<MASK>")

        edge_weights = compute_token_weights(
            token_list=self.vocab.edge_tokens,
            distribution=self.distributions.edge_type_distribution,
            special_token_names=edge_special_tokens,
            weight_alpha=self.cat_weighting.weight_alpha,
            type_loss_token_weights=self.type_loss_token_weights,
        )
        self.register_buffer("edge_token_weights", edge_weights)

        charge_weights = compute_token_weights(
            token_list=self.vocab.charge_tokens,
            distribution=self.distributions.charge_type_distribution,
            special_token_names=[],  # no special tokens for charges
            weight_alpha=self.cat_weighting.weight_alpha,
            type_loss_token_weights=self.type_loss_token_weights,
        )
        self.register_buffer("charge_token_weights", charge_weights)

        self.metrics, self.stability_metrics = init_metrics()

        self.save_hyperparameters()
        self.model = hydra.utils.instantiate(model)

        self.edge_aligner = EdgeAligner()

        # Initialize unified loss weight wrapper with dictionary structure
        manual_weights = {
            "do_sub_a": float(self.l_do_sub_a),
            "sub_a_class": float(self.l_sub_a_class),
            "do_sub_e": float(self.l_do_sub_e),
            "sub_e_class": float(self.l_sub_e_class),
            "do_del": float(self.l_do_del),
            "do_ins": float(self.l_do_ins),
            "ins_rate": float(self.l_ins_rate),
            "ins_gmm": float(self.l_ins_gmm),
            "ins_e": float(self.l_ins_e),
            "x": float(self.l_x),
            "c": float(self.l_c),
            # Global budget prediction losses
            "global_ins_budget": float(getattr(self, "l_global_ins_budget", 1.0)),
            "global_del_budget": float(getattr(self, "l_global_del_budget", 1.0)),
        }
        self.loss_weight_wrapper = UnifiedWeightedLoss(
            manual_weights=manual_weights,
            component_keys=manual_weights.keys(),
            use_learnable=use_learnable_loss_weights,
        )

        self.is_compiled = False

    def compile(self):
        """Compile the model using torch.compile."""
        if self.is_compiled:
            return
        print("Compiling model...")
        self.model = torch.compile(self.model, dynamic=True)
        self.is_compiled = True

    def forward(self, x):
        # Define the forward pass of your model here
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

    def do_action_loss(self, do_action_pred, num_actions, batch, num_graphs):
        """Calculate the do action loss for the given do action predictions.
        Is used for all actions (substitutions, deletions, insertions).

        Args:
            do_action_pred: The predicted do action values.
            num_actions: The number of actions for each node.
            batch: The batch indices for each node.
            num_graphs: The number of graphs in the batch.
        """
        do_action = num_actions > 0.0

        # calculate dynamic weighting for the rate loss
        num_total_actions = do_action.sum()
        num_total = do_action.shape[0]
        num_no_action = num_total - num_total_actions

        # TODO we could add an exponential moving averate of the pos weight for stability
        pos_weight = num_no_action / num_total_actions if num_total_actions > 0 else 1.0
        pos_weight = torch.tensor(pos_weight, device=self.device)

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

        do_action_loss = do_action_loss.mean()

        return do_action_loss

    def rate_loss(self, rate_pred, num_actions, batch, num_graphs):
        """
        Calculate the rate loss for the given rate predictions and number of actions.
        Is only used for insertions which can have multiple insertions per node.
        For the other actions, we use the do_action_loss.

        Supports two strategies:
        - "poisson": Poisson NLL loss for regression (rate_pred is scalar)
        - "classification": Cross-entropy loss (rate_pred is logits over classes)

        Args:
            rate_pred: The predicted rate values (scalar for poisson, logits for classification).
            num_actions: The integer number of actions for each node.
            batch: The batch indices for each node.
            num_graphs: The number of graphs in the batch.
        """

        do_action = num_actions > 0.0
        do_action = do_action.view(-1)

        if self.ins_rate_strategy == "classification":
            # Classification: use cross-entropy loss
            # rate_pred shape: (N, num_classes)
            # num_actions needs to be converted to class indices (clamped to valid range)
            num_classes = rate_pred.shape[-1]
            targets = num_actions.long().view(-1).clamp(0, num_classes - 1)

            # Only apply loss for nodes that have actions
            rate_loss = F.cross_entropy(
                rate_pred[do_action],
                targets[do_action],
                reduction="none",
            )
        else:
            # Poisson: use Poisson NLL loss (original behavior)
            # Only apply rate loss for nodes that have actions.
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
        rate_loss = rate_loss[batch_has_modified].mean()

        return rate_loss

    def class_loss(
        self, class_pred, class_target, class_weights, num_actions, batch, num_graphs
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

        class_loss = class_loss[batch_has_modified].mean()

        return class_loss

    def global_budget_loss(self, budget_pred, target_budget, num_graphs):
        """
        Calculate the global budget loss for graph-level predictions.

        This is a cross-entropy classification loss where we predict the total
        number of insertions/deletions remaining to reach the target size.

        Args:
            budget_pred: Shape (num_graphs, num_classes) - logits for budget prediction
            target_budget: Shape (num_graphs,) - target number of insertions/deletions
            num_graphs: The number of graphs in the batch.

        Returns:
            loss: Scalar cross-entropy loss averaged over graphs
        """
        # Clamp target to valid range [0, num_classes - 1]
        num_classes = budget_pred.shape[-1]
        target_budget = target_budget.long().clamp(0, num_classes - 1)

        # Cross-entropy loss
        loss = F.cross_entropy(
            budget_pred,
            target_budget,
            reduction="mean",
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

        # Validate: check for cross-batch edges in training batch
        validate_no_cross_batch_edges(
            mols_t.edge_index, mols_t.batch, "training shared_step mols_t"
        )
        validate_no_cross_batch_edges(
            mols_1.edge_index, mols_1.batch, "training shared_step mols_1"
        )

        # randomized self-conditioning with p = 0.5 during training
        is_random_self_conditioning = (torch.rand(1) > 0.5).item()

        preds = self.model(
            mols_t,
            t.view(-1, 1),
            is_random_self_conditioning=is_random_self_conditioning,
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

        # Global budget predictions (graph-level classifiers)
        global_ins_budget_pred = preds.get("global_ins_budget_head", None)
        global_del_budget_pred = preds.get("global_del_budget_head", None)

        # NOTE charges not in mol_t, so we need to predict them for all atoms
        pred_all_atoms = torch.ones_like(mols_1.c, dtype=torch.bool)
        c_loss = self.class_loss(
            c_pred,
            mols_1.c,
            self.charge_token_weights,
            pred_all_atoms,
            mols_t.batch,
            mols_t.num_graphs,
        )

        # 1. Handle substitutions

        #### Handle atom type substitutions
        do_sub_a_loss = self.do_action_loss(
            do_sub_a_head, mols_t.lambda_a_sub, mols_t.batch, mols_t.num_graphs
        )
        sub_a_class_loss = self.class_loss(
            a_pred,
            mols_1.a,
            self.atom_type_weights,
            mols_t.lambda_a_sub,
            mols_t.batch,
            mols_t.num_graphs,
        )
        sub_a_loss = do_sub_a_loss + sub_a_class_loss

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
            do_sub_e_head_triu, do_sub_e_target_triu, e_batch_triu, mols_t.num_graphs
        )
        sub_e_class_loss = self.class_loss(
            e_pred_triu,
            e_target_triu,
            self.edge_token_weights,
            do_sub_e_target_triu,
            e_batch_triu,
            mols_t.num_graphs,
        )
        sub_e_loss = do_sub_e_loss + sub_e_class_loss

        # Compute weighted components for logging using wrapper weights
        weights = self.loss_weight_wrapper.get_weight_tensors(self.device)

        sub_loss_weighted = sub_a_loss + sub_e_loss

        # Log substitution losses with subgroups
        self.log_dict(
            {
                "loss/sub/a/action": do_sub_a_loss,
                "loss/sub/a/class": sub_a_class_loss,
                "loss/sub/a/weighted": sub_a_loss,
                "loss/sub/e/action": do_sub_e_loss,
                "loss/sub/e/class": sub_e_class_loss,
                "loss/sub/e/weighted": sub_e_loss,
                "loss/sub/weighted": sub_a_loss + sub_e_loss,
                "loss/sub/c": c_loss,
            },
            prog_bar=False,
            logger=True,
        )

        del_loss_weighted = torch.tensor(0.0, device=self.device)
        ins_loss_weighted = torch.tensor(0.0, device=self.device)
        ins_loss_rate = torch.tensor(0.0, device=self.device)
        ins_loss_gmm = torch.tensor(0.0, device=self.device)
        ins_loss_e = torch.tensor(0.0, device=self.device)
        if self.n_atoms_strategy != "fixed":
            # 2. Handle deletions (no class changes here!)
            do_del_loss = self.do_action_loss(
                do_del_head, mols_t.lambda_del, mols_t.batch, mols_t.num_graphs
            )

            # 3. Handle insertions
            do_ins_loss = self.do_action_loss(
                do_ins_head, mols_t.lambda_ins, mols_t.batch, mols_t.num_graphs
            )

            # Count number of insertions per node
            # Initialize with zeros for all nodes
            ins_loss_gmm = torch.tensor(0.0, device=self.device)
            ins_loss_e = torch.tensor(0.0, device=self.device)

            # indices of nodes in mol_t that spawn/predict each insertion
            spawn_node_idx = ins_targets.spawn_node_idx

            # num_inserts_per_node = torch.zeros(mols_t.num_nodes, device=self.device)

            if spawn_node_idx.numel() > 0:
                # Count how many insertions each node spawns
                num_inserts_per_node = mols_t.n_ins

                # mask indicating which nodes are focal for insertion
                ins_loss_rate = self.rate_loss(
                    ins_rate_pred, num_inserts_per_node, mols_t.batch, mols_t.num_graphs
                )

                # mask indicating in which graph we have inserts
                batch_has_inserts = (
                    unsorted_segment_sum(
                        torch.ones((spawn_node_idx.shape[0], 1), device=self.device),
                        mols_t.batch[spawn_node_idx],
                        mols_t.num_graphs,
                    )
                    > 0
                )

                # we must take the NLL for the closest nodes only
                gmm_dict_pred = {
                    "mu": gmm_pred_dict["mu"][spawn_node_idx],
                    "sigma": gmm_pred_dict["sigma"][spawn_node_idx],
                    "pi": gmm_pred_dict["pi"][spawn_node_idx],
                    "a_probs": gmm_pred_dict["a_probs"][spawn_node_idx],
                    "c_probs": gmm_pred_dict["c_probs"][spawn_node_idx],
                }

                if self.cat_weighting.cat_strategy == "uniform-sample":
                    gmm_loss, _ = typed_gmm_loss(
                        gmm_dict_pred,
                        ins_targets.x,
                        ins_targets.a,
                        ins_targets.c,
                        self.atom_type_weights,
                        self.charge_token_weights,
                        reduction="none",
                    )
                else:
                    raise ValueError("Untyped GMM loss is not implemented yet.")
                    gmm_loss = untyped_gmm_loss(
                        gmm_dict_pred,
                        ins_targets.x,
                        ins_targets.batch,
                    )

                ins_loss_gmm = gmm_loss.view(-1)

                # reduce the loss
                ins_loss_gmm = unsorted_segment_mean(
                    ins_loss_gmm.view(-1, 1),
                    mols_t.batch[spawn_node_idx],
                    mols_t.num_graphs,
                )

                ins_loss_gmm = ins_loss_gmm[batch_has_inserts].mean()

                self.log_dict(
                    {
                        "loss/ins/gmm": ins_loss_gmm,
                        "loss/ins/gmm_weighted": weights["ins_gmm"] * ins_loss_gmm,
                    },
                    prog_bar=False,
                    logger=True,
                )

                # Compute insertion edge loss if available
                # Validate indices are within bounds
                # n_nodes = mols_t.num_nodes
                spawn_idx = ins_targets.ins_edge_spawn_idx
                target_idx = ins_targets.ins_edge_target_idx

                # make sure there are insertions to predict edges for
                assert spawn_idx.numel() > 0, (
                    "No insertions spawn points to predict edges for"
                )
                assert target_idx.numel() > 0, (
                    "No insertions target points to predict edges for"
                )

                # Get edge predictions using the insertion edge head
                ins_edge_logits = self.model.predict_insertion_edges(
                    out_dict=preds,
                    batch=mols_t.batch,
                    spawn_node_idx=spawn_idx,
                    target_node_idx=target_idx,
                    hard_sampling=True,
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

                    # reduce the loss
                    ins_loss_e = unsorted_segment_mean(
                        ins_loss_e.view(-1, 1),
                        mols_t.batch[spawn_idx],
                        mols_t.num_graphs,
                    )

                    ins_loss_e = ins_loss_e[batch_has_inserts].mean()

                    self.log_dict(
                        {
                            "loss/ins/e": ins_loss_e,
                            "loss/ins/e_weighted": weights["ins_e"] * ins_loss_e,
                        },
                        prog_bar=False,
                        logger=True,
                    )

            else:
                # NOTE edge_head and gmm_head unused, throws an error (unused_params)
                # NOTE therefore we add a dummy loss wrt. edge_head and gmm_head
                edge_head_loss = sum(
                    p.sum() for p in self.model.ins_edge_head.parameters()
                )
                gmm_head_loss = sum(
                    p.sum() for p in self.model.ins_gmm_head.parameters()
                )
                ins_loss_e = 0.0 * edge_head_loss
                ins_loss_gmm = 0.0 * gmm_head_loss

            # Log insertion and deletion losses with subgroups
            self.log_dict(
                {
                    "loss/del/action": do_del_loss,
                    "loss/ins/rate": ins_loss_rate,
                    "loss/ins/rate_weighted": weights["ins_rate"] * ins_loss_rate,
                },
                prog_bar=False,
                logger=True,
            )

        # 4. Calculate the flow matching loss
        # Only compute the loss for nodes that are not to be deleted
        # TODO edge case all deletes would lead to unused params error, but is highly unlikely
        to_delete_mask = mols_t.lambda_del > 0.0
        x_loss = F.mse_loss(
            x_pred[~to_delete_mask], mols_1.x[~to_delete_mask], reduction="none"
        )
        x_loss = unsorted_segment_mean(
            x_loss, mols_t.batch[~to_delete_mask], mols_t.num_graphs
        )
        # mask indicating in which graph we have deletes
        present_graphs_mask = torch.zeros(
            mols_t.num_graphs, dtype=torch.bool, device=self.device
        )
        present_graphs_mask[mols_t.batch[to_delete_mask]] = True

        x_loss = x_loss[present_graphs_mask].mean()

        # 4.5 Compute global budget losses (graph-level classifiers)
        # Target: n_ins_missing and n_del_missing computed during interpolation
        # These are the counts of insertions/deletions still needed to reach the target
        global_ins_budget_loss = torch.tensor(0.0, device=self.device)
        global_del_budget_loss = torch.tensor(0.0, device=self.device)

        if self.n_atoms_strategy != "fixed":
            # Use precomputed graph-level counts from interpolation
            # These are automatically batched by PyG as graph-level attributes
            ins_budget_target = mols_t.n_ins_missing  # Shape: (num_graphs,)
            del_budget_target = mols_t.n_del_missing  # Shape: (num_graphs,)

            # Compute losses if heads are available
            if global_ins_budget_pred is not None:
                global_ins_budget_loss = self.global_budget_loss(
                    global_ins_budget_pred,
                    ins_budget_target,
                    mols_t.num_graphs,
                )

            if global_del_budget_pred is not None:
                global_del_budget_loss = self.global_budget_loss(
                    global_del_budget_pred,
                    del_budget_target,
                    mols_t.num_graphs,
                )

            # Log global budget losses
            self.log_dict(
                {
                    "loss/global_budget/ins": global_ins_budget_loss,
                    "loss/global_budget/del": global_del_budget_loss,
                    "stats/ins_budget_target_mean": ins_budget_target.float().mean(),
                    "stats/del_budget_target_mean": del_budget_target.float().mean(),
                },
                prog_bar=False,
                logger=True,
            )
        else:
            global_ins_budget_loss = 0.0 * sum(
                p.sum() for p in self.model.global_ins_budget_head.parameters()
            )
            global_del_budget_loss = 0.0 * sum(
                p.sum() for p in self.model.global_del_budget_head.parameters()
            )

        # 5. Combine all losses using the unified wrapper
        loss_components = {
            "do_sub_a": do_sub_a_loss,
            "sub_a_class": sub_a_class_loss,
            "do_sub_e": do_sub_e_loss,
            "sub_e_class": sub_e_class_loss,
            "do_del": do_del_loss,
            "do_ins": do_ins_loss,
            "ins_rate": ins_loss_rate,
            "ins_gmm": ins_loss_gmm,
            "ins_e": ins_loss_e,
            "x": x_loss,
            "c": c_loss,
            # Global budget losses
            "global_ins_budget": global_ins_budget_loss,
            "global_del_budget": global_del_budget_loss,
        }
        loss = self.loss_weight_wrapper(loss_components)

        # Log learnable weights for monitoring (if using learnable weights)
        if self.loss_weight_wrapper.use_learnable:
            self.log_dict(
                {
                    "learnable_weight/sub/a/rate": weights["do_sub_a"],
                    "learnable_weight/sub/a/class": weights["sub_a_class"],
                    "learnable_weight/sub/e/rate": weights["do_sub_e"],
                    "learnable_weight/sub/e/class": weights["sub_e_class"],
                    "learnable_weight/del/rate": weights["do_del"],
                    "learnable_weight/ins/action": weights["do_ins"],
                    "learnable_weight/ins/rate": weights["ins_rate"],
                    "learnable_weight/ins/gmm": weights["ins_gmm"],
                    "learnable_weight/ins/e": weights["ins_e"],
                    "learnable_weight/x": weights["l_x"],
                    "learnable_weight/c": weights["l_c"],
                },
                prog_bar=False,
                logger=True,
            )

        # For backward compatibility, compute edit_flow_loss and x_loss_weighted for logging
        del_loss_weighted = weights["do_del"] * do_del_loss
        ins_loss_weighted = (
            weights["ins_rate"] * ins_loss_rate
            + weights["ins_gmm"] * ins_loss_gmm
            + weights["ins_e"] * ins_loss_e
        )
        edit_flow_loss = sub_loss_weighted + del_loss_weighted + ins_loss_weighted
        x_loss_weighted = weights["x"] * x_loss

        loss = self.safe_loss(loss)

        # Log final losses and statistics
        n_inserts = (mols_t.lambda_ins > 0.0).sum()
        n_deletes = (mols_t.lambda_del > 0.0).sum()
        self.log_dict(
            {
                "loss/sub": sub_loss_weighted,
                "loss/del": del_loss_weighted,
                "loss/ins": ins_loss_weighted,
                "loss/edit_flow": edit_flow_loss,
                "loss/move/x": x_loss,
                "loss/move/x_weighted": x_loss_weighted,
                "loss/move": x_loss_weighted,
                "stats/n_ins": n_inserts,
                "stats/n_del": n_deletes,
            },
            prog_bar=False,
            logger=True,
        )

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

        return eval_metrics

    def test_step(self, batch, batch_idx):
        # Define the test step logic here
        pass

    def predict_step(self, batch, batch_idx):
        return self.sample(batch, batch_idx, return_traj=True)

    def sample(self, batch, batch_idx, return_traj: bool = False):
        """
        Inference step for flow matching.

        Args:
            batch: Batch of data (for now, we'll use batch size to determine number of graphs)
            batch_idx: Batch index
            return_traj: If True, return trajectory for each molecule. If False, return final state only.

        Returns:
            If return_traj is False: MoleculeBatch - final sampled molecules
            If return_traj is True: List[List[MoleculeData]] - trajectory for each molecule
        """
        self.model.set_inference()
        if self.vocab.atom_tokens is None:
            raise ValueError(
                "vocab.atom_tokens must be set before prediction. "
                "Call set_vocab() first."
            )

        # Get batch size from the batch (assuming it's similar to training)
        mol_t, _ = batch

        batch_size = mol_t.batch_size

        _ = mol_t.remove_com()

        # Validate: check for cross-batch edges in initial sample
        validate_no_cross_batch_edges(
            mol_t.edge_index, mol_t.batch, "sample initial mol_t"
        )

        # Start at t=0
        t = torch.zeros(batch_size, device=self.device)
        step_sizes = self.integrator.get_time_steps()

        # Trajectory storage
        mol_traj = [mol_t.clone()]

        # previous outputs for self-conditioning. none at the beginning
        preds = None

        # Integration loop: integrate from t=0 to t=1
        for step_size in step_sizes:
            batch_id = mol_t.batch

            # Get model predictions
            preds = self.model(
                mol_t,
                t.view(-1, 1),
                prev_outs=preds,
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
            # Note: do_ins is computed inside integrate_step_gnn from num_ins_pred
            ins_rate_output = preds["ins_rate_head"]  # Shape depends on strategy

            if self.ins_rate_strategy == "classification":
                # Classification: sample from categorical distribution
                # ins_rate_output shape: (N, num_classes) - logits
                ins_probs = F.softmax(ins_rate_output, dim=-1)
                num_ins_pred = (
                    torch.distributions.Categorical(probs=ins_probs).sample().float()
                )
            else:
                # Poisson: use the predicted rate directly
                num_ins_pred = ins_rate_output  # Shape (N, 1) or (N,)
                # Ensure num_ins_pred has correct shape
                if num_ins_pred.ndim > 1:
                    num_ins_pred = num_ins_pred.squeeze(-1)

            # if we fix the number of atoms, we will not use the jump process
            if self.n_atoms_strategy == "fixed":
                num_ins_pred = torch.zeros_like(num_ins_pred)
                do_del_probs = torch.zeros_like(do_del_probs)

            # Get insertion edge head if available
            ins_edge_head = getattr(self.model, "ins_edge_head", None)

            # Extract global insertion budget prediction (graph-level)
            global_ins_budget = None
            global_ins_budget_logits = preds.get("global_ins_budget_head", None)
            if (
                global_ins_budget_logits is not None
                and self.n_atoms_strategy != "fixed"
            ):
                # TODO add temperature sampling to take higher confidence predictions
                temperature = 0.7
                global_ins_budget_logits = global_ins_budget_logits / temperature

                # Sample from categorical distribution to get the predicted budget
                ins_budget_probs = F.softmax(global_ins_budget_logits, dim=-1)
                global_ins_budget = (
                    torch.distributions.Categorical(probs=ins_budget_probs)
                    .sample()
                    .float()
                )  # Shape: (num_graphs,)

            global_del_budget = None
            global_del_budget_logits = preds.get("global_del_budget_head", None)
            if (
                global_del_budget_logits is not None
                and self.n_atoms_strategy != "fixed"
            ):
                # TODO add temperature sampling to take higher confidence predictions
                temperature = 0.7
                global_del_budget_logits = global_del_budget_logits / temperature

                # Sample from categorical distribution to get the predicted budget
                del_budget_probs = F.softmax(global_del_budget_logits, dim=-1)
                global_del_budget = (
                    torch.distributions.Categorical(probs=del_budget_probs)
                    .sample()
                    .float()
                )  # Shape: (num_graphs,)

            # print(global_del_budget)

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
                global_ins_budget=global_ins_budget,
                global_del_budget=global_del_budget,
            )

            # Validate: check for cross-batch edges after integration step
            validate_no_cross_batch_edges(
                mol_t.edge_index,
                mol_t.batch,
                f"sample after integrate_step t={t[0].item():.3f}",
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


if __name__ == "__main__":
    # Instantiate the LightningModule and run the training loop
    model = LightningModuleRates()
    trainer = pl.Trainer()
    trainer.fit(model)
