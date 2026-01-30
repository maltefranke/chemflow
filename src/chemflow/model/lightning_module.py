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
                    "l_sub_a_rate": 1.0,
                    "l_sub_a_class": 1.0,
                    # "l_sub_c_rate": 1.0,
                    # "l_sub_c_class": 1.0,
                    "l_sub_e_rate": 1.0,
                    "l_sub_e_class": 1.0,
                    "l_del_rate": 1.0,
                    "l_ins_rate": 1.0,
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
            "sub_a_rate": float(self.l_sub_a_rate),
            "sub_a_class": float(self.l_sub_a_class),
            "sub_e_rate": float(self.l_sub_e_rate),
            "sub_e_class": float(self.l_sub_e_class),
            "del_rate": float(self.l_del_rate),
            "ins_rate": float(self.l_ins_rate),
            "ins_gmm": float(self.l_ins_gmm),
            "ins_e": float(self.l_ins_e),
            "x": float(self.l_x),
            "c": float(self.l_c),
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

    def edit_flow_loss(
        self,
        rate_pred,
        target_rate,
        batch,
        num_graphs,
        class_pred=None,
        class_target=None,
        class_weights=None,
    ):
        # EditFlow loss (eq. 23 in the paper):
        # L = sum(x != x_t)[u_t(x|x_t)]
        #   + sum(i=0 to N_nodes)[1{is_different} * log(u_t(ins/del/sub|x_t))]

        # Rate loss (EditFlow's term 1 in eq. 23)
        # Intuitively, we want to do as few edits as possible

        # Correct edits loss (EditFlows, term 2 in eq. 23)
        # Intuitively, we want to do as many correct edits as possible
        # u_t(insert) = lambda_insert * Q_insert
        # <--> log[u_t(insert)] = log[lambda_insert * Q_insert]
        # <--> log[u_t(insert)] = log[lambda_insert] + log[Q_insert]
        # similar for delete and substitute

        # This simiplies to a simple poisson nll loss and a cross entropy loss term
        # Loss is L = sum(lambda_pred - weight * log(lambda_pred)) + sum(weight * log(Q_target))
        #           = sum(poisson_nll(lambda_pred, weight)) + sum(weight * cross_entropy(class_pred, class_target))
        # where weight is the target rate

        # calculate dynamic weighting for the rate loss
        is_modified = target_rate.view(-1) > 0.0
        num_modified = is_modified.sum()
        num_total = target_rate.shape[0]
        num_unmodified = num_total - num_modified

        """total = num_modified + num_unmodified
        w_modified = total / (2.0 * num_modified)
        w_unmodified = total / (2.0 * num_unmodified)

        if num_modified == 0 or num_unmodified == 0:
            w_modified = 1.0
            w_unmodified = 1.0"""

        w_modified = 1.0
        w_unmodified = 1.0

        # Apply weights for Rate Loss
        weights = (
            is_modified.to(torch.int32) * w_modified
            + (1 - is_modified.to(torch.int32)) * w_unmodified
        )

        # Rate Loss Calculation
        # TODO need to double check if target_rate is the actual rate, or number of actions
        rate_loss = F.poisson_nll_loss(
            rate_pred.view(-1),
            target_rate.view(-1),
            log_input=False,
            reduction="none",
            full=True,
        )
        rate_loss = rate_loss * weights

        rate_loss = unsorted_segment_mean(rate_loss.view(-1, 1), batch, num_graphs)
        rate_loss = rate_loss.mean()

        # Class Loss Calculation
        class_loss_val = torch.tensor(0.0, device=self.device)

        if class_pred is not None and class_target is not None:  # noqa: SIM102
            # 1. Slice: Only process nodes that are actually modified
            # This makes the operation O(N_modified) instead of O(N_total)
            if is_modified.any():
                pred_sub = class_pred[is_modified]
                target_sub = class_target[is_modified]
                rate_sub = target_rate.view(-1)[is_modified]
                batch_sub = batch[is_modified]

                # 2. Compute Element-wise Weighted Loss
                # The 'weight' param applies the class imbalance penalty (e.g. x100 for rare classes)
                ce_loss = F.cross_entropy(
                    pred_sub, target_sub, weight=class_weights, reduction="none"
                )

                # Apply the target rate (lambda) scaling
                # This corresponds to the term: lambda_target * CrossEntropy
                weighted_loss_elements = rate_sub * ce_loss

                # 3. Aggregate per Graph using Mean
                # unsorted_segment_mean sums the weighted elements and divides by the COUNT
                # of items in that segment.
                # Result: A graph with 1 rare error (weight 100) gets high loss.
                graph_class_loss = unsorted_segment_mean(
                    weighted_loss_elements.view(-1, 1), batch_sub, num_graphs
                )

                # 4. Filter and Average
                # We must only average over graphs that actually appeared in batch_sub.
                # Using 'graph_class_loss > 0' is risky if loss is genuinely 0 (perfect prediction).
                # Instead, create a mask of present graphs.
                present_graphs_mask = torch.zeros(
                    num_graphs, dtype=torch.bool, device=self.device
                )
                present_graphs_mask[batch_sub] = True

                class_loss_val = graph_class_loss[present_graphs_mask].mean()

        return rate_loss, class_loss_val

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

        gmm_pred_dict = preds["gmm_head"]

        ins_rate_pred = preds["ins_rate_head"]
        del_rate_pred = preds["del_rate_head"]
        sub_rate_pred_a = preds["sub_rate_a_head"]
        sub_rate_pred_e = preds["sub_rate_e_head"]

        c_loss = F.cross_entropy(
            c_pred, mols_1.c, weight=self.charge_token_weights, reduction="none"
        )

        c_loss = unsorted_segment_mean(
            c_loss.view(-1, 1), mols_t.batch, mols_t.num_graphs
        )

        c_loss = c_loss.mean()

        # 1. Handle substitutions

        #### Handle atom type substitutions
        # Mask indicating which nodes need to be substituted
        sub_loss_a_rate, sub_loss_a_class = self.edit_flow_loss(
            sub_rate_pred_a,
            mols_t.lambda_a_sub,
            mols_t.batch,
            mols_t.num_graphs,
            a_pred,
            mols_1.a,
            class_weights=self.atom_type_weights,
        )

        #### Handle edge type substitutions
        edge_infos = self.edge_aligner.align_edges(
            source_group=(mols_t.edge_index, [e_pred, sub_rate_pred_e]),
            target_group=(mols_1.edge_index, [mols_1.e, mols_t.lambda_e_sub]),
        )
        e_batch_triu = mols_t.batch[edge_infos["edge_index"][0][0]]
        e_pred_triu, sub_rate_pred_e_triu = edge_infos["edge_attr"][:2]
        e_target_triu, sub_rate_target_e_triu = edge_infos["edge_attr"][2:]

        sub_loss_e_rate, sub_loss_e_class = self.edit_flow_loss(
            sub_rate_pred_e_triu,
            sub_rate_target_e_triu,
            e_batch_triu,
            mols_t.num_graphs,
            e_pred_triu,
            e_target_triu,
            class_weights=self.edge_token_weights,
        )

        # Compute weighted components for logging using wrapper weights
        weights = self.loss_weight_wrapper.get_weight_tensors(self.device)

        sub_loss_a_weighted = (
            weights["sub_a_rate"] * sub_loss_a_rate
            + weights["sub_a_class"] * sub_loss_a_class
        )

        sub_loss_e_weighted = (
            weights["sub_e_rate"] * sub_loss_e_rate
            + weights["sub_e_class"] * sub_loss_e_class
        )
        sub_loss_weighted = (
            sub_loss_a_weighted + sub_loss_e_weighted  # + sub_loss_c_weighted
        )

        # Log substitution losses with subgroups
        self.log_dict(
            {
                "loss/sub/a/rate": sub_loss_a_rate,
                "loss/sub/a/class": sub_loss_a_class,
                "loss/sub/a/weighted": sub_loss_a_weighted,
                # "loss/sub/c/rate": sub_loss_c_rate,
                # "loss/sub/c/class": sub_loss_c_class,
                # "loss/sub/c/weighted": sub_loss_c_weighted,
                "loss/sub/e/rate": sub_loss_e_rate,
                "loss/sub/e/class": sub_loss_e_class,
                "loss/sub/e/weighted": sub_loss_e_weighted,
                "loss/sub/weighted": sub_loss_weighted,
                "loss/sub/c": c_loss,
            },
            prog_bar=False,
            logger=True,
        )

        del_loss_weighted = torch.tensor(0.0, device=self.device)
        del_loss = torch.tensor(0.0, device=self.device)
        ins_loss_weighted = torch.tensor(0.0, device=self.device)
        ins_loss_rate = torch.tensor(0.0, device=self.device)
        ins_loss_gmm = torch.tensor(0.0, device=self.device)
        ins_loss_e = torch.tensor(0.0, device=self.device)
        if self.n_atoms_strategy != "fixed":
            # TODO need to add weighting between deletion and insertion loss
            # 2. Handle deletions (no class changes here!)
            del_loss, _ = self.edit_flow_loss(
                del_rate_pred,
                mols_t.lambda_del,  # TODO rate or number of actions?
                mols_t.batch,
                mols_t.num_graphs,
            )

            # 3. Handle insertions
            # mask indicating which nodes are focal for insertion
            ins_loss_rate, _ = self.edit_flow_loss(
                ins_rate_pred,
                mols_t.lambda_ins,  # TODO rate or number of actions?
                mols_t.batch,
                mols_t.num_graphs,
            )

            ins_loss_gmm = torch.tensor(0.0, device=self.device)
            ins_loss_e = torch.tensor(0.0, device=self.device)

            # TODO need to add fail-safe for no insertions like in the adjusted model
            if ins_targets.spawn_node_idx.numel() > 0:
                # spawn_node_idx: indices of nodes in mol_t that spawn/predict each insertion
                spawn_node_idx = ins_targets.spawn_node_idx

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

                # weight the gmm loss by the target rate
                ins_loss_gmm = gmm_loss.view(-1) * mols_t.lambda_ins[spawn_node_idx]

                # reduce the loss
                ins_loss_gmm = unsorted_segment_mean(
                    ins_loss_gmm.view(-1, 1),
                    mols_t.batch[spawn_node_idx],
                    mols_t.num_graphs,
                )
                ins_loss_gmm = ins_loss_gmm[batch_has_inserts].mean()

            # Compute insertion edge loss if available
            # Validate indices are within bounds
            n_nodes = mols_t.num_nodes
            spawn_idx = ins_targets.ins_edge_spawn_idx
            target_idx = ins_targets.ins_edge_target_idx

            if spawn_idx.max() < n_nodes and target_idx.max() < n_nodes:
                # Get edge predictions using the insertion edge head
                ins_edge_logits = self.model.predict_insertion_edges(
                    out_dict=preds,
                    batch=mols_t.batch,
                    spawn_node_idx=spawn_idx,
                    target_node_idx=target_idx,
                    hard_sampling=True,
                )
                if ins_edge_logits is not None and ins_edge_logits.numel() > 0:
                    ins_edge_weights = mols_t.lambda_ins[spawn_idx]

                    ins_loss_e = F.cross_entropy(
                        ins_edge_logits,
                        ins_targets.ins_edge_types,
                        weight=self.edge_token_weights,
                        reduction="none",
                    )

                    # weight the loss by the target rate
                    # NOTE All inserted edges will have a non-zero target rate
                    # NOTE Therefore no more filtering needed
                    ins_loss_e = ins_loss_e * ins_edge_weights

                    # reduce the loss
                    ins_loss_e = unsorted_segment_mean(
                        ins_loss_e.view(-1, 1),
                        mols_t.batch[spawn_idx],
                        mols_t.num_graphs,
                    )

                    ins_loss_e = ins_loss_e[batch_has_inserts].mean()

            # Compute weighted components for logging using wrapper weights
            del_loss_weighted = weights["del_rate"] * del_loss
            ins_loss_weighted = (
                weights["ins_rate"] * ins_loss_rate
                + weights["ins_gmm"] * ins_loss_gmm
                + weights["ins_e"] * ins_loss_e
            )

            # Log insertion and deletion losses with subgroups
            self.log_dict(
                {
                    "loss/del/rate": del_loss,
                    "loss/del/weighted": del_loss_weighted,
                    "loss/ins/rate": ins_loss_rate,
                    "loss/ins/gmm": ins_loss_gmm,
                    "loss/ins/e": ins_loss_e,
                    "loss/ins/weighted": ins_loss_weighted,
                    "loss/ins/rate_weighted": weights["ins_rate"] * ins_loss_rate,
                    "loss/ins/gmm_weighted": weights["ins_gmm"] * ins_loss_gmm,
                    "loss/ins/e_weighted": weights["ins_e"] * ins_loss_e,
                },
                prog_bar=False,
                logger=True,
            )

        # 4. Calculate the flow matching loss
        # Only compute the loss for nodes that are not to be deleted
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

        # 5. Combine all losses using the unified wrapper
        loss_components = {
            "sub_a_rate": sub_loss_a_rate,
            "sub_a_class": sub_loss_a_class,
            "sub_e_rate": sub_loss_e_rate,
            "sub_e_class": sub_loss_e_class,
            "del_rate": del_loss,
            "ins_rate": ins_loss_rate,
            "ins_gmm": ins_loss_gmm,
            "ins_e": ins_loss_e,
            "x": x_loss,
            "c": c_loss,
        }
        loss = self.loss_weight_wrapper(loss_components)

        # Log learnable weights for monitoring (if using learnable weights)
        if self.loss_weight_wrapper.use_learnable:
            self.log_dict(
                {
                    "learnable_weight/sub/a/rate": weights["sub_a_rate"],
                    "learnable_weight/sub/a/class": weights["sub_a_class"],
                    "learnable_weight/sub/e/rate": weights["sub_e_rate"],
                    "learnable_weight/sub/e/class": weights["sub_e_class"],
                    "learnable_weight/del/rate": weights["del_rate"],
                    "learnable_weight/ins/rate": weights["ins_rate"],
                    "learnable_weight/ins/gmm": weights["ins_gmm"],
                    "learnable_weight/ins/e": weights["ins_e"],
                    "learnable_weight/x": weights["x"],
                    "learnable_weight/c": weights["c"],
                },
                prog_bar=False,
                logger=True,
            )

        # For backward compatibility, compute edit_flow_loss and x_loss_weighted for logging
        del_loss_weighted = weights["del_rate"] * del_loss
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

        Returns:
            Dictionary containing final samples
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

            # Process rates
            ins_rate_pred = preds["ins_rate_head"]
            del_rate_pred = preds["del_rate_head"]
            sub_rate_pred_a = preds["sub_rate_a_head"]
            sub_rate_pred_e = preds["sub_rate_e_head"]

            # if we fix the number of atoms, we will not use the jump process
            if self.n_atoms_strategy == "fixed":
                ins_rate_pred = torch.zeros_like(ins_rate_pred)
                del_rate_pred = torch.zeros_like(del_rate_pred)

            # Get insertion edge head if available
            ins_edge_head = getattr(self.model, "ins_edge_head", None)

            # Integrate one step (edge prediction happens inside if head is provided)
            mol_t = self.integrator.integrate_step_gnn(
                mol_t=mol_t.clone(),
                mol_1_pred=mol_1_pred.clone(),
                sub_rate_a=sub_rate_pred_a,
                sub_rate_e=sub_rate_pred_e,
                del_rate=del_rate_pred,
                ins_rate=ins_rate_pred,
                ins_gmm_preds=gmm_dict_pred,
                t=t,
                dt=step_size,
                h_latent=preds.get("h_latent"),
                ins_edge_head=ins_edge_head,
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

            if torch.all(t >= 1.0):
                # correct the coordinates
                mol_1_pred_cloned = mol_1_pred.clone()
                mol_1_pred_cloned.x = (
                    mol_1_pred_cloned.x * self.distributions.coordinate_std
                )

                # save final predicted state to trajectory
                mol_traj.append(mol_1_pred_cloned)
            else:
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
                for mol_t in traj_lists:
                    traj_per_mol[i].append(mol_t[i])

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
