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
)
from external_code.egnn import unsorted_segment_mean

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
                    "l_sub_c_rate": 1.0,
                    "l_sub_c_class": 1.0,
                    "l_sub_e_rate": 1.0,
                    "l_sub_e_class": 1.0,
                    "l_del_rate": 1.0,
                    "l_ins_rate": 1.0,
                    "l_ins_gmm": 1.0,
                    "l_ins_e": 1.0,
                    "l_x": 1.0,
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
            "sub_c_rate": float(self.l_sub_c_rate),
            "sub_c_class": float(self.l_sub_c_class),
            "sub_e_rate": float(self.l_sub_e_rate),
            "sub_e_class": float(self.l_sub_e_class),
            "del_rate": float(self.l_del_rate),
            "ins_rate": float(self.l_ins_rate),
            "ins_gmm": float(self.l_ins_gmm),
            "ins_e": float(self.l_ins_e),
            "x": float(self.l_x),
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

        # NOTE: target rate must also include 0 rates!
        # This will automatically make term 2 zero for nodes that are not modified
        rate_loss = F.poisson_nll_loss(
            rate_pred.view(-1),
            target_rate.view(-1),
            log_input=False,
            reduction="none",
            full=True,
        )
        # NOTE: first normalize by number of nodes / graphs
        # Otherwise, nodes with more atoms will have more weight by design
        rate_loss = unsorted_segment_mean(rate_loss.view(-1, 1), batch, num_graphs)
        rate_loss = rate_loss.mean()

        class_loss = torch.tensor(0.0, device=self.device)
        if class_pred is not None and class_target is not None:
            class_loss = target_rate.view(-1) * F.cross_entropy(
                class_pred,
                class_target,
                weight=class_weights,
                reduction="none",
            )
            class_loss = unsorted_segment_mean(
                class_loss.view(-1, 1), batch, num_graphs
            )
            class_loss = class_loss.mean()

        # then take the mean
        return rate_loss, class_loss

    def one_flow_loss(
        self,
        rate_pred,
        target_rate,
        do_action_pred,
        batch,
        num_graphs,
        class_pred=None,
        class_target=None,
        class_weights=None,
    ):
        do_action = target_rate > 0.0
        rate_loss = F.poisson_nll_loss(
            rate_pred[do_action].view(-1),
            target_rate[do_action].view(-1),
            log_input=False,
            reduction="none",
            full=True,
        )

        # NOTE: first normalize by number of nodes / graphs
        # Otherwise, nodes with more atoms will have more weight by design
        rate_loss = unsorted_segment_mean(
            rate_loss.view(-1, 1), batch[do_action], num_graphs
        )
        rate_loss = rate_loss.mean()

        # NOTE: we add a BCE for do action or not do action
        do_action_target = F.one_hot(do_action.view(-1), num_classes=2)
        do_action_loss = F.binary_cross_entropy(
            do_action_pred.view(-1, 2),
            do_action_target.view(-1, 2),
            reduction="none",
        )
        do_action_loss = unsorted_segment_mean(
            do_action_loss.view(-1, 1), batch, num_graphs
        )
        do_action_loss = do_action_loss.mean()

        class_loss = torch.tensor(0.0, device=self.device)
        if class_pred is not None and class_target is not None:
            # NOTE: compared to edit_flow_loss, we do not weight the class loss by the target rate
            class_loss = F.cross_entropy(
                class_pred,
                class_target,
                weight=class_weights,
                reduction="none",
            )
            class_loss = unsorted_segment_mean(
                class_loss.view(-1, 1), batch, num_graphs
            )
            class_loss = class_loss.mean()

        return rate_loss, class_loss, do_action_loss

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
        sub_rate_pred_c = preds["sub_rate_c_head"]
        sub_rate_pred_e = preds["sub_rate_e_head"]

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
        sub_loss_c_rate, sub_loss_c_class = self.edit_flow_loss(
            sub_rate_pred_c,
            mols_t.lambda_c_sub,
            mols_t.batch,
            mols_t.num_graphs,
            c_pred,
            mols_1.c,
            class_weights=self.charge_token_weights,
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
        sub_loss_c_weighted = (
            weights["sub_c_rate"] * sub_loss_c_rate
            + weights["sub_c_class"] * sub_loss_c_class
        )
        sub_loss_e_weighted = (
            weights["sub_e_rate"] * sub_loss_e_rate
            + weights["sub_e_class"] * sub_loss_e_class
        )
        sub_loss_weighted = (
            sub_loss_a_weighted + sub_loss_c_weighted + sub_loss_e_weighted
        )

        # Log substitution losses with subgroups
        self.log_dict(
            {
                "loss/sub/a/rate": sub_loss_a_rate,
                "loss/sub/a/class": sub_loss_a_class,
                "loss/sub/a/weighted": sub_loss_a_weighted,
                "loss/sub/c/rate": sub_loss_c_rate,
                "loss/sub/c/class": sub_loss_c_class,
                "loss/sub/c/weighted": sub_loss_c_weighted,
                "loss/sub/e/rate": sub_loss_e_rate,
                "loss/sub/e/class": sub_loss_e_class,
                "loss/sub/e/weighted": sub_loss_e_weighted,
                "loss/sub/weighted": sub_loss_weighted,
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
            # 2. Handle deletions (no class changes here!)
            del_loss, _ = self.edit_flow_loss(
                del_rate_pred,
                mols_t.lambda_del,
                mols_t.batch,
                mols_t.num_graphs,
            )

            # 3. Handle insertions
            # mask indicating which nodes are focal for insertion
            ins_loss_rate, _ = self.edit_flow_loss(
                ins_rate_pred,
                mols_t.lambda_ins,
                mols_t.batch,
                mols_t.num_graphs,
            )

            ins_loss_gmm = torch.tensor(0.0, device=self.device)
            ins_loss_e = torch.tensor(0.0, device=self.device)

            if ins_targets.spawn_node_idx.numel() > 0:
                # spawn_node_idx: indices of nodes in mol_t that spawn/predict each insertion
                spawn_node_idx = ins_targets.spawn_node_idx

                gmm_dict_pred = {
                    "mu": gmm_pred_dict["mu"][spawn_node_idx],
                    "sigma": gmm_pred_dict["sigma"][spawn_node_idx],
                    "pi": gmm_pred_dict["pi"][spawn_node_idx],
                    "a_probs": gmm_pred_dict["a_probs"][spawn_node_idx],
                    "c_probs": gmm_pred_dict["c_probs"][spawn_node_idx],
                }

                # TODO we must take the NLL for the closest nodes only
                if self.cat_weighting.cat_strategy == "uniform-sample":
                    sigma_t = self.ins_noise_scale * (1 - t)
                    sigma_t = sigma_t[mols_t.batch][spawn_node_idx]

                    gmm_loss = typed_gmm_loss(
                        gmm_dict_pred,
                        ins_targets.x,
                        ins_targets.a,
                        ins_targets.c,
                        sigma_t,
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
                gmm_loss = gmm_loss.view(-1) * mols_t.lambda_ins[spawn_node_idx]

                # reduce the loss
                ins_loss_gmm = unsorted_segment_mean(
                    gmm_loss.view(-1, 1),
                    mols_t.batch[spawn_node_idx],
                    mols_t.num_graphs,
                )

                ins_loss_gmm = ins_loss_gmm.mean()

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
                    ins_loss_e = ins_loss_e * ins_edge_weights

                    # reduce the loss
                    ins_loss_e = unsorted_segment_mean(
                        ins_loss_e.view(-1, 1),
                        mols_t.batch[spawn_idx],
                        mols_t.num_graphs,
                    )
                    ins_loss_e = ins_loss_e.mean()

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
        x_loss = F.mse_loss(x_pred, mols_1.x)

        # 5. Combine all losses using the unified wrapper
        loss_components = {
            "sub_a_rate": sub_loss_a_rate,
            "sub_a_class": sub_loss_a_class,
            "sub_c_rate": sub_loss_c_rate,
            "sub_c_class": sub_loss_c_class,
            "sub_e_rate": sub_loss_e_rate,
            "sub_e_class": sub_loss_e_class,
            "del_rate": del_loss,
            "ins_rate": ins_loss_rate,
            "ins_gmm": ins_loss_gmm,
            "ins_e": ins_loss_e,
            "x": x_loss,
        }
        loss = self.loss_weight_wrapper(loss_components)

        # Log learnable weights for monitoring (if using learnable weights)
        if self.loss_weight_wrapper.use_learnable:
            self.log_dict(
                {
                    "learnable_weight/sub/a/rate": weights["sub_a_rate"],
                    "learnable_weight/sub/a/class": weights["sub_a_class"],
                    "learnable_weight/sub/c/rate": weights["sub_c_rate"],
                    "learnable_weight/sub/c/class": weights["sub_c_class"],
                    "learnable_weight/sub/e/rate": weights["sub_e_rate"],
                    "learnable_weight/sub/e/class": weights["sub_e_class"],
                    "learnable_weight/del/rate": weights["del_rate"],
                    "learnable_weight/ins/rate": weights["ins_rate"],
                    "learnable_weight/ins/gmm": weights["ins_gmm"],
                    "learnable_weight/ins/e": weights["ins_e"],
                    "learnable_weight/x": weights["x"],
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
            sub_rate_pred_c = preds["sub_rate_c_head"]
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
                sub_rate_c=sub_rate_pred_c,
                sub_rate_e=sub_rate_pred_e,
                del_rate=del_rate_pred,
                ins_rate=ins_rate_pred,
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

            if torch.all(t >= 1.0):
                # save final predicted state to trajectory
                mol_traj.append(mol_1_pred.clone())
            else:
                # Save  state to trajectory
                mol_traj.append(mol_t.clone())

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
