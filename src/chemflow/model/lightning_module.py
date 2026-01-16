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
        noise_params: DictConfig = None,
        cat_weighting: DictConfig = None,
        time_dist: DictConfig = None,
        vocab: Vocab = None,
        distributions: Distributions = None,
    ):
        super().__init__()

        self.vocab = vocab
        self.distributions = distributions

        self.cat_weighting = cat_weighting

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

        if noise_params is None:
            noise_params = DictConfig(
                {
                    "cat_noise_level": 0.0,
                    "coord_noise_level": 0.0,
                }
            )
        self.noise_params = noise_params

        # Set default loss weights if not provided
        if loss_weights is None:
            loss_weights = DictConfig(
                {
                    "l_x": 1.0,
                    "l_ins": 1.0,
                    "l_del": 1.0,
                    "l_sub_rate_a": 1.0,
                    "l_sub_class_a": 1.0,
                    "l_sub_rate_c": 1.0,
                    "l_sub_class_c": 1.0,
                    "l_sub_rate_e": 1.0,
                    "l_sub_class_e": 1.0,
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
        )
        class_loss = torch.tensor(0.0, device=self.device)
        if class_pred is not None and class_target is not None:
            class_loss = target_rate.view(-1) * F.cross_entropy(
                class_pred,
                class_target,
                weight=class_weights,
                reduction="none",
            )
        loss = rate_loss + class_loss

        # NOTE: first normalize by number of nodes / graphs
        # Otherwise, nodes with more atoms will have more weight by design
        loss = unsorted_segment_mean(loss.view(-1, 1), batch, num_graphs)

        # then take the mean
        return loss.mean()

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
        sub_loss_a = self.edit_flow_loss(
            sub_rate_pred_a,
            mols_t.lambda_a_sub,
            mols_t.batch,
            mols_t.num_graphs,
            a_pred,
            mols_1.a,
            class_weights=self.atom_type_weights,
        )
        self.log("sub_loss_a", sub_loss_a, prog_bar=False, logger=True)

        sub_loss_c = self.edit_flow_loss(
            sub_rate_pred_c,
            mols_t.lambda_c_sub,
            mols_t.batch,
            mols_t.num_graphs,
            c_pred,
            mols_1.c,
            class_weights=self.charge_token_weights,
        )
        self.log("sub_loss_c", sub_loss_c, prog_bar=False, logger=True)

        #### Handle edge type substitutions
        edge_infos = self.edge_aligner.align_edges(
            source_group=(mols_t.edge_index, [e_pred, sub_rate_pred_e]),
            target_group=(mols_1.edge_index, [mols_1.e, mols_t.lambda_e_sub]),
        )
        e_batch_triu = mols_t.batch[edge_infos["edge_index"][0][0]]
        e_pred_triu, sub_rate_pred_e_triu = edge_infos["edge_attr"][:2]
        e_target_triu, sub_rate_target_e_triu = edge_infos["edge_attr"][2:]

        sub_loss_e = self.edit_flow_loss(
            sub_rate_pred_e_triu,
            sub_rate_target_e_triu,
            e_batch_triu,
            mols_t.num_graphs,
            e_pred_triu,
            e_target_triu,
            class_weights=self.edge_token_weights,
        )
        self.log("sub_loss_e", sub_loss_e, prog_bar=False, logger=True)

        sub_loss = sub_loss_a + sub_loss_c + sub_loss_e
        self.log("sub_loss", sub_loss, prog_bar=False, logger=True)

        del_loss = torch.tensor(0.0, device=self.device)
        ins_loss = torch.tensor(0.0, device=self.device)
        if self.n_atoms_strategy != "fixed":
            # 2. Handle deletions (no class changes here!)
            del_loss = self.edit_flow_loss(
                del_rate_pred, mols_t.lambda_del, mols_t.batch, mols_t.num_graphs
            )
            self.log("del_loss", del_loss, prog_bar=False, logger=True)

            # 3. Handle insertions
            # mask indicating which nodes are focal for insertion
            ins_loss_rate = self.edit_flow_loss(
                ins_rate_pred, mols_t.lambda_ins, mols_t.batch, mols_t.num_graphs
            )
            self.log("ins_loss_rate", ins_loss_rate, prog_bar=False, logger=True)

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
                    gmm_loss = typed_gmm_loss(
                        gmm_dict_pred,
                        ins_targets.x,
                        ins_targets.a,
                        ins_targets.c,
                        ins_targets.batch,
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
                self.log("ins_loss_gmm", ins_loss_gmm, prog_bar=False, logger=True)

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
                    self.log("ins_loss_e", ins_loss_e, prog_bar=False, logger=True)

            ins_loss = ins_loss_rate + ins_loss_gmm + ins_loss_e
            self.log("ins_loss", ins_loss, prog_bar=False, logger=True)

        # 4. Combine the edit losses
        edit_flow_loss = sub_loss + del_loss + ins_loss
        self.log("edit_flow_loss", edit_flow_loss, prog_bar=False, logger=True)

        # 5. Calculate the flow matching loss
        x_loss = F.mse_loss(x_pred, mols_1.x)
        x_loss = x_loss * self.l_x
        self.log("x_loss", x_loss, prog_bar=False, logger=True)

        # 6. Combine the EditFlow loss and the flow matching loss
        loss = edit_flow_loss + x_loss

        loss = self.safe_loss(loss)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        return
        batched_mols = self.sample(batch, batch_idx, return_traj=False)

        # List of MoleculeData objects
        mols = batched_mols.to_data_list()

        # List of RDKit molecules or None
        rdkit_mols = [
            mol.to_rdkit_mol(
                self.vocab.atom_tokens, self.vocab.edge_tokens, self.vocab.charge_tokens
            )
            for mol in mols
        ]
        eval_metrics = calc_metrics_(rdkit_mols, self.metrics)
        print(eval_metrics)

        for key, value in eval_metrics.items():
            self.log(
                f"val_{key}",
                value,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                batch_size=len(mols),
            )

        return eval_metrics

        return

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
        # Instantiate optimizer from config
        optimizer_cfg = dict(
            OmegaConf.to_container(self.optimizer_config.optimizer, resolve=True)
        )
        optimizer_cfg["params"] = self.model.parameters()
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
