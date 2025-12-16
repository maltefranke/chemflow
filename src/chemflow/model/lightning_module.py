import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
import hydra
from chemflow.losses import typed_gmm_loss, rate_loss
from chemflow.losses import gmm_loss as untyped_gmm_loss
from torch_geometric.nn import knn_graph


from chemflow.flow_matching.integration import Integrator
from chemflow.utils import (
    token_to_index,
    compute_token_weights,
    get_canonical_upper_triangle_with_index,
    symmetrize_upper_triangle,
)
from external_code.egnn import unsorted_segment_mean
from chemflow.flow_matching.interpolation import Interpolator
from chemflow.flow_matching.gmm import compute_equivariant_gmm

from chemflow.dataset.molecule_data import MoleculeData, PointCloud, MoleculeBatch


class LightningModule(pl.LightningModule):
    def __init__(
        self,
        model: DictConfig = None,
        loss_weights: DictConfig = None,
        optimizer_config: DictConfig = None,
        weight_alpha: float = 1.0,
        k_nn_edges: int = 20,
        N_samples: int = 20,
        K: int = 10,
        D: int = 3,
        n_atoms_strategy: str = "fixed",
        cat_strategy: str = "uniform-sample",  # "mask" or "uniform-sample"
        type_loss_token_weights: str = "training",  # "uniform" or "training"
        num_integration_steps: int = 100,
        cat_noise_level: float = 0.0,
        coord_noise_level: float = 0.0,
    ):
        super().__init__()

        # Will be set via setter method
        self.atom_tokens = None
        self.edge_tokens = None
        self.charge_tokens = None
        self.atom_mask_index = None
        self.death_token_index = None
        self.interpolator = None
        self.integrator = None

        self.weight_alpha = weight_alpha

        self.cat_strategy = cat_strategy
        self.N_samples = N_samples
        self.k_nn_edges = k_nn_edges
        self.K = K
        self.D = D
        self.n_atoms_strategy = n_atoms_strategy
        self.cat_strategy = cat_strategy
        self.type_loss_token_weights = type_loss_token_weights
        self.cat_noise_level = cat_noise_level
        self.num_integration_steps = num_integration_steps
        self.coord_noise_level = coord_noise_level
        self.time_dist = torch.distributions.Uniform(0.0, 1.0)

        # Set default loss weights if not provided
        if loss_weights is None:
            loss_weights = DictConfig(
                {
                    "a_loss": 1.0,
                    "x_loss": 1.0,
                    "gmm_loss": 1.0,
                    "death_rate_loss": 1.0,
                    "birth_rate_loss": 1.0,
                }
            )
        self.loss_weights = loss_weights

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

        self.save_hyperparameters()
        self.model = hydra.utils.instantiate(model)

        self.is_compiled = False

    def compile(self):
        """Compile the model using torch.compile."""
        if self.is_compiled:
            return
        print("Compiling model...")
        self.model = torch.compile(self.model, dynamic=True)
        self.is_compiled = True

    def set_tokens_and_distribution(
        self,
        atom_tokens: list[str],
        edge_tokens: list[str],
        charge_tokens: list[str],
        atom_type_distribution: torch.Tensor,
        edge_type_distribution: torch.Tensor,
        charge_type_distribution: torch.Tensor,
    ):
        """Set tokens and distributions after initialization."""
        self.atom_tokens = atom_tokens
        self.edge_tokens = edge_tokens
        self.charge_tokens = charge_tokens

        if self.n_atoms_strategy != "fixed":
            self.atom_death_token_index = token_to_index(self.atom_tokens, "<DEATH>")
        if self.cat_strategy == "mask":
            self.atom_mask_index = token_to_index(self.atom_tokens, "<MASK>")
            self.edge_mask_index = token_to_index(self.edge_tokens, "<MASK>")

        self.interpolator = Interpolator(
            self.atom_tokens,
            self.edge_tokens,
            self.charge_tokens,
            atom_type_distribution.to(self.device),
            edge_type_distribution.to(self.device),
            cat_strategy=self.cat_strategy,
            n_atoms_strategy=self.n_atoms_strategy,
            N_samples=self.N_samples,
        )
        self.integrator = Integrator(
            self.atom_tokens,
            edge_tokens=self.edge_tokens,
            edge_type_distribution=edge_type_distribution.to(
                "cuda" if torch.cuda.is_available() else "cpu"
            ),
            K=self.K,
            D=self.D,
            cat_strategy=self.cat_strategy,
            n_atoms_strategy=self.n_atoms_strategy,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        # Always compute token distribution weights for weighted cross-entropy loss
        atom_special_tokens = []
        if self.cat_strategy == "mask":
            atom_special_tokens.append("<MASK>")
        if self.n_atoms_strategy != "fixed":
            atom_special_tokens.append("<DEATH>")

        atom_type_weights = compute_token_weights(
            token_list=self.atom_tokens,
            distribution=atom_type_distribution,
            special_token_names=atom_special_tokens,
            weight_alpha=self.weight_alpha,
            type_loss_token_weights=self.type_loss_token_weights,
        )
        self.register_buffer("atom_type_weights", atom_type_weights)

        # Compute edge token distribution weights for weighted cross-entropy loss
        edge_special_tokens = ["<NO_BOND>"]
        if self.cat_strategy == "mask":
            edge_special_tokens.append("<MASK>")

        edge_weights = compute_token_weights(
            token_list=self.edge_tokens,
            distribution=edge_type_distribution,
            special_token_names=edge_special_tokens,
            weight_alpha=self.weight_alpha,
            type_loss_token_weights=self.type_loss_token_weights,
        )
        self.register_buffer("edge_token_weights", edge_weights)

        charge_weights = compute_token_weights(
            token_list=self.charge_tokens,
            distribution=charge_type_distribution,
            special_token_names=[],  # no special tokens for charges
            weight_alpha=self.weight_alpha,
            type_loss_token_weights=self.type_loss_token_weights,
        )
        self.register_buffer("charge_token_weights", charge_weights)

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

    def shared_step(self, batch, batch_idx):
        self.model.set_training()
        # Define the training step logic here
        if self.atom_tokens is None:
            raise ValueError(
                "atom_tokens must be set before training. "
                "Call set_tokens_and_distribution() first."
            )

        samples_batched, targets_batched = batch

        batch_size = targets_batched.batch_size

        # interpolate
        t = self.time_dist.sample((batch_size,)).to(self.device)
        mols_t, targets = self.interpolator.interpolate_different_size(
            samples_batched,
            targets_batched,
            t,
        )

        # TODO make choice flexible
        # build kNN graph
        # edge_index = knn_graph(xt, k=self.k_nn_edges, batch=xt_batch_id)

        # build a fully connected graph per batch
        # edge_index = build_fully_connected_edge_index(xt_batch_id)

        # edge_type_ids = et[edge_index[0], edge_index[1]]

        # random self-conditioning
        is_random_self_conditioning = (torch.rand(1) > 0.5).item()

        preds = self.model(
            mols_t,
            t.view(-1, 1),
            is_random_self_conditioning=is_random_self_conditioning,
        )

        a_pred = preds["atom_type_head"]
        x_pred = preds["pos_head"]
        edge_type_pred = preds["edge_type_head"]
        gmm_pred = preds["gmm_head"]
        charge_pred = preds["charge_head"]

        gmm_dict = compute_equivariant_gmm(
            gmm_pred,
            mols_t.x,
            mols_t.batch,
            self.K,
            len(self.atom_tokens) if self.cat_strategy == "uniform-sample" else 0,
        )

        net_rate_pred = preds["net_rate_head"]
        death_rate_pred = F.relu(-net_rate_pred)
        birth_rate_pred = F.relu(net_rate_pred)

        # Calculate losses
        # only do class prediction on non-mask tokens
        if self.atom_type_weights is None:
            raise ValueError(
                "atom_type_weights must be set before training. "
                "Call set_tokens_and_distribution() first."
            )

        if self.cat_strategy == "mask":
            # predict final class only for mask tokens
            mask = targets["mols_1"].a != self.atom_mask_index
        else:
            # predict final class for all tokens
            mask = torch.ones_like(targets["mols_1"].a, dtype=torch.bool)

        if mask.sum() > 0:
            # Always use weighted cross-entropy loss
            a_loss = F.cross_entropy(
                a_pred[mask],
                targets["mols_1"].a[mask],
                weight=self.atom_type_weights,
            )
        else:
            # all tokens are correct, so no loss
            a_loss = torch.tensor(0.0, device=self.device)

        if self.edge_token_weights is None:
            raise ValueError(
                "edge_token_weights must be set before training. "
                "Call set_tokens_and_distribution() first."
            )

        edge_index_pred, e_pred_triu = get_canonical_upper_triangle_with_index(
            mols_t.edge_index, edge_type_pred
        )
        edge_index_target, e_target_triu = get_canonical_upper_triangle_with_index(
            targets["mols_1"].edge_index, targets["mols_1"].e
        )

        assert torch.all(edge_index_pred == edge_index_target), (
            "The edge indices must be the same."
        )

        e_loss = F.cross_entropy(
            e_pred_triu,
            e_target_triu,
            weight=self.edge_token_weights,
        )

        c_loss = F.cross_entropy(
            charge_pred,
            targets["mols_1"].c,
            weight=self.charge_token_weights,
        )

        x_loss = F.mse_loss(x_pred, targets["mols_1"].x)

        # TODO must add charges here later
        if self.cat_strategy == "uniform-sample":
            gmm_loss = typed_gmm_loss(
                gmm_dict,
                targets["atoms_to_birth"].x,
                targets["atoms_to_birth"].a,
                targets["atoms_to_birth"].batch,
            )
        else:
            gmm_loss = untyped_gmm_loss(
                gmm_dict,
                targets["atoms_to_birth"].x,
                targets["atoms_to_birth"].a,
                targets["atoms_to_birth"].batch,
            )

        death_rate_loss = rate_loss(death_rate_pred, targets["death_rate_target"])

        birth_rate_loss = rate_loss(birth_rate_pred, targets["birth_rate_target"])

        loss = (
            self.loss_weights.a_loss * a_loss
            + self.loss_weights.x_loss * x_loss
            + self.loss_weights.e_loss * e_loss
            + self.loss_weights.c_loss * c_loss
            + self.loss_weights.gmm_loss * gmm_loss
            + self.loss_weights.death_rate_loss * death_rate_loss
            + self.loss_weights.birth_rate_loss * birth_rate_loss
        )
        self.log("loss", loss, prog_bar=False, logger=True, batch_size=batch_size)
        self.log("a_loss", a_loss, prog_bar=False, logger=True, batch_size=batch_size)
        self.log("x_loss", x_loss, prog_bar=False, logger=True, batch_size=batch_size)
        self.log("e_loss", e_loss, prog_bar=False, logger=True, batch_size=batch_size)
        self.log(
            "gmm_loss", gmm_loss, prog_bar=False, logger=True, batch_size=batch_size
        )
        self.log(
            "death_rate_loss",
            death_rate_loss,
            prog_bar=False,
            logger=True,
            batch_size=batch_size,
        )
        self.log(
            "birth_rate_loss",
            birth_rate_loss,
            prog_bar=False,
            logger=True,
            batch_size=batch_size,
        )
        loss = self.safe_loss(loss)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # Define the validation step logic here
        loss = self.shared_step(batch, batch_idx)
        # Log val_loss at epoch level for ReduceLROnPlateau scheduler
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch[0].batch_size,
        )
        return loss

    def test_step(self, batch, batch_idx):
        # Define the test step logic here
        pass

    def predict_step(self, batch, batch_idx):
        """
        Inference step for flow matching.

        Args:
            batch: Batch of data (for now, we'll use batch size to determine number of graphs)
            batch_idx: Batch index

        Returns:
            Dictionary containing final samples
        """
        self.model.set_inference()
        if self.atom_tokens is None:
            raise ValueError(
                "tokens must be set before prediction. "
                "Call set_tokens_and_distribution() first."
            )

        # For inference, we start from noise and integrate forward from t=0 to t=1
        # For now, assume batch contains batch_size or we use a default
        # In practice, you might want to pass initial conditions or use a prior

        # Get batch size from the batch (assuming it's similar to training)
        mol_t, _ = batch

        batch_size = mol_t.batch_size
        batch_id = mol_t.batch

        _ = mol_t.remove_com()

        # Time parameters
        num_steps = self.num_integration_steps
        dt = 1.0 / num_steps
        t = torch.zeros(batch_size, device=self.device)  # Start at t=0

        # Hyperparameters
        cat_noise_level = self.cat_noise_level

        # Trajectory storage
        mol_traj = [mol_t.clone()]

        # previous outputs for self-conditioning. none at the beginning
        preds = None

        # Integration loop: integrate from t=0 to t=1
        for _ in range(num_steps):
            # TODO make choice flexible
            # Build kNN graph
            # edge_index = knn_graph(xt, k=self.k_nn_edges, batch=batch_id)
            # edge_index = build_fully_connected_edge_index(batch_id)
            # edge_type_ids = et[edge_index[0], edge_index[1]]

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

            c_pred = preds["charge_head"]  # (N_total, num_classes)
            c_pred = F.softmax(c_pred, dim=-1)

            # get only triu part of the edge types
            e_pred = preds["edge_type_head"]
            e_pred = F.softmax(e_pred, dim=-1)

            mol_1_pred = MoleculeBatch(
                x=x1_pred,
                a=a_pred,  # one-hot
                c=c_pred,  # one-hot
                e=e_pred,  # one-hot
                edge_index=mol_t.edge_index.clone(),
                batch=batch_id.clone(),
            )

            # (num_graphs, K + 2*K*D + K*N_types)
            gmm_pred = preds["gmm_head"]
            gmm_dict = compute_equivariant_gmm(
                gmm_pred,
                mol_1_pred.x,
                mol_1_pred.batch,
                self.K,
                len(self.atom_tokens) if self.cat_strategy == "uniform-sample" else 0,
            )

            # Process rates
            net_rate_pred = preds["net_rate_head"]  # (num_graphs, 1)
            # if we fix the number of atoms, we will not use the jump process
            if self.n_atoms_strategy == "fixed":
                death_rate = torch.zeros_like(net_rate_pred).squeeze(-1)
                birth_rate = torch.zeros_like(net_rate_pred).squeeze(-1)
            else:
                death_rate = F.relu(-net_rate_pred).squeeze(-1)  # (num_graphs,)
                birth_rate = F.relu(net_rate_pred).squeeze(-1)  # (num_graphs,)

            # Integrate one step
            mol_t = self.integrator.integrate_step_gnn(
                mol_t=mol_t,
                mol_1_pred=mol_1_pred,
                global_death_rate=death_rate,
                birth_rate=birth_rate,
                birth_gmm_dict=gmm_dict,
                t=t,
                dt=dt,
                cat_noise_level=cat_noise_level,
            )

            # remove mean from xt for each batch
            _ = mol_t.remove_com()

            # Update time forward
            # Number of graphs stays constant (batch_size)
            t = t + dt

            # Save  state to trajectory
            mol_traj.append(mol_t.clone())

        # rectify the trajectory such that we get a traj for each molecule
        traj_lists = [mol_traj_i.to_data_list() for mol_traj_i in mol_traj]

        traj_per_mol = [[] for _ in range(batch_size)]
        for i in range(batch_size):
            for mol_t in traj_lists:
                traj_per_mol[i].append(mol_t[i])

        # Return results
        return traj_per_mol

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
    model = LightningModule()
    trainer = pl.Trainer()
    trainer.fit(model)
