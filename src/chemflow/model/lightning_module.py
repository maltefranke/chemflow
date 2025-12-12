import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
import hydra
from chemflow.losses import typed_gmm_loss, rate_loss
from chemflow.losses import gmm_loss as untyped_gmm_loss
from torch_geometric.nn import knn_graph

from chemflow.flow_matching.integration import Integrator
from chemflow.utils import token_to_index
from external_code.egnn import unsorted_segment_mean
from chemflow.flow_matching.interpolation import Interpolator
from chemflow.flow_matching.gmm import compute_equivariant_gmm


class LightningModule(pl.LightningModule):
    def __init__(
        self,
        model: DictConfig = None,
        loss_weights: DictConfig = None,
        optimizer_config: DictConfig = None,
        weight_alpha: float = 1.0,
        k_nn_edges: int = 20,
        typed_gmm: bool = True,
        N_samples: int = 20,
        K: int = 10,
        D: int = 3,
        n_atoms_strategy: str = "fixed",
        cat_strategy: str = "uniform-sample", # "mask" or "uniform-sample"
        type_loss_token_weights: str = "uniform", # "uniform" or "training"
        num_integration_steps: int = 100,
        cat_noise_level: float = 1.0,
        coord_noise_level: float = 0.0,
    ):
        super().__init__()

        # Will be set via setter method
        self.tokens = None
        self.mask_index = None
        self.death_token_index = None
        self.interpolator = None
        self.integrator = None

        self.weight_alpha = weight_alpha

        self.typed_gmm = typed_gmm
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
        # Lightning will handle device placement automatically

    def set_tokens_and_distribution(
        self,
        tokens: list[str],
        atom_type_distribution: torch.Tensor,
    ):
        """Set tokens and atom type distribution after initialization."""
        self.tokens = tokens
        self.mask_index = token_to_index(self.tokens, "<MASK>")
        self.death_token_index = token_to_index(self.tokens, "<DEATH>")

        self.interpolator = Interpolator(
            self.tokens,
            atom_type_distribution.to(self.device),
            typed_gmm=self.typed_gmm,
            N_samples=self.N_samples,
        )
        self.integrator = Integrator(
            self.tokens,
            self.K,
            self.D,
            self.typed_gmm,
            self.cat_strategy,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        # Always compute token distribution weights for weighted cross-entropy loss
        weights = self._compute_token_weights(atom_type_distribution)
        self.register_buffer("token_weights", weights)

    def _compute_token_weights(
        self, atom_type_distribution: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute token weights with special handling for special tokens.

        Args:
            atom_type_distribution: Distribution of atom types from training data

        Returns:
            Weights tensor with same shape as atom_type_distribution
        """
        # --- 1. Initial Weight Calculation (Inverse Frequency) ---
        epsilon = 1e-8
        weights = 1.0 / (atom_type_distribution + epsilon)
        weights = weights**self.weight_alpha  # Apply alpha scaling

        # --- 2. Isolate & Normalize REGULAR Tokens ---
        special_token_indices = {self.mask_index, self.death_token_index}
        all_indices = set(range(len(self.tokens)))
        # Convert to list for indexing
        regular_token_indices = list(all_indices - special_token_indices)

        if not regular_token_indices:
            # Fallback if no regular tokens (unlikely, but good to handle)
            final_weights = torch.ones_like(weights)
            final_weights[self.mask_index] = 0.0
            # Death token gets a default 'high' weight of 1.0
            final_weights[self.death_token_index] = 1.0
            return final_weights

        # Get weights for only the regular tokens
        regular_weights = weights[regular_token_indices]

        # Normalize *only* the regular weights to have a mean of 1.0
        mean_regular_weight = regular_weights.mean()
        if mean_regular_weight > 0:
            regular_weights = regular_weights / mean_regular_weight

        # Find the max and min weight *after* normalization
        max_regular_weight = regular_weights.max()
        min_regular_weight = regular_weights.min()

        # --- 3. Build Final Weights Tensor ---
        # Start with zeros
        final_weights = torch.zeros_like(weights)

        # Assign normalized regular weights to their correct positions
        final_weights[regular_token_indices] = regular_weights

        # Assign special token weights based on the normalized regular weights
        # TODO: death token weight could be improved / set dynamically
        # TODO: e.g. look at N_t and compare to data, then weight accordingly
        final_weights[self.mask_index] = 0.0
        final_weights[self.death_token_index] = min_regular_weight
        if self.type_loss_token_weights == "uniform":
            final_weights = torch.ones_like(final_weights)

        return final_weights

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
        # Define the training step logic here
        if self.tokens is None:
            raise ValueError(
                "tokens must be set before training. "
                "Call set_tokens_and_distribution() first."
            )

        samples_batched, targets_batched = batch

        batch_size = batch[-1]["N_atoms"].shape[-1]

        x0 = samples_batched["coord"]
        x1 = targets_batched["coord"]

        a0_ind = samples_batched["atom_types"]
        a1_ind = targets_batched["atom_types"]
        # to one-hot
        a0 = F.one_hot(a0_ind, num_classes=len(self.tokens))
        a1 = F.one_hot(a1_ind, num_classes=len(self.tokens))

        N = samples_batched["N_atoms"]
        batch_size = len(N)

        samples_batch_id = samples_batched["batch_index"]
        targets_batch_id = targets_batched["batch_index"]

        # interpolate
        #t = torch.rand(batch_size, device=self.device)
        
        t = self.time_dist.sample((batch_size,)).to(self.device)
        # NOTE WARNING t is not sampled
        # t = 0.65 + 0.35 * torch.rand(batch_size, device=self.device)
        xt, at, xt_batch_id, targets = self.interpolator.interpolate_different_size(
            x0, a0, samples_batch_id, x1, a1, targets_batch_id, t
        )
        # convert one-hot interpolated types back to indices, as input to nn.Embedding
        at_ind = torch.argmax(at, dim=-1)

        # build kNN graph
        edge_index = knn_graph(xt, k=self.k_nn_edges, batch=xt_batch_id)

        preds = self.model(at_ind, xt, edge_index, t.view(-1, 1), batch=xt_batch_id)

        a_pred = preds["class_head"]
        x_pred = preds["pos_head"]
        gmm_pred = preds["gmm_head"]
        gmm_dict = compute_equivariant_gmm(
            gmm_pred,
            xt,
            xt_batch_id,
            self.K,
            len(self.tokens) if self.typed_gmm else 0,
        )

        net_rate_pred = preds["net_rate_head"]
        death_rate_pred = F.relu(-net_rate_pred)
        birth_rate_pred = F.relu(net_rate_pred)
        # Calculate losses
        # only do class prediction on non-mask tokens
        if self.token_weights is None:
            raise ValueError(
                "token_weights must be set before training. "
                "Call set_tokens_and_distribution() first."
            )

        if self.cat_strategy == "mask":
            # predict final class only for mask tokens
            mask = at_ind != self.mask_index
        else:
            # predict final class for all tokens
            mask = torch.ones_like(at_ind, dtype=torch.bool)

        if mask.sum() > 0:
            # Always use weighted cross-entropy loss
            a_loss = F.cross_entropy(
                a_pred[mask],
                targets["target_c"][mask].argmax(dim=-1),
                weight=self.token_weights,
            )
        else:
            # all tokens are correct, so no loss
            a_loss = torch.tensor(0.0, device=self.device)

        x_loss = F.mse_loss(x_pred, targets["target_x"])

        if self.typed_gmm:
            gmm_loss = typed_gmm_loss(
                gmm_dict,
                targets["birth_locations"],
                targets["birth_types"],
                targets["birth_batch_ids"],
            )
        else:
            gmm_loss = untyped_gmm_loss(
                gmm_dict,
                targets["birth_locations"],
                targets["birth_batch_ids"],
            )

        death_rate_loss = rate_loss(death_rate_pred, targets["death_rate_target"])

        birth_rate_loss = rate_loss(birth_rate_pred, targets["birth_rate_target"])

        loss = (
            self.loss_weights.a_loss * a_loss
            + self.loss_weights.x_loss * x_loss
            + self.loss_weights.gmm_loss * gmm_loss
            + self.loss_weights.death_rate_loss * death_rate_loss
            + self.loss_weights.birth_rate_loss * birth_rate_loss
        )
        self.log("loss", loss, prog_bar=False, logger=True, batch_size=batch_size)
        self.log("a_loss", a_loss, prog_bar=False, logger=True, batch_size=batch_size)
        self.log("x_loss", x_loss, prog_bar=False, logger=True, batch_size=batch_size)
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
        self.log("train_loss", loss, prog_bar=False, logger=True)
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
            batch_size=batch[-1]["N_atoms"].shape[-1],
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
        if self.tokens is None or self.mask_index is None:
            raise ValueError(
                "tokens must be set before prediction. "
                "Call set_tokens_and_distribution() first."
            )

        # For inference, we start from noise and integrate forward from t=0 to t=1
        # For now, assume batch contains batch_size or we use a default
        # In practice, you might want to pass initial conditions or use a prior

        # Get batch size from the batch (assuming it's similar to training)
        samples_batched, _ = batch

        xt = samples_batched["coord"]
        at_ind = samples_batched["atom_types"]
        at = F.one_hot(at_ind, num_classes=len(self.tokens))

        N = samples_batched["N_atoms"]
        batch_size = len(N)

        batch_id = samples_batched["batch_index"]

        # Time parameters
        num_steps = self.num_integration_steps
        dt = 1.0 / num_steps
        t = torch.zeros(batch_size, device=self.device)  # Start at t=0

        # Hyperparameters
        cat_noise_level = self.cat_noise_level

        # Trajectory storage
        xt_trajectory = [xt.clone()]
        at_trajectory = [at_ind.clone()]
        batch_id_trajectory = [batch_id.clone()]
        x_pred_trajectory = []
        a_pred_trajectory = []

        # Integration loop: integrate from t=0 to t=1
        for _ in range(num_steps):

            # Build kNN graph
            edge_index = knn_graph(xt, k=self.k_nn_edges, batch=batch_id)

            # Get model predictions
            preds = self.model(
                at_ind, xt, edge_index, t.view(-1, 1), batch=batch_id
            )
            # Extract predictions
            type_pred = preds["class_head"]  # (N_total, num_classes)
            type_pred = F.softmax(type_pred, dim=-1)

            x1_pred = preds["pos_head"]  # (N_total, D)

            # (num_graphs, K + 2*K*D + K*N_types)
            gmm_pred = preds["gmm_head"]
            gmm_dict = compute_equivariant_gmm(
                gmm_pred,
                xt,
                batch_id,
                self.K,
                len(self.tokens) if self.typed_gmm else 0,
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
            xt, at, batch_id = self.integrator.integrate_step_gnn(
                x_pred=x1_pred,
                type_pred=type_pred,
                global_death_rate=death_rate,
                birth_rate=birth_rate,
                birth_gmm_dict=gmm_dict,
                xt=xt,
                ct=at.float(),
                batch_id=batch_id,
                t=t,
                dt=dt,
                mask_index=self.mask_index,
                death_token_index=self.death_token_index,
                cat_noise_level=cat_noise_level,
            )

            # remove mean from xt for each batch
            xt_mean_batch = unsorted_segment_mean(xt, batch_id, batch_size)
            xt = xt - xt_mean_batch[batch_id]

            # Update time forward
            # Number of graphs stays constant (batch_size)
            t = t + dt

            # Save  state to trajectory
            at_ind = torch.argmax(at, dim=-1)
            xt_trajectory.append(xt.clone())
            at_trajectory.append(at_ind.clone())
            batch_id_trajectory.append(batch_id.clone())
            x_pred_trajectory.append(x1_pred.clone())
            a_pred_trajectory.append(type_pred.clone())

        # Return results
        return {
            "final_coord": xt,
            "final_atom_types": at_ind,
            "final_batch_index": batch_id,
            "xt_trajectory": xt_trajectory,
            "at_trajectory": at_trajectory,
            "batch_id_trajectory": batch_id_trajectory,
            "x_pred_trajectory": x_pred_trajectory,
            "a_pred_trajectory": a_pred_trajectory,
        }

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
            optimizer.zero_grad()
        else:
            # No NaN gradients found, proceed with the optimizer step
            optimizer.step()


if __name__ == "__main__":
    # Instantiate the LightningModule and run the training loop
    model = LightningModule()
    trainer = pl.Trainer()
    trainer.fit(model)
