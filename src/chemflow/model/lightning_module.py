import torch
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from omegaconf import DictConfig
import hydra
from chemflow.losses import typed_gmm_loss, rate_loss
from torch_geometric.nn import knn_graph

from chemflow.flow_matching.interpolation import interpolate_different_size
from chemflow.flow_matching.integration import integrate_step_gnn
from chemflow.utils import token_to_index


class LightningModule(pl.LightningModule):
    def __init__(self, tokens: list[str], model: DictConfig):
        super().__init__()
        self.tokens = tokens
        self.mask_index = token_to_index(tokens, "<MASK>")
        self.death_token_index = token_to_index(tokens, "<DEATH>")

        self.model = hydra.utils.instantiate(model)
        self.model.to(self.device)

    def forward(self, x):
        # Define the forward pass of your model here
        pass

    def shared_step(self, batch, batch_idx):
        # Define the training step logic here
        samples_batched, targets_batched = batch

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
        t = torch.rand(batch_size, device=self.device)
        xt, at, xt_batch_id, targets = interpolate_different_size(
            x0, a0, samples_batch_id, x1, a1, targets_batch_id, t, self.tokens
        )
        # convert one-hot interpolated types back to indices, as input to nn.Embedding
        at_ind = torch.argmax(at, dim=-1)

        # build kNN graph
        # TODO: make k a hyperparameter
        edge_index = knn_graph(xt, k=10, batch=xt_batch_id)

        preds = self.model(at_ind, xt, edge_index, batch=xt_batch_id)

        a_pred = preds["class_head"]
        x_pred = preds["pos_head"]
        gmm_pred = preds["gmm_head"]

        net_rate_pred = preds["net_rate_head"]
        death_rate_pred = F.relu(-net_rate_pred)
        birth_rate_pred = F.relu(net_rate_pred)

        # Calculate losses
        # only do class prediction on non-mask tokens
        mask = at_ind != self.mask_index
        a_loss = F.cross_entropy(a_pred[mask], targets["target_c"][mask])

        x_loss = F.mse_loss(x_pred, targets["target_x"])

        gmm_loss = typed_gmm_loss(
            gmm_pred,
            targets["birth_locations"],
            targets["birth_types"],
            targets["birth_batch_ids"],
            D=x0.shape[-1],
            K=10,  # TODO make K a hyperparameter
            N_types=len(self.tokens),
        )

        death_rate_loss = rate_loss(death_rate_pred, targets["death_rate_target"])

        birth_rate_loss = rate_loss(birth_rate_pred, targets["birth_rate_target"])

        loss = a_loss + x_loss + gmm_loss + death_rate_loss + birth_rate_loss
        self.log("a_loss", a_loss, prog_bar=True)
        self.log("x_loss", x_loss, prog_bar=True)
        self.log("gmm_loss", gmm_loss, prog_bar=True)
        self.log("death_rate_loss", death_rate_loss, prog_bar=True)
        self.log("birth_rate_loss", birth_rate_loss, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        # self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # Define the validation step logic here
        loss = self.shared_step(batch, batch_idx)
        self.log("val_loss", loss, prog_bar=True)
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
        D = xt.shape[-1]

        xt_mask = torch.ones(xt.shape[0], dtype=torch.bool, device=self.device)
        batch_id = samples_batched["batch_index"]

        # Time parameters
        num_steps = 100  # TODO: make this a hyperparameter
        dt = 1.0 / num_steps
        t = torch.zeros(batch_size, device=self.device)  # Start at t=0

        # Hyperparameters
        K = 10  # TODO: make this a hyperparameter
        cat_noise_level = 0.0  # TODO: make this a hyperparameter

        # Trajectory storage
        xt_trajectory = [xt.clone()]
        at_trajectory = [at_ind.clone()]
        xt_mask_trajectory = [xt_mask.clone()]

        # Integration loop: integrate from t=0 to t=1
        for _ in range(num_steps):
            # Build kNN graph
            edge_index = knn_graph(xt, k=10, batch=batch_id)

            # Get model predictions
            with torch.no_grad():
                preds = self.model(at_ind, xt, edge_index, batch=batch_id)

            # Extract predictions
            type_pred = preds["class_head"]  # (N_total, num_classes)
            velocity = preds["pos_head"]  # (N_total, D)
            # (num_graphs, K + 2*K*D + K*N_types)
            gmm_pred = preds["gmm_head"]
            net_rate_pred = preds["net_rate_head"]  # (num_graphs, 1)

            # Process rates
            death_rate = F.relu(-net_rate_pred).squeeze(-1)  # (num_graphs,)
            birth_rate = F.relu(net_rate_pred).squeeze(-1)  # (num_graphs,)

            # Integrate one step
            xt, at, xt_mask, batch_id = integrate_step_gnn(
                velocity=velocity,
                type_pred=type_pred,
                global_death_rate=death_rate,
                birth_rate=birth_rate,
                birth_gmm_params=gmm_pred,
                xt=xt,
                ct=at.float(),
                xt_mask=xt_mask,
                batch_id=batch_id,
                t=t,
                dt=dt,
                K=K,
                D=D,
                mask_index=self.mask_index,
                death_token_index=self.death_token_index,
                cat_noise_level=cat_noise_level,
                device=self.device,
            )

            # Update time forward
            # Number of graphs stays constant (batch_size)
            t = t + dt

            # Early stopping if all graphs are empty
            if xt_mask.sum() == 0:
                break

            # Save  state to trajectory
            at_ind = torch.argmax(at, dim=-1)
            xt_trajectory.append(xt.clone())
            at_trajectory.append(at_ind.clone())
            xt_mask_trajectory.append(xt_mask.clone())


        # Return results
        return {
            "final_coord": xt,
            "final_atom_types": at_ind,
            "final_batch_index": batch_id,
            "final_mask": xt_mask,
            "xt_trajectory": xt_trajectory,
            "at_trajectory": at_trajectory,
            "xt_mask_trajectory": xt_mask_trajectory,
        }

    def configure_optimizers(self):
        # Define your optimizer and learning rate scheduler here
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.85, patience=10
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }


if __name__ == "__main__":
    # Instantiate the LightningModule and run the training loop
    model = LightningModule()
    trainer = pl.Trainer()
    trainer.fit(model)
