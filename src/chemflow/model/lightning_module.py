import torch
import pytorch_lightning as pl

import torch.nn as nn
import torch.optim as optim

from omegaconf import DictConfig
import hydra

import torch.nn.functional as F
from chemflow.losses import typed_gmm_loss, rate_loss
from torch_geometric.nn import knn_graph

from chemflow.flow_matching.interpolation import interpolate_different_size
from chemflow.utils import token_to_index


class LightningModule(pl.LightningModule):
    def __init__(self, tokens: list[str], model: DictConfig):
        super().__init__()
        self.tokens = tokens
        self.mask_index = token_to_index(tokens, "<MASK>")

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
        M = targets_batched["N_atoms"]
        batch_size = len(N)

        samples_batch_id = samples_batched["batch_index"]
        targets_batch_id = targets_batched["batch_index"]

        # interpolate
        t = torch.rand(batch_size)
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
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # Define the validation step logic here
        loss = self.shared_step(batch, batch_idx)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        # Define the test step logic here
        pass

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
