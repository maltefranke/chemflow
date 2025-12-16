from src.external_code.egnn import EGNN
from omegaconf import DictConfig
import torch.nn as nn
import torch
import hydra
import random

from src.chemflow.model.embedding import SinusoidalEmbedding
from src.chemflow.model.self_conditioning import SelfConditioningResidualLayer

from chemflow.dataset.molecule_data import MoleculeBatch


class EGNNWithEdgeType(EGNN):
    def forward(self, h, x, edges, edge_attr):
        h = self.embedding_in(h)
        for i in range(0, self.n_layers):
            h, x, edge_attr = self._modules["gcl_%d" % i](
                h, edges, x, edge_attr=edge_attr
            )
        h = self.embedding_out(h)
        edge_attr = self.embedding_out(edge_attr)
        return h, x, edge_attr


class BaseEGNN(nn.Module):
    """Base class for EGNN models. Embeds the atom feats and passes the data through the EGNN."""

    def __init__(
        self,
        atom_type_embedding_args: DictConfig,
        edge_type_embedding_args: DictConfig,
        egnn_args: DictConfig,
    ):
        super().__init__()
        self.atom_type_embedding = hydra.utils.instantiate(atom_type_embedding_args)
        self.edge_type_embedding = hydra.utils.instantiate(edge_type_embedding_args)
        self.egnn = hydra.utils.instantiate(egnn_args)

    def forward(self, atom_feats, coord, edge_index, edge_type_ids=None):
        # first embed the atom feats
        h = self.atom_type_embedding(atom_feats)

        if edge_type_ids is not None:
            edge_type_embeddings = self.edge_type_embedding(edge_type_ids)
            edge_attr = edge_type_embeddings
        else:
            edge_attr = None

        edge_index = (edge_index[0], edge_index[1])

        # then pass the data through the EGNN
        h, coord, edge_attr = self.egnn(h, coord, edge_index, edge_attr)

        return h, coord, edge_attr


class EGNNwithHeads(BaseEGNN):
    """EGNN model with heads. Passes the data through an embedding layer, an EGNN and then through the heads."""

    def __init__(
        self,
        atom_type_embedding_args: DictConfig,
        edge_type_embedding_args: DictConfig,
        egnn_args: DictConfig,
        heads_args: DictConfig,
        self_conditioning: bool = False,
    ):
        super().__init__(atom_type_embedding_args, edge_type_embedding_args, egnn_args)
        self.time_embedding = nn.Sequential(
            nn.Linear(1, atom_type_embedding_args["out_nf"]),
            nn.SiLU(),
            nn.Linear(
                atom_type_embedding_args["out_nf"], atom_type_embedding_args["out_nf"]
            ),
        )
        self.sinusoidal_embedding = SinusoidalEmbedding(
            atom_type_embedding_args["out_nf"]
        )
        self.heads = hydra.utils.instantiate(heads_args)

        self.self_conditioning = self_conditioning

        # Initialize self-conditioning residual layer if enabled
        if self.self_conditioning:
            n_atom_types = atom_type_embedding_args["in_nf"]
            n_bond_types = edge_type_embedding_args["in_nf"]
            node_embedding_dim = atom_type_embedding_args["out_nf"]
            edge_embedding_dim = edge_type_embedding_args["out_nf"]

            self.self_conditioning_residual_layer = SelfConditioningResidualLayer(
                n_atom_types=n_atom_types,
                n_bond_types=n_bond_types,
                node_embedding_dim=node_embedding_dim,
                edge_embedding_dim=edge_embedding_dim,
                rbf_dim=16,  # Can be made configurable
                rbf_dmax=5.0,  # Can be made configurable
            )
        else:
            self.self_conditioning_residual_layer = None

        self.training = True

    def set_training(self):
        self.training = True

    def set_inference(self):
        self.training = False

    def embed(self, atom_feats, edge_index, t, batch, edge_type_ids=None):
        N_nodes = torch.bincount(batch)

        h = self.atom_type_embedding(atom_feats)

        # calculate conditioning embeddings
        N_nodes_embedding = self.sinusoidal_embedding(N_nodes)[batch]
        t_embedding = self.time_embedding(t)[batch]
        h = h + N_nodes_embedding + t_embedding

        edge_index = (edge_index[0], edge_index[1])

        if edge_type_ids is not None:
            edge_attr = self.edge_type_embedding(edge_type_ids)
        else:
            edge_attr = None

        return h, edge_index, edge_attr

    def denoise_graph(self, h, coord, edge_index, edge_attr, batch):
        # then pass the data through the EGNN
        h, coord, edge_attr = self.egnn(h, coord, edge_index, edge_attr=edge_attr)

        # Pass through heads
        if batch is not None:
            out_dict = self.heads(h, batch, edge_attr=edge_attr)
        else:
            # If no batch info provided, only node-level heads will work
            out_dict = self.heads(h, batch=None, edge_attr=edge_attr)

        out_dict["pos_head"] = coord

        return out_dict

    def forward(
        self,
        mols_t: MoleculeBatch,
        t: torch.Tensor,
        prev_outs=None,
        is_random_self_conditioning: bool = False,
    ):
        """
        Forward pass through EGNN with heads.

        Args:
            atom_feats: Node features
            coord: Node coordinates
            edge_index: Edge indices
            edge_type_ids: Edge type ids (optional)
            batch: Batch assignment for each node (required for graph-level heads)
            prev_outs: Previous outputs from the model for self-conditioning (optional)
            is_random_self_conditioning: Coin

        Returns:
            Dictionary mapping head names to their outputs
        """
        x, a, c, e, edge_index, batch = mols_t.unpack()
        h, edge_index, e = self.embed(a, edge_index, t, batch, e)

        # NOTE using FlowMol3 self-conditioning logic here. Their description:
        # if we are using self-conditoning, and prev_outs is None, then
        # we must be in the process of training a self-conditioning model, and need to
        # enter the following logic branch:
        # with p = 0.5, we do a gradient-stopped pass through denoise_graph to get
        # predicted endpoint, then set prev_dst_dict to this predicted endpoint
        # for the other 0.5 of the time, we do nothing!
        # if in the first timestep of inference, we need to first generate the endpoint

        if self.self_conditioning and prev_outs is None:
            train_self_condition = self.training and is_random_self_conditioning
            inference_first_step = not self.training and (t == 0).all().item()

            if train_self_condition or inference_first_step:
                with torch.no_grad():
                    prev_outs = self.denoise_graph(h, x, edge_index, e, batch)

        if self.self_conditioning and prev_outs is not None:
            # if prev_outs is not none, we need to pass through the self-conditioning residual block

            # Handle case where edge_attr might be None
            # Create dummy edge attributes if not provided (shouldn't happen in practice)
            if e is None:
                print("No edge attr provided, creating dummy edge attr")
                e = torch.zeros(
                    edge_index[0].shape[0],
                    self.edge_type_embedding.embedding_dim,
                    device=h.device,
                    dtype=h.dtype,
                )

            h, x, e = self.self_conditioning_residual_layer(
                h=h,
                coord=x,
                edge_index=edge_index,
                edge_attr=e,
                prev_outs=prev_outs,
                atom_types=a,  # Pass atom type indices
            )

        outs = self.denoise_graph(h, x, edge_index, e, batch)
        return outs
