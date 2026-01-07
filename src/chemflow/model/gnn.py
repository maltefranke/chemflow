from src.external_code.egnn import EGNN
from omegaconf import DictConfig, OmegaConf
import torch.nn as nn
import torch
import hydra
import random

from src.chemflow.model.embedding import (
    CountEmbedding,
    TimeEmbedding,
    RBFEmbedding,
)
from src.chemflow.model.self_conditioning import SelfConditioningResidualLayer

from chemflow.dataset.molecule_data import MoleculeBatch


class EGNNWithEdgeType(EGNN):
    def __init__(self, *args, rbf_embedding_args: DictConfig = None, **kwargs):
        super().__init__(*args, **kwargs)
        out_node_nf = kwargs.get("out_node_nf", 0)

        """# RBF embedding for distance encoding
        if rbf_embedding_args is not None:
            # Check if it's already instantiated (Hydra may auto-instantiate nested configs)
            if isinstance(rbf_embedding_args, DictConfig):
                # It's a config, instantiate it ourselves
                self.rbf_embedding = hydra.utils.instantiate(rbf_embedding_args)
                rbf_out_dim = rbf_embedding_args.get(
                    "out_dim", rbf_embedding_args.get("num_rbf", 16)
                )
            else:
                # Already instantiated by Hydra, use as-is
                self.rbf_embedding = rbf_embedding_args
                # Get out_dim from the RBFEmbedding object
                rbf_out_dim = getattr(self.rbf_embedding, "out_dim", 16)
        else:
            # Fallback to default if not provided
            self.rbf_embedding = RBFEmbedding(num_rbf=16, rbf_dmax=10.0)
            rbf_out_dim = 16"""

        self.rbf_embedding = hydra.utils.instantiate(rbf_embedding_args)
        rbf_out_dim = rbf_embedding_args.get(
            "out_dim", rbf_embedding_args.get("num_rbf", 16)
        )

        # Update input dimension to account for RBF embedding instead of raw distance
        self.edge_embedding_out = nn.Linear(2 * out_node_nf + rbf_out_dim, out_node_nf)

    def forward(self, h, x, edges, edge_attr):
        h = self.embedding_in(h)
        for i in range(0, self.n_layers):
            h, x, _ = self._modules["gcl_%d" % i](h, edges, x, edge_attr=edge_attr)
        h = self.embedding_out(h)

        rows, cols = edges

        # Collect features for every edge
        h_i = h[rows]  # [E, hidden]
        h_j = h[cols]  # [E, hidden]

        # Calculate final distances (actual distance, not squared)
        dist_vec = x[rows] - x[cols]  # [E, 3]
        dist = torch.norm(dist_vec, dim=1)  # [E]

        # Embed distances using RBF
        dist_emb = self.rbf_embedding(dist)  # [E, rbf_out_dim]

        # Concatenate: [Source Node, Target Node, RBF Distance Embedding]
        edge_inputs = torch.cat([h_i, h_j, dist_emb], dim=-1)

        # Predict
        edge_emb = self.edge_embedding_out(edge_inputs)

        return h, x, edge_emb


class BaseEGNN(nn.Module):
    """Base class for EGNN models. Embeds the atom feats and passes the data through the EGNN."""

    def __init__(
        self,
        atom_type_embedding_args: DictConfig,
        edge_type_embedding_args: DictConfig,
        charge_embedding_args: DictConfig,
        time_embedding_args: DictConfig,
        node_count_embedding_args: DictConfig,
        egnn_args: DictConfig,
    ):
        super().__init__()
        self.atom_type_embedding = hydra.utils.instantiate(atom_type_embedding_args)
        self.charge_embedding = hydra.utils.instantiate(charge_embedding_args)
        self.edge_type_embedding = hydra.utils.instantiate(edge_type_embedding_args)
        self.time_embedding = hydra.utils.instantiate(time_embedding_args)
        self.node_count_embedding = hydra.utils.instantiate(node_count_embedding_args)
        self.egnn = hydra.utils.instantiate(egnn_args)


class EGNNwithHeads(BaseEGNN):
    """EGNN model with heads. Passes the data through an embedding layer, an EGNN and then through the heads."""

    def __init__(
        self,
        atom_type_embedding_args: DictConfig,
        edge_type_embedding_args: DictConfig,
        charge_embedding_args: DictConfig,
        time_embedding_args: DictConfig,
        node_count_embedding_args: DictConfig,
        egnn_args: DictConfig,
        heads_args: DictConfig,
        gmm_head_args: DictConfig,
        self_conditioning: bool = False,
        ins_edge_head_args: DictConfig = None,
    ):
        super().__init__(
            atom_type_embedding_args,
            edge_type_embedding_args,
            charge_embedding_args,
            time_embedding_args,
            node_count_embedding_args,
            egnn_args,
        )

        self.heads = hydra.utils.instantiate(heads_args)
        self.gmm_head = hydra.utils.instantiate(gmm_head_args)

        # Insertion edge head for predicting edges between new insertions and existing nodes
        if ins_edge_head_args is not None:
            self.ins_edge_head = hydra.utils.instantiate(ins_edge_head_args)
        else:
            self.ins_edge_head = None

        self.self_conditioning = self_conditioning

        # Initialize self-conditioning residual layer if enabled
        if self.self_conditioning:
            n_atom_types = atom_type_embedding_args["num_embeddings"]
            n_bond_types = edge_type_embedding_args["num_embeddings"]
            node_embedding_dim = atom_type_embedding_args["out_dim"]
            edge_embedding_dim = edge_type_embedding_args["out_dim"]

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

    def embed(self, atom_feats, charges, edge_index, t, batch, edge_type_ids=None):
        N_nodes = torch.bincount(batch)

        a_embed = self.atom_type_embedding(atom_feats)
        c_embed = self.charge_embedding(charges)

        # calculate conditioning embeddings
        N_nodes_embedding = self.node_count_embedding(N_nodes)[batch]
        t_embedding = self.time_embedding(t)[batch]

        # Concatenate all embeddings instead of adding
        embeddings_to_concat = [a_embed, c_embed, N_nodes_embedding, t_embedding]

        h = torch.cat(embeddings_to_concat, dim=-1)

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

        out_dict["gmm_head"] = self.gmm_head(h, coord, edge_index)

        # Store latent features for insertion edge prediction
        out_dict["h_latent"] = h

        return out_dict

    def predict_insertion_edges(
        self,
        out_dict: dict,
        batch: torch.Tensor,
        spawn_node_idx: torch.Tensor,
        target_node_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict edge types between insertion points and existing nodes.

        Args:
            out_dict: Output dictionary from denoise_graph (contains h_latent, gmm_head)
            batch: Batch assignment for each node
            spawn_node_idx: Indices of spawn nodes (nodes that predict/trigger insertions)
            target_node_idx: Indices of existing nodes to predict edges to

        Returns:
            edge_logits: [E_ins, n_edge_types] - Edge type logits for each pair
        """
        if self.ins_edge_head is None:
            return None

        h = out_dict["h_latent"]
        x = out_dict["pos_head"]
        gmm_dict = out_dict["gmm_head"]

        return self.ins_edge_head(
            h=h,
            x=x,
            gmm_dict=gmm_dict,
            batch=batch,
            spawn_node_idx=spawn_node_idx,
            target_node_idx=target_node_idx,
        )

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
        h, edge_index, e = self.embed(a, c, edge_index, t, batch, e)

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
