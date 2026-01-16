from typing import Dict, Optional, Tuple, Any
from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
import hydra
from chemflow.dataset.molecule_data import MoleculeBatch


class EmbeddingBackbone(nn.Module):
    """
    Embedding Module.
    Embeds atom features, charges, edge types, time, and node counts before passing to the backbone.
    """

    def __init__(
        self,
        atom_type_embedding_args: DictConfig,
        edge_type_embedding_args: DictConfig,
        charge_embedding_args: DictConfig,
        time_embedding_args: DictConfig,
        node_count_embedding_args: DictConfig,
    ):
        super().__init__()
        self.atom_type_embedding = hydra.utils.instantiate(atom_type_embedding_args)
        self.charge_embedding = hydra.utils.instantiate(charge_embedding_args)
        self.edge_type_embedding = hydra.utils.instantiate(edge_type_embedding_args)
        self.time_embedding = hydra.utils.instantiate(time_embedding_args)
        self.node_count_embedding = hydra.utils.instantiate(node_count_embedding_args)

    def forward(
        self,
        a: torch.Tensor,
        c: torch.Tensor,
        e: torch.Tensor,
        edge_index: torch.Tensor,
        t: torch.Tensor,
        batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        N_nodes = torch.bincount(batch)

        a_embed = self.atom_type_embedding(a)
        c_embed = self.charge_embedding(c)

        # Calculate conditioning embeddings
        # Ensure we index correctly based on the batch size
        N_nodes_embedding = self.node_count_embedding(N_nodes)[batch]
        t_embedding = self.time_embedding(t)[batch]

        # Concatenate all embeddings
        embeddings_to_concat = [a_embed, c_embed, N_nodes_embedding, t_embedding]
        h_0 = torch.cat(embeddings_to_concat, dim=-1)

        # Process edge embeddings
        e_embed = self.edge_type_embedding(e)

        # Ensure edge_index is formatted correctly (tuple for some backbones, tensor for others)
        # Assuming tuple is required based on original code
        edge_index_tuple = (edge_index[0], edge_index[1])

        return h_0, edge_index_tuple, e_embed


class BackboneWithHeads(nn.Module):
    """
    Full Model: Embedding -> EGNN Backbone -> Prediction Heads.
    """

    def __init__(
        self,
        # Embedding args
        atom_type_embedding_args: DictConfig,
        edge_type_embedding_args: DictConfig,
        charge_embedding_args: DictConfig,
        time_embedding_args: DictConfig,
        node_count_embedding_args: DictConfig,
        # Backbone model args
        backbone_model_args: DictConfig,
        # Heads args
        heads_args: DictConfig,
        ins_gmm_head_args: DictConfig,
        ins_edge_head_args: DictConfig,
    ):
        super().__init__()

        # 1. Instantiate the Embedding Layer manually using the provided args
        self.embedding_backbone = EmbeddingBackbone(
            atom_type_embedding_args=atom_type_embedding_args,
            edge_type_embedding_args=edge_type_embedding_args,
            charge_embedding_args=charge_embedding_args,
            time_embedding_args=time_embedding_args,
            node_count_embedding_args=node_count_embedding_args,
        )

        # 2. Instantiate the main Backbone (e.g., EGNN)
        self.backbone = hydra.utils.instantiate(backbone_model_args)

        # 3. Instantiate Heads
        self.heads = hydra.utils.instantiate(heads_args)

        # 4. Instantiate Insertion Heads
        self.ins_edge_head = hydra.utils.instantiate(ins_edge_head_args)
        self.ins_gmm_head = hydra.utils.instantiate(ins_gmm_head_args)

    def set_training(self):
        pass

    def set_inference(self):
        pass

    def predict_insertion_edges(
        self,
        out_dict: Dict[str, Any],
        batch: torch.Tensor,
        spawn_node_idx: torch.Tensor,
        target_node_idx: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Predict edge types between insertion points and existing nodes."""

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
    ) -> Dict[str, Any]:
        """Forward pass through Embedding -> Backbone -> Heads."""
        x, a, c, e, edge_index, batch = mols_t.unpack()

        # 1. Embedding Pass
        h_0, edge_index_tuple, e_embed = self.embedding_backbone(
            a, c, e, edge_index, t, batch
        )

        # 2. Backbone Pass
        # self.backbone returns features, coordinates, and edge attrs
        h, x_out, e_out = self.backbone(h_0, x, edge_index_tuple, e_embed)

        # 3. Heads Pass
        out_dict = self.heads(h, batch, e_out)

        out_dict["pos_head"] = x_out
        out_dict["gmm_head"] = self.ins_gmm_head(h, x_out, edge_index_tuple)

        # Store latent features
        out_dict["h_latent"] = h

        # Enforce positive rates
        for key in out_dict:
            if "rate" in key:
                out_dict[key] = F.softplus(out_dict[key])

        return out_dict
