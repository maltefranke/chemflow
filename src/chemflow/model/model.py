from typing import Dict, Optional, Tuple, Any
from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
import hydra
from chemflow.dataset.molecule_data import MoleculeBatch


class FeatureProjector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.activation = nn.SiLU()
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = self.activation(x)
        return self.norm(x)


class EmbeddingBackbone(nn.Module):
    """
    Embedding Module.
    Embeds atom features, edge types, time, and node counts before passing to the backbone.
    Optionally embeds molecular properties for property-conditional generation.
    """

    def __init__(
        self,
        atom_type_embedding_args: DictConfig,
        edge_type_embedding_args: DictConfig,
        time_embedding_args: DictConfig,
        node_count_embedding_args: DictConfig,
        property_embedding_args: Optional[DictConfig] = None,
        bond_degree_embedding_args: Optional[DictConfig] = None,
        natoms_cfg_embedding_args: Optional[DictConfig] = None,
        *,
        h0_input_dim: int,
        h0_projection_dim: int,
        e_input_dim: int,
        e_projection_dim: int,
    ):
        super().__init__()
        self.atom_type_embedding = hydra.utils.instantiate(atom_type_embedding_args)
        self.edge_type_embedding = hydra.utils.instantiate(edge_type_embedding_args)
        self.time_embedding = hydra.utils.instantiate(time_embedding_args)
        self.node_count_embedding = hydra.utils.instantiate(node_count_embedding_args)

        self.property_embedding = None
        if property_embedding_args is not None:
            self.property_embedding = hydra.utils.instantiate(property_embedding_args)

        self.bond_degree_embedding = None
        if bond_degree_embedding_args is not None:
            self.bond_degree_embedding = hydra.utils.instantiate(
                bond_degree_embedding_args
            )

        self.natoms_cfg_embedding = None
        if natoms_cfg_embedding_args is not None:
            self.natoms_cfg_embedding = hydra.utils.instantiate(
                natoms_cfg_embedding_args
            )

        self.h0_projection = FeatureProjector(h0_input_dim, h0_projection_dim)
        self.e_projection = FeatureProjector(e_input_dim, e_projection_dim)

    def forward(
        self,
        a: torch.Tensor,
        e: torch.Tensor,
        edge_index: torch.Tensor,
        t: torch.Tensor,
        batch: torch.Tensor,
        properties: Optional[torch.Tensor] = None,
        property_drop_mask: Optional[torch.Tensor] = None,
        target_n_atoms: Optional[torch.Tensor] = None,
        natoms_drop_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        N_nodes = torch.bincount(batch)

        a_embed = self.atom_type_embedding(a)

        # Calculate conditioning embeddings
        # Ensure we index correctly based on the batch size
        N_nodes_embedding = self.node_count_embedding(N_nodes)[batch]
        t_embedding = self.time_embedding(t)[batch]

        # Concatenate all embeddings
        embeddings_to_concat = [a_embed, N_nodes_embedding, t_embedding]

        if self.property_embedding is not None:
            num_graphs = N_nodes.shape[0]
            prop_embed = self.property_embedding(
                properties, property_drop_mask, batch_size=num_graphs
            )
            embeddings_to_concat.append(prop_embed[batch])

        if self.natoms_cfg_embedding is not None:
            num_graphs = N_nodes.shape[0]
            natoms_embed = self.natoms_cfg_embedding(
                target_n_atoms, natoms_drop_mask, batch_size=num_graphs
            )
            embeddings_to_concat.append(natoms_embed[batch])

        if self.bond_degree_embedding is not None:
            struct_embed = self.bond_degree_embedding(e, edge_index[0], a.shape[0])
            embeddings_to_concat.append(struct_embed)

        h_0 = torch.cat(embeddings_to_concat, dim=-1)
        h_0 = self.h0_projection(h_0)

        # Process edge embeddings
        e_embed = self.edge_type_embedding(e)
        e_embed = self.e_projection(e_embed)

        # Ensure edge_index is formatted correctly (tuple for some backbones, tensor for others)
        edge_index_tuple = (edge_index[0], edge_index[1])

        return h_0, edge_index_tuple, e_embed


class BackboneWithHeads(nn.Module):
    """
    Full Model: Embedding -> EGNN Backbone -> Prediction Heads.
    """

    def __init__(
        self,
        embedding_backbone_args: DictConfig,
        # Backbone model args
        backbone_model_args: DictConfig,
        # Heads args
        heads_args: DictConfig,
        ins_gmm_head_args: DictConfig,
        ins_edge_head_args: DictConfig,
    ):
        super().__init__()

        # 1. Instantiate EmbeddingBackbone from a bundled config block
        self.embedding_backbone = hydra.utils.instantiate(embedding_backbone_args)

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
        mols_t: MoleculeBatch,
        out_dict: Dict[str, Any],
        spawn_node_idx: torch.Tensor,
        existing_node_idx: torch.Tensor,
        ins_x: torch.Tensor,
        ins_a: torch.Tensor,
        ins_c: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Predict edge types between inserted nodes and existing current-state nodes."""

        x = mols_t.x
        h = out_dict["h_latent"]

        return self.ins_edge_head(
            h=h,
            x=x,
            spawn_node_idx=spawn_node_idx,
            existing_node_idx=existing_node_idx,
            ins_x=ins_x,
            ins_a=ins_a,
            ins_c=ins_c,
        )

    def forward(
        self,
        mols_t: MoleculeBatch,
        t: torch.Tensor,
        prev_outs=None,
        is_random_self_conditioning: bool = False,
        properties: Optional[torch.Tensor] = None,
        property_drop_mask: Optional[torch.Tensor] = None,
        target_n_atoms: Optional[torch.Tensor] = None,
        natoms_drop_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Forward pass through Embedding -> Backbone -> Heads."""
        x, a, c, e, edge_index, batch = mols_t.unpack()

        # 1. Embedding Pass
        h_0, edge_index_tuple, e_embed = self.embedding_backbone(
            a,
            e,
            edge_index,
            t,
            batch,
            properties=properties,
            property_drop_mask=property_drop_mask,
            target_n_atoms=target_n_atoms,
            natoms_drop_mask=natoms_drop_mask,
        )

        # 2. Backbone Pass
        # self.backbone returns features, coordinates, and edge attrs
        h, x_out, e_out = self.backbone(h_0, x, edge_index_tuple, e_embed, batch)

        # 3. Heads Pass
        out_dict = self.heads(h, batch, e_out)

        out_dict["pos_head"] = x_out
        out_dict["gmm_head"] = self.ins_gmm_head(h, x, edge_index_tuple)

        # Store latent features
        out_dict["h_latent"] = h

        # Enforce positive rates
        for key in out_dict:
            if "rate" in key:
                out_dict[key] = F.softplus(out_dict[key])

        return out_dict
