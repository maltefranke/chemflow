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
        bond_degree_embedding_args: Optional[DictConfig] = None,
        cfg_embedding_args: Optional[DictConfig] = None,
        scaffold_mask_embedding_args: Optional[DictConfig] = None,
        edge_scaffold_embedding_args: Optional[DictConfig] = None,
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

        self.bond_degree_embedding = None
        if bond_degree_embedding_args is not None:
            self.bond_degree_embedding = hydra.utils.instantiate(
                bond_degree_embedding_args
            )

        self.scaffold_mask_embedding = None
        if scaffold_mask_embedding_args is not None:
            self.scaffold_mask_embedding = hydra.utils.instantiate(
                scaffold_mask_embedding_args
            )

        self.edge_scaffold_embedding = None
        if edge_scaffold_embedding_args is not None:
            self.edge_scaffold_embedding = hydra.utils.instantiate(
                edge_scaffold_embedding_args
            )

        self.cfg_embedding = None
        if cfg_embedding_args is not None:
            self.cfg_embedding = hydra.utils.instantiate(cfg_embedding_args)

        self.h0_projection = FeatureProjector(h0_input_dim, h0_projection_dim)
        self.e_projection = FeatureProjector(e_input_dim, e_projection_dim)

    def forward(
        self,
        a: torch.Tensor,
        e: torch.Tensor,
        edge_index: torch.Tensor,
        t: torch.Tensor,
        batch: torch.Tensor,
        scaffold_mask: Optional[torch.Tensor] = None,
        cfg_inputs: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        N_nodes = torch.bincount(batch)

        a_embed = self.atom_type_embedding(a)

        N_nodes_embedding = self.node_count_embedding(N_nodes)[batch]
        t_embedding = self.time_embedding(t)[batch]

        embeddings_to_concat = [a_embed, N_nodes_embedding, t_embedding]

        if self.cfg_embedding is not None:
            num_graphs = N_nodes.shape[0]
            cfg_embed = self.cfg_embedding(
                cfg_inputs if cfg_inputs is not None else {},
                batch_size=num_graphs,
            )
            embeddings_to_concat.append(cfg_embed[batch])

        if self.bond_degree_embedding is not None:
            struct_embed = self.bond_degree_embedding(e, edge_index[0], a.shape[0])
            embeddings_to_concat.append(struct_embed)

        if self.scaffold_mask_embedding is not None:
            if scaffold_mask is None:
                scaffold_mask = torch.zeros(a.shape[0], dtype=torch.long, device=a.device)
            scaffold_embed = self.scaffold_mask_embedding(scaffold_mask)
            embeddings_to_concat.append(scaffold_embed)

        h_0 = torch.cat(embeddings_to_concat, dim=-1)
        h_0 = self.h0_projection(h_0)

        # Process edge embeddings
        e_embed = self.edge_type_embedding(e)

        if self.edge_scaffold_embedding is not None:
            if scaffold_mask is None:
                edge_scaffold_type = torch.zeros(e.shape[0], dtype=torch.long, device=e.device)
            else:
                edge_scaffold_type = scaffold_mask[edge_index[0]] + scaffold_mask[edge_index[1]]
            e_embed = torch.cat([e_embed, self.edge_scaffold_embedding(edge_scaffold_type)], dim=-1)

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
        a = mols_t.a
        h = out_dict["h_latent"]

        return self.ins_edge_head(
            h=h,
            x=x,
            node_atom_types=a,
            spawn_node_idx=spawn_node_idx,
            existing_node_idx=existing_node_idx,
            ins_x=ins_x,
            ins_a=ins_a,
            ins_c=ins_c,
        )

    def predict_insertion_edges_ins_to_ins(
        self,
        mols_t: MoleculeBatch,
        out_dict: Dict[str, Any],
        spawn_src_idx: torch.Tensor,
        ins_x_src: torch.Tensor,
        ins_a_src: torch.Tensor,
        ins_c_src: torch.Tensor,
        ins_a_dst: torch.Tensor,
        ins_x_dst: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Predict edge types between two inserted nodes."""
        h = out_dict["h_latent"]
        return self.ins_edge_head.forward_ins_to_ins(
            h=h,
            x=mols_t.x,
            spawn_src_idx=spawn_src_idx,
            ins_x_src=ins_x_src,
            ins_a_src=ins_a_src,
            ins_c_src=ins_c_src,
            ins_a_dst=ins_a_dst,
            ins_x_dst=ins_x_dst,
        )

    def forward(
        self,
        mols_t: MoleculeBatch,
        t: torch.Tensor,
        prev_outs=None,
        is_random_self_conditioning: bool = False,
        cfg_inputs: Optional[dict] = None,
    ) -> Dict[str, Any]:
        """Forward pass through Embedding -> Backbone -> Heads."""
        x, a, c, e, edge_index, batch = mols_t.unpack()
        scaffold_mask = getattr(mols_t, 'scaffold_mask', None)

        # 1. Embedding Pass
        h_0, edge_index_tuple, e_embed = self.embedding_backbone(
            a, e, edge_index, t, batch,
            scaffold_mask=scaffold_mask,
            cfg_inputs=cfg_inputs,
        )

        # 2. Backbone Pass
        # self.backbone returns features, coordinates, and edge attrs
        h, x_out, e_out = self.backbone(h_0, x, edge_index_tuple, e_embed, batch)

        # Reintroduce input embeddings (time, property, node-count conditioning)
        # via residual skip to counteract washout through the backbone
        h = h + h_0

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
