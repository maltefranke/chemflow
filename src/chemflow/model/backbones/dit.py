"""
DiT (Diffusion Transformer) backbone for molecular generation.

Adapted from: https://github.com/facebookresearch/DiT
and https://github.com/facebookresearch/all-atom-diffusion-transformer

Uses adaptive Layer Normalization (adaLN-Zero) for conditioning instead of
concatenating conditioning embeddings to node features as in the standard
Transformer backbone. The conditioning signal (time, node count, optional
molecular properties) is injected via scale/shift/gate parameters in each
transformer block.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import hydra
from typing import Any
from omegaconf import DictConfig
from torch_geometric.utils import to_dense_batch, scatter

from chemflow.dataset.molecule_data import MoleculeBatch


# ---------------------------------------------------------------------------
#  Building blocks
# ---------------------------------------------------------------------------


class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks."""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        norm_layer=None,
        bias=True,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.norm = (
            norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        )
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


def modulate(x, shift, scale):
    """Apply adaptive modulation: x * (1 + scale) + shift.

    Args:
        x: (B, N, d) token features
        shift: (B, d) shift parameters
        scale: (B, d) scale parameters
    """
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    """Transformer block with adaptive layer norm zero (adaLN-Zero) conditioning."""

    def __init__(self, hidden_dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads=num_heads, dropout=0, bias=True, batch_first=True
        )
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=hidden_dim,
            hidden_features=mlp_hidden_dim,
            act_layer=lambda: nn.GELU(approximate="tanh"),
            drop=0,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_dim, 6 * hidden_dim, bias=True)
        )

    def forward(self, x, c, key_padding_mask):
        """
        Args:
            x: (B, N, d) token features
            c: (B, d) graph-level conditioning
            key_padding_mask: (B, N) True for padding positions
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=1)
        )
        _x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = (
            x
            + gate_msa.unsqueeze(1)
            * self.attn(
                _x, _x, _x, key_padding_mask=key_padding_mask, need_weights=False
            )[0]
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


class FinalLayer(nn.Module):
    """Final DiT layer with adaLN conditioning."""

    def __init__(self, hidden_dim, out_dim):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_dim, out_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_dim, 2 * hidden_dim, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


# ---------------------------------------------------------------------------
#  DiT backbone
# ---------------------------------------------------------------------------


class DiTBackbone(nn.Module):
    """Core DiT backbone for molecular graphs.

    Unlike the standard Transformer backbone which receives pre-concatenated
    conditioning in the node features, this backbone receives atom embeddings
    and graph-level conditioning separately.  Conditioning is injected via
    adaLN-Zero in every block.

    Returns ``(h, x_out, e_out)`` matching the Transformer backbone interface.
    """

    def __init__(
        self,
        d_input: int,
        d_cond: int,
        d_model: int = 512,
        out_node_nf: int = 256,
        num_layers: int = 12,
        nhead: int = 16,
        mlp_ratio: float = 4.0,
        in_edge_nf: int = 128,
        rbf_embedding_args: DictConfig | None = None,
    ):
        super().__init__()
        self.d_model = d_model

        self.input_proj = nn.Sequential(
            nn.Linear(d_input, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        self.pos_embedding = nn.Sequential(
            nn.Linear(3, d_model, bias=False),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        self.cond_proj = nn.Sequential(
            nn.Linear(d_cond, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        self.blocks = nn.ModuleList(
            [DiTBlock(d_model, nhead, mlp_ratio=mlp_ratio) for _ in range(num_layers)]
        )

        self.final_layer = FinalLayer(d_model, d_model)

        self.node_out = nn.Sequential(
            nn.Linear(d_model, out_node_nf),
            nn.SiLU(),
            nn.Linear(out_node_nf, out_node_nf),
        )

        self.pos_out = nn.Sequential(
            nn.Linear(out_node_nf, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 3),
        )

        self.rbf_embedding = hydra.utils.instantiate(rbf_embedding_args)
        rbf_out_dim = rbf_embedding_args.get(
            "out_dim", rbf_embedding_args.get("num_rbf", 16)
        )

        edge_in_dim = 2 * out_node_nf + in_edge_nf + rbf_out_dim
        self.edge_output = nn.Sequential(
            nn.LayerNorm(edge_in_dim),
            nn.Linear(edge_in_dim, out_node_nf),
            nn.LayerNorm(out_node_nf),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(
        self,
        a_embed: torch.Tensor,
        x: torch.Tensor,
        cond: torch.Tensor,
        edge_index: tuple[torch.Tensor, torch.Tensor],
        e_embed: torch.Tensor,
        batch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            a_embed: (N, d_input) atom feature embeddings
            x: (N, 3) atomic positions
            cond: (B, d_cond) graph-level conditioning (time + count + props)
            edge_index: tuple (row, col) each of shape (E,)
            e_embed: (E, in_edge_nf) edge feature embeddings
            batch: (N,) batch assignment

        Returns:
            h: (N, out_node_nf) node features
            x_out: (N, 3) predicted positions
            e_out: (E, out_node_nf) edge features
        """
        batch_size = cond.shape[0]

        h = self.input_proj(a_embed) + self.pos_embedding(x)
        c = self.cond_proj(cond)

        num_nodes = scatter(
            batch.new_ones(x.size(0)), batch, dim=0, dim_size=batch_size, reduce="sum"
        )
        max_num_nodes = num_nodes.max()

        h_padded, atom_mask = to_dense_batch(
            h, batch, batch_size=batch_size, max_num_nodes=max_num_nodes
        )

        for block in self.blocks:
            h_padded = block(h_padded, c, ~atom_mask)

        h_padded = self.final_layer(h_padded, c)
        h = h_padded[atom_mask]

        h = self.node_out(h)

        rows, cols = edge_index
        h_i = h[rows]
        h_j = h[cols]

        dist_vec = x[rows] - x[cols]
        dist = torch.norm(dist_vec, dim=1)
        dist_emb = self.rbf_embedding(dist)

        edge_inputs = torch.cat([h_i, h_j, e_embed, dist_emb], dim=-1)
        e_out = self.edge_output(edge_inputs)

        x_out = self.pos_out(h)

        return h, x_out, e_out


# ---------------------------------------------------------------------------
#  Embedding layer (DiT-specific: keeps conditioning separate from input)
# ---------------------------------------------------------------------------


class DiTEmbedding(nn.Module):
    """Embedding module for DiT.

    Produces atom embeddings, edge embeddings, and a *graph-level*
    conditioning vector (time + node count + optional properties +
    optional target n_atoms for CFG).

    Unlike ``EmbeddingBackbone`` in model.py which concatenates all
    conditioning into the node features, this module returns conditioning
    as a separate tensor so DiTBackbone can feed it through adaLN.

    Exposes ``property_embedding`` and ``natoms_cfg_embedding`` for
    classifier-free guidance compatibility with the lightning module.
    """

    def __init__(
        self,
        atom_type_embedding_args: DictConfig,
        edge_type_embedding_args: DictConfig,
        time_embedding_args: DictConfig,
        node_count_embedding_args: DictConfig,
        property_embedding_args: DictConfig | None = None,
        bond_degree_embedding_args: DictConfig | None = None,
        natoms_cfg_embedding_args: DictConfig | None = None,
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

    def forward(
        self,
        a: torch.Tensor,
        e: torch.Tensor,
        edge_index: torch.Tensor,
        t: torch.Tensor,
        batch: torch.Tensor,
        properties: torch.Tensor | None = None,
        property_drop_mask: torch.Tensor | None = None,
        target_n_atoms: torch.Tensor | None = None,
        natoms_drop_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            a_embed: (N, atom_dim) atom embeddings (per-node)
            e_embed: (E, edge_dim) edge embeddings (per-edge)
            cond: (B, cond_dim) graph-level conditioning vector
        """
        N_nodes = torch.bincount(batch)
        num_graphs = N_nodes.shape[0]

        a_embed = self.atom_type_embedding(a)
        e_embed = self.edge_type_embedding(e)

        if self.bond_degree_embedding is not None:
            struct_embed = self.bond_degree_embedding(e, edge_index[0], a.shape[0])
            a_embed = torch.cat([a_embed, struct_embed], dim=-1)

        t_embed = self.time_embedding(t)
        n_embed = self.node_count_embedding(N_nodes)

        cond_parts = [t_embed, n_embed]

        if self.property_embedding is not None:
            prop_embed = self.property_embedding(
                properties, property_drop_mask, batch_size=num_graphs
            )
            cond_parts.append(prop_embed)

        if self.natoms_cfg_embedding is not None:
            natoms_embed = self.natoms_cfg_embedding(
                target_n_atoms, natoms_drop_mask, batch_size=num_graphs
            )
            cond_parts.append(natoms_embed)

        cond = torch.cat(cond_parts, dim=-1)

        return a_embed, e_embed, cond


# ---------------------------------------------------------------------------
#  Top-level model (drop-in replacement for BackboneWithHeads)
# ---------------------------------------------------------------------------


class DiTBackboneWithHeads(nn.Module):
    """Full DiT Model: Embedding -> DiT Backbone -> Prediction Heads.

    Drop-in replacement for ``BackboneWithHeads`` with the same constructor
    signature, forward signature, and output dictionary format.
    """

    def __init__(
        self,
        atom_type_embedding_args: DictConfig,
        edge_type_embedding_args: DictConfig,
        time_embedding_args: DictConfig,
        node_count_embedding_args: DictConfig,
        backbone_model_args: DictConfig,
        heads_args: DictConfig,
        ins_gmm_head_args: DictConfig,
        ins_edge_head_args: DictConfig,
        property_embedding_args: DictConfig | None = None,
        bond_degree_embedding_args: DictConfig | None = None,
        natoms_cfg_embedding_args: DictConfig | None = None,
    ):
        super().__init__()

        self.embedding_backbone = DiTEmbedding(
            atom_type_embedding_args=atom_type_embedding_args,
            edge_type_embedding_args=edge_type_embedding_args,
            time_embedding_args=time_embedding_args,
            node_count_embedding_args=node_count_embedding_args,
            property_embedding_args=property_embedding_args,
            bond_degree_embedding_args=bond_degree_embedding_args,
            natoms_cfg_embedding_args=natoms_cfg_embedding_args,
        )

        self.backbone = hydra.utils.instantiate(backbone_model_args)
        self.heads = hydra.utils.instantiate(heads_args)
        self.ins_gmm_head = hydra.utils.instantiate(ins_gmm_head_args)
        self.ins_edge_head = hydra.utils.instantiate(ins_edge_head_args)

    def set_training(self):
        pass

    def set_inference(self):
        pass

    def predict_insertion_edges(
        self,
        mols_t: MoleculeBatch,
        out_dict: dict[str, Any],
        spawn_node_idx: torch.Tensor,
        existing_node_idx: torch.Tensor,
        ins_x: torch.Tensor,
        ins_a: torch.Tensor,
        ins_c: torch.Tensor,
    ) -> torch.Tensor | None:
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
        out_dict: dict[str, Any],
        spawn_src_idx: torch.Tensor,
        ins_x_src: torch.Tensor,
        ins_a_src: torch.Tensor,
        ins_c_src: torch.Tensor,
        ins_a_dst: torch.Tensor,
        ins_x_dst: torch.Tensor,
    ) -> torch.Tensor | None:
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
        properties: torch.Tensor | None = None,
        property_drop_mask: torch.Tensor | None = None,
        target_n_atoms: torch.Tensor | None = None,
        natoms_drop_mask: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Forward pass through Embedding -> DiT Backbone -> Heads."""
        x, a, _c, e, edge_index, batch = mols_t.unpack()

        a_embed, e_embed, cond = self.embedding_backbone(
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

        edge_index_tuple = (edge_index[0], edge_index[1])

        h, x_out, e_out = self.backbone(
            a_embed,
            x,
            cond,
            edge_index_tuple,
            e_embed,
            batch,
        )

        out_dict = self.heads(h, batch, e_out)
        out_dict["pos_head"] = x_out
        out_dict["gmm_head"] = self.ins_gmm_head(h, x, edge_index_tuple)
        out_dict["h_latent"] = h

        for key in out_dict:
            if "rate" in key:
                out_dict[key] = F.softplus(out_dict[key])

        return out_dict
