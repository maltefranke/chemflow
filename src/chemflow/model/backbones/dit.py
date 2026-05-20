"""
DiT (Diffusion Transformer) backbone for molecular generation.

Adapted from: https://github.com/facebookresearch/DiT
and https://github.com/facebookresearch/all-atom-diffusion-transformer

Uses adaptive Layer Normalization (adaLN-Zero) for conditioning instead of
concatenating conditioning embeddings to node features as in the standard
Transformer backbone. The conditioning signal (time, node count, optional
molecular properties) is injected via scale/shift/gate parameters in each
transformer block.

Runs fully in packed (total_N, d) layout — no dense padding inside the
backbone. Self-attention dispatches to flash-attn varlen when available
(GPU + fp16/bf16 + flash-attn installed) and falls back to SDPA with a
key-padding mask otherwise.
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


try:
    from flash_attn import flash_attn_varlen_qkvpacked_func  # type: ignore

    _HAS_FLASH_ATTN = True
except ImportError:  # pragma: no cover - exercised when flash-attn is absent
    flash_attn_varlen_qkvpacked_func = None
    _HAS_FLASH_ATTN = False


# ---------------------------------------------------------------------------
#  Attention helper
# ---------------------------------------------------------------------------


def _attn_varlen(
    qkv: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    batch_idx: torch.Tensor,
    attn_bias: torch.Tensor | None = None,
    atom_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Variable-length self-attention on packed sequences.

    Args:
        qkv: (total_N, 3, H, D) packed q/k/v.
        cu_seqlens: (B+1,) int32 cumulative sequence lengths.
        max_seqlen: largest sequence length in the batch (Python int).
        batch_idx: (total_N,) batch assignment, used by the SDPA path.
        attn_bias: optional (B, H, max_seqlen, max_seqlen) float bias added
            to attention logits (padded keys carry ``-inf``). When given,
            flash-attn is bypassed (it can't take arbitrary bias) and the
            dense SDPA path runs.
        atom_mask: (B, max_seqlen) bool, True at valid atoms. Required when
            ``attn_bias`` is provided so we don't recompute it.

    Returns:
        (total_N, H, D)
    """
    use_flash = (
        attn_bias is None
        and _HAS_FLASH_ATTN
        and qkv.is_cuda
        and qkv.dtype in (torch.float16, torch.bfloat16)
    )
    if use_flash:
        return flash_attn_varlen_qkvpacked_func(
            qkv, cu_seqlens, max_seqlen, dropout_p=0.0, causal=False
        )

    # Dense SDPA path. Used when (a) flash-attn is unavailable, or (b) we
    # have a pair bias that flash-attn varlen can't consume.
    qkv_padded, mask = to_dense_batch(qkv, batch_idx, max_num_nodes=max_seqlen)
    if atom_mask is None:
        atom_mask = mask
    q, k, v = qkv_padded.unbind(dim=2)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    attn_mask = attn_bias if attn_bias is not None else atom_mask[:, None, None, :]
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    out = out.transpose(1, 2)
    return out[atom_mask]


# ---------------------------------------------------------------------------
#  Pair bias for attention (distance + edge type)
# ---------------------------------------------------------------------------


class PairBias(nn.Module):
    """Per-head additive attention bias from pairwise distance + edge type.

    Logically the bias is::

        bias[b, h, i, j] = down_proj( GELU( Linear(concat([rbf(||x_i - x_j||),
                                                           edge_features[i, j]])) ) )[h]

    where ``edge_features[i, j] = e_embed`` at edges in ``edge_index`` and
    zero elsewhere. We exploit ``Linear(concat[a, b]) = Linear_a(a) +
    Linear_b(b)`` to avoid materialising the dense edge tensor: project
    distances dense and *scatter-add* the edge contribution at edge positions
    only. The GELU + final ``down_proj`` is the non-linearity that makes this
    strictly more expressive than the previous two-linear formulation.

    ``down_proj`` is zero-init so the bias is exactly 0 on valid pairs at
    init; blocks gate via adaLN-Zero so contribution gates in during training.
    Padded pairs are ``-inf``-masked here once so the bias can be shared by
    reference across all blocks (one ``(B, H, N, N)`` allocation per forward,
    not per layer). ``down_proj`` is left at its Xavier init (inherited from
    the parent backbone's ``_basic_init``).
    """

    def __init__(
        self,
        rbf_embedding: nn.Module,
        rbf_out_dim: int,
        edge_nf: int,
        num_heads: int,
        hidden_dim: int,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.rbf_embedding = rbf_embedding
        self.dist_to_hidden = nn.Linear(rbf_out_dim, hidden_dim, bias=True)
        self.edge_to_hidden = nn.Linear(edge_nf, hidden_dim, bias=False)
        self.act = nn.GELU(approximate="tanh")
        self.down_proj = nn.Linear(hidden_dim, num_heads, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: tuple[torch.Tensor, torch.Tensor],
        e_embed: torch.Tensor,
        batch: torch.Tensor,
        batch_size: int,
        max_seqlen: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            bias: (B, num_heads, max_seqlen, max_seqlen) additive attention
                bias, already ``-inf`` at padded pairs so softmax masks them.
            atom_mask: (B, max_seqlen) True at valid atoms (cached for the
                attention path so it doesn't repad).
        """
        x_dense, atom_mask = to_dense_batch(
            x, batch, batch_size=batch_size, max_num_nodes=max_seqlen
        )
        diff = x_dense.unsqueeze(2) - x_dense.unsqueeze(1)
        dist = diff.norm(dim=-1)
        rbf = self.rbf_embedding(dist.reshape(-1)).reshape(*dist.shape, -1)
        h = self.dist_to_hidden(rbf)  # (B, N, N, hidden_dim)

        rows, cols = edge_index
        counts = scatter(
            batch.new_ones(batch.size(0)), batch, dim=0,
            dim_size=batch_size, reduce="sum",
        )
        cum = F.pad(counts.cumsum(0), (1, 0))[:-1]
        b_idx = batch[rows]
        local_rows = rows - cum[b_idx]
        local_cols = cols - cum[b_idx]
        h[b_idx, local_rows, local_cols] = (
            h[b_idx, local_rows, local_cols] + self.edge_to_hidden(e_embed)
        )

        h = self.act(h)
        bias = self.down_proj(h)  # (B, N, N, H)
        bias = bias.permute(0, 3, 1, 2).contiguous()  # (B, H, N, N)

        pad_mask = ~(atom_mask.unsqueeze(2) & atom_mask.unsqueeze(1))
        bias.masked_fill_(pad_mask.unsqueeze(1), float("-inf"))
        return bias, atom_mask


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


def modulate(x, shift, scale, batch_idx):
    """Apply adaptive modulation in packed layout: x * (1 + scale) + shift.

    Args:
        x: (total_N, d) packed token features
        shift: (B, d) per-graph shift parameters
        scale: (B, d) per-graph scale parameters
        batch_idx: (total_N,) batch assignment for gather
    """
    return x * (1 + scale[batch_idx]) + shift[batch_idx]


class DiTBlock(nn.Module):
    """Transformer block with adaLN-Zero conditioning, packed-sequence layout.

    Self-attention is computed with flash-attn varlen on CUDA (when available
    and the inputs are fp16/bf16) and SDPA on the fallback path. The MLP and
    LayerNorms operate on the packed ``(total_N, d)`` tensor so no compute is
    wasted on padded positions.
    """

    def __init__(self, hidden_dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
            )
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim, bias=True)
        self.proj = nn.Linear(hidden_dim, hidden_dim, bias=True)
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

    def forward(self, x, c, batch_idx, cu_seqlens, max_seqlen,
                pair_bias=None, atom_mask=None):
        """
        Args:
            x: (total_N, d) packed token features
            c: (B, d) graph-level conditioning
            batch_idx: (total_N,) batch assignment
            cu_seqlens: (B+1,) int32 cumulative sequence lengths
            max_seqlen: largest sequence length in the batch (Python int)
            pair_bias: optional (B, H, max_seqlen, max_seqlen) pair bias
                already ``-inf``-masked at padded pairs.
            atom_mask: (B, max_seqlen) True at valid atoms; required iff
                ``pair_bias`` is given.
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=1)
        )

        _x = modulate(self.norm1(x), shift_msa, scale_msa, batch_idx)
        qkv = self.qkv(_x).view(-1, 3, self.num_heads, self.head_dim)
        attn_out = _attn_varlen(
            qkv, cu_seqlens, max_seqlen, batch_idx,
            attn_bias=pair_bias, atom_mask=atom_mask,
        )
        attn_out = attn_out.reshape(-1, self.hidden_dim)
        attn_out = self.proj(attn_out)

        x = x + gate_msa[batch_idx] * attn_out
        x = x + gate_mlp[batch_idx] * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp, batch_idx)
        )
        return x


class FinalLayer(nn.Module):
    """Final DiT layer with adaLN conditioning, packed-sequence layout."""

    def __init__(self, hidden_dim, out_dim):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_dim, out_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_dim, 2 * hidden_dim, bias=True)
        )

    def forward(self, x, c, batch_idx):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale, batch_idx)
        return self.linear(x)


# ---------------------------------------------------------------------------
#  DiT backbone
# ---------------------------------------------------------------------------


class DiTBackbone(nn.Module):
    """Core DiT backbone for molecular graphs.

    Operates fully in packed ``(total_N, d)`` layout internally — no
    ``to_dense_batch`` for the main forward path. Self-attention is the only
    operator that needs to know about sequence boundaries; it gets them via
    ``cu_seqlens``.

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
        pair_bias_hidden_dim: int = 32,
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

        self.pair_bias = PairBias(
            rbf_embedding=self.rbf_embedding,
            rbf_out_dim=rbf_out_dim,
            edge_nf=in_edge_nf,
            num_heads=nhead,
            hidden_dim=pair_bias_hidden_dim,
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

        # Pair-bias init: ``down_proj`` keeps the Xavier init from
        # ``_basic_init`` above, so the bias is non-trivial from step 0. The
        # bias is added inside softmax and the whole attention output is then
        # gated by ``gate_msa(c)`` (adaLN-Zero) downstream, so we don't need
        # an extra per-block bias gate.

    def encode(
        self,
        a_embed: torch.Tensor,
        x: torch.Tensor,
        cond: torch.Tensor,
        batch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project inputs and compute ``cu_seqlens`` for flash-attn varlen.

        Returns:
            h: (total_N, d_model) packed token features.
            cu_seqlens: (B+1,) int32 cumulative sequence lengths.
            c: (B, d_model) projected graph-level conditioning.
        """
        batch_size = cond.shape[0]
        h = self.input_proj(a_embed) + self.pos_embedding(x)
        c = self.cond_proj(cond)

        counts = scatter(
            batch.new_ones(batch.size(0)),
            batch,
            dim=0,
            dim_size=batch_size,
            reduce="sum",
        )
        cu_seqlens = F.pad(counts.cumsum(dim=0), (1, 0)).to(torch.int32)
        return h, cu_seqlens, c

    def decode(
        self,
        h: torch.Tensor,
        c: torch.Tensor,
        batch_idx: torch.Tensor,
        x: torch.Tensor,
        edge_index: tuple[torch.Tensor, torch.Tensor],
        e_embed: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply final layer + node/pos/edge heads on packed inputs."""
        h = self.final_layer(h, c, batch_idx)
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

    def forward(
        self,
        a_embed: torch.Tensor,
        x: torch.Tensor,
        cond: torch.Tensor,
        edge_index: tuple[torch.Tensor, torch.Tensor],
        e_embed: torch.Tensor,
        batch: torch.Tensor,
        max_seqlen: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            a_embed: (N, d_input) atom feature embeddings
            x: (N, 3) atomic positions
            cond: (B, d_cond) graph-level conditioning (time + count + props)
            edge_index: tuple (row, col) each of shape (E,)
            e_embed: (E, in_edge_nf) edge feature embeddings
            batch: (N,) batch assignment
            max_seqlen: largest sequence length in the batch (precomputed
                CPU-side in ``MoleculeBatch.from_data_list``; flash-attn varlen
                requires a Python int and computing it here would graph-break).

        Returns:
            h: (N, out_node_nf) node features
            x_out: (N, 3) predicted positions
            e_out: (E, out_node_nf) edge features
        """
        h, cu_seqlens, c = self.encode(a_embed, x, cond, batch)
        pair_bias, atom_mask = self.pair_bias(
            x, edge_index, e_embed, batch, cond.shape[0], max_seqlen,
        )

        for block in self.blocks:
            h = block(
                h, c, batch, cu_seqlens, max_seqlen,
                pair_bias=pair_bias, atom_mask=atom_mask,
            )

        return self.decode(h, c, batch, x, edge_index, e_embed)


# ---------------------------------------------------------------------------
#  Embedding layer (DiT-specific: keeps conditioning separate from input)
# ---------------------------------------------------------------------------


class DiTEmbedding(nn.Module):
    """Embedding module for DiT.

    Produces atom embeddings, edge embeddings, and a *graph-level*
    conditioning vector (time + node count + optional CFG signals).

    Unlike ``EmbeddingBackbone`` in model.py which concatenates all
    conditioning into the node features, this module returns conditioning
    as a separate tensor so DiTBackbone can feed it through adaLN.

    Exposes ``cfg_embedding`` (a :class:`chemflow.model.cfg.CFGEmbedding`) for
    classifier-free guidance compatibility with the lightning module.
    """

    def __init__(
        self,
        atom_type_embedding_args: DictConfig,
        edge_type_embedding_args: DictConfig,
        time_embedding_args: DictConfig,
        node_count_embedding_args: DictConfig,
        cfg_embedding_args: DictConfig | None = None,
    ):
        super().__init__()
        self.atom_type_embedding = hydra.utils.instantiate(atom_type_embedding_args)
        self.edge_type_embedding = hydra.utils.instantiate(edge_type_embedding_args)
        self.time_embedding = hydra.utils.instantiate(time_embedding_args)
        self.node_count_embedding = hydra.utils.instantiate(node_count_embedding_args)

        self.cfg_embedding = None
        if cfg_embedding_args is not None:
            self.cfg_embedding = hydra.utils.instantiate(cfg_embedding_args)

    def forward(
        self,
        mols: MoleculeBatch,
        t: torch.Tensor,
        *,
        overrides: dict[str, torch.Tensor] | None = None,
        drop_masks: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            a_embed: (N, atom_dim) atom embeddings (per-node)
            e_embed: (E, edge_dim) edge embeddings (per-edge)
            cond: (B, cond_dim) graph-level conditioning vector
        """
        a, e, _edge_index, batch = mols.a, mols.e, mols.edge_index, mols.batch
        N_nodes = torch.bincount(batch)
        num_graphs = N_nodes.shape[0]

        a_embed = self.atom_type_embedding(a)
        e_embed = self.edge_type_embedding(e)

        t_embed = self.time_embedding(t)
        n_embed = self.node_count_embedding(N_nodes)

        cond_parts = [t_embed, n_embed]

        if self.cfg_embedding is not None:
            cfg_embed = self.cfg_embedding(
                batch_size=num_graphs,
                overrides=overrides,
                drop_masks=drop_masks,
            )
            cond_parts.append(cfg_embed)

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
        cfg_embedding_args: DictConfig | None = None,
    ):
        super().__init__()

        self.embedding_backbone = DiTEmbedding(
            atom_type_embedding_args=atom_type_embedding_args,
            edge_type_embedding_args=edge_type_embedding_args,
            time_embedding_args=time_embedding_args,
            node_count_embedding_args=node_count_embedding_args,
            cfg_embedding_args=cfg_embedding_args,
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
        overrides: dict[str, torch.Tensor] | None = None,
        drop_masks: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, Any]:
        """Forward pass through Embedding -> DiT Backbone -> Heads."""
        x, a, _c, e, edge_index, batch = mols_t.unpack()

        a_embed, e_embed, cond = self.embedding_backbone(
            mols_t, t,
            overrides=overrides,
            drop_masks=drop_masks,
        )

        edge_index_tuple = (edge_index[0], edge_index[1])

        h, x_out, e_out = self.backbone(
            a_embed,
            x,
            cond,
            edge_index_tuple,
            e_embed,
            batch,
            mols_t.max_seqlen,
        )

        out_dict = self.heads(h, batch, e_out)
        out_dict["pos_head"] = x_out
        out_dict["gmm_head"] = self.ins_gmm_head(h, x, edge_index_tuple)
        out_dict["h_latent"] = h

        return out_dict

    @staticmethod
    def apply_activations(out_dict: dict[str, Any]) -> dict[str, Any]:
        """Apply post-head activations (softplus on rate logits).

        Separated from forward() so that CFGGuidance can apply CFG on raw logits
        first and then activate. Must be called by callers of forward() before
        the outputs are consumed.
        """
        for key in out_dict:
            if "rate" in key:
                out_dict[key] = F.softplus(out_dict[key])
        return out_dict
