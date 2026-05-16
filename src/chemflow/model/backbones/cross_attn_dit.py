"""
Cross-attention DiT backbone for atom-wise + edge-wise conditioning.

* **Atom-wise (sequence-form) conditioning** is injected via gated
  cross-attention layers (Flamingo-style) inserted before each ``DiTBlock``
  so atoms can attend to a separate sequence of conditioning tokens. The
  cross-attention output is multiplied by ``tanh(gate)`` where ``gate`` is a
  learnable scalar initialised to 0, so at the start of training the model
  is identity-equivalent to a vanilla DiT backbone. Cross-attention logits
  accept an additive per-pair bias of shape ``(B, K, N_padded)`` to seed
  prior coupling between specific conditioning tokens and specific atoms.

* **Edge-wise (per-edge) conditioning** is injected via a small gated linear
  projection added to ``e_embed`` before the edge readout. Since the
  conditioning is already 1-1 aligned with edges no attention is needed;
  cost is ``O(E · d_edge_ctx · d)``. The gate is initialised to 0 so the
  identity-on-init property is preserved.

Composition (not inheritance): :class:`CrossAttnDiTBackbone` *holds* a
:class:`chemflow.model.backbones.dit.DiTBackbone` instance as ``self.dit`` and
a parallel ``self.cross_attn_blocks`` ModuleList. Its forward calls the DiT's
:meth:`encode`/:meth:`decode` helpers and iterates ``self.dit.blocks`` so the
cross-attention can be interleaved between the existing DiT blocks. To load a
pretrained DiT, use :meth:`CrossAttnDiTBackbone.load_dit_state_dict`, which
forwards to ``self.dit.load_state_dict``.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torch_geometric.utils import to_dense_batch

from chemflow.dataset.molecule_data import MoleculeBatch
from chemflow.model.backbones.dit import (
    DiTBackbone,
    DiTBackboneWithHeads,
    Mlp,
)


# ---------------------------------------------------------------------------
#  Gated cross-attention block (Flamingo-style)
# ---------------------------------------------------------------------------


class GatedCrossAttention(nn.Module):
    """Gated cross-attention: atoms (queries) attend to conditioning tokens.

    Both the attention and feed-forward branches are wrapped in a tanh gate
    on a learnable scalar initialised to 0, so the layer is the identity at
    init (Flamingo, https://arxiv.org/abs/2204.14198).

    Attention logits accept an additive per-pair bias of shape ``(B, K, N)``
    (token-then-atom indexing); ``bias[b, k, n]`` is added to the logit of
    atom ``n`` attending to token ``k``.
    """

    def __init__(
        self,
        hidden_dim: int,
        ctx_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
            )
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.norm_q = nn.LayerNorm(hidden_dim)
        self.norm_kv = nn.LayerNorm(ctx_dim)

        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.kv_proj = nn.Linear(ctx_dim, 2 * hidden_dim, bias=True)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)

        self.attn_gate = nn.Parameter(torch.zeros(1))

        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.norm_ffn = nn.LayerNorm(hidden_dim)
        self.ffn = Mlp(
            in_features=hidden_dim,
            hidden_features=mlp_hidden,
            act_layer=lambda: nn.GELU(approximate="tanh"),
            drop=0,
        )
        self.ffn_gate = nn.Parameter(torch.zeros(1))

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        nn.init.zeros_(self.attn_gate)
        nn.init.zeros_(self.ffn_gate)

    def forward(
        self,
        x: torch.Tensor,
        ctx: torch.Tensor,
        ctx_mask: torch.Tensor | None,
        attn_bias: torch.Tensor | None,
        query_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, N, d) atom features (queries).
            ctx: (B, K, d_ctx) conditioning tokens.
            ctx_mask: (B, K) bool, True for valid tokens. ``None`` attends to all.
            attn_bias: (B, K, N) additive bias on attention logits, or ``None``.
            query_mask: (B, N) bool, True for valid atoms. Used to zero out
                cross-attention output at padded query positions.

        Returns:
            (B, N, d) updated features.
        """
        B, N, _ = x.shape
        K = ctx.shape[1]
        H, Dh = self.num_heads, self.head_dim

        q = self.q_proj(self.norm_q(x))
        kv = self.kv_proj(self.norm_kv(ctx))
        k, v = kv.chunk(2, dim=-1)

        q = q.view(B, N, H, Dh).transpose(1, 2)  # (B, H, N, Dh)
        k = k.view(B, K, H, Dh).transpose(1, 2)
        v = v.view(B, K, H, Dh).transpose(1, 2)

        attn_mask: torch.Tensor | None = None
        if attn_bias is not None or ctx_mask is not None:
            if attn_bias is not None:
                attn_mask = attn_bias.transpose(1, 2).unsqueeze(1).to(q.dtype)
            else:
                attn_mask = torch.zeros((B, 1, N, K), dtype=q.dtype, device=q.device)
            if ctx_mask is not None:
                attn_mask = attn_mask.masked_fill(
                    ~ctx_mask[:, None, None, :], float("-inf")
                )

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).contiguous().view(B, N, -1)
        out = self.out_proj(out)

        if query_mask is not None:
            out = out * query_mask.unsqueeze(-1).to(out.dtype)

        x = x + torch.tanh(self.attn_gate) * out
        x = x + torch.tanh(self.ffn_gate) * self.ffn(self.norm_ffn(x))
        return x


# ---------------------------------------------------------------------------
#  Backbone (composes a DiTBackbone with parallel cross-attention blocks)
# ---------------------------------------------------------------------------


class CrossAttnDiTBackbone(nn.Module):
    """DiT backbone augmented with Flamingo-style gated cross-attention.

    Holds a :class:`DiTBackbone` (``self.dit``) plus a parallel ``ModuleList``
    of :class:`GatedCrossAttention` modules (``self.cross_attn_blocks``).
    The forward delegates input projection / final layer / heads to
    ``self.dit.encode`` and ``self.dit.decode``, and interleaves a gated
    cross-attention block before each ``self.dit.blocks[i]``.

    Loading a pretrained vanilla DiT is just::

        model.load_dit_state_dict(dit_backbone_sd)

    which forwards to ``self.dit.load_state_dict`` — the cross-attention
    blocks keep their zero-gated init so the augmented model behaves
    identically to the loaded DiT until the gates train away from 0.
    """

    def __init__(
        self,
        d_input: int,
        d_cond: int,
        ctx_dim: int,
        d_model: int = 512,
        out_node_nf: int = 256,
        num_layers: int = 12,
        nhead: int = 16,
        mlp_ratio: float = 4.0,
        in_edge_nf: int = 128,
        rbf_embedding_args: DictConfig | None = None,
        pair_bias_hidden_dim: int = 32,
        edge_ctx_dim: int | None = None,
    ):
        super().__init__()
        self.ctx_dim = ctx_dim
        self.edge_ctx_dim = edge_ctx_dim
        self.num_layers = num_layers

        self.dit = DiTBackbone(
            d_input=d_input,
            d_cond=d_cond,
            d_model=d_model,
            out_node_nf=out_node_nf,
            num_layers=num_layers,
            nhead=nhead,
            mlp_ratio=mlp_ratio,
            in_edge_nf=in_edge_nf,
            rbf_embedding_args=rbf_embedding_args,
            pair_bias_hidden_dim=pair_bias_hidden_dim,
        )

        self.cross_attn_blocks = nn.ModuleList(
            [
                GatedCrossAttention(d_model, ctx_dim, nhead, mlp_ratio=mlp_ratio)
                for _ in range(num_layers)
            ]
        )

        if edge_ctx_dim is not None:
            self.edge_cond_proj = nn.Sequential(
                nn.Linear(edge_ctx_dim, in_edge_nf),
                nn.GELU(),
                nn.Linear(in_edge_nf, in_edge_nf),
            )
            self.edge_cond_gate = nn.Parameter(torch.zeros(1))
            self._init_edge_cond_weights()
        else:
            self.edge_cond_proj = None
            self.register_parameter("edge_cond_gate", None)

    def _init_edge_cond_weights(self):
        for module in self.edge_cond_proj.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        nn.init.zeros_(self.edge_cond_gate)

    def forward(
        self,
        a_embed: torch.Tensor,
        x: torch.Tensor,
        cond: torch.Tensor,
        edge_index: tuple[torch.Tensor, torch.Tensor],
        e_embed: torch.Tensor,
        batch: torch.Tensor,
        max_seqlen: int,
        cond_tokens: torch.Tensor | None = None,
        cond_token_mask: torch.Tensor | None = None,
        cond_token_attn_bias: torch.Tensor | None = None,
        edge_cond: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Same as :meth:`DiTBackbone.forward` plus cross-attention and
        edge-conditioning kwargs.

        Args:
            cond_tokens: (B, K, ctx_dim) per-token conditioning sequence. If
                ``None`` the gated cross-attention blocks are skipped (cheaper
                and equivalent to gate=0).
            cond_token_mask: (B, K) bool, True at valid tokens.
            cond_token_attn_bias: (B, K, N_padded) additive bias on
                cross-attention logits. ``N_padded`` is the max number of
                atoms across the current batch as produced by
                ``to_dense_batch``.
            edge_cond: (E, edge_ctx_dim) per-edge conditioning features
                aligned 1-to-1 with ``edge_index`` / ``e_embed``. The
                projection is gated to 0 at init, so providing this is
                identity-equivalent to omitting it on a freshly loaded DiT.
                Requires the backbone to have been constructed with
                ``edge_ctx_dim`` set.

        See ``DiTBackbone.forward`` for the remaining arguments.
        """
        h, cu_seqlens, c = self.dit.encode(a_embed, x, cond, batch)

        if cond_tokens is not None and cond_token_attn_bias is not None:
            if cond_token_attn_bias.shape[-1] != max_seqlen:
                raise ValueError(
                    "cond_token_attn_bias last dim "
                    f"{cond_token_attn_bias.shape[-1]} != padded num atoms "
                    f"{max_seqlen}. Bias must match the padded length produced "
                    "by to_dense_batch."
                )

        batch_size = cond.shape[0]
        pair_bias, atom_mask = self.dit.pair_bias(
            x, edge_index, e_embed, batch, batch_size, max_seqlen,
        )
        for cross_attn, block in zip(self.cross_attn_blocks, self.dit.blocks):
            if cond_tokens is not None:
                h_dense, _ = to_dense_batch(
                    h, batch, batch_size=batch_size, max_num_nodes=max_seqlen
                )
                h_dense = cross_attn(
                    h_dense,
                    cond_tokens,
                    cond_token_mask,
                    cond_token_attn_bias,
                    query_mask=atom_mask,
                )
                h = h_dense[atom_mask]
            h = block(
                h, c, batch, cu_seqlens, max_seqlen,
                pair_bias=pair_bias, atom_mask=atom_mask,
            )

        if edge_cond is not None:
            if self.edge_cond_proj is None:
                raise ValueError(
                    "edge_cond was provided but the backbone was constructed "
                    "without edge_ctx_dim. Set edge_ctx_dim to enable per-edge "
                    "conditioning."
                )
            if edge_cond.shape[0] != e_embed.shape[0]:
                raise ValueError(
                    f"edge_cond first dim {edge_cond.shape[0]} != number of "
                    f"edges {e_embed.shape[0]}; edge_cond must be aligned to "
                    "edge_index / e_embed."
                )
            e_embed = e_embed + torch.tanh(self.edge_cond_gate) * self.edge_cond_proj(
                edge_cond
            )

        return self.dit.decode(h, c, batch, x, edge_index, e_embed)

    def load_dit_state_dict(
        self,
        dit_state_dict: dict[str, torch.Tensor],
        strict: bool = True,
    ) -> Any:
        """Load a vanilla ``DiTBackbone`` state_dict into ``self.dit``.

        Cross-attention blocks are unaffected and keep their zero-gated init.
        """
        return self.dit.load_state_dict(dit_state_dict, strict=strict)


# ---------------------------------------------------------------------------
#  Top-level wrapper (extends DiTBackboneWithHeads)
# ---------------------------------------------------------------------------


class CrossAttnDiTBackboneWithHeads(DiTBackboneWithHeads):
    """:class:`DiTBackboneWithHeads` with cross-attention forward kwargs.

    Constructor is identical to the parent — set ``backbone_model_args._target_``
    to ``chemflow.model.backbones.cross_attn_dit.CrossAttnDiTBackbone`` to swap
    the backbone in via Hydra. The forward simply forwards the new kwargs into
    ``self.backbone``.
    """

    def forward(
        self,
        mols_t: MoleculeBatch,
        t: torch.Tensor,
        prev_outs=None,
        is_random_self_conditioning: bool = False,
        overrides: dict[str, torch.Tensor] | None = None,
        drop_masks: dict[str, torch.Tensor] | None = None,
        cond_tokens: torch.Tensor | None = None,
        cond_token_mask: torch.Tensor | None = None,
        cond_token_attn_bias: torch.Tensor | None = None,
        edge_cond: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        x, a, _c, e, edge_index, batch = mols_t.unpack()

        a_embed, e_embed, cond = self.embedding_backbone(
            mols_t,
            t,
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
            cond_tokens=cond_tokens,
            cond_token_mask=cond_token_mask,
            cond_token_attn_bias=cond_token_attn_bias,
            edge_cond=edge_cond,
        )

        out_dict = self.heads(h, batch, e_out)
        out_dict["pos_head"] = x_out
        out_dict["gmm_head"] = self.ins_gmm_head(h, x, edge_index_tuple)
        out_dict["h_latent"] = h

        return out_dict
