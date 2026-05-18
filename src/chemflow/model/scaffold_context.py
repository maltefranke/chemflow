"""Scaffold context encoder + cross-attn DiT wrapper for scaffold fine-tuning.

Two components:

* :class:`ScaffoldContextEncoder` — turns the scaffold atoms of ``mol_t``
  (selected via ``mol_t.scaffold_mask``) into per-token cross-attention
  inputs ``(cond_tokens, cond_token_mask, cond_token_attn_bias)`` for a
  :class:`CrossAttnDiTBackbone`, *and* a per-edge ``edge_cond`` tensor
  flagging scaffold/decoration edge categories. Both are derived from raw
  atom features (position, type, charge) and the live ``edge_index``, so
  the conditioning stays in sync with the molecule as the integrator
  deletes/inserts atoms.

* :class:`ScaffoldCrossAttnDiTBackboneWithHeads` — a thin subclass of
  :class:`CrossAttnDiTBackboneWithHeads` that, when ``mols_t.scaffold_mask``
  is present, builds the scaffold context internally and forwards it via
  the existing cross-attention / edge-conditioning kwargs.

Both pieces are additive — without ``scaffold_mask`` on the input batch
the wrapper behaves exactly like the parent class, and the cross-attn
and edge_cond gates start at zero (identity), so an unconditional
checkpoint loaded via :func:`load_pretrained_into_lightning_model`
produces unchanged predictions at the first training step.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_batch

from chemflow.dataset.molecule_data import MoleculeBatch
from chemflow.model.backbones.cross_attn_dit import CrossAttnDiTBackboneWithHeads


# Categorical edge types based on scaffold membership of the two endpoints.
_EDGE_DEC_DEC = 0  # both endpoints in decoration
_EDGE_SC_DEC = 1   # exactly one endpoint in the scaffold
_EDGE_SC_SC = 2    # both endpoints in the scaffold
_N_EDGE_SCAFFOLD_TYPES = 3


class ScaffoldContextEncoder(nn.Module):
    """Encode the scaffold of ``mol_t`` into cross-attention + edge inputs.

    For each batch element, the scaffold atoms (where ``scaffold_mask == 1``)
    are projected to ``ctx_dim`` features and padded to a common K. The
    per-edge ``edge_cond`` flags each edge as decoration-decoration,
    scaffold-decoration, or scaffold-scaffold — recomputed each forward so
    it tracks any deletions or insertions performed by the integrator.

    Args:
        num_atom_types: Size of the atom-type vocabulary.
        num_charges: Size of the charge vocabulary.
        atom_embed_dim: Width of the atom-type embedding.
        charge_embed_dim: Width of the charge embedding.
        pos_embed_dim: Width of the 3D-position MLP projection.
        ctx_dim: Output token dimension. Must match
            ``CrossAttnDiTBackbone.ctx_dim``.
        edge_ctx_dim: Output edge dimension. Must match
            ``CrossAttnDiTBackbone.edge_ctx_dim`` when edge conditioning is
            enabled; set to ``None`` to skip edge conditioning entirely.
        pair_bias_init: Additive bias on the (token k → atom k) attention
            logit at init. 0 disables the bias.
    """

    def __init__(
        self,
        num_atom_types: int,
        num_charges: int,
        atom_embed_dim: int = 64,
        charge_embed_dim: int = 16,
        pos_embed_dim: int = 32,
        ctx_dim: int = 128,
        edge_ctx_dim: int | None = 32,
        pair_bias_init: float = 0.0,
    ):
        super().__init__()
        self.ctx_dim = ctx_dim
        self.edge_ctx_dim = edge_ctx_dim
        self.pair_bias_init = pair_bias_init

        self.atom_embed = nn.Embedding(num_atom_types, atom_embed_dim)
        self.charge_embed = nn.Embedding(num_charges, charge_embed_dim)
        self.pos_proj = nn.Sequential(
            nn.Linear(3, pos_embed_dim),
            nn.GELU(),
            nn.Linear(pos_embed_dim, pos_embed_dim),
        )

        feat_dim = atom_embed_dim + charge_embed_dim + pos_embed_dim
        self.token_proj = nn.Sequential(
            nn.Linear(feat_dim, ctx_dim),
            nn.GELU(),
            nn.Linear(ctx_dim, ctx_dim),
        )

        if edge_ctx_dim is not None:
            self.edge_scaffold_embed = nn.Embedding(
                _N_EDGE_SCAFFOLD_TYPES, edge_ctx_dim
            )
        else:
            self.edge_scaffold_embed = None

    def _build_node_tokens(
        self, mols_t: MoleculeBatch, B: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Return ``(cond_tokens, cond_token_mask, cond_token_attn_bias)``."""
        sc_mask = mols_t.scaffold_mask.bool()
        batch = mols_t.batch
        device = mols_t.x.device

        sc_idx = sc_mask.nonzero(as_tuple=False).flatten()
        if sc_idx.numel() == 0:
            return (
                torch.zeros((B, 0, self.ctx_dim), device=device, dtype=mols_t.x.dtype),
                torch.zeros((B, 0), device=device, dtype=torch.bool),
                None,
            )

        sc_batch = batch[sc_idx]
        a_embed = self.atom_embed(mols_t.a[sc_idx])
        c_embed = self.charge_embed(mols_t.c[sc_idx])
        x_embed = self.pos_proj(mols_t.x[sc_idx].to(dtype=a_embed.dtype))
        feats = torch.cat([a_embed, c_embed, x_embed], dim=-1)
        tokens = self.token_proj(feats)

        cond_tokens, cond_token_mask = to_dense_batch(tokens, sc_batch, batch_size=B)

        cond_token_attn_bias: torch.Tensor | None = None
        if self.pair_bias_init != 0.0:
            # TODO can the following be vectorized?
            N_pad = int(mols_t.max_seqlen)
            K = cond_tokens.shape[1]
            cond_token_attn_bias = torch.zeros(
                (B, K, N_pad), device=device, dtype=cond_tokens.dtype
            )
            # For each batch b, identify the dense-layout positions of its
            # scaffold atoms inside mol_t and bias attention from token k to
            # the k-th scaffold position. Computed from the live
            # scaffold_mask so it stays correct after del/ins steps.
            for b in range(B):
                node_indices_b = (batch == b).nonzero(as_tuple=False).flatten()
                sc_positions = (
                    sc_mask[node_indices_b].nonzero(as_tuple=False).flatten()
                )
                k_b = sc_positions.numel()
                if k_b == 0:
                    continue
                cond_token_attn_bias[
                    b, torch.arange(k_b, device=device), sc_positions
                ] = self.pair_bias_init

        return cond_tokens, cond_token_mask, cond_token_attn_bias

    def _build_edge_cond(self, mols_t: MoleculeBatch) -> torch.Tensor | None:
        """Categorise each edge as dec-dec / scaffold-dec / scaffold-scaffold.

        Recomputed each forward from the live ``edge_index`` and
        ``scaffold_mask`` so deletions / insertions stay in sync.
        """
        if self.edge_scaffold_embed is None:
            return None
        edge_index = mols_t.edge_index
        sc_mask = mols_t.scaffold_mask.bool()
        src_sc = sc_mask[edge_index[0]]
        dst_sc = sc_mask[edge_index[1]]
        edge_types = (src_sc.long() + dst_sc.long()).clamp(max=_EDGE_SC_SC)
        # 0 = dec-dec (neither sc), 1 = scaffold-dec (exactly one), 2 = sc-sc (both)
        return self.edge_scaffold_embed(edge_types)

    def forward(
        self,
        mols_t: MoleculeBatch,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        """Build ``(cond_tokens, cond_token_mask, cond_token_attn_bias, edge_cond)``."""
        batch = mols_t.batch
        B = int(batch.max().item()) + 1 if batch.numel() > 0 else 1

        cond_tokens, cond_token_mask, cond_token_attn_bias = self._build_node_tokens(
            mols_t, B
        )
        edge_cond = self._build_edge_cond(mols_t)
        return cond_tokens, cond_token_mask, cond_token_attn_bias, edge_cond


class ScaffoldCrossAttnDiTBackboneWithHeads(CrossAttnDiTBackboneWithHeads):
    """:class:`CrossAttnDiTBackboneWithHeads` with built-in scaffold context.

    Builds ``(cond_tokens, cond_token_mask, cond_token_attn_bias, edge_cond)``
    from ``mols_t.scaffold_mask`` on every forward and passes them through
    to the parent's cross-attention / edge-conditioning path. Caller-supplied
    kwargs win (override).

    Constructor takes one extra Hydra-instantiable arg
    ``scaffold_encoder_args`` on top of the parent's signature.
    """

    def __init__(
        self,
        *args,
        scaffold_encoder_args: dict | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        import hydra

        if scaffold_encoder_args is None:
            raise ValueError(
                "scaffold_encoder_args is required for "
                "ScaffoldCrossAttnDiTBackboneWithHeads"
            )
        self.scaffold_encoder = hydra.utils.instantiate(scaffold_encoder_args)

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
        target_props: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        if (
            getattr(mols_t, "scaffold_mask", None) is not None
            and cond_tokens is None
        ):
            sc_tokens, sc_mask, sc_bias, sc_edge_cond = self.scaffold_encoder(mols_t)
            cond_tokens = sc_tokens
            cond_token_mask = sc_mask
            cond_token_attn_bias = sc_bias
            if edge_cond is None:
                edge_cond = sc_edge_cond

        if target_props is None:
            target_props = getattr(mols_t, "target_props", None)

        return super().forward(
            mols_t,
            t,
            prev_outs=prev_outs,
            is_random_self_conditioning=is_random_self_conditioning,
            overrides=overrides,
            drop_masks=drop_masks,
            cond_tokens=cond_tokens,
            cond_token_mask=cond_token_mask,
            cond_token_attn_bias=cond_token_attn_bias,
            edge_cond=edge_cond,
            target_props=target_props,
        )
