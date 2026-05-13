"""Classifier-free guidance orchestrator.

Three responsibilities:
  1. `build_overrides(...)` — call each signal's `extract()` on `mols_1` to
     get a `{name: tensor}` dict consumed by the model forward.
  2. `sample_drop_masks(...)` — per-signal Bernoulli dropout for training.
  3. `guided_predict(...)` — run uncond + per-signal-cond forwards and blend.

The model forward never calls `signal.extract()`; it only consumes the
already-extracted `overrides` dict.  This keeps the embedding pure (no
"target vs current state" ambiguity) and lets us route the same machinery
through both the training and sampling code paths.
"""

from __future__ import annotations

from typing import Any

import torch

from chemflow.dataset.molecule_data import MoleculeBatch
from chemflow.model.cfg.embedding import CFGEmbedding
from chemflow.model.cfg.signals import CFGMode, ConditioningSignal


class CFGGuidance:
    """Owns inference-time CFG and training-time signal dropout."""

    def __init__(
        self,
        model: torch.nn.Module,
        atom_tokens: list[str] | None = None,
        batched_replica: bool = True,
    ):
        self.model = model
        self.atom_tokens = list(atom_tokens) if atom_tokens else None
        self.batched_replica = bool(batched_replica)

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    @property
    def cfg_embedding(self) -> CFGEmbedding | None:
        return getattr(getattr(self.model, "embedding_backbone", None), "cfg_embedding", None)

    @property
    def signals(self) -> list[ConditioningSignal]:
        emb = self.cfg_embedding
        return list(emb.signals) if emb is not None else []

    @property
    def signal_names(self) -> list[str]:
        return [s.name for s in self.signals]

    def has_signal(self, name: str) -> bool:
        return any(s.name == name for s in self.signals)

    def get_signal(self, name: str) -> ConditioningSignal | None:
        for s in self.signals:
            if s.name == name:
                return s
        return None

    def build_ctx(self, ins_targets=None) -> dict[str, Any]:
        """`ctx` dict shared by every signal's `extract()` call."""
        ctx: dict[str, Any] = {}
        if self.atom_tokens is not None:
            ctx["atom_tokens"] = self.atom_tokens
        if ins_targets is not None:
            ctx["ins_targets"] = ins_targets
        return ctx

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def build_overrides(
        self,
        mols_for_extract,
        ctx: dict[str, Any],
        user_overrides: dict[str, torch.Tensor | None] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Per-signal extracted values, with `user_overrides` taking precedence.

        Used both at training time (where `mols_for_extract == mols_1` and
        `user_overrides is None`) and at inference (where the user may inject
        a target via `user_overrides`).
        """
        user_overrides = user_overrides or {}
        out: dict[str, torch.Tensor] = {}
        for s in self.signals:
            if s.name in user_overrides and user_overrides[s.name] is not None:
                out[s.name] = user_overrides[s.name]
                continue
            v = s.extract(mols_for_extract, ctx)
            if v is not None:
                out[s.name] = v
        return out

    def sample_drop_masks(
        self,
        batch_size: int,
        device: torch.device,
        training: bool,
    ) -> dict[str, torch.Tensor]:
        """Bernoulli drop masks per signal, only during training."""
        if not training:
            return {}
        out: dict[str, torch.Tensor] = {}
        for s in self.signals:
            if s.dropout_prob > 0.0:
                out[s.name] = torch.rand(batch_size, device=device) < s.dropout_prob
        return out

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @staticmethod
    def apply_cfg(
        preds_cond: dict,
        preds_uncond: dict,
        guidance_scale: float,
        mode: CFGMode,
    ) -> dict:
        """Blend cond/uncond head outputs per `mode`."""
        w = guidance_scale
        out: dict[str, Any] = {}
        for key, v_cond in preds_cond.items():
            v_uncond = preds_uncond[key]
            if isinstance(v_cond, dict):
                out[key] = CFGGuidance.apply_cfg(v_cond, v_uncond, w, mode)
                continue
            if mode == "linear":
                out[key] = v_uncond + w * (v_cond - v_uncond)
            elif mode == "rate":
                log_c = torch.log(v_cond.clamp_min(1e-12))
                log_u = torch.log(v_uncond.clamp_min(1e-12))
                out[key] = torch.exp(log_u + w * (log_c - log_u))
            elif mode == "cond_only":
                out[key] = v_cond
            else:
                raise ValueError(f"Unknown CFG mode: {mode!r}")
        return out

    def guided_predict(
        self,
        model,
        mol_t,
        t: torch.Tensor,
        prev_preds,
        overrides: dict[str, torch.Tensor],
    ) -> dict:
        """One CFG step: returns activated, post-blend predictions.

        `overrides` is the already-built per-signal value dict — see
        `build_overrides()`.  Signals with `guidance_scale <= 0` or with no
        value present are simply not steered.
        """
        active = [
            s for s in self.signals
            if s.guidance_scale > 0.0 and overrides.get(s.name) is not None
        ]

        # No active CFG signals → run a single uncond forward (still passing
        # any override values, since dropout_prob=0 means the model expects to
        # see the conditioning signal at inference; the legacy code did the
        # same via the `scales_all_unit` branch).
        if not active:
            preds = model(
                mol_t,
                t.view(-1, 1),
                prev_outs=prev_preds,
                overrides=overrides,
                drop_masks=None,
            )
            model.apply_activations(preds)
            return preds

        # Fast path: all active signals have unit guidance → cond == guided,
        # so a single conditional forward suffices.
        if all(s.guidance_scale == 1.0 for s in active):
            preds = model(
                mol_t,
                t.view(-1, 1),
                prev_outs=prev_preds,
                overrides=overrides,
                drop_masks=None,
            )
            model.apply_activations(preds)
            return preds

        if self.batched_replica:
            return self._guided_predict_batched(model, mol_t, t, prev_preds, overrides, active)
        return self._guided_predict_sequential(model, mol_t, t, prev_preds, overrides, active)

    # ------------------------------------------------------------------
    # Sequential path (uncond + K cond forwards, in order)
    # ------------------------------------------------------------------

    def _guided_predict_sequential(
        self,
        model,
        mol_t,
        t: torch.Tensor,
        prev_preds,
        overrides: dict[str, torch.Tensor],
        active: list[ConditioningSignal],
    ) -> dict:
        device = mol_t.batch.device
        bs = int(mol_t.num_graphs)
        all_drop = {s.name: _all_true(bs, device) for s in self.signals}

        preds = model(
            mol_t,
            t.view(-1, 1),
            prev_outs=prev_preds,
            overrides=overrides,
            drop_masks=all_drop,
        )

        for s in active:
            cond_drop = dict(all_drop)
            cond_drop[s.name] = _all_false(bs, device)
            preds_cond = model(
                mol_t,
                t.view(-1, 1),
                prev_outs=prev_preds,
                overrides=overrides,
                drop_masks=cond_drop,
            )
            preds = self.apply_cfg(preds_cond, preds, s.guidance_scale, s.cfg_mode)

        model.apply_activations(preds)
        return preds

    # ------------------------------------------------------------------
    # Batched-replica path: one mega-forward, then slice and blend
    # ------------------------------------------------------------------

    def _guided_predict_batched(
        self,
        model,
        mol_t,
        t: torch.Tensor,
        prev_preds,
        overrides: dict[str, torch.Tensor],
        active: list[ConditioningSignal],
    ) -> dict:
        K = len(active)
        device = mol_t.batch.device
        bs = int(mol_t.num_graphs)
        n_replicas = K + 1  # uncond + one per active signal

        big_mol_t = _replicate_mol_batch(mol_t, n_replicas)
        big_t = t.view(-1, 1).repeat(n_replicas, 1)
        big_prev = _replicate_prev_preds(prev_preds, n_replicas)

        # Per-replica drop pattern: replica 0 drops all signals; replica k
        # (1..K) leaves only the k-th active signal undropped.
        all_signal_names = [s.name for s in self.signals]
        big_drop: dict[str, torch.Tensor] = {}
        for name in all_signal_names:
            mask = torch.ones(bs * n_replicas, dtype=torch.bool, device=device)
            for k, s in enumerate(active, start=1):
                if s.name == name:
                    mask[k * bs : (k + 1) * bs] = False
            big_drop[name] = mask

        # Tile each override along the replica axis.
        big_overrides = {
            name: v.repeat((n_replicas,) + (1,) * (v.ndim - 1))
            for name, v in overrides.items()
        }

        big_preds = model(
            big_mol_t,
            big_t,
            prev_outs=big_prev,
            overrides=big_overrides,
            drop_masks=big_drop,
        )

        replicas = _slice_replica_preds(big_preds, n_replicas)
        preds = replicas[0]
        for k, s in enumerate(active, start=1):
            preds = self.apply_cfg(replicas[k], preds, s.guidance_scale, s.cfg_mode)

        model.apply_activations(preds)
        return preds


# =====================================================================
# Replica plumbing
# =====================================================================


def _all_true(n: int, device: torch.device) -> torch.Tensor:
    return torch.ones(n, dtype=torch.bool, device=device)


def _all_false(n: int, device: torch.device) -> torch.Tensor:
    return torch.zeros(n, dtype=torch.bool, device=device)


def _replicate_mol_batch(mol_t: MoleculeBatch, n_replicas: int) -> MoleculeBatch:
    """Tile a `MoleculeBatch` `n_replicas` times along the graph axis.

    Replica `k` gets graph IDs offset by `k * B` and node IDs offset by
    `k * N`, so the batched forward sees a single big graph-batch with
    `n_replicas * B` graphs.
    """
    B = int(mol_t.num_graphs)
    N = int(mol_t.x.shape[0])

    x = mol_t.x.repeat(n_replicas, 1)
    a = mol_t.a.repeat(n_replicas)
    c = mol_t.c.repeat(n_replicas)
    e = mol_t.e.repeat(n_replicas)

    edge_offsets = (
        torch.arange(n_replicas, device=mol_t.edge_index.device).repeat_interleave(
            mol_t.edge_index.shape[1]
        )
        * N
    )
    edge_index = mol_t.edge_index.repeat(1, n_replicas) + edge_offsets.unsqueeze(0)

    batch_offsets = (
        torch.arange(n_replicas, device=mol_t.batch.device).repeat_interleave(N) * B
    )
    batch = mol_t.batch.repeat(n_replicas) + batch_offsets

    big = MoleculeBatch(
        x=x,
        a=a,
        c=c,
        e=e,
        edge_index=edge_index,
        batch=batch,
    )
    big.max_seqlen = mol_t.max_seqlen
    big._num_graphs = B * n_replicas
    nodes_per_graph = torch.bincount(batch, minlength=B * n_replicas)
    big.ptr = torch.cat(
        [
            torch.zeros(1, dtype=nodes_per_graph.dtype, device=nodes_per_graph.device),
            nodes_per_graph.cumsum(0),
        ]
    )

    # Carry through per-graph attributes that downstream code might read.
    skip = {"x", "a", "c", "e", "edge_index", "batch", "ptr"}
    for k, v in mol_t.__dict__.items():
        if k in skip or k.startswith("_"):
            continue
        if isinstance(v, torch.Tensor):
            setattr(big, k, v.repeat((n_replicas,) + (1,) * (v.ndim - 1)))
        elif isinstance(v, list):
            setattr(big, k, v * n_replicas)
        elif isinstance(v, str):
            setattr(big, k, [v] * n_replicas)
        else:
            setattr(big, k, v)

    return big


def _replicate_prev_preds(prev_preds, n_replicas: int):
    if prev_preds is None:
        return None
    out: dict[str, Any] = {}
    for k, v in prev_preds.items():
        out[k] = _tile_value(v, n_replicas)
    return out


def _tile_value(v, n_replicas: int):
    if isinstance(v, torch.Tensor):
        return v.repeat((n_replicas,) + (1,) * (v.ndim - 1))
    if isinstance(v, dict):
        return {kk: _tile_value(vv, n_replicas) for kk, vv in v.items()}
    return v


def _slice_replica_preds(big_preds: dict, n_replicas: int) -> list[dict]:
    """Reverse the replication: split per-replica predictions into a list of dicts.

    Every tensor was tiled `n_replicas` times along its first axis, so the
    per-replica chunk is `tensor.shape[0] // n_replicas`.
    """
    return [_slice_one_replica(big_preds, k, n_replicas) for k in range(n_replicas)]


def _slice_one_replica(v, k: int, n_replicas: int):
    if isinstance(v, torch.Tensor):
        chunk = v.shape[0] // n_replicas
        return v[k * chunk : (k + 1) * chunk]
    if isinstance(v, dict):
        return {kk: _slice_one_replica(vv, k, n_replicas) for kk, vv in v.items()}
    return v
