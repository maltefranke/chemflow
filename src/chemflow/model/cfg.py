from __future__ import annotations

import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import Crippen, GetPeriodicTable

from chemflow.dataset.qm9 import QM9_PROPERTY_NAMES
from chemflow.model.embedding import (
    CountEmbedding,
    SinusoidalEncoding,
)

_PERIODIC_TABLE = GetPeriodicTable()


class CFGAdapter:
    """Classifier-free guidance adapter.

    Manages per-signal dropout, guidance scales, and multi-pass CFG
    inference for property, n_atoms, and molecular-weight conditioning.
    """

    # Head categories for type-aware classifier-free guidance.
    # All keys here receive linear logit-space interpolation:
    #   guided = uncond + w * (cond - uncond)
    # Activations (softplus for rates, softmax for class heads) are applied
    # *after* CFG by the caller, not inside the model forward pass.
    _cfg_linear_keys = {
        "do_del_head",
        "do_sub_a_head",
        "do_sub_e_head",
        # Rate logits — softplus is applied after CFG
        "ins_rate_head",
        # Class logits — softmax is applied after CFG (in the inference loop)
        "atom_type_head",
        "charge_head",
        "edge_type_head",
        "pos_head"
    }
    _cfg_rate_keys: set[str] = set()
    _cfg_use_cond_keys = {
    }

    def __init__(
        self,
        model: torch.nn.Module,
        cfg_dropout_prob: float = 0.0,
        cfg_guidance_scale: float = 0.0,
        natoms_cfg_dropout_prob: float = 0.15,
        natoms_cfg_guidance_scale: float = 2.5,
        mw_cfg_dropout_prob: float = 0.15,
        mw_cfg_guidance_scale: float = 0.0,
        logp_cfg_dropout_prob: float = 0.0,
        logp_cfg_guidance_scale: float = 0.0,
        atom_tokens: list[str] | None = None,
        property_names: list[str] | None = None,
    ):
        self.model = model
        self.cfg_dropout_prob = float(cfg_dropout_prob)
        self.cfg_guidance_scale = float(cfg_guidance_scale)
        self.natoms_cfg_dropout_prob = float(natoms_cfg_dropout_prob)
        self.natoms_cfg_guidance_scale = float(natoms_cfg_guidance_scale)
        self.mw_cfg_dropout_prob = float(mw_cfg_dropout_prob)
        self.mw_cfg_guidance_scale = float(mw_cfg_guidance_scale)
        self.logp_cfg_dropout_prob = float(logp_cfg_dropout_prob)
        self.logp_cfg_guidance_scale = float(logp_cfg_guidance_scale)
        self.atom_tokens = atom_tokens

        if property_names is not None:
            unknown = [n for n in property_names if n not in QM9_PROPERTY_NAMES]
            if unknown:
                raise ValueError(
                    f"Unknown property name(s): {unknown}. "
                    f"Valid names: {sorted(QM9_PROPERTY_NAMES)}"
                )
            self.property_indices: list[int] | None = [QM9_PROPERTY_NAMES[n] for n in property_names]
        else:
            self.property_indices = None

    @property
    def _cfg_embedding(self):
        return getattr(self.model.embedding_backbone, "cfg_embedding", None)

    @property
    def _has_property_conditioning(self) -> bool:
        e = self._cfg_embedding
        return e is not None and getattr(e, "property_encoder", None) is not None

    @property
    def _has_natoms_cfg(self) -> bool:
        e = self._cfg_embedding
        return e is not None and getattr(e, "natoms_encoder", None) is not None

    @property
    def _has_mw_cfg(self) -> bool:
        e = self._cfg_embedding
        return e is not None and getattr(e, "mw_encoder", None) is not None

    @property
    def _has_logp_cfg(self) -> bool:
        e = self._cfg_embedding
        return e is not None and getattr(e, "logp_encoder", None) is not None

    def extract_properties(self, mols_t) -> torch.Tensor | None:
        if not self._has_property_conditioning:
            return None
        if hasattr(mols_t, "y") and mols_t.y is not None:
            y = mols_t.y.float()
            if self.property_indices is not None:
                y = y[:, self.property_indices]
            return y
        return None

    def extract_target_n_atoms(
        self,
        mols,
        ins_targets=None,
    ) -> torch.Tensor | None:
        """Per-graph target atom count for ``n_atoms`` conditioning.

        During training ``mols`` is ``target_state`` — the OT-aligned target
        after filtering to nodes that exist at time ``t`` (i.e. ``sub``,
        ``keep_del`` and ``keep_ins``).  ``keep_del`` nodes are padded
        (``is_auxiliary=True``) because they were matched to dummies on the
        target side and will be deleted before reaching ``mol_1``.  The real
        final atom count is therefore

            ``(~target_state.is_auxiliary).sum()``      (sub + keep_ins)
            ``+ future_ins_nodes.num_nodes``            (remaining insertions)

        which differs from ``mol_t``'s current node count and is what the
        model actually needs to condition on.  When ``ins_targets`` is not
        provided (e.g. inference, where ``mols`` already holds the final
        target) we fall back to the straight bincount.
        """
        if not self._has_natoms_cfg:
            return None
        device = mols.batch.device
        num_graphs = mols.num_graphs
        if ins_targets is not None and hasattr(mols, "is_auxiliary"):
            is_real = (~mols.is_auxiliary).view(-1).to(torch.long)
            real_per_graph = torch.zeros(num_graphs, dtype=torch.long, device=device)
            real_per_graph.scatter_add_(0, mols.batch, is_real)
            ins_batch = getattr(ins_targets, "batch", None)
            if ins_batch is not None and ins_batch.numel() > 0:
                ins_per_graph = torch.bincount(ins_batch, minlength=num_graphs)
            else:
                ins_per_graph = torch.zeros(
                    num_graphs, dtype=torch.long, device=device
                )
            return real_per_graph + ins_per_graph
        return torch.bincount(mols.batch, minlength=num_graphs)

    def extract_target_mw(
        self, mols, batch: torch.Tensor | None = None
    ) -> torch.Tensor | None:
        if not self._has_mw_cfg or self.atom_tokens is None:
            return None
        b = batch if batch is not None else mols.batch
        return compute_molecular_weight(mols.a, self.atom_tokens, b, mols.num_graphs)

    def extract_target_logp(self, mols) -> torch.Tensor | None:
        """Per-graph RDKit Crippen logP, computed on-the-fly from SMILES.

        Expects ``mols.smiles`` to be either a single string (single graph) or
        a list of canonical SMILES of length ``num_graphs`` (the form PyG
        ``Batch`` uses when batching string attributes).  Returns ``None`` if
        SMILES are unavailable so the embedding falls back to its null token.
        """
        if not self._has_logp_cfg:
            return None
        smiles = getattr(mols, "smiles", None)
        if smiles is None:
            return None
        device = mols.x.device if hasattr(mols, "x") else None
        return compute_logp(smiles, device=device)

    def _sample_drop_mask(
        self,
        prob: float,
        batch_size: int,
        device: torch.device,
        training: bool,
    ) -> torch.Tensor | None:
        if prob <= 0.0 or not training:
            return None
        return torch.rand(batch_size, device=device) < prob

    def should_use_property_cfg(self, properties) -> bool:
        return (
            self.cfg_guidance_scale > 0.0
            and properties is not None
            and self._has_property_conditioning
        )

    def should_use_natoms_cfg(self, target_n_atoms) -> bool:
        return (
            self.natoms_cfg_guidance_scale > 0.0
            and target_n_atoms is not None
            and self._has_natoms_cfg
        )

    def should_use_mw_cfg(self, target_mw) -> bool:
        return (
            self.mw_cfg_guidance_scale > 0.0
            and target_mw is not None
            and self._has_mw_cfg
        )

    def should_use_logp_cfg(self, target_logp) -> bool:
        return (
            self.logp_cfg_guidance_scale > 0.0
            and target_logp is not None
            and self._has_logp_cfg
        )

    def apply_cfg(
        self,
        preds_cond: dict,
        preds_uncond: dict,
        guidance_scale: float,
    ) -> dict:
        """Type-aware classifier-free guidance."""
        guided = {}
        w = guidance_scale

        for key, v_cond in preds_cond.items():
            v_uncond = preds_uncond[key]

            if key in self._cfg_linear_keys:
                guided[key] = v_uncond + w * (v_cond - v_uncond)

            elif key in self._cfg_use_cond_keys:
                guided[key] = v_cond

            elif key in self._cfg_rate_keys:
                log_cond = torch.log(v_cond.clamp_min(1e-12))
                log_uncond = torch.log(v_uncond.clamp_min(1e-12))
                guided[key] = torch.exp(log_uncond + w * (log_cond - log_uncond))

            else:
                guided[key] = v_cond

        return guided

    def get_training_inputs(
        self,
        mols_t,
        mols_1,
        device: torch.device,
        training: bool,
        ins_targets=None,
    ) -> dict:
        """Build the ``cfg_inputs`` dict for a training forward pass."""
        bs = mols_t.num_graphs
        return {
            "properties": self.extract_properties(mols_t),
            "property_drop_mask": self._sample_drop_mask(
                self.cfg_dropout_prob,
                bs,
                device,
                training,
            ),
            "target_n_atoms": self.extract_target_n_atoms(mols_1, ins_targets),
            "natoms_drop_mask": self._sample_drop_mask(
                self.natoms_cfg_dropout_prob,
                bs,
                device,
                training,
            ),
            "target_mw": self.extract_target_mw(mols_1),
            "mw_drop_mask": self._sample_drop_mask(
                self.mw_cfg_dropout_prob,
                bs,
                device,
                training,
            ),
            "target_logp": self.extract_target_logp(mols_1),
            "logp_drop_mask": self._sample_drop_mask(
                self.logp_cfg_dropout_prob,
                bs,
                device,
                training,
            ),
        }

    def _uncond_cfg_inputs(self) -> dict:
        """Return a cfg_inputs dict with every signal set to None."""
        return {
            "properties": None,
            "property_drop_mask": None,
            "target_n_atoms": None,
            "natoms_drop_mask": None,
            "target_mw": None,
            "mw_drop_mask": None,
            "target_logp": None,
            "logp_drop_mask": None,
        }

    def guided_predict(
        self,
        model,
        mol_t,
        t: torch.Tensor,
        prev_preds,
        cfg_inputs: dict,
    ) -> dict:
        """Run CFG inference: unconditional + per-signal guided passes.

        All model calls return raw logits (no activations).  CFG is applied on
        the raw outputs, then activations are applied once at the end.
        """
        target_n_atoms = cfg_inputs.get("target_n_atoms")
        target_mw = cfg_inputs.get("target_mw")
        target_logp = cfg_inputs.get("target_logp")
        properties = cfg_inputs.get("properties")

        use_natoms = self.should_use_natoms_cfg(target_n_atoms)
        use_mw = self.should_use_mw_cfg(target_mw)
        use_logp = self.should_use_logp_cfg(target_logp)
        use_prop = self.should_use_property_cfg(properties)

        # At guidance_scale == 1 the linear interpolation
        #     guided = uncond + 1 * (cond - uncond) = cond
        # collapses to the conditional pass for every linear-key head, so
        # running the uncond forward is pure overhead.  When every active
        # signal has a unit scale we can replicate the general path's final
        # output with a single model call and skip the uncond pass entirely.
        # Semantics below mirror the sequential loop: the *last* active
        # signal's conditional inputs determine the forward (property cond
        # uses the full cfg_inputs, mw / natoms / logp cond use the isolated
        # signal only).
        scales_all_unit = (
            (not use_natoms or self.natoms_cfg_guidance_scale == 1.0)
            and (not use_mw or self.mw_cfg_guidance_scale == 1.0)
            and (not use_logp or self.logp_cfg_guidance_scale == 1.0)
            and (not use_prop or self.cfg_guidance_scale == 1.0)
        )
        if scales_all_unit:
            if use_prop:
                active_cfg_inputs = cfg_inputs
            elif use_logp:
                active_cfg_inputs = {
                    **self._uncond_cfg_inputs(),
                    "target_logp": target_logp,
                }
            elif use_mw:
                active_cfg_inputs = {
                    **self._uncond_cfg_inputs(),
                    "target_mw": target_mw,
                }
            elif use_natoms:
                active_cfg_inputs = {
                    **self._uncond_cfg_inputs(),
                    "target_n_atoms": target_n_atoms,
                }
            else:
                active_cfg_inputs = self._uncond_cfg_inputs()
            preds = model(
                mol_t,
                t.view(-1, 1),
                prev_outs=prev_preds,
                cfg_inputs=active_cfg_inputs,
            )
            model.apply_activations(preds)
            return preds

        uncond = self._uncond_cfg_inputs()

        preds = model(
            mol_t,
            t.view(-1, 1),
            prev_outs=prev_preds,
            cfg_inputs=uncond,
        )

        if use_natoms:
            cond = {**uncond, "target_n_atoms": target_n_atoms}
            preds_cond = model(
                mol_t,
                t.view(-1, 1),
                prev_outs=prev_preds,
                cfg_inputs=cond,
            )
            preds = self.apply_cfg(
                preds_cond,
                preds,
                self.natoms_cfg_guidance_scale,
            )

        if use_mw:
            cond = {**uncond, "target_mw": target_mw}
            preds_cond = model(
                mol_t,
                t.view(-1, 1),
                prev_outs=prev_preds,
                cfg_inputs=cond,
            )
            preds = self.apply_cfg(
                preds_cond,
                preds,
                self.mw_cfg_guidance_scale,
            )

        if use_logp:
            cond = {**uncond, "target_logp": target_logp}
            preds_cond = model(
                mol_t,
                t.view(-1, 1),
                prev_outs=prev_preds,
                cfg_inputs=cond,
            )
            preds = self.apply_cfg(
                preds_cond,
                preds,
                self.logp_cfg_guidance_scale,
            )

        if use_prop:
            preds_cond = model(
                mol_t,
                t.view(-1, 1),
                prev_outs=prev_preds,
                cfg_inputs=cfg_inputs,
            )
            preds = self.apply_cfg(
                preds_cond,
                preds,
                self.cfg_guidance_scale,
            )

        # Apply activations after CFG so guidance operates on raw logits.
        model.apply_activations(preds)

        return preds


def compute_molecular_weight(
    atom_indices: torch.Tensor,
    atom_tokens: list[str],
    batch: torch.Tensor | None = None,
    num_graphs: int | None = None,
) -> torch.Tensor:
    """Compute molecular weight per graph from atom token indices.

    Uses RDKit's periodic table for accurate atomic weights.

    Args:
        atom_indices: (N,) integer tensor of atom type indices.
        atom_tokens: ordered list of element symbols (vocab).
        batch: (N,) graph membership for each atom.
        num_graphs: number of graphs in the batch.

    Returns:
        (num_graphs,) tensor of molecular weights in Daltons.
        If batch / num_graphs are not provided, returns a scalar.
    """
    weights = torch.tensor(
        [_PERIODIC_TABLE.GetAtomicWeight(tok) for tok in atom_tokens],
        dtype=torch.float,
        device=atom_indices.device,
    )
    per_atom_mw = weights[atom_indices.long()]

    if batch is not None and num_graphs is not None:
        mw = torch.zeros(num_graphs, device=atom_indices.device)
        mw.scatter_add_(0, batch, per_atom_mw)
        return mw

    return per_atom_mw.sum().unsqueeze(0)


def compute_logp(
    smiles: list[str] | str,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Compute RDKit Crippen MolLogP for each SMILES on the fly.

    Args:
        smiles: A single canonical SMILES or a list of length ``num_graphs``.
            Empty / unparsable strings yield 0.0 (the trained null prior).
        device: Device for the returned tensor.

    Returns:
        ``(num_graphs,)`` float tensor of logP values.
    """
    if isinstance(smiles, str):
        smiles = [smiles]
    values: list[float] = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s) if s else None
        values.append(float(Crippen.MolLogP(mol)) if mol is not None else 0.0)
    return torch.tensor(values, dtype=torch.float, device=device)


def _encode_signal(
    value: torch.Tensor | None,
    encoder: nn.Module,
    null_emb: nn.Parameter,
    drop_mask: torch.Tensor | None,
    batch_size: int,
) -> torch.Tensor:
    """Shared logic for encoding a single CFG signal with dropout."""
    if value is None:
        return null_emb.unsqueeze(0).expand(batch_size, -1)
    
    if value.ndim == 1:
        value = value.unsqueeze(-1) if value.dtype.is_floating_point else value
    emb = encoder(value)

    if drop_mask is not None:
        null = null_emb.unsqueeze(0).expand_as(emb)
        emb = torch.where(drop_mask.unsqueeze(-1), null, emb)
    else:
        # drop_mask is None when dropout_prob=0.  Still route null_emb through
        # the autograd graph (with zero weight) so DDP does not flag it as an
        # unused parameter, while leaving the output numerically unchanged.
        emb = emb + null_emb.sum() * 0
    return emb


class UnifiedCFGEmbedding(nn.Module):
    """Unified classifier-free guidance embedding.

    Bundles property, n_atoms, and molecular weight conditioning into a
    single module.  Each signal has its own sub-encoder and learnable null
    embedding; the sub-embeddings are concatenated and projected to
    ``out_dim``.

    Accepts a ``cfg_inputs`` dict so callers do not need separate kwargs
    per signal.

    Input dict keys (all optional):
        properties, property_drop_mask,
        target_n_atoms, natoms_drop_mask,
        target_mw, mw_drop_mask

    Output: Float Tensor [Batch, out_dim]
    """

    def __init__(
        self,
        out_dim: int,
        # Property conditioning (num_properties=0 disables)
        num_properties: int = 0,
        property_hidden_dim: int = 128,
        # N-atoms conditioning
        use_natoms: bool = False,
        natoms_encoder: nn.Module | None = None,
        # MW conditioning
        use_mw: bool = False,
        mw_sinusoidal_dim: int = 64,
        mw_max_period: float = 1000.0,
        # LogP conditioning (RDKit Crippen, computed on the fly from SMILES)
        use_logp: bool = False,
        logp_sinusoidal_dim: int = 64,
        # QM9 logP roughly spans [-3, 6]; period 20 gives well-separated phases.
        logp_max_period: float = 20.0,
    ):
        super().__init__()
        self.out_dim = out_dim
        internal_dim = 0

        self.property_encoder = None
        self._property_null = None
        if num_properties > 0:
            prop_out = property_hidden_dim
            self.property_encoder = nn.Sequential(
                nn.Linear(num_properties, property_hidden_dim),
                nn.LayerNorm(property_hidden_dim),
                nn.SiLU(),
                nn.Linear(property_hidden_dim, prop_out),
            )
            self._property_null = nn.Parameter(torch.randn(prop_out) * 0.02)
            internal_dim += prop_out

        self.natoms_encoder = None
        self._natoms_null = None
        if use_natoms:
            if natoms_encoder is None:
                raise ValueError(
                    "use_natoms=True requires a `natoms_encoder` module "
                    "(provide it as a Hydra _target_ sub-block)."
                )
            if not hasattr(natoms_encoder, "out_dim"):
                raise AttributeError(
                    f"natoms_encoder {type(natoms_encoder).__name__} must "
                    "expose an `out_dim` attribute so the null embedding "
                    "can be sized."
                )
            self.natoms_encoder = natoms_encoder
            natoms_dim = int(natoms_encoder.out_dim)
            self._natoms_null = nn.Parameter(torch.randn(natoms_dim) * 0.02)
            internal_dim += natoms_dim

        self.mw_encoder = None
        self._mw_null = None
        if use_mw:
            self._mw_sinusoidal = SinusoidalEncoding(
                mw_sinusoidal_dim, max_period=mw_max_period
            )
            self.mw_encoder = nn.Sequential(
                nn.Linear(mw_sinusoidal_dim, mw_sinusoidal_dim),
                nn.SiLU(),
                nn.Linear(mw_sinusoidal_dim, mw_sinusoidal_dim),
            )
            self._mw_null = nn.Parameter(torch.randn(mw_sinusoidal_dim) * 0.02)
            internal_dim += mw_sinusoidal_dim

        self.logp_encoder = None
        self._logp_null = None
        if use_logp:
            self._logp_sinusoidal = SinusoidalEncoding(
                logp_sinusoidal_dim, max_period=logp_max_period
            )
            self.logp_encoder = nn.Sequential(
                nn.Linear(logp_sinusoidal_dim, logp_sinusoidal_dim),
                nn.SiLU(),
                nn.Linear(logp_sinusoidal_dim, logp_sinusoidal_dim),
            )
            self._logp_null = nn.Parameter(torch.randn(logp_sinusoidal_dim) * 0.02)
            internal_dim += logp_sinusoidal_dim

        if internal_dim > 0:
            self.projection = nn.Sequential(
                nn.Linear(internal_dim, out_dim),
                nn.SiLU(),
                nn.Linear(out_dim, out_dim),
            )
        else:
            self.projection = None

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _encode_mw(self, value, drop_mask, batch_size):
        if value is None:
            return self._mw_null.unsqueeze(0).expand(batch_size, -1)
        if value.ndim == 1:
            value = value.unsqueeze(-1)
        emb = self.mw_encoder(self._mw_sinusoidal(value))
        if drop_mask is not None:
            null = self._mw_null.unsqueeze(0).expand_as(emb)
            emb = torch.where(drop_mask.unsqueeze(-1), null, emb)
        return emb

    def _encode_logp(self, value, drop_mask, batch_size):
        if value is None:
            return self._logp_null.unsqueeze(0).expand(batch_size, -1)
        if value.ndim == 1:
            value = value.unsqueeze(-1)
        emb = self.logp_encoder(self._logp_sinusoidal(value))
        if drop_mask is not None:
            null = self._logp_null.unsqueeze(0).expand_as(emb)
            emb = torch.where(drop_mask.unsqueeze(-1), null, emb)
        else:
            emb = emb + self._logp_null.sum() * 0
        return emb

    def forward(
        self,
        cfg_inputs: dict,
        batch_size: int,
    ) -> torch.Tensor:
        parts: list[torch.Tensor] = []

        if self.property_encoder is not None:
            parts.append(
                _encode_signal(
                    cfg_inputs.get("properties"),
                    self.property_encoder,
                    self._property_null,
                    cfg_inputs.get("property_drop_mask"),
                    batch_size,
                )
            )

        if self.natoms_encoder is not None:
            parts.append(
                _encode_signal(
                    cfg_inputs.get("target_n_atoms"),
                    self.natoms_encoder,
                    self._natoms_null,
                    cfg_inputs.get("natoms_drop_mask"),
                    batch_size,
                )
            )

        if self.mw_encoder is not None:
            parts.append(
                self._encode_mw(
                    cfg_inputs.get("target_mw"),
                    cfg_inputs.get("mw_drop_mask"),
                    batch_size,
                )
            )

        if self.logp_encoder is not None:
            parts.append(
                self._encode_logp(
                    cfg_inputs.get("target_logp"),
                    cfg_inputs.get("logp_drop_mask"),
                    batch_size,
                )
            )

        if not parts:
            device = next(self.parameters()).device
            return torch.zeros(
                batch_size,
                self.out_dim,
                device=device,
            )

        return self.projection(torch.cat(parts, dim=-1))


@torch.no_grad()
def evaluate_cfg_steering(
    module,
    dataloader,
    target_n_atoms: int,
    num_batches: int | None = None,
) -> dict:
    """Evaluate how well n_atoms CFG steers generation toward a target size.

    Samples molecules from ``dataloader`` with ``target_n_atoms`` overridden
    and reports statistics about the actually generated atom counts.

    Args:
        module: The lightning module (must expose ``.sample()``).
        dataloader: A predict/test dataloader yielding ``(mol_t, mol_1)`` batches.
        target_n_atoms: The desired atom count to condition on.
        num_batches: If set, only process this many batches (useful for quick checks).

    Returns:
        Dictionary with keys:
            target_n_atoms, mean_n_atoms, std_n_atoms,
            median_n_atoms, exact_match_rate, within_1_rate,
            within_2_rate, n_molecules, all_n_atoms (list).
    """
    all_n_atoms: list[int] = []

    for batch_idx, batch in enumerate(dataloader):
        if num_batches is not None and batch_idx >= num_batches:
            break

        mol_t, mol_1 = batch
        # In standalone eval (outside Lightning Trainer predict), dataloader
        # batches stay on CPU; move them to the same device as the module.
        mol_t = mol_t.to(module.device)
        mol_1 = mol_1.to(module.device)
        batch_size = mol_t.batch_size
        override = torch.full(
            (batch_size,), target_n_atoms, dtype=torch.long, device=module.device
        )

        generated = module.sample(
            batch,
            batch_idx,
            return_traj=False,
            target_n_atoms_override=override,
        )

        # Fast path: avoid expensive Python-side to_data_list() conversion.
        # Per-graph atom counts are directly available from the batch vector.
        batch_n_atoms = torch.bincount(
            generated.batch, minlength=generated.batch_size
        ).to(dtype=torch.long)
        all_n_atoms.extend(batch_n_atoms.cpu().tolist())

    if not all_n_atoms:
        return {"target_n_atoms": target_n_atoms, "n_molecules": 0}

    counts = torch.tensor(all_n_atoms, dtype=torch.float)
    exact = (counts == target_n_atoms).float().mean().item()
    within_1 = ((counts - target_n_atoms).abs() <= 1).float().mean().item()
    within_2 = ((counts - target_n_atoms).abs() <= 2).float().mean().item()

    return {
        "target_n_atoms": target_n_atoms,
        "mean_n_atoms": counts.mean().item(),
        "std_n_atoms": counts.std().item(),
        "median_n_atoms": counts.median().item(),
        "exact_match_rate": exact,
        "within_1_rate": within_1,
        "within_2_rate": within_2,
        "n_molecules": len(all_n_atoms),
        "all_n_atoms": all_n_atoms,
    }
