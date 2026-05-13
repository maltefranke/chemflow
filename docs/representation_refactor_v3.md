# Representation Refactor — Design Review (v3)

A **`Representation`** enum with three values:

| Value | Uses `c`? | Uses chemical edges (`e`)? |
|---|---|---|
| `pointcloud` | no | no |
| `charged_pointcloud` | yes | no |
| `geometric_graph` | yes | yes |

A **`Capabilities`** dataclass declared per dataset class:

```python
@dataclass(frozen=True)
class Capabilities:
    provides_charges: bool
    provides_topology: bool
```

At startup, `validate_representation(caps, mode)` checks `mode.requires_X ⊆ caps.provides_X` and raises before any data loads.

`MoleculeData` stays a single class. Datasets emit canonical (richest) samples using the canonical vocab. The projection boundary in `FlowMatchingDatasetWrapper` emits the **model-facing samples** the rest of the pipeline consumes — same canonical vocab in all modes, but with topology removed in non-topology modes and charges removed only in non-charge modes.

**Model-facing projection output** (what the interpolator/model receive):

| Mode | `x` | `a` | `c` | `edge_index` | `e` |
|---|---|---|---|---|---|
| pointcloud | real | real | zeros(N) | fully_connected(N) | zeros(E) — all `<NO_BOND>` |
| charged_pointcloud | real | real | real | fully_connected(N) | zeros(E) — all `<NO_BOND>` |
| geometric_graph | real | real | real | sparse bond graph | bond types |

**Note on `e=0` in non-topology modes.** Token 0 of the canonical edge vocab is `<NO_BOND>` (the existing convention from `Preprocessing.NO_BOND_TOKEN`). In non-topology modes we relabel *every* runtime edge as `<NO_BOND>`, including atom pairs that are actually bonded in the source molecule. So `e=0` here is the **inert default edge token on runtime message-passing edges** — not a supervised statement about source chemistry. The mislabel is safe only because edge losses and RDKit metrics are gated off (§4.8). We do not introduce a `<RUNTIME_EDGE>` token because the canonical `<NO_BOND>` slot serves the same dummy-token role without expanding the vocab.

**Note on `c=index0` in pointcloud mode.** Charge tokens are sorted-int strings (e.g. `["-1", "0", "1"]`), so `index 0` is *not* always the neutral token. Projection resolves the neutral charge by name (`token_to_index(vocab.charge_tokens, "0")`) and falls back to literal index 0 when "0" isn't in the vocab. The fallback is just a constant whose embedding the model absorbs into bias — harmless because charge loss is gated off in pointcloud.

**Terminology note: "non-topology" vs "non-charge" are independent.** `charged_pointcloud` is *non-topology* but charges *are* supervised. The loss/inference gating tables distinguish these axes by `mode.requires_topology` and `mode.requires_charges` independently — never as a single "non-graph" flag.

## 2. Canonical vs Model-Facing Artifacts

Preprocessing caches **canonical** dataset facts once. The cache is shared across all representations — no per-mode duplication, no re-preprocessing on mode switch.

```text
preprocessing  ->  canonical Vocab + canonical Distributions  (shared cache)
                          │
                          ├──>  loss_weight_distributions = deepcopy(canonical)
                          │     (real class frequencies; loss weighting + metrics)
                          │
                          └──>  init_uniform_prior(canonical) -> uniform priors
                                          │
                                          └──>  project_distributions_to_representation(...)
                                                (edge priors pinned to <NO_BOND>
                                                 when not requires_topology;
                                                 charge priors pinned to "0"
                                                 when not requires_charges;
                                                 atom and n_atom priors unchanged)
                                                          │
                                                          ▼
                                          datamodule.distributions:
                                            - dataset (reads coord_std)
                                            - FlowMatchingDatasetWrapper.sample_prior_graph
                                            - interpolator's _cat_atom/_cat_charge/_cat_edge
                                          lightning_module.distributions:
                                            - integrator.integrate_step (insertions + edges)
```

**The two channels already exist in code.** `run.py` already constructs two parallel distribution objects: `loss_weight_distributions` (canonical) for weighting and metrics, and `token_prior_distribution` (uniform) for everything that samples. The refactor adds one projection call on the latter; the routing wiring is unchanged.

**Why distribution projection is the only projection we need (no vocab projection).** Canonical vocab works in all modes — heads/embeddings adapt via `${len:${data.vocab.*_tokens}}`. The only mode-dependent thing in the data path is what *samples* from those priors during `sample_prior_graph` / interpolation / integration. Under uncorrected canonical priors, non-topology samples would draw real bond/charge tokens that disagree with the all-zero projected targets and trip the very gating we're trying to avoid (e.g. `need_e_sub > 0` → edge_type_head re-enters the loss graph with garbage).

## 3. The new module: `chemflow/dataset/representation.py`

A small, pure-function module. No classes beyond the enum + dataclass.

```python
# src/chemflow/dataset/representation.py
from dataclasses import dataclass, replace
from enum import Enum

import torch
import torch.nn.functional as F

from chemflow.dataset.molecule_data import MoleculeData
from chemflow.dataset.vocab import Distributions


class Representation(str, Enum):
    POINTCLOUD = "pointcloud"
    CHARGED_POINTCLOUD = "charged_pointcloud"
    GEOMETRIC_GRAPH = "geometric_graph"

    @property
    def requires_charges(self) -> bool:
        return self != Representation.POINTCLOUD

    @property
    def requires_topology(self) -> bool:
        return self == Representation.GEOMETRIC_GRAPH


@dataclass(frozen=True)
class Capabilities:
    provides_charges: bool
    provides_topology: bool


def validate_representation(caps: Capabilities, mode: Representation) -> None:
    """Raise if dataset cannot satisfy the requested representation."""
    ...


def build_fully_connected_edge_index(n: int, device=None) -> torch.Tensor:
    """Directed, no-self-loop, symmetric edge_index of shape (2, N*(N-1))."""
    ...


def project_molecule_to_representation(
    mol: MoleculeData, vocab: Vocab, mode: Representation
) -> MoleculeData:
    """Project a (possibly partial) canonical MoleculeData onto the run's
    representation. In non-topology modes: edge_index = fully_connected,
    e = zeros (= <NO_BOND>, token 0 of canonical edge vocab by construction).
    In pointcloud: c is filled with the canonical index of "0" (resolved by
    name via vocab.charge_tokens; falls back to 0 if "0" is absent — a constant
    the model can absorb into bias). geometric_graph: returns mol unchanged.
    """
    ...


def project_distributions_to_representation(
    distributions: Distributions, vocab: Vocab, mode: Representation
) -> Distributions:
    """Project canonical distributions onto the run's representation.

    Mirrors `project_molecule_to_representation` but for the priors that
    `sample_prior_graph`, the interpolator's internal Categoricals, and the
    integrator's insertion sampler draw from. In non-topology modes
    `edge_type_distribution` is replaced by a one-hot at the canonical
    `<NO_BOND>` token; in pointcloud `charge_type_distribution` is replaced by
    a one-hot at the neutral charge index (resolved via vocab; falls back to 0
    if "0" isn't a token). All other fields are returned unchanged so the
    atom-type prior, n_atoms prior, and coordinate_std flow through.
    """
    ...
```

**Deleted** (vs. previous design): `build_edge_tokens`, `build_charge_tokens`, `build_edge_distribution`, `build_charge_distribution`, plus the `<RUNTIME_EDGE>` / `<NO_CHARGE>` constants. No vocab projection — model uses canonical vocab. The single `project_distributions_to_representation` helper replaces all four old `build_*` functions.

## 4. File-by-file changes

### 4.1 New files

| Path | Purpose |
|---|---|
| `src/chemflow/dataset/representation.py` | The module above. |
| `configs/data/tmqm.yaml` | New dataset config (pointcloud-only). |
| `src/chemflow/dataset/tmqm.py` | New dataset class. |

### 4.2 [configs/default.yaml](../configs/default.yaml)

```yaml
representation: geometric_graph  # pointcloud | charged_pointcloud | geometric_graph
```

Single source of truth for projection, loss gating, metric selection, and checkpoint monitor.

### 4.3 [configs/data/qm9.yaml](../configs/data/qm9.yaml), [configs/data/geom.yaml](../configs/data/geom.yaml), `configs/data/tmqm.yaml`

**Shared cache paths — not scoped by representation:**

```yaml
preprocessing:
  atom_tokens_path:   ${data.preprocessing.root}/processed/atom_tokens.txt
  edge_tokens_path:   ${data.preprocessing.root}/processed/edge_tokens.txt
  charge_tokens_path: ${data.preprocessing.root}/processed/charge_tokens.txt
  distributions_path: ${data.preprocessing.root}/processed/distributions.pt
```

Thread `representation` into the `FlowMatchingDatasetWrapper` constructor and the lightning module. Do not thread it into preprocessing, interpolation, integration, or the per-dataset classes.

### 4.4 [src/chemflow/dataset/qm9.py](../src/chemflow/dataset/qm9.py), [src/chemflow/dataset/geom.py](../src/chemflow/dataset/geom.py), `src/chemflow/dataset/tmqm.py`

Each per-dataset class declares its `CAPABILITIES`. Datasets are **representation-agnostic**: they parse source data into canonical `MoleculeData` using the canonical vocab, full stop. No projection, no `self.representation` attribute, no mode branches.

```python
class FlowMatchingQM9Dataset(RevisedQM9):
    CAPABILITIES = Capabilities(provides_charges=True, provides_topology=True)

    def __init__(self, root, vocab, distributions, ...):
        self.vocab = vocab
        self.distributions = distributions

    def __getitem__(self, index):
        data = super().__getitem__(index)
        coord = (data.pos - data.pos.mean(dim=0)) / self.distributions.coordinate_std
        a = ... # canonical atom token indices
        e = ... # canonical edge token indices
        c = ... # canonical charge token indices
        return MoleculeData(x=coord, a=a, c=c, e=e, edge_index=data.edge_index)
```

For datasets whose source already has some fields missing (e.g. TMQM with no charges, no bonds), `__getitem__` returns `MoleculeData(x, a, c=None, e=None, edge_index=None)`. The wrapper's projection is additive there (§4.6).

**Contract: `.get()` returns raw, `.__getitem__()` returns canonical.** Preprocessing iterates `.get(i)` to discover the canonical vocab and distributions *before* any vocab exists. So `.get(i)` MUST return raw source data with PyG-style fields (`z, pos, edge_index, edge_attr, charges`) — not the canonical `MoleculeData`. This is PyG's standard contract: `get` = raw cached sample, `__getitem__` = transformed.

QM9 inherits this from `InMemoryDataset` automatically (we override `__getitem__` but not `get`). **GEOM violates it** at [geom.py:450](../src/chemflow/dataset/geom.py#L450) — `def get(self, index): return self.__getitem__(index)` delegates to the canonical builder, which needs `self.vocab` (not yet built at preprocessing time) and returns `MoleculeData` (not the raw PyG fields preprocessing expects). Fix:

```python
class FlowMatchingGEOMDataset(GEOM):
    def get(self, index):
        return super().get(index)   # raw GEOM record from the parent

    def __getitem__(self, index):
        data = super().__getitem__(index)  # raw, then build canonical MoleculeData
        ...
        return MoleculeData(...)
```

New dataset classes (TMQM, future) must follow this contract.

### 4.5 [src/chemflow/dataset/preprocessing.py](../src/chemflow/dataset/preprocessing.py)

`Preprocessing` is **representation-agnostic**. It always emits the canonical superset:

- atom tokens, edge tokens (`<NO_BOND>` + bond types), charge tokens — all real
- atom-type, edge-type, charge-type distributions — all real
- n_atoms_distribution, coordinate_std
- `pairwise_distance_histogram`, `radius_of_gyration_histogram` (pointcloud target stats, in Å, always computed)

The previous `if requires_topology / requires_charges` branches in `_compute_tokens_from_data` and `_compute_distributions` are removed. The `representation` parameter is removed from `Preprocessing.__init__`.

For very large datasets where pointcloud target histogram computation is expensive (O(N²) per molecule), gate behind a `compute_pointcloud_metric_targets: true` flag; default true.

**Canonical dummy-vocab convention for absent source fields.** If the dataset source does not provide bonds (e.g. TMQM), preprocessing must still emit a one-element placeholder vocab + distribution so that `${len:${data.vocab.edge_tokens}}` resolves to a non-zero head output dim:

```text
no source charges:  charge_tokens = ["0"],          charge_type_distribution = [1.0]
no source bonds:    edge_tokens   = ["<NO_BOND>"],  edge_type_distribution   = [1.0]
```

This is the *canonical* state for those datasets — not a representation projection. With this convention, head/embedding sizes are always well-defined regardless of dataset capabilities, and projection on top of these singleton-canonical fields is a no-op for the absent axis. Datasets declare their `Capabilities` so `validate_representation` still prevents asking for a representation richer than the dataset supports.

### 4.6 [src/chemflow/dataset/flow_matching_wrapper.py](../src/chemflow/dataset/flow_matching_wrapper.py)

**This is where projection lives.** The wrapper owns all the representation-aware transforms in the data path: it projects the canonical sample from the base dataset, draws a projected-prior sample, and interpolates the pair. Constructor gains `representation: Representation` and `vocab: Vocab` (already takes `distributions`). Inside `__getitem__`:

```python
def __getitem__(self, index):
    target = self.base_dataset[index]                         # canonical MoleculeData
    target = project_molecule_to_representation(              # → model-facing
        target, self.vocab, self.representation,
    )
    sample = sample_prior_graph(self.distributions, ...)      # distributions already projected
    if self.stage == "train":
        return self.interpolator.interpolate_single(sample, target, t.item())
    return sample, target
```

`self.distributions` is the projected `token_prior_distribution` (Channel B, see §2 and §4.10), so the `sample` already has `e=0` whenever topology is disabled, and `c=neutral_idx` whenever charges are disabled. In those modes, the corresponding `need_*_sub` terms stay identically zero and the interpolator's inactive corruption paths are natural no-ops.

**Why here and not in the dataset.** Co-locating projection with interpolation means *all* representation-aware data work happens in one file. The dataset stays pure (parse source → canonical, no mode awareness). The get/`__getitem__` contract becomes the standard PyG one (raw vs. transformed) instead of overloaded with an extra "projected" layer. Preprocessing iterates `.get(i)` for raw fields with no risk of triggering projection or hitting a not-yet-built vocab.

### 4.7 [src/chemflow/flow_matching/interpolation.py](../src/chemflow/flow_matching/interpolation.py), [src/chemflow/flow_matching/integration.py](../src/chemflow/flow_matching/integration.py), [src/chemflow/flow_matching/sampling.py](../src/chemflow/flow_matching/sampling.py)

**No `representation` parameter threaded.** Projected priors and projected molecules together make inactive topology/charge paths natural no-ops:

```text
project_molecule_to_representation:           target.e = 0,  target.c = neutral_idx
project_distributions_to_representation:      edge_dist = one-hot(<NO_BOND>)
                                              charge_dist = one-hot(neutral_idx)
                                            │
                          ┌─────────────────┼────────────────────┐
                          ▼                                      ▼
        sampling.py + interpolation.py            integration.py
   sample.e = Categorical(edge_dist) = 0         insertion edges sampled from edge_dist = 0
   need_e_sub = (sample.e != target.e) = 0       joined runtime edges: token 0 = <NO_BOND>
   lambda_e_sub = 0   → sub_e corruption no-op
```

The integrator stays representation-agnostic; `sample()` owns inference-side clamping and passes `ins_edge_head=None` for non-topology modes. See §4.8(d) and §4.8(e).

### 4.8 [src/chemflow/model/lightning_module.py](../src/chemflow/model/lightning_module.py)

`LightningModuleRates` gains a `representation` attribute (threaded from `run.py`). Used at four sites:

**(a) Loss gating in `training_step`.** Two reasons gates are applied; only the first is a DDP correctness issue:

The gating follows the nested-loss structure from the diagram:
**all modes** supervise `x`, `do_sub_a`, `sub_a_class`, `do_del`, `ins_rate`, and the **x / a** components of `ins_gmm`. **`requires_charges`** adds `c` and the **c** component of `ins_gmm`. **`requires_topology`** adds `do_sub_e`, `sub_e_class`, `ins_e`, `ins_e_ii`.

| Loss | Gated when | Why |
|---|---|---|
| `sub_e_class` (edge_type_head) | `not requires_topology` | **DDP correctness.** `need_e_sub = (mol_t.e != target.e) = 0` → `lambda_e_sub ≡ 0` → `to_sub_edge_mask` empty → `class_loss` short-circuits → `edge_type_head` not in autograd graph → DDP unused-param error. |
| `do_sub_e`, `ins_e`, `ins_e_ii` | `not requires_topology` | **Semantic.** BCE / CE on dummy targets is well-defined (DDP-safe) but trains heads on no-information tasks. |
| `c` (charge_head) | `not requires_charges` | **Semantic.** `c_loss` uses `non_del_mask` (not `lambda_c_sub`), so DDP-safe, but every target is the neutral-charge index. |
| **`ins_gmm` c-term** (c_probs sub-head) | `not requires_charges` | **Semantic.** `typed_gmm_loss` aggregates `log_prob_x + log_prob_a + log_prob_c`. In non-charge modes the c-target is the constant neutral_idx, so the c-term trains a constant predictor. We split the joint, keep x/a, drop c. The `c_probs` sub-head needs a DDP dummy because its params would otherwise leave the graph. |

Replacements use **graph-preserving** dummies on the head outputs (or parameter-sums where the head isn't called):

```python
do_sub_e_loss    = torch.zeros(num_graphs, 1, device=...) + 0.0 * do_sub_e_head.sum()
sub_e_class_loss = torch.zeros(num_graphs, 1, device=...) + 0.0 * e_pred.sum()
c_loss           = torch.zeros(num_graphs, 1, device=...) + 0.0 * c_pred.sum()  # requires_charges=False

ins_edge_head_dummy = 0.0 * sum(p.sum() for p in self.model.ins_edge_head.parameters())
ins_loss_e    += ins_edge_head_dummy
ins_loss_e_ii += ins_edge_head_dummy

# typed_gmm_loss exposes the per-component log-probs so the lightning module
# can sum only the active ones. c_probs is a slice of `scalar_mlp`'s output
# (heads.py:257), not a submodule — so the DDP dummy attaches to the output
# tensor, not to a parameter list.
if not self.representation.requires_charges:
    ins_loss_gmm = -(log_prob_mix + log_prob_x + log_prob_a).logsumexp(-1).mean()
    ins_loss_gmm += 0.0 * gmm_pred_dict["c_probs"].sum()
else:
    ins_loss_gmm = -(log_prob_mix + log_prob_x + log_prob_a + log_prob_c).logsumexp(-1).mean()
```

`typed_gmm_loss` needs a small refactor to return the per-component log-probs (or an `include_c: bool` flag); the joint reduction moves to the caller. Equivalent to the existing aggregation when all terms are kept.

**(b) `class_loss` defensive fix in [losses.py](../src/chemflow/model/losses.py).** The empty-mask short-circuit at line 238 used to return `torch.zeros(...)` — a detached constant that dropped `class_pred` from the graph. Now returns `torch.zeros(...) + 0.0 * class_pred.sum()`. Good hygiene regardless of representation; would have caught the `edge_type_head` bug on its own.

**(c) RDKit-mol gating in `validation_step` and `predict_step`.** Skipped in non-topology modes (no chemical bonds to perceive). Pointcloud-specific metrics (§4.9) run instead in validation.

**(d) `ins_edge_head` short-circuit at inference.** In `sample`, [line 1026](../src/chemflow/model/lightning_module.py#L1026): pass `ins_edge_head=None` in non-topology modes. The integrator's fallback path samples insertion edge types from the projected edge prior (one-hot at `<NO_BOND>`), avoiding both wasted forward compute and the implicit "trained head drives inference" semantics.

**(e) Inference-side prediction clamping.** Because the model heads gated in §4.8(a) stay at near-init weights, their inference outputs are essentially random samples over the full canonical vocab. Under the v2 singleton design this was self-correcting (output dim 1 → only one possible sample). Under v3 the heads are full-size, so we must clamp their inference outputs to the canonical defaults in `sample()` — parallel to §4.8(d).

The clamping happens **immediately after the Categorical samples** for `c_pred`/`e_pred` are drawn from the heads ([lightning_module.py:1058,1066,1082](../src/chemflow/model/lightning_module.py#L1058)), before `mol_1_pred` is constructed:

```python
# In lightning_module.sample, right after the existing Categorical().sample() calls:

if not self.representation.requires_topology:
    e_pred.fill_(0)            # all <NO_BOND>
    do_sub_e_probs.zero_()     # never substitute an edge

if not self.representation.requires_charges:
    neutral_idx = (
        self.vocab.charge_tokens.index("0")
        if "0" in self.vocab.charge_tokens
        else 0
    )
    c_pred.fill_(neutral_idx)
```

**GMM-inserted charges** are a separate concern. After integration, inserted atoms get their `c` from sampling `c_probs` inside `sample_from_typed_gmm` (called by the integrator). Two options:

- **Preferred — thread a flag through `integrate_step_gnn`.** Add a `force_charge_idx: int | None = None` parameter. When non-None, the inserted-atom GMM sampler emits that index instead of drawing from `c_probs`. Pass `neutral_idx` in non-charge modes from `sample()`. This is clean because insertion-time is the only moment where `c_probs` actually drives a write — fixing it at the source avoids needing to identify "newly inserted nodes" after the integration step.
- **Avoid: post-hoc override.** Trying to clamp `new_atoms.c = neutral_idx` after `integrator.integrate_step` returns is brittle because by then inserted atoms have been merged with existing nodes — you have to recover the insertion mask before overwriting, and any future integrator refactor breaks the assumption.

Per-mode summary of what's gated at inference:

| Mode | `e_pred` | `do_sub_e_probs` | `c_pred` | `ins_edge_head` | GMM-inserted c |
|---|---|---|---|---|---|
| `pointcloud` | 0 | 0 | neutral_idx | None | neutral_idx |
| `charged_pointcloud` | 0 | 0 | sampled normally | None | sampled normally |
| `geometric_graph` | sampled normally | sampled normally | sampled normally | head | sampled normally |

The integrator itself stays representation-agnostic — `sample()` sanitizes the predictions before handing them off.

**Head/embedding sizes** follow the canonical vocab via `${len:${data.vocab.*_tokens}}` in [configs/heads/heads.yaml](../configs/heads/heads.yaml) and [configs/embeddings/](../configs/embeddings/). Heads are full-size in all modes. **Cross-mode checkpoint compatibility — shape only, not semantically trained.** Pointcloud checkpoints load into geometric_graph mode (and vice versa) because all parameter shapes match. But the gated heads (edge, do_sub_e, c, c_probs) in a pointcloud checkpoint are near-init — they need real training before geometric_graph inference is meaningful. Useful as a starting point, not as a finished cross-mode model.

### 4.9 Pointcloud metrics — [src/chemflow/utils/pointcloud_metrics.py](../src/chemflow/utils/pointcloud_metrics.py)

The RDKit pipeline is gated off in non-topology modes (§4.8(c)) and a parallel tensor-based metric family runs in its place. All metrics consume a `MoleculeBatch` and undo dataset coord normalization with `coord_scale=coordinate_std` so bin edges (Å) compare like-for-like with target stats.

| `val/pc/*` key | What |
|---|---|
| `atom_count_kl` | KL(gen ‖ target) over per-mol atom counts |
| `atom_type_kl` | KL(gen ‖ target) over atom-type histogram |
| `pair_dist_l1` | Mean L1 between gen/target distance histograms over **target-populated** atom-type pairs — empty-gen pairs count as max-pain L1 |
| `rog_l1` | L1 between gen/target radius-of-gyration histograms |
| `min_dist_violation` | Fraction of mols with smallest pairwise distance < 0.5 Å |
| `range_overflow` | Fraction of distances/RoGs outside the histogram range (silent-clamping warning) |
| `plots/atom_count`, `plots/atom_type`, `plots/rog` | GT-vs-gen marginal figures (1D); per-element-pair plots deferred |

Target stats (`pairwise_distance_histogram`, `radius_of_gyration_histogram`) are computed in canonical preprocessing and live in the shared cache. **DDP**: the plotting path enters `m.sync_context()` on every rank before rank 0 reads `gen_hist`, mirroring the RDKit-marginal pattern.

**Deferred**: target-derived bin edges, MMD with RBF kernels, per-element-pair distance plots.

### 4.10 [run.py](../run.py)

Before instantiating preprocessing, validate the representation against the dataset's class-level capabilities:

```python
mode = Representation(cfg.representation)
dataset_cls = hydra.utils.get_class(cfg.data.datamodule.datasets.train._target_)
validate_representation(dataset_cls.CAPABILITIES, mode)
```

After preprocessing, add **one** projection call on the sampling-side distributions channel. The vocab is canonical and passed everywhere unchanged. **The two distribution channels already exist** in current `run.py` — `loss_weight_distributions` (canonical) and `token_prior_distribution` (uniform priors for sampling); we just project the latter:

```python
canonical_vocab          = preprocessing.vocab
canonical_distributions  = preprocessing.distributions

# Channel A — canonical real frequencies. Unchanged.
loss_weight_distributions = deepcopy(canonical_distributions)

# Channel B — priors used by every Categorical sampling call. Now projected.
token_prior_distribution = init_uniform_prior(canonical_distributions)
token_prior_distribution = project_distributions_to_representation(   # <-- one new line
    token_prior_distribution, canonical_vocab, mode
)
```

That's the entire plumbing change. Existing call-site routing already does the right thing because both channels were already separately threaded:

| Consumer | Distributions channel | Field accessed |
|---|---|---|
| Dataset `__getitem__` (via datamodule) | B (projected) | `coordinate_std` — preserved by both `init_uniform_prior` and projection |
| `FlowMatchingDatasetWrapper.sample_prior_graph` (via datamodule) | B (projected) | edge / charge / atom / n_atoms priors |
| `Interpolator._cat_*` Categoricals (via datamodule) | B (projected) | edge / charge / atom priors |
| `RateIntegrator` (via lightning_module → `self.distributions`) | B (projected) | edge / charge / atom priors |
| Loss-weight setup, RDKit metrics, pointcloud metrics | A (canonical) | real class-frequency tensors and pointcloud target stats |

No new `sampling_distributions` argument anywhere. No new datamodule constructor argument; `setup()` threads the existing canonical `vocab` plus top-level `representation` into `FlowMatchingDatasetWrapper`.

SMILES-dependent metrics (e.g. `Novelty`) are topology-only:

```python
train_smiles = (
    base_dataset.get_all_smiles()
    if mode.requires_topology and hasattr(base_dataset, "get_all_smiles")
    else None
)
```

### 4.11 [configs/callbacks/callbacks.yaml](../configs/callbacks/callbacks.yaml)

```yaml
# @package _global_
callbacks:
  monitor_metric: ${if:${eq:${representation},geometric_graph},val/validity,val/pc/pair_dist_l1}
  monitor_metric_mode: ${if:${eq:${representation},geometric_graph},max,min}
  every_n_epochs_checkpoint:
    save_top_k: 0      # no monitor on this callback; last-only
    save_last: True
```

### 4.12 Files that **don't change**

| File | Why |
|---|---|
| [molecule_data.py](../src/chemflow/dataset/molecule_data.py) | `MoleculeData` already accepts optional fields. `PointCloud` class can be cleaned up later. |
| Backbone wrappers (`semla_bb.py`, `egnn.py`, `gvp.py`, `dit.py`) | Receive a fully-shaped `edge_index` and `e` in every mode. |
| `external_code/semla.py` | Upstream `SemlaGenerator` unused (SemlaBB calls `EquiInvDynamics` directly). |
| `heads.py`, `gmm.py` | Always built from canonical vocab size; no mode-dependent head shapes. |
| `assignment.py`, `losses.py` | Operate on tensors regardless of semantic meaning. Gating happens at the call site. |

## 5. The data path, end-to-end

### QM9 Pointcloud

```
configs/default.yaml:   representation: pointcloud
                ↓
run.py:                 validate_representation(QM9.CAPABILITIES, pointcloud) → OK
                ↓
Preprocessing:          canonical (shared cache)
                        atom_tokens   = ["C","F","H","N","O"]
                        edge_tokens   = ["<NO_BOND>","1","2","3","4"]
                        charge_tokens = ["0"]
                        distributions = real canonical
                ↓
run.py:                 token_prior_distribution = project_distributions_to_representation(
                            init_uniform_prior(canonical), canonical_vocab, pointcloud)
                        # edge_type_distribution    = [1, 0, 0, 0, 0]  (NO_BOND only)
                        # charge_type_distribution  = [1]              (QM9 has only "0")
                        # atom_type_distribution, n_atoms_distribution → unchanged uniform
                ↓
QM9Dataset.__getitem__: build canonical MoleculeData(x, a, c=real, e=real, edge_index=bonds)
                        (no projection, no representation flag — pure parsing)
                ↓
Wrapper.__getitem__:    project_molecule_to_representation(target, vocab, pointcloud)
                        → MoleculeData(x, a,
                                       c=neutral_idx(N),
                                       edge_index=fully_connected(N),
                                       e=zeros(E)=<NO_BOND>)
                        sample_prior_graph(projected_distributions, ...)
                        → e=zeros, c=neutral_idx prior  (matches target)
                        interpolator.interpolate_single(sample, target, t)
                ↓
Interpolator:           need_e_sub = 0; need_c_sub = 0 only when charges are disabled
                        → inactive corruption paths are no-ops
                ↓
Model forward:          backbones see (x, a, c=0, edge_index=dense, e=0)
                        canonical vocab → heads are full-size in all modes
                ↓
Loss:                   active:  x, do_sub_a, sub_a_class, do_del, ins_rate, ins_gmm
                        gated:   c, do_sub_e, sub_e_class, ins_e, ins_e_ii
                ↓
Validation:             tensor-based pointcloud metrics + marginal plots
```

### QM9 Charged Pointcloud

Same canonical cache. `project_distributions_to_representation` projects only the edge distribution; the charge distribution stays as the uniform prior so insertions sample real charges. `project_molecule_to_representation` keeps real `c` and zeros `edge_index`/`e`.

### QM9 Geometric Graph

Same canonical cache. `project_distributions_to_representation` returns input unchanged. `project_molecule_to_representation` is a no-op. RDKit metrics run.

### TMQM Pointcloud

TMQM emits `MoleculeData(x, a, c=None, e=None, edge_index=None)`. Projection fills `c=neutral_idx(N)` (falls back to literal `0` if "0" isn't in the discovered TMQM charge vocab), `edge_index=fully_connected(N)`, `e=zeros(E)`. Capabilities validation prevents geometric_graph mode at startup.

## 6. Summary of what's deleted vs. v2

| Removed | Why |
|---|---|
| `<RUNTIME_EDGE>` token | Replaced by canonical `<NO_BOND>` as the inert runtime edge token. |
| `<NO_CHARGE>` token | Replaced by canonical charge-0 token. |
| `build_edge_tokens`, `build_charge_tokens` | Vocab is canonical everywhere; no projection. |
| `build_edge_distribution`, `build_charge_distribution` | Replaced by single `project_distributions_to_representation` helper. |
| `${representation}/` in cache paths | Cache is shared. |
| `representation` parameter in `Preprocessing.__init__` | Preprocessing is canonical / mode-agnostic. |
| `if requires_topology/requires_charges` branches in `Preprocessing._compute_*` | Same. |
| `if requires_topology/requires_charges` branches in dataset `__getitem__` | Datasets parse canonical; wrapper projection handles modes. |

**Net effect**: one preprocessing per dataset, cross-mode checkpoint compatibility, simpler routing diagram. The only remaining mode-aware pieces in the data path are `project_molecule_to_representation` and `project_distributions_to_representation`, both invoked once at known sites.

## 7. Bonus schematic: the two projections

```
                 Canonical preprocessing cache
                 vocab + real training distributions
                              │
             ┌────────────────┴────────────────┐
             │                                 │
             │ Target side                     │ Prior side
             │                                 │
base_dataset[index]                   init_uniform_prior(distributions)
→ canonical MoleculeData              → uniform token prior
  x, a, c, e, edge_index                    │
             │                              │
             ▼                              ▼
project_molecule_to_representation    project_distributions_to_representation
(called in FlowMatchingDatasetWrapper) (called in run.py before datamodule)
             │                              │
             ▼                              ▼
target / mol_1                         token_prior_distribution
mode-facing MoleculeData               passed into datamodule
                                       │
                                       ▼
                                 sample_prior_graph(...)
                                       │
                                       ▼
                                  sample / mol_0

                      sample / mol_0 + target / mol_1 + t
                                      │
                                      ▼
                     interpolator.interpolate_single(...)
                                      │
                                      ▼
                               mol_t, mol_1, ins_targets
```

In pointcloud mode, the target-side projection makes `mol_1.e` all `<NO_BOND>`
and `mol_1.c` the neutral dummy. The prior-side projection makes
`sample_prior_graph` draw the same dummy edge/charge tokens, so `mol_0`,
`mol_t`, and `mol_1` all agree that disabled fields are inert.

## 8. Training heads and gates

`LightningModuleRates.shared_step` still reads every head from the model output:

```python
a_pred = preds["atom_type_head"]
x_pred = preds["pos_head"]
e_pred = preds["edge_type_head"]
c_pred = preds["charge_head"]

ins_rate_pred = preds["ins_rate_head"]
gmm_pred_dict = preds["gmm_head"]

do_del_head = preds["do_del_head"]
do_sub_a_head = preds["do_sub_a_head"]
do_sub_e_head = preds["do_sub_e_head"]
```

The representation gates decide which supervised losses carry real signal:

| Prediction key | Loss / use | Gate | In gated-off modes |
|---|---|---|---|
| `pos_head` (`x_pred`) | coordinate movement / GMM position terms | always active | never gated |
| `atom_type_head` (`a_pred`) | atom substitution class + GMM atom type terms | always active | never gated |
| `do_del_head` | deletion action loss | always active | never gated |
| `do_sub_a_head` | atom substitution action loss | always active | never gated |
| `ins_rate_head` (`ins_rate_pred`) | insertion rate loss | always active unless `n_atoms_strategy == "fixed"` | fixed-atom runs zero insertion/deletion dynamics |
| `charge_head` (`c_pred`) | charge class loss | `representation.requires_charges` | zero-gradient dummy loss keeps `charge_head` in DDP graph |
| `gmm_head["c_probs"]` | inserted-atom charge term in typed GMM loss | `representation.requires_charges` | charge term excluded; zero-gradient dummy keeps `c_probs` in DDP graph |
| `edge_type_head` (`e_pred`) | edge substitution class loss | `representation.requires_topology` | zero-gradient dummy loss keeps `edge_type_head` in DDP graph |
| `do_sub_e_head` | edge substitution action loss | `representation.requires_topology` | zero-gradient dummy loss keeps `do_sub_e_head` in DDP graph |
| `ins_edge_head` | inserted-to-existing and inserted-to-inserted edge losses | `representation.requires_topology` | supervised edge insertion losses skipped; parameter-sum dummy keeps the head in DDP graph |

Mode summary:

| Representation | `requires_charges` | `requires_topology` | Active chemical heads |
|---|---:|---:|---|
| `pointcloud` | false | false | atom / position / deletion / insertion-count paths only |
| `charged_pointcloud` | true | false | pointcloud paths + charge paths |
| `geometric_graph` | true | true | all heads carry real supervised signal |

## 9. Inference heads and clamps

During generation, `LightningModuleRates.sample()` still reads the same model
heads, then clamps inactive predictions before calling
`RateIntegrator.integrate_step_gnn`.

| Prediction / argument | Used for | Gate | Clamp / override location | In gated-off modes |
|---|---|---|---|---|
| `pos_head` | predicted target coordinates `mol_1_pred.x` | always active | none | never clamped |
| `atom_type_head` | predicted target atom types `mol_1_pred.a` | always active | none | never clamped |
| `do_sub_a_head` | atom substitution decisions | always active | none | never clamped |
| `do_del_head` | deletion decisions | active unless `n_atoms_strategy == "fixed"` | `sample()` | fixed-atom runs set `do_del_probs = 0` |
| `ins_rate_head` | insertion count/rate decisions | active unless `n_atoms_strategy == "fixed"` | `sample()` | fixed-atom runs set `num_ins_pred = 0` |
| `charge_head` | predicted target charges `mol_1_pred.c` | `representation.requires_charges` | `sample()` | `c_pred` filled with neutral charge index |
| `edge_type_head` | predicted target edge types `mol_1_pred.e` | `representation.requires_topology` | `sample()` | `e_pred` filled with `<NO_BOND>` index `0` |
| `do_sub_e_head` | edge substitution decisions | `representation.requires_topology` | `sample()` | `do_sub_e_probs = 0` |
| `ins_edge_head` | predicted edges from inserted atoms | `representation.requires_topology` | `sample()` argument | passed as `None`; integrator falls back to projected edge prior |
| `gmm_head["c_probs"]` | charges of newly inserted atoms | `representation.requires_charges` | `integration.py` via `force_charge_idx` | sampled charges overwritten with neutral charge index |
| `gmm_head["mu"]`, `gmm_head["sigma"]`, `gmm_head["pi"]`, `gmm_head["a_probs"]` | position and atom type of newly inserted atoms | always active | none | never clamped |

The split is intentional: `sample()` clamps predictions that become `mol_1_pred`
or integration decisions directly. Inserted charges are different because they are
sampled inside `RateIntegrator.sample_insertions()` from `gmm_head["c_probs"]`;
therefore `sample()` passes `force_charge_idx`, and `integration.py` overwrites
`new_atoms.c` immediately after GMM sampling in non-charge modes.
