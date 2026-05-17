# Multi-GPU tmQM hang/NaN: data-dependent insertion branch under DDP

## TL;DR

Multi-GPU SemLA training on tmQM hung (4090/ETH: 30 min NCCL watchdog →
`SIGABRT`) or crashed with a NaN (`Categorical` Simplex `ValueError`,
A100/Mila). Root cause: the GMM/insertion loss is computed inside a
**data-dependent Python branch** that some DDP ranks skip when their
micro-batch has no insertion ("spawn") nodes. Under
`ddp_find_unused_parameters_false`, a rank that skips the branch produces no
gradient for the GMM-head parameters, the gradient all-reduce desyncs across
ranks, and the (separate, hand-rolled) Muon `all_gather` then amplifies the
desync into a hard hang. The fix is a one-line, numerically-zero keep-alive
that keeps the GMM head in the autograd graph on every rank every step.

Status: **fixed and confirmed in production** (see [Current behaviour](#current-behaviour)).

---

## Symptoms

- **ETH (4×RTX 4090):** training reached `Epoch 0: 57%|841/1469` then hung;
  after ~30 min the NCCL watchdog aborted all ranks. Every traceback pointed at
  `dist.all_gather` in [src/external_code/muon.py:248](../src/external_code/muon.py#L248)
  inside the Muon optimizer step.
- **Mila (4×A100):** ran longer, then crashed with
  `ValueError: Expected parameter probs ... Simplex() ... found invalid values: tensor([[[nan, ...`
  from `Categorical(probs=pi)` in
  [src/chemflow/model/gmm.py](../src/chemflow/model/gmm.py).
- **QM9, same 4-GPU + Muon setup:** trained cleanly past the equivalent point
  (loss 33 → 2.66 over 10 epochs, no desync, no NaN). The problem was
  tmQM-specific, not a universal Muon/cluster/launcher issue.

---

## Root cause

### The data-dependent branch

In `shared_step`, insertion handling is gated on whether *this rank's*
micro-batch contains any spawn nodes:

- [src/chemflow/model/lightning_module.py:454](../src/chemflow/model/lightning_module.py#L454)
  — `spawn_node_idx = ins_targets.spawn_node_idx`
- [src/chemflow/model/lightning_module.py:472](../src/chemflow/model/lightning_module.py#L472)
  — `if spawn_node_idx.numel() > 0:` … the whole GMM-head loss
  ([typed_gmm_loss call, line 490](../src/chemflow/model/lightning_module.py#L490))
  lives inside this block.

`spawn_node_idx` is derived from data, so the branch is taken on some ranks and
not others within the same global step.

### Why that breaks DDP

The configured strategy is
[`ddp_find_unused_parameters_false`](../configs/trainer/trainer.yaml#L7).
DDP builds fixed gradient buckets and requires **every parameter to receive a
gradient on every rank, every step**. `find_unused_parameters_false` neither
tolerates nor detects parameters that get no gradient.

- Rank with spawns → enters the branch → `ins_gmm_head` parameters are in the
  autograd graph → they get gradients.
- Rank with **zero** spawns → skips the branch → `ins_gmm_head` gets **no
  gradient** → the gradient all-reduce mismatches across ranks → desync.

The Muon optimizer adds a *second* hand-rolled collective
([src/external_code/muon.py:248](../src/external_code/muon.py#L248)) with no
tolerance for divergence, which turns the soft DDP inconsistency into a
30-minute NCCL watchdog hang. Muon is the **amplifier**, not the origin.

### The "smoking gun": the idiom already existed at the wrong scope

The codebase already uses the standard DDP keep-alive idiom — a
numerically-zero reference to all head parameters — but only in the
`requires_topology == False` branch:

- [src/chemflow/model/lightning_module.py:613](../src/chemflow/model/lightning_module.py#L613)
  — `gmm_head_loss = sum(p.sum() for p in self.model.ins_gmm_head.parameters())`
- [src/chemflow/model/lightning_module.py:617](../src/chemflow/model/lightning_module.py#L617)
  — `ins_loss_gmm = 0.0 * gmm_head_loss`

Inside the `requires_topology == True` path, the nested empty-spawn case had no
equivalent fallback: `ins_loss_gmm` was initialised to a bare
`torch.tensor(0.0, device=self.device)` — a constant **disconnected from the
head parameters** — so a topology-on rank with zero spawns left `ins_gmm_head`
ungradiented.

### Note on the empty-mean NaN

`typed_gmm_loss` with `reduction="mean"` would compute `torch.mean` of an
empty tensor (`nan`) when there are no targets — see
[src/chemflow/model/losses.py:127-128](../src/chemflow/model/losses.py#L127).
This call site uses `reduction="none"` and is guarded by the `numel() > 0`
branch, so that specific NaN path is **not** the trigger here. The observed
A100 NaN is a downstream consequence of the DDP desync corrupting parameters,
not an empty-mean.

### Secondary, latent (not the trigger)

The distributed Muon implementation has independent defects, currently dormant
because the desync trigger is removed:

- [src/external_code/muon.py:231-232](../src/external_code/muon.py#L231)
  (and the duplicate at [lines 89-90](../src/external_code/muon.py#L89)):
  padding count `world_size - len(params) % world_size` is `world_size` (not
  `0`) when evenly divisible; correct form is `(ws - len % ws) % ws`.
- [src/external_code/muon.py:248](../src/external_code/muon.py#L248):
  `dist.all_gather` assumes every rank contributes the same-shaped tensor, but
  each rank owns a differently-shaped parameter; the padding tensors are
  uninitialised `torch.empty_like`. Written for transformer stacks of
  identically-shaped matrices, fragile on a heterogeneous EGNN/SemLA backbone.

---

## The fix

Initialise `ins_loss_gmm` with the **same zero-weighted head reference the
`requires_topology == False` branch already uses**, instead of a disconnected
constant, so the GMM head is always in the autograd graph regardless of data:

[src/chemflow/model/lightning_module.py:442](../src/chemflow/model/lightning_module.py#L442)

```python
# BEFORE — constant, not connected to any head parameter
ins_loss_gmm = torch.tensor(0.0, device=self.device)

# AFTER — 0-weighted reference to every ins_gmm_head parameter
ins_loss_gmm = 0.0 * sum(
    p.sum() for p in self.model.ins_gmm_head.parameters()
)
```

Properties:

- **Numerically a no-op** (`0.0 *`): does not change the loss value or training
  dynamics.
- **Structurally unconditional**: every `ins_gmm_head` parameter receives a
  (zero) gradient on every rank every step, satisfying the
  `find_unused_parameters_false` contract even when a rank has no spawns.
- **Overwritten on normal steps**: when `spawn_node_idx.numel() > 0`, the real
  GMM loss replaces this value
  ([line 490+](../src/chemflow/model/lightning_module.py#L490)); the real loss
  already connects to the head, so behaviour on insertion-bearing steps is
  unchanged.

### How it was confirmed

During the investigation a temporary per-rank spawn-count print, a
non-finite-activation forward hook, and a short DDP collective timeout were
added (env-gated) to observe the divergence and fail fast. They served their
purpose and have since been removed; the evidence they produced is recorded in
[Current behaviour](#current-behaviour). The only code change retained is the
one-line keep-alive fix above.

---

## Current behaviour

Verified on ETH (4×RTX 4090, `data=tmqm model=semla cfg=uncond
representation=pointcloud`), run `SLURM_OUTPUT-66833393.log`:

- The spawn log shows the predicted divergence at the exact historical failure
  step: `[spawn] rank=1 batch_idx=841 n_spawn=0` while other ranks had
  insertions (step 841 = the original `Epoch 0: 57%|841/1469` hang point).
  A second divergence occurred at `batch_idx=107`.
- With the fix, training **passed through both divergence steps** and continued
  to `batch_idx=1468` and into epoch 1.
- Zero NCCL watchdog/`ALLGATHER` desync events, zero `[nonfinite]` hook
  triggers, no NaN, no abort.

Conclusion: the mechanism is directly observed (a rank with `n_spawn=0` at the
exact failure step) and the keep-alive fix resolves it.

### Recommended follow-ups

- The production [submission_scripts/euler/tmqm_semla_uncond.sh](../submission_scripts/euler/tmqm_semla_uncond.sh)
  carries no diagnostic env vars; keep it that way for long runs (a short DDP
  collective timeout in particular would abort a healthy 14 h run on any
  transient slow step).
- Audit other data-dependent branches in `shared_step` (e.g. the insertion-edge
  head path around
  [lightning_module.py:498+](../src/chemflow/model/lightning_module.py#L498))
  for the same class of bug; some already have dummy keep-alives — confirm they
  are at the correct scope.
- Optionally fix the latent Muon defects (padding modulo; shape-agnostic
  gather) or stop sharding Muon across ranks, before scaling up or changing
  parameter shapes.
