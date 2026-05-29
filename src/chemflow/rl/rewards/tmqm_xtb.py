"""xTB-based tmQM rewards: validity and validity-gated HOMO-LUMO gap.

Designed for transition-metal-complex point clouds, where RDKit whole-molecule
validity is undefined (the metal breaks bond perception). A single GFN2-xTB
relaxation per graph feeds every reward here, so the expensive geometry
optimization is computed once and reused:

* ``tmqm_validity`` — structure/geometry validity in [0, 1].
* ``tmqm_gap`` — ``validity * relaxed_gap_eV``: HOMO-LUMO gap maximization gated
  by validity (PL-MOGA-style fitness on the equilibrium geometry, with the
  validity factor blocking fragment/no-metal/non-relaxing reward hacks).
* ``tmqm_validity_gate`` — wrapper to gate any other base reward by validity.

Validity is scored as a weighted blend behind hard structural gates:

    gate  = I[exactly one transition metal]
          * I[coordination number in range]
          * I[xTB single-point/relax completed]
    geom  = 1.0 if the 30-step GFN2 relaxation converged,
            else clamp(exp(-(force_rms - fmax) / scale), 0, 1)   # partial credit
    reward = gate * (w_lig * ligand_valid_fraction + w_geom * geom)

where ``ligand_valid_fraction`` comes from molSimplify ligand extraction +
per-ligand RDKit ``DetermineBonds`` sanitization (the metal-ligand bonds are cut,
each organic fragment is validated independently).

Rationale (see the design discussion / eval_outputs/tmqm_force_calib_relax):
- The hard gates kill the dominant reward hack — dissolving the complex into a
  gas of small high-property fragments shows up as no-metal / out-of-range
  coordination and is zeroed.
- ``ligand_valid_fraction`` is soft because train itself is only ~93%
  all-ligands-valid under this extractor, so it can't be a hard wall.
- ``geom`` is continuous (not a binary convergence flag) because 30-step
  convergence is only ~41% even on good structures (it's ~99% at 100 steps);
  the binary flag throws away gradient, partial credit on residual force gives
  GRPO a smooth signal.

xTB relaxation is the expensive part, so it runs only on graphs that pass the
cheap structural gates, parallelized across the batch with a process pool
(``CHEMFLOW_XTB_JOBS`` env, default ``os.cpu_count()-1``).
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from multiprocessing import get_context
from typing import Callable

import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import rdDetermineBonds

from chemflow.dataset.representation import Representation
from chemflow.utils.xtb_eval import compute_props
from chemflow.utils.xtb_threads import configure_single_thread

from .common import _as_tensor
from .spec import RewardSpec, WrapperSpec

RDLogger.DisableLog("rdApp.*")  # type: ignore[attr-defined]
_PT = Chem.GetPeriodicTable()

# Set CHEMFLOW_TMQM_DEBUG=1 to print per-reward-call timing (struct gate vs xTB
# relax wall-time, throughput, and the effective OMP_NUM_THREADS).
_DEBUG = bool(os.environ.get("CHEMFLOW_TMQM_DEBUG"))

_POINTCLOUD_REPS = frozenset({
    Representation.POINTCLOUD,
    Representation.CHARGED_POINTCLOUD,
    Representation.GEOMETRIC_GRAPH,
})

# --- defaults (mirror notebook XTB_CFG_OPT_FAST for the relaxation) ---
_DEFAULT_MIN_CN = 2
_DEFAULT_MAX_CN = 9
_DEFAULT_W_LIG = 0.5
_DEFAULT_W_GEOM = 0.5
_DEFAULT_RELAX_STEPS = 30
_DEFAULT_RELAX_FMAX = 0.2
_DEFAULT_GEOM_SCALE = 0.5  # eV/Å; residual-force decay for non-converged geoms
_XTB_ACCURACY = 2.0
_XTB_MAX_ITER = 150
# molSimplify ligand charge candidates (tmQM ligands are commonly charged).
_CHARGE_CANDIDATES = (0, -1, 1, -2)
_COV_FACTOR = 1.3

_TM_RANGES = ((21, 30), (39, 48), (57, 80))  # Sc-Zn, Y-Cd, La-Hg
_Z_LOOKUP_CACHE: dict[tuple[str, ...], np.ndarray] = {}


def _is_transition_metal(z: int) -> bool:
    return any(lo <= z <= hi for lo, hi in _TM_RANGES)


def _coord_scale(module) -> float:
    coord_std = getattr(module.distributions, "coordinate_std", None)
    if coord_std is None:
        return 1.0
    return float(coord_std.item() if hasattr(coord_std, "item") else coord_std)


def _z_lookup(atom_tokens: list[str]) -> np.ndarray:
    key = tuple(atom_tokens)
    out = _Z_LOOKUP_CACHE.get(key)
    if out is None:
        out = np.array(
            [int(_PT.GetAtomicNumber(t)) for t in atom_tokens], dtype=np.int64
        )
        _Z_LOOKUP_CACHE[key] = out
    return out


# ----------------------------------------------------------------------------
# Cheap structural scoring (molSimplify + RDKit, no xTB).
# ----------------------------------------------------------------------------
@dataclass
class _StructScore:
    single_metal: bool
    coord_ok: bool
    coordination_number: int
    ligand_valid_fraction: float
    n_ligands: int


def _ligand_valid(z_frag: np.ndarray, pos_frag: np.ndarray) -> bool:
    """True if a ligand fragment infers bonds + sanitizes for any candidate charge."""
    for charge in _CHARGE_CANDIDATES:
        rw = Chem.RWMol()
        for zi in z_frag.tolist():
            rw.AddAtom(Chem.Atom(int(zi)))
        mol = rw.GetMol()
        conf = Chem.Conformer(int(len(z_frag)))
        for i, xyz in enumerate(pos_frag.tolist()):
            conf.SetAtomPosition(i, (float(xyz[0]), float(xyz[1]), float(xyz[2])))
        mol.AddConformer(conf, assignId=True)
        try:
            rdDetermineBonds.DetermineBonds(
                mol,
                charge=int(charge),
                covFactor=_COV_FACTOR,
                allowChargedFragments=True,
                embedChiral=False,
                useVdw=False,
                useHueckel=False,
            )
            if Chem.SanitizeMol(mol, catchErrors=True) == Chem.SanitizeFlags.SANITIZE_NONE:
                return True
        except Exception:
            continue
    return False


def _struct_score(
    z: np.ndarray, pos: np.ndarray, *, min_cn: int, max_cn: int
) -> _StructScore:
    """molSimplify single-metal + coordination + ligand validity. Failures are
    reported as a non-passing _StructScore rather than raising."""
    # Lazy import so the rest of the rewards package doesn't hard-depend on it.
    from molSimplify.Classes.atom3D import atom3D
    from molSimplify.Classes.mol3D import mol3D

    fail = _StructScore(False, False, 0, 0.0, 0)
    try:
        mol = mol3D()
        for zi, xyz in zip(z.tolist(), pos.tolist()):
            mol.addAtom(atom3D(_PT.GetElementSymbol(int(zi)), xyz=[float(v) for v in xyz]))
        mol.createMolecularGraph(oct=True)
        metals = [int(i) for i in mol.findMetal(transition_metals_only=True, include_X=False)]
    except Exception:
        return fail
    if len(metals) != 1:
        return fail
    metal_idx = metals[0]
    try:
        binding = [int(i) for i in mol.getBondedAtoms(metal_idx)]
    except Exception:
        return fail
    cn = len(binding)
    coord_ok = min_cn <= cn <= max_cn

    # Extract ligand fragments (cut metal-ligand bonds) and validate each.
    frags: dict[frozenset, None] = {}
    for b in binding:
        try:
            frag = frozenset(
                int(i) for i in mol.findsubMol(b, metal_idx, smart=False) if int(i) != metal_idx
            )
        except Exception:
            continue
        if frag:
            frags[frag] = None
    n_lig = len(frags)
    if n_lig == 0:
        return _StructScore(True, coord_ok, cn, 0.0, 0)
    n_valid = 0
    for frag in frags:
        idx = sorted(frag)
        if _ligand_valid(z[idx], pos[idx]):
            n_valid += 1
    return _StructScore(True, coord_ok, cn, n_valid / n_lig, n_lig)


# ----------------------------------------------------------------------------
# xTB relaxation (parallel across the gated subset).
# ----------------------------------------------------------------------------
def _relax_worker(task: tuple[np.ndarray, np.ndarray, int, float]) -> tuple[bool, bool, float]:
    z, pos, steps, fmax = task
    try:
        res = compute_props(
            z,
            pos,
            optimize=True,
            opt_fmax=fmax,
            opt_max_steps=steps,
            optimizer="bfgs",
            accuracy=_XTB_ACCURACY,
            max_iterations=_XTB_MAX_ITER,
        )
        return (
            True,
            bool(res["opt_converged"]),
            float(res["force_rms_eV_per_Ang"]),
            float(res["gap_eV"]),
        )
    except Exception:
        return False, False, float("nan"), float("nan")


def _xtb_jobs() -> int:
    env = os.environ.get("CHEMFLOW_XTB_JOBS")
    if env:
        return max(1, int(env))
    # Respect the Slurm/cgroup CPU allocation: os.cpu_count() reports the whole
    # node, which would massively oversubscribe a `--cpus-per-task`-limited job.
    try:
        n = len(os.sched_getaffinity(0))
    except AttributeError:  # not on Linux
        n = os.cpu_count() or 2
    return max(1, n - 1)


def _relax_batch(
    tasks: list[tuple[np.ndarray, np.ndarray, int, float]]
) -> list[tuple[bool, bool, float, float]]:
    if not tasks:
        return []
    n_jobs = min(_xtb_jobs(), len(tasks))
    if n_jobs <= 1:
        return [_relax_worker(t) for t in tasks]
    # CRITICAL: set the single-thread env in the PARENT before spawning. Spawn
    # children re-import run_grpo -> tblite at bootstrap, BEFORE the pool
    # `initializer` runs, so the initializer alone is too late and tblite would
    # grab every core per worker (~40x thrash). Children inherit the parent env
    # at spawn time, so setting it here (it doesn't disturb the parent's
    # already-initialized OpenMP) is what actually fixes the threading. The
    # `initializer` stays as belt-and-suspenders.
    configure_single_thread()
    ctx = get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=n_jobs, mp_context=ctx, initializer=configure_single_thread
    ) as ex:
        return list(ex.map(_relax_worker, tasks))


# ----------------------------------------------------------------------------
# Core scorer: ONE xTB relaxation per graph yields both validity and gap, shared
# by every reward / wrapper below so the expensive relaxation is never repeated.
# ----------------------------------------------------------------------------
@dataclass
class _BatchScore:
    validity: np.ndarray  # (N,) weighted-blend validity in [0, 1]
    gap_eV: np.ndarray    # (N,) relaxed HOMO-LUMO gap; 0.0 where invalid/undefined
    diag: dict[str, float]


def _score_batch(
    module,
    trajectory,
    *,
    min_cn: int,
    max_cn: int,
    w_lig: float,
    w_geom: float,
    relax_steps: int,
    relax_fmax: float,
    geom_scale: float,
) -> _BatchScore:
    z_lookup = _z_lookup(module.vocab.atom_tokens)
    scale = _coord_scale(module)
    graphs = trajectory.mol_final.to_data_list()
    n = len(graphs)

    t_struct = time.perf_counter()
    structs: list[_StructScore] = []
    zpos: list[tuple[np.ndarray, np.ndarray]] = []
    for g in graphs:
        a = g.a
        if a.dim() > 1:
            a = a.argmax(dim=-1)
        z = z_lookup[a.detach().cpu().numpy()]
        pos = (g.x.detach().cpu().numpy() * scale).astype(float)
        zpos.append((z, pos))
        if not np.isfinite(pos).all():
            structs.append(_StructScore(False, False, 0, 0.0, 0))
        else:
            structs.append(_struct_score(z, pos, min_cn=min_cn, max_cn=max_cn))
    dt_struct = time.perf_counter() - t_struct

    # Only relax graphs that clear the cheap hard gates (single metal + coord).
    gate_pass = [s.single_metal and s.coord_ok for s in structs]
    relax_idx = [i for i, ok in enumerate(gate_pass) if ok]
    tasks = [(zpos[i][0], zpos[i][1], relax_steps, relax_fmax) for i in relax_idx]
    t_relax = time.perf_counter()
    relax_res = _relax_batch(tasks)
    dt_relax = time.perf_counter() - t_relax

    if _DEBUG:
        n_relax = len(tasks)
        per = dt_relax / max(n_relax, 1)
        print(
            f"[tmqm_xtb] graphs={n} struct_gate_pass={n_relax} "
            f"| struct {dt_struct:.1f}s | relax {dt_relax:.1f}s "
            f"({per:.2f}s/mol, {_xtb_jobs()} workers) "
            f"| OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS','<unset>')}",
            flush=True,
        )

    xtb_ok = np.zeros(n, dtype=bool)
    converged = np.zeros(n, dtype=bool)
    force_rms = np.full(n, np.nan, dtype=float)
    gap = np.full(n, np.nan, dtype=float)
    for i, (ok, conv, frms, g_eV) in zip(relax_idx, relax_res):
        xtb_ok[i] = ok
        converged[i] = conv
        force_rms[i] = frms
        gap[i] = g_eV

    validity = np.zeros(n, dtype=np.float32)
    geom_scores = np.zeros(n, dtype=np.float32)
    for i, s in enumerate(structs):
        gate = s.single_metal and s.coord_ok and xtb_ok[i]
        if not gate:
            continue
        if converged[i]:
            geom = 1.0
        else:
            geom = float(np.clip(np.exp(-(force_rms[i] - relax_fmax) / geom_scale), 0.0, 1.0))
        geom_scores[i] = geom
        validity[i] = w_lig * s.ligand_valid_fraction + w_geom * geom

    # Gap is meaningful only on completed relaxations with a defined gap; elsewhere
    # 0 (those graphs already have validity 0, so the gap reward is 0 there too).
    gap_clean = np.where(np.isfinite(gap), gap, 0.0).astype(np.float32)
    gap_clean = np.maximum(gap_clean, 0.0)

    finite_gap = gap[np.isfinite(gap)]
    diag = {
        "tmqm_p_single_metal": float(np.mean([s.single_metal for s in structs])),
        "tmqm_p_coord_ok": float(np.mean([s.coord_ok for s in structs])),
        "tmqm_p_struct_gate": float(np.mean(gate_pass)),
        "tmqm_p_xtb_ok": float(xtb_ok.mean()),
        "tmqm_p_converged": float(converged.mean()),
        "tmqm_coordination_mean": float(np.mean([s.coordination_number for s in structs])),
        "tmqm_ligand_valid_fraction_mean": float(
            np.mean([s.ligand_valid_fraction for s in structs])
        ),
        "tmqm_geom_score_mean": float(geom_scores.mean()),
        "tmqm_validity_mean": float(validity.mean()),
        "tmqm_gap_mean": float(finite_gap.mean()) if finite_gap.size else 0.0,
        "tmqm_gap_median": float(np.median(finite_gap)) if finite_gap.size else 0.0,
    }
    return _BatchScore(validity=validity, gap_eV=gap_clean, diag=diag)


def tmqm_validity_reward(module, trajectory) -> tuple[torch.Tensor, dict[str, float]]:
    """Weighted-blend tmQM validity in [0, 1] per graph (see module docstring)."""
    score = _score_batch(
        module,
        trajectory,
        min_cn=_DEFAULT_MIN_CN,
        max_cn=_DEFAULT_MAX_CN,
        w_lig=_DEFAULT_W_LIG,
        w_geom=_DEFAULT_W_GEOM,
        relax_steps=_DEFAULT_RELAX_STEPS,
        relax_fmax=_DEFAULT_RELAX_FMAX,
        geom_scale=_DEFAULT_GEOM_SCALE,
    )
    return _as_tensor(score.validity.tolist(), trajectory.mol_final.x.device), score.diag


def tmqm_gap_reward(module, trajectory) -> tuple[torch.Tensor, dict[str, float]]:
    """Validity-gated HOMO-LUMO gap maximization: ``reward = validity * gap_eV``.

    One xTB relaxation per graph produces both the validity blend and the
    relaxed gap, so the gap is measured on the same equilibrium geometry the
    validity check passed (the PL-MOGA fitness convention). The multiplicative
    validity factor blocks the dominant reward hacks: structures that fragment,
    lose the metal, or don't relax cleanly get near-zero validity and thus
    near-zero reward regardless of their (often spurious) gap.
    """
    score = _score_batch(
        module,
        trajectory,
        min_cn=_DEFAULT_MIN_CN,
        max_cn=_DEFAULT_MAX_CN,
        w_lig=_DEFAULT_W_LIG,
        w_geom=_DEFAULT_W_GEOM,
        relax_steps=_DEFAULT_RELAX_STEPS,
        relax_fmax=_DEFAULT_RELAX_FMAX,
        geom_scale=_DEFAULT_GEOM_SCALE,
    )
    rewards = score.validity * score.gap_eV
    diag = dict(score.diag)
    diag["tmqm_gap_reward_mean"] = float(rewards.mean())
    return _as_tensor(rewards.tolist(), trajectory.mol_final.x.device), diag


def tmqm_validity_gate_wrapper(
    base_reward_fn: Callable,
    *,
    min_cn: int = _DEFAULT_MIN_CN,
    max_cn: int = _DEFAULT_MAX_CN,
    w_lig: float = _DEFAULT_W_LIG,
    w_geom: float = _DEFAULT_W_GEOM,
    relax_steps: int = _DEFAULT_RELAX_STEPS,
    relax_fmax: float = _DEFAULT_RELAX_FMAX,
    geom_scale: float = _DEFAULT_GEOM_SCALE,
) -> Callable:
    """Multiply a base reward (e.g. a HOMO-LUMO gap reward) by tmQM validity in
    [0, 1]. Use this to gate a property objective behind structure/geometry
    validity so the policy can't reward-hack through broken complexes."""

    def wrapped(module, trajectory) -> tuple[torch.Tensor, dict[str, float]]:
        r, diag = base_reward_fn(module, trajectory)
        diag_out = dict(diag)
        score = _score_batch(
            module,
            trajectory,
            min_cn=min_cn,
            max_cn=max_cn,
            w_lig=w_lig,
            w_geom=w_geom,
            relax_steps=relax_steps,
            relax_fmax=relax_fmax,
            geom_scale=geom_scale,
        )
        mask = _as_tensor(score.validity.tolist(), r.device)
        r_gated = r * mask
        diag_out.update(score.diag)
        diag_out["reward_mean_pre_tmqm_validity"] = float(r.mean())
        diag_out["reward_mean_post_tmqm_validity"] = float(r_gated.mean())
        return r_gated, diag_out

    return wrapped


TMQM_VALIDITY_SPEC = RewardSpec(
    fn=tmqm_validity_reward,
    supported_representations=_POINTCLOUD_REPS,
)
TMQM_GAP_SPEC = RewardSpec(
    fn=tmqm_gap_reward,
    supported_representations=_POINTCLOUD_REPS,
)
TMQM_VALIDITY_GATE_SPEC = WrapperSpec(
    make=tmqm_validity_gate_wrapper,
    supported_representations=_POINTCLOUD_REPS,
)
