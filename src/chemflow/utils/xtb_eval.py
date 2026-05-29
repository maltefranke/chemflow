"""GFN2-xTB property evaluation via tblite for tmQM-style comparisons.

Two entry points:

* ``compute_props`` — single MoleculeData (or arrays) → dict of GFN2-xTB properties.
* ``batch_compute_props`` — iterate over a list, tolerate failures, return a
  pandas DataFrame.

The notebook ``notebooks/tmqm_eval.ipynb`` consumes these.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from ase import Atoms
from ase.optimize import BFGS, FIRE, LBFGS
from rdkit import Chem
from tblite.ase import TBLite
from tblite.interface import Calculator

# Hartree -> eV. tblite returns orbital energies in Hartree.
HARTREE_TO_EV = 27.211386245988
# atomic units of dipole (e * Bohr) -> Debye.
EBOHR_TO_DEBYE = 2.541746229
# Gradient unit conversion: 1 Hartree/Bohr = 51.4220674... eV/Å.
HA_PER_BOHR_TO_EV_PER_ANG = HARTREE_TO_EV / 0.52917721067


def _load_atom_tokens(tokens_path: str | Path) -> list[str]:
    with open(tokens_path) as f:
        return [line.strip() for line in f if line.strip()]


def _tokens_to_z(atom_tokens: list[str]) -> np.ndarray:
    """Map vocab index -> atomic number via rdkit periodic table."""
    table = Chem.GetPeriodicTable()
    return np.array([table.GetAtomicNumber(t) for t in atom_tokens], dtype=int)


def mol_to_zxyz(
    mol, vocab_z: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """MoleculeData / PyG Data -> (atomic_numbers[N], positions_angstrom[N,3]).

    Supports both layouts:
      * generated mols: ``.a`` (vocab index) + ``.x`` (positions).
      * raw tmQM items: ``.z`` (atomic numbers) + ``.pos`` (positions).
    """
    z_attr = getattr(mol, "z", None)
    if z_attr is not None:
        z = z_attr.detach().cpu().numpy().astype(int)
    else:
        a = mol.a.detach().cpu().numpy()
        if a.ndim > 1:
            a = a.argmax(axis=-1)
        z = vocab_z[a]

    pos = getattr(mol, "x", None)
    if pos is None:
        pos = mol.pos
    x = pos.detach().cpu().numpy().astype(float)
    return z, x


def _homo_lumo_from_orbitals(
    energies: np.ndarray, occupations: np.ndarray
) -> tuple[float, float, float]:
    """Find HOMO/LUMO (eV) + gap from orbital energies (Ha) + occupations.

    Works for restricted (single spin channel) and unrestricted (sums over
    spin if shape is [2, n_orb]).
    """
    e = np.asarray(energies)
    occ = np.asarray(occupations)
    if e.ndim == 2:  # unrestricted
        e = e.reshape(-1)
        occ = occ.reshape(-1)
    order = np.argsort(e)
    e_sorted = e[order]
    occ_sorted = occ[order]
    occupied = occ_sorted > 0.5
    if not occupied.any() or occupied.all():
        return float("nan"), float("nan"), float("nan")
    homo_idx = np.where(occupied)[0].max()
    lumo_idx = homo_idx + 1
    homo = float(e_sorted[homo_idx] * HARTREE_TO_EV)
    lumo = float(e_sorted[lumo_idx] * HARTREE_TO_EV)
    return homo, lumo, lumo - homo


def _configure_xtb_env(
    *,
    omp_threads: int | str | None = None,
    omp_stacksize: str | None = None,
    omp_max_active_levels: int | str | None = 1,
    omp_schedule: str | None = None,
    blas_threads: int | str | None = None,
) -> None:
    """Set OpenMP/BLAS env knobs used by tblite.

    These must be set BEFORE `import tblite` for the OpenMP runtime to pick
    them up. Setting them inside a worker after tblite has loaded is a no-op
    for thread count. The right pattern is: set them at the top of the
    entrypoint (notebook first cell, or Slurm script) before any chemflow
    import that pulls in tblite. For ProcessPoolExecutor with spawn context,
    children inherit the parent env, so setting them once at the top works.
    """
    if omp_threads is not None:
        os.environ["OMP_NUM_THREADS"] = str(omp_threads)
    if omp_stacksize is not None:
        os.environ["OMP_STACKSIZE"] = str(omp_stacksize)
    if omp_max_active_levels is not None:
        os.environ["OMP_MAX_ACTIVE_LEVELS"] = str(omp_max_active_levels)
    if omp_schedule is not None:
        os.environ["OMP_SCHEDULE"] = str(omp_schedule)
    if blas_threads is not None:
        os.environ["MKL_NUM_THREADS"] = str(blas_threads)
        os.environ["OPENBLAS_NUM_THREADS"] = str(blas_threads)


def _singlepoint(
    numbers: np.ndarray,
    positions_ang: np.ndarray,
    *,
    method: str = "GFN2-xTB",
    accuracy: float = 1.0,
    max_iterations: int = 250,
) -> dict:
    """tblite GFN2 single-point. positions in Angstrom -> Bohr internally."""
    BOHR = 0.52917721067
    pos_bohr = positions_ang / BOHR
    calc = Calculator(method, numbers, pos_bohr)
    calc.set("verbosity", 0)
    calc.set("accuracy", accuracy)
    calc.set("max-iter", max_iterations)
    res = calc.singlepoint()
    energy_ha = float(res.get("energy"))
    dipole_au = np.asarray(res.get("dipole"))  # e * Bohr
    homo, lumo, gap = _homo_lumo_from_orbitals(
        res.get("orbital-energies"), res.get("orbital-occupations")
    )
    charges = np.asarray(res.get("charges"))
    grad_au = np.asarray(res.get("gradient"))  # (N, 3) Hartree/Bohr
    forces_ev_ang = -grad_au * HA_PER_BOHR_TO_EV_PER_ANG
    per_atom_norm = np.linalg.norm(forces_ev_ang, axis=1)
    return {
        "energy_eV": energy_ha * HARTREE_TO_EV,
        "homo_eV": homo,
        "lumo_eV": lumo,
        "gap_eV": gap,
        "dipole_Debye": float(np.linalg.norm(dipole_au) * EBOHR_TO_DEBYE),
        "charges": charges,
        "force_rms_eV_per_Ang": float(np.sqrt(np.mean(forces_ev_ang ** 2))),
        "force_max_eV_per_Ang": float(per_atom_norm.max()) if per_atom_norm.size else float("nan"),
    }


def _metal_charge(numbers: np.ndarray, charges: np.ndarray) -> float:
    """Mulliken charge on the transition-metal center.

    tmQM defines exactly one metal per complex. We pick the atom with the
    highest atomic number among the transition metals; if none present,
    return NaN.
    """
    TM_RANGES = [(21, 30), (39, 48), (57, 80)]  # Sc-Zn, Y-Cd, La-Hg
    mask = np.zeros_like(numbers, dtype=bool)
    for lo, hi in TM_RANGES:
        mask |= (numbers >= lo) & (numbers <= hi)
    if not mask.any():
        return float("nan")
    idx = np.where(mask)[0][np.argmax(numbers[mask])]
    return float(charges[idx])


_OPTIMIZERS = {"bfgs": BFGS, "fire": FIRE, "lbfgs": LBFGS}


def _optimize_ase(
    numbers: np.ndarray,
    positions_ang: np.ndarray,
    fmax: float,
    max_steps: int,
    *,
    method: str = "GFN2-xTB",
    accuracy: float = 1.0,
    max_iterations: int = 250,
    optimizer: str = "bfgs",
) -> tuple[np.ndarray, bool]:
    """GFN2 opt via ASE+tblite. Returns (opt_positions, converged).

    `optimizer` selects the ASE optimizer: 'bfgs' (default), 'fire', 'lbfgs'.
    FIRE is generally more robust on far-from-equilibrium starts (e.g. generated
    pointclouds); BFGS converges in fewer steps once close to a minimum.
    """
    try:
        opt_cls = _OPTIMIZERS[optimizer.lower()]
    except KeyError as e:
        raise ValueError(
            f"optimizer must be one of {sorted(_OPTIMIZERS)}, got {optimizer!r}"
        ) from e
    symbols = [Chem.GetPeriodicTable().GetElementSymbol(int(z)) for z in numbers]
    atoms = Atoms(symbols=symbols, positions=positions_ang)
    atoms.calc = TBLite(
        method=method,
        verbosity=0,
        accuracy=accuracy,
        max_iterations=max_iterations,
    )
    opt = opt_cls(atoms, logfile=None)
    converged = opt.run(fmax=fmax, steps=max_steps)
    return atoms.get_positions(), bool(converged)


def compute_props(
    numbers: np.ndarray,
    positions_ang: np.ndarray,
    optimize: bool = False,
    opt_fmax: float = 0.1,
    opt_max_steps: int = 200,
    method: str = "GFN2-xTB",
    accuracy: float = 1.0,
    max_iterations: int = 250,
    optimizer: str = "bfgs",
) -> dict:
    """Full GFN2 property bundle. ``optimize=True`` runs the configured
    optimizer (BFGS / FIRE / LBFGS) first.

    Polarizability is omitted — tblite-python doesn't expose static
    polarizability via a single call. Compute it separately if needed.
    """
    opt_converged = None
    if optimize:
        positions_ang, opt_converged = _optimize_ase(
            numbers,
            positions_ang,
            opt_fmax,
            opt_max_steps,
            method=method,
            accuracy=accuracy,
            max_iterations=max_iterations,
            optimizer=optimizer,
        )
    props = _singlepoint(
        numbers,
        positions_ang,
        method=method,
        accuracy=accuracy,
        max_iterations=max_iterations,
    )
    charges = props.pop("charges")
    props["metal_charge"] = _metal_charge(numbers, charges)
    props["opt_converged"] = opt_converged
    props["n_atoms"] = int(len(numbers))
    return props


@dataclass
class XtbConfig:
    method: str = "GFN2-xTB"
    optimize: bool = False
    opt_fmax: float = 0.1
    opt_max_steps: int = 200
    optimizer: str = "bfgs"   # 'bfgs' | 'fire' | 'lbfgs'
    accuracy: float = 1.0
    max_iterations: int = 250
    # Outer molecule-level parallelism. Keep 1 for serial behavior.
    n_jobs: int = 1
    # "process" (recommended for many small molecules — each worker has its
    # own OpenMP pool, no contention) or "thread" (single OpenMP pool shared).
    # Process workers inherit env from the parent, so configure OMP_NUM_THREADS=1
    # in the parent BEFORE importing tblite for the cleanest result.
    executor: str = "process"
    # Inner tblite/BLAS threading per worker. For n_jobs>1 with executor="process",
    # 1 is best (each worker gets one core, no oversubscription).
    omp_threads: int | str | None = None
    blas_threads: int | str | None = None
    omp_stacksize: str | None = None
    omp_max_active_levels: int | str | None = 1
    omp_schedule: str | None = None


def _compute_props_worker(args: tuple[int, np.ndarray, np.ndarray, XtbConfig]) -> dict:
    i, z, x, cfg = args
    row = compute_props(
        z,
        x,
        optimize=cfg.optimize,
        opt_fmax=cfg.opt_fmax,
        opt_max_steps=cfg.opt_max_steps,
        method=cfg.method,
        accuracy=cfg.accuracy,
        max_iterations=cfg.max_iterations,
        optimizer=cfg.optimizer,
    )
    row["index"] = i
    row["status"] = "ok"
    return row


def _error_row(i: int, mol=None, exc: Exception | None = None) -> dict:
    status = "err"
    if exc is not None:
        status = f"err: {type(exc).__name__}: {exc}"
    row = {"index": i, "status": status}
    if mol is not None:
        row["n_atoms"] = int(len(mol.a)) if hasattr(mol, "a") else None
    return row


def batch_compute_props(
    mols: list,
    vocab_z: np.ndarray,
    config: XtbConfig | None = None,
    progress: bool = True,
    label: str = "",
) -> pd.DataFrame:
    """Run ``compute_props`` over a list of MoleculeData. Failures -> NaN row."""
    cfg = config or XtbConfig()
    rows: list[dict | None] = [None] * len(mols)

    _configure_xtb_env(
        omp_threads=cfg.omp_threads,
        omp_stacksize=cfg.omp_stacksize,
        omp_max_active_levels=cfg.omp_max_active_levels,
        omp_schedule=cfg.omp_schedule,
        blas_threads=cfg.blas_threads,
    )

    tasks = []
    for i, mol in enumerate(mols):
        try:
            z, x = mol_to_zxyz(mol, vocab_z)
        except Exception as e:
            rows[i] = _error_row(i, mol, e)
            continue
        tasks.append((i, z, x, cfg))

    if cfg.n_jobs <= 1:
        task_iter = tasks
        if progress:
            try:
                from tqdm.auto import tqdm

                task_iter = tqdm(tasks, total=len(tasks), desc=label or "xtb")
            except ImportError:
                pass
        for task in task_iter:
            i = task[0]
            try:
                rows[i] = _compute_props_worker(task)
            except Exception as e:
                rows[i] = _error_row(i, exc=e)
    else:
        if cfg.executor == "process":
            ctx = mp.get_context("spawn")
            ex_cls = ProcessPoolExecutor
            ex_kwargs = {"max_workers": cfg.n_jobs, "mp_context": ctx}
        elif cfg.executor == "thread":
            ex_cls = ThreadPoolExecutor
            ex_kwargs = {"max_workers": cfg.n_jobs}
        else:
            raise ValueError(
                f"executor must be 'process' or 'thread', got {cfg.executor!r}"
            )
        with ex_cls(**ex_kwargs) as ex:
            futures = {ex.submit(_compute_props_worker, task): task[0] for task in tasks}
            future_iter = as_completed(futures)
            if progress:
                try:
                    from tqdm.auto import tqdm

                    future_iter = tqdm(
                        future_iter,
                        total=len(futures),
                        desc=f"{label or 'xtb'} {cfg.executor}x{cfg.n_jobs}",
                    )
                except ImportError:
                    pass
            for fut in future_iter:
                i = futures[fut]
                try:
                    rows[i] = fut.result()
                except Exception as e:
                    rows[i] = _error_row(i, exc=e)

    return pd.DataFrame([r if r is not None else _error_row(i) for i, r in enumerate(rows)])


def load_atom_tokens_z(tokens_path: str | Path) -> np.ndarray:
    """Convenience: vocab atom_tokens.txt -> per-index atomic number array."""
    return _tokens_to_z(_load_atom_tokens(tokens_path))
