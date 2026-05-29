"""OpenMP/BLAS thread config for tblite worker processes.

Deliberately imports nothing heavy (no tblite, no torch). When used as a
``ProcessPoolExecutor(initializer=...)`` it runs in each worker *before* the
first task is unpickled — and therefore before that worker imports tblite — so
the OpenMP runtime picks up a single-threaded config. This is the regime tblite
recommends for many small molecules: parallelize over molecules (one process
each), one thread per process, no oversubscription.

See https://tblite.readthedocs.io/en/latest/tutorial/parallel.html
"""

from __future__ import annotations

import os


def configure_single_thread() -> None:
    """Force one OpenMP/BLAS thread for this (worker) process; cap stack/levels.

    Thread counts are forced to ``1`` (correct for one-molecule-per-process).
    Stack size and nested-level cap are ``setdefault`` so a launcher can
    override them.
    """
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    # Guard against OpenMP stack overflow on larger systems; virtual only.
    os.environ.setdefault("OMP_STACKSIZE", "1G")
    # No nested parallelism inside a single-thread worker.
    os.environ.setdefault("OMP_MAX_ACTIVE_LEVELS", "1")
