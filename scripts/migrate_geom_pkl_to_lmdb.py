"""Migrate already-processed GEOM pickle files to LMDB in-place.

Reads each ``<split>_data.pkl`` (a pickled ``list[bytes]`` where each entry is
the output of ``mol_to_bytes``) and streams those bytes into
``<split>_data.lmdb/`` alongside it, without re-running RDKit processing.

Memory note: loading the pickled list requires roughly as much RAM as the
source file on disk (~28GB for GEOM train). Run on a node with enough memory,
or re-run ``preprocess_geom.py`` from the raw ``.pickle`` files instead.

Usage:
    python scripts/migrate_geom_pkl_to_lmdb.py \\
        --processed_dir /iopsstor/scratch/cscs/frankem/data/data/geom/processed

Optional flags:
    --splits train val test         Subset of splits to migrate.
    --delete_pkl                    Remove source .pkl after successful write.
"""

import argparse
import os
import pickle
from pathlib import Path

from chemflow.dataset.geom import LMDB_MAP_SIZE, write_bytes_to_lmdb


def migrate_split(
    pkl_path: Path,
    lmdb_path: Path,
    map_size: int,
    delete_pkl: bool,
) -> None:
    if lmdb_path.exists():
        print(f"[skip] {lmdb_path} already exists")
        return
    if not pkl_path.exists():
        print(f"[skip] {pkl_path} not found")
        return

    print(f"Loading {pkl_path} ({pkl_path.stat().st_size / 1e9:.1f} GB)...")
    with open(pkl_path, "rb") as f:
        mol_bytes_list = pickle.load(f)
    print(f"  {len(mol_bytes_list)} entries loaded")

    n_written, n_failed = write_bytes_to_lmdb(
        mol_bytes_list,
        str(lmdb_path),
        total=len(mol_bytes_list),
        map_size=map_size,
        desc=f"Migrating {pkl_path.name}",
    )
    print(f"  Wrote: {n_written}  |  Skipped-None: {n_failed}")
    print(f"  LMDB: {lmdb_path}")

    del mol_bytes_list

    if delete_pkl:
        os.remove(pkl_path)
        print(f"  Deleted {pkl_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--processed_dir",
        type=str,
        required=True,
        help="Directory containing <split>_data.pkl files.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Which splits to migrate.",
    )
    parser.add_argument(
        "--map_size",
        type=int,
        default=LMDB_MAP_SIZE,
        help="LMDB map size in bytes (virtual address reservation, not disk).",
    )
    parser.add_argument(
        "--delete_pkl",
        action="store_true",
        help="Remove the source .pkl file after a successful LMDB write.",
    )
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    if not processed_dir.is_dir():
        raise NotADirectoryError(processed_dir)

    for split in args.splits:
        print(f"\n{'=' * 60}\nMigrating {split}\n{'=' * 60}")
        migrate_split(
            pkl_path=processed_dir / f"{split}_data.pkl",
            lmdb_path=processed_dir / f"{split}_data.lmdb",
            map_size=args.map_size,
            delete_pkl=args.delete_pkl,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
