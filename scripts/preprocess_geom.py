"""Standalone preprocessing script for the GEOM dataset.

Converts raw GEOM pickle files (containing (smiles, [conformers]) tuples)
into an LMDB-backed store for fast, memory-mapped loading at training time.

Usage:
    python -m chemflow.dataset.preprocess_geom --data_path /path/to/geom_data

Expected directory layout::

    data_path/
      raw/
        train_data.pickle
        val_data.pickle
        test_data.pickle

After preprocessing::

    data_path/
      raw/
        ...
      processed/
        train_data.lmdb/    (LMDB env: data.mdb + lock.mdb)
        val_data.lmdb/
        test_data.lmdb/
        train_smiles.txt
        val_smiles.txt
        test_smiles.txt
"""

import argparse
import pickle
from pathlib import Path

from tqdm import tqdm

from chemflow.dataset.geom import (
    LEN_KEY,
    LMDB_MAP_SIZE,
    open_read_env,
    process_conformers_to_lmdb,
)

RAW_SPLITS = {
    "train": "train_data.pickle",
    "val": "val_data.pickle",
    "test": "test_data.pickle",
}

LMDB_SPLITS = {
    "train": "train_data.lmdb",
    "val": "val_data.lmdb",
    "test": "test_data.lmdb",
}

SMILES_FILES = {
    "train": "train_smiles.txt",
    "val": "val_smiles.txt",
    "test": "test_smiles.txt",
}


def write_smiles_sidecar(lmdb_path: Path, smiles_path: Path) -> None:
    """Read an LMDB env once and write the unique-SMILES sidecar."""
    unique: set[str] = set()
    env = open_read_env(str(lmdb_path))
    try:
        with env.begin() as txn, txn.cursor() as cur:
            for key, value in tqdm(cur, desc=f"SMILES {smiles_path.stem}"):
                if bytes(key) == LEN_KEY:
                    continue
                unique.add(pickle.loads(bytes(value))["smiles"])
    finally:
        env.close()
    with open(smiles_path, "w") as f:
        f.write("\n".join(sorted(unique)))


def process_split(
    raw_filepath: Path,
    lmdb_path: Path,
    smiles_path: Path,
    num_workers: int | None = None,
    chunksize: int = 128,
    map_size: int = LMDB_MAP_SIZE,
) -> None:
    """Process a split from raw RDKit mols into an LMDB env + SMILES sidecar."""
    print(f"Loading raw data from {raw_filepath}...")
    with open(raw_filepath, "rb") as f:
        raw_data = pickle.load(f)

    all_conformers = [
        conformer
        for _smiles, conformers_list in tqdm(raw_data, desc="Preparing data")
        for conformer in conformers_list
    ]
    del raw_data

    print(
        f"Processing {len(all_conformers)} conformers "
        f"with multiprocessing (workers={num_workers}, chunksize={chunksize})..."
    )
    n_written, n_failed = process_conformers_to_lmdb(
        all_conformers,
        str(lmdb_path),
        num_workers=num_workers,
        chunksize=chunksize,
        map_size=map_size,
    )
    del all_conformers

    print(f"  Wrote: {n_written}  |  Failed: {n_failed}")
    print(f"  LMDB: {lmdb_path}")

    write_smiles_sidecar(lmdb_path, smiles_path)
    print(f"  SMILES: {smiles_path}")


def main(args: argparse.Namespace) -> None:
    data_path = Path(args.data_path)
    raw_path = data_path / "raw"
    save_path = data_path / "processed"
    save_path.mkdir(parents=True, exist_ok=True)

    for split, raw_filename in RAW_SPLITS.items():
        raw_file = raw_path / raw_filename
        if not raw_file.exists():
            print(f"Skipping {split}: {raw_file} not found")
            continue

        lmdb_path = save_path / LMDB_SPLITS[split]
        smiles_path = save_path / SMILES_FILES[split]
        print(f"\n{'=' * 60}")
        print(f"Processing {split} split")
        print(f"{'=' * 60}")
        process_split(
            raw_file,
            lmdb_path,
            smiles_path,
            num_workers=args.num_workers,
            chunksize=args.chunksize,
            map_size=args.map_size,
        )

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess GEOM dataset into an LMDB-backed store."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Root directory containing raw/ subfolder with GEOM pickle files.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of worker processes for conformer processing.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=128,
        help="Multiprocessing chunk size for worker dispatch.",
    )
    parser.add_argument(
        "--map_size",
        type=int,
        default=LMDB_MAP_SIZE,
        help="LMDB map size in bytes (virtual address reservation, not disk).",
    )
    args = parser.parse_args()
    main(args)
