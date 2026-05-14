import os
import pickle
from multiprocessing import get_context
from typing import Iterable

import lmdb
import torch
from rdkit import Chem
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.data import download_url
from tqdm import tqdm

from chemflow.dataset.molecule_data import MoleculeData
from chemflow.dataset.vocab import Distributions, Vocab
from chemflow.utils.rdkit_utils import (
    BOND_IDX_MAP,
    mol_is_valid,
    sanitize_mol_correctly,
    smiles_from_mol,
)
from chemflow.utils.utils import (
    edge_types_to_symmetric,
    token_to_index,
    z_to_atom_types,
)
from external_code.geom_drugs_preprocessing import process_geom_drugs

# Source of the raw GEOM-Drugs pickle files used by the GEOM-Drugs Revisited
# paper (Nikitin et al., 2025, arXiv:2505.00169). Three files live here:
# train_data.pickle, val_data.pickle, test_data.pickle.
GEOM_RAW_URL = "https://bits.csb.pitt.edu/files/geom_raw"

RAW_FILENAMES = {
    "train": "train_data.pickle",
    "val": "val_data.pickle",
    "test": "test_data.pickle",
}

PICKLE_PROTOCOL = 4

# 128 GiB virtual address reservation for the LMDB file. This is NOT disk
# usage — the file grows sparsely with actual data. Large enough to hold the
# full processed GEOM train split with headroom.
LMDB_MAP_SIZE = 128 * (1024**3)

# Reserved metadata key for entry count. Integer data keys are 8-byte
# big-endian, so this textual key cannot collide with them.
LEN_KEY = b"__len__"


def _int_key(i: int) -> bytes:
    return i.to_bytes(8, "big")


def process_one_conformer(mol: Chem.Mol):
    """Process a single RDKit conformer and extract features.

    Returns a Data object or None if processing fails.
    """
    if mol is None or not mol_is_valid(mol, allow_charged=True):
        return None

    mol = sanitize_mol_correctly(mol)
    if mol is None:
        return None

    # Force Kekulé form so aromatic ring bonds become SINGLE/DOUBLE rather than
    # BondType.AROMATIC. Sanitization above re-perceives aromaticity, undoing
    # the kekulization applied by the GEOM-Drugs Revisited pre-filter; this
    # call restores it so AROMATIC never reaches the bond-type tensor (Nikitin
    # et al., 2025 — kekulization is the paper's recommended fix for the
    # valency ambiguity of aromatic bonds).
    try:
        Chem.Kekulize(mol, clearAromaticFlags=True)
    except Exception:
        return None

    try:
        N = mol.GetNumAtoms()
        if N >= 72:
            return None
        conf = mol.GetConformer()
        pos = conf.GetPositions()
        pos = torch.tensor(pos, dtype=torch.float)

        charges = []
        atomic_number = []
        for atom in mol.GetAtoms():
            atomic_number.append(atom.GetAtomicNum())
            charges.append(atom.GetFormalCharge())

        z = torch.tensor(atomic_number, dtype=torch.long)
        charges = torch.tensor(charges, dtype=torch.int64)

        rows, cols, edge_types = [], [], []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            rows += [start, end]
            cols += [end, start]
            edge_types += 2 * [BOND_IDX_MAP[bond.GetBondType()]]

        if len(rows) == 0:
            edge_index = torch.tensor([[], []], dtype=torch.long)
            edge_type = torch.tensor([], dtype=torch.long)
        else:
            edge_index = torch.tensor([rows, cols], dtype=torch.long)
            edge_type = torch.tensor(edge_types, dtype=torch.long)

            perm = (edge_index[0] * N + edge_index[1]).argsort()
            edge_index = edge_index[:, perm]
            edge_type = edge_type[perm]

        smiles = smiles_from_mol(mol, canonical=True) or ""

        data = Data(
            z=z,
            pos=pos,
            charges=charges,
            edge_index=edge_index,
            smiles=smiles,
            edge_attr=edge_type,
        )

        return data
    except Exception as e:
        print(f"Error processing molecule: {e}")
        return None


def mol_to_bytes(data: Data) -> bytes:
    """Serialize a processed molecule Data object to compact pickle bytes.

    Stores only the raw tensor data as a plain dict, avoiding the overhead
    of pickling full PyG Data objects.
    """
    dict_repr = {
        "z": data.z,
        "pos": data.pos,
        "charges": data.charges,
        "edge_index": data.edge_index,
        "edge_attr": data.edge_attr,
        "smiles": data.smiles,
    }
    return pickle.dumps(dict_repr, protocol=PICKLE_PROTOCOL)


def mol_from_bytes(data: bytes) -> Data:
    """Deserialize pickle bytes back into a PyG Data object."""
    obj = pickle.loads(data)
    return Data(
        z=obj["z"],
        pos=obj["pos"],
        charges=obj["charges"],
        edge_index=obj["edge_index"],
        smiles=obj["smiles"],
        edge_attr=obj["edge_attr"],
    )


def process_one_conformer_to_bytes(mol: Chem.Mol) -> bytes | None:
    """Process one conformer and return serialized bytes if valid."""
    data = process_one_conformer(mol)
    if data is None:
        return None
    return mol_to_bytes(data)


def open_read_env(lmdb_path: str) -> lmdb.Environment:
    """Open an LMDB environment configured for multi-worker read access.

    ``lock=False`` + ``readonly=True`` allows multiple processes (DataLoader
    workers) to share the mmap without the writer lock. ``readahead=False``
    is better for random access patterns typical in training.
    """
    return lmdb.open(
        lmdb_path,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        subdir=True,
        max_readers=512,
    )


def read_lmdb_length(lmdb_path: str) -> int:
    """Return the number of molecules stored in an LMDB env."""
    env = open_read_env(lmdb_path)
    try:
        with env.begin() as txn:
            raw = txn.get(LEN_KEY)
            if raw is not None:
                return int(raw)
            return txn.stat()["entries"]
    finally:
        env.close()


def write_bytes_to_lmdb(
    bytes_iter: Iterable[bytes | None],
    lmdb_path: str,
    total: int | None = None,
    map_size: int = LMDB_MAP_SIZE,
    commit_every: int = 10_000,
    desc: str = "Writing LMDB",
) -> tuple[int, int]:
    """Stream serialized molecule bytes into a new LMDB environment.

    ``None`` entries in the iterator are counted as failures and skipped.
    """
    os.makedirs(os.path.dirname(lmdb_path) or ".", exist_ok=True)
    env = lmdb.open(
        lmdb_path,
        map_size=map_size,
        subdir=True,
        meminit=False,
        writemap=False,
        max_readers=512,
    )
    n_written = 0
    n_failed = 0
    try:
        txn = env.begin(write=True)
        for out in tqdm(bytes_iter, total=total, desc=desc):
            if out is None:
                n_failed += 1
                continue
            txn.put(_int_key(n_written), out)
            n_written += 1
            if n_written % commit_every == 0:
                txn.commit()
                txn = env.begin(write=True)
        txn.put(LEN_KEY, str(n_written).encode())
        txn.commit()
    finally:
        env.close()
    return n_written, n_failed


def process_conformers_to_lmdb(
    conformers: list[Chem.Mol],
    lmdb_path: str,
    num_workers: int | None = None,
    chunksize: int = 128,
    map_size: int = LMDB_MAP_SIZE,
    commit_every: int = 10_000,
) -> tuple[int, int]:
    """Process conformers in parallel and stream serialized bytes into LMDB.

    Writes happen as workers complete so peak memory stays bounded regardless
    of dataset size (unlike accumulating a full in-memory list first).
    """
    if num_workers is None:
        cpu_count = os.cpu_count() or 1
        num_workers = max(1, cpu_count - 1)

    if num_workers <= 1:
        iterator = (process_one_conformer_to_bytes(c) for c in conformers)
        return write_bytes_to_lmdb(
            iterator,
            lmdb_path,
            total=len(conformers),
            map_size=map_size,
            commit_every=commit_every,
            desc="Processing conformers",
        )

    ctx = get_context("spawn")
    with ctx.Pool(processes=num_workers) as pool:
        outputs = pool.imap_unordered(
            process_one_conformer_to_bytes, conformers, chunksize=chunksize
        )
        return write_bytes_to_lmdb(
            outputs,
            lmdb_path,
            total=len(conformers),
            map_size=map_size,
            commit_every=commit_every,
            desc="Processing conformers",
        )


class GEOM(Dataset):
    """GEOM dataset backed by an LMDB environment.

    The processed data lives in ``<root>/processed/<split>_data.lmdb/`` as a
    key-value store with 8-byte big-endian integer keys mapping to the same
    compact pickle-dict bytes produced by :func:`mol_to_bytes`.

    Reads are memory-mapped: the dataset does not load into RAM, and multiple
    DataLoader workers share the mmap. The LMDB handle is opened lazily on
    first access so it's safe under both ``fork`` and ``spawn`` worker start
    methods.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        vocab: Vocab = None,
        distributions: Distributions = None,
    ):
        self.root = root
        self.split = split
        self.vocab = vocab
        self.distributions = distributions

        self.raw_dir = os.path.join(root, "raw")
        # Output dir of the GEOM-Drugs Revisited pre-filter (Nikitin et al.,
        # 2025). Sits between ``raw/`` and ``processed/`` and holds the
        # sanitized pickle files plus the empirical ``valency_dict.json``.
        self.filtered_dir = os.path.join(root, "raw_filtered")
        self.processed_dir = os.path.join(root, "processed")

        os.makedirs(self.processed_dir, exist_ok=True)

        self.lmdb_path = os.path.join(self.processed_dir, f"{split}_data.lmdb")

        if not os.path.exists(self.lmdb_path):
            self.download()
            self.prefilter()
            self.process()

        self._length = read_lmdb_length(self.lmdb_path)
        self._env: lmdb.Environment | None = None

    def _ensure_env(self) -> lmdb.Environment:
        if self._env is None:
            self._env = open_read_env(self.lmdb_path)
        return self._env

    def __getstate__(self):
        # Strip the live env handle so DataLoader workers using spawn can
        # pickle the dataset. Each worker reopens lazily on first access.
        state = self.__dict__.copy()
        state["_env"] = None
        return state

    def download(self) -> None:
        """Fetch raw GEOM-Drugs pickle files from bits.csb.pitt.edu.

        Mirrors the PyG ``InMemoryDataset.download`` pattern: each of the three
        split pickles is downloaded into ``raw/`` if it is not already present.
        Files that already exist on disk are left untouched so re-running
        training does not re-fetch ~8 GB.
        """
        os.makedirs(self.raw_dir, exist_ok=True)
        for fname in RAW_FILENAMES.values():
            target = os.path.join(self.raw_dir, fname)
            if os.path.exists(target):
                continue
            print(f"Downloading {fname} from {GEOM_RAW_URL} ...")
            download_url(f"{GEOM_RAW_URL}/{fname}", self.raw_dir)

    def prefilter(self) -> None:
        """Apply the GEOM-Drugs Revisited pre-filter to all available splits.

        Defers to ``external_code.geom_drugs_preprocessing.process_geom_drugs``,
        which sanitizes/kekulizes molecules, drops disconnected fragments, runs
        a covalent-radius topology check on every conformer, and writes the
        cleaned pickle files plus a ``valency_dict.json`` into ``raw_filtered/``.

        Runs once per dataset root: if every filtered pickle is already on disk
        the step is skipped.
        """
        all_present = all(
            os.path.exists(os.path.join(self.filtered_dir, fname))
            for fname in RAW_FILENAMES.values()
        )
        if all_present:
            return

        print("Applying GEOM-Drugs Revisited pre-filter to all splits ...")
        process_geom_drugs(self.raw_dir, self.filtered_dir)

        # Reclaim ~8 GB by truncating each raw pickle whose filtered copy now
        # exists. A zero-byte sentinel is left in place so ``download()`` still
        # short-circuits on subsequent runs.
        for fname in RAW_FILENAMES.values():
            raw_path = os.path.join(self.raw_dir, fname)
            filtered_path = os.path.join(self.filtered_dir, fname)
            if (
                os.path.exists(filtered_path)
                and os.path.exists(raw_path)
                and os.path.getsize(raw_path) > 0
            ):
                with open(raw_path, "wb"):
                    pass
                print(f"  Truncated raw {fname} (filtered copy in {self.filtered_dir})")

    def process(self):
        """Process pre-filtered pickle files and write them into LMDB."""
        raw_file = os.path.join(self.filtered_dir, f"{self.split}_data.pickle")

        if not os.path.exists(raw_file):
            raise FileNotFoundError(f"Filtered data file not found: {raw_file}")

        print(f"Loading filtered data from {raw_file}...")
        with open(raw_file, "rb") as f:
            raw_data = pickle.load(f)

        print("Preparing conformers for processing...")
        all_conformers = [
            conformer
            for _smiles, conformers_list in tqdm(raw_data, desc="Preparing data")
            for conformer in conformers_list
        ]
        del raw_data

        print(f"Total conformers to process: {len(all_conformers)}")
        n_written, n_failed = process_conformers_to_lmdb(all_conformers, self.lmdb_path)
        print(f"Successfully processed: {n_written} molecules")
        print(f"Failed: {n_failed} molecules")
        del all_conformers

        self._write_smiles_sidecar()

    def _write_smiles_sidecar(self) -> None:
        """Build the unique-SMILES sidecar by reading the LMDB env once."""
        smiles_path = os.path.join(self.processed_dir, f"{self.split}_smiles.txt")
        unique: set[str] = set()
        env = open_read_env(self.lmdb_path)
        try:
            with env.begin() as txn, txn.cursor() as cur:
                for key, value in tqdm(cur, desc="Collecting SMILES"):
                    if bytes(key) == LEN_KEY:
                        continue
                    unique.add(pickle.loads(bytes(value))["smiles"])
        finally:
            env.close()
        with open(smiles_path, "w") as f:
            f.write("\n".join(sorted(unique)))

    def get_all_smiles(self) -> list[str]:
        smiles_path = os.path.join(self.processed_dir, f"{self.split}_smiles.txt")
        with open(smiles_path) as f:
            return f.read().splitlines()

    def iter_bytes(self) -> Iterable[bytes]:
        """Iterate raw serialized molecule bytes in insertion order."""
        env = self._ensure_env()
        with env.begin() as txn:
            for i in range(self._length):
                raw = txn.get(_int_key(i))
                if raw is None:
                    continue
                yield bytes(raw)

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        env = self._ensure_env()
        with env.begin(buffers=True) as txn:
            raw = txn.get(_int_key(index))
        if raw is None:
            raise IndexError(index)
        return mol_from_bytes(bytes(raw))

    def get(self, index):
        return self.__getitem__(index)


class FlowMatchingGEOMDataset(GEOM):
    def __init__(
        self,
        root,
        vocab: Vocab,
        distributions: Distributions,
        transform=None,
        pre_transform=None,
        split="train",
    ):
        super().__init__(root, split, transform, pre_transform)

        self.vocab = vocab
        self.distributions = distributions

    def __getitem__(self, index):
        data = super().__getitem__(index)

        # remove center of mass
        coord = data.pos - data.pos.mean(dim=0)
        if self.distributions.coordinate_std is not None:
            coord = coord / self.distributions.coordinate_std

        atom_types = data.z
        atom_types = z_to_atom_types(atom_types.tolist())
        atom_types = [
            token_to_index(self.vocab.atom_tokens, token) for token in atom_types
        ]
        atom_types = torch.tensor(atom_types, dtype=torch.long)

        # add 1 to the edge types to make them 1-indexed
        # 0 is no bond, 1 is single, 2 is double, 3 is triple, 4 is aromatic
        edge_types = data.edge_attr
        edge_types = edge_types_to_symmetric(
            data.edge_index, edge_types, data.num_nodes
        )

        edge_types = edge_types[data.edge_index[0], data.edge_index[1]]
        edge_types = edge_types.to(torch.long)

        charges = data.charges.tolist()
        charges = [
            token_to_index(self.vocab.charge_tokens, str(charge)) for charge in charges
        ]
        charges = torch.tensor(charges, dtype=torch.long)

        data = MoleculeData(
            x=coord, a=atom_types, e=edge_types, c=charges, edge_index=data.edge_index
        )
        return data

    def get(self, index):
        return self.__getitem__(index)
