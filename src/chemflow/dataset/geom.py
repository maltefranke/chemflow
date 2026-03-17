import os
import pickle
from multiprocessing import get_context

import torch
from rdkit import Chem
from torch.utils.data import Dataset
from torch_geometric.data import Data
from tqdm import tqdm

from chemflow.dataset.molecule_data import MoleculeData
from chemflow.dataset.vocab import Distributions, Vocab
from chemflow.utils.rdkit import BOND_IDX_MAP, mol_is_valid, sanitize_mol_correctly
from chemflow.utils.utils import (
    edge_types_to_symmetric,
    token_to_index,
    z_to_atom_types,
)

PICKLE_PROTOCOL = 4


def process_one_conformer(mol: Chem.Mol):
    """Process a single RDKit conformer and extract features.

    Returns a Data object or None if processing fails.
    """
    if mol is None or not mol_is_valid(mol, allow_charged=True):
        return None

    mol = sanitize_mol_correctly(mol)
    if mol is None:
        return None

    try:
        N = mol.GetNumAtoms()
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

        smiles = Chem.MolToSmiles(mol, isomericSmiles=False)

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


def save_dataset_bytes(mol_bytes_list: list[bytes], filepath: str) -> None:
    """Save a list of serialized molecule bytes to a single pickle file."""
    with open(filepath, "wb") as f:
        pickle.dump(mol_bytes_list, f, protocol=PICKLE_PROTOCOL)


def load_dataset_bytes(filepath: str) -> list[bytes]:
    """Load a list of serialized molecule bytes from a pickle file."""
    with open(filepath, "rb") as f:
        return pickle.load(f)


def process_one_conformer_to_bytes(mol: Chem.Mol) -> bytes | None:
    """Process one conformer and return serialized bytes if valid."""
    data = process_one_conformer(mol)
    if data is None:
        return None
    return mol_to_bytes(data)


def process_conformers_parallel(
    conformers: list[Chem.Mol],
    num_workers: int | None = None,
    chunksize: int = 128,
) -> tuple[list[bytes], int]:
    """Process conformers in parallel with multiprocessing.

    Args:
        conformers: RDKit conformers to process.
        num_workers: Number of processes. Defaults to ``max(cpu_count - 1, 1)``.
        chunksize: Chunk size for ``imap_unordered``.
    """
    if num_workers is None:
        cpu_count = os.cpu_count() or 1
        num_workers = max(1, cpu_count - 1)

    if num_workers <= 1:
        mol_bytes_list = []
        n_failed = 0
        for conformer in tqdm(conformers, desc="Processing conformers"):
            out = process_one_conformer_to_bytes(conformer)
            if out is None:
                n_failed += 1
            else:
                mol_bytes_list.append(out)
        return mol_bytes_list, n_failed

    mol_bytes_list = []
    n_failed = 0
    ctx = get_context("spawn")
    with ctx.Pool(processes=num_workers) as pool:
        outputs = pool.imap_unordered(
            process_one_conformer_to_bytes, conformers, chunksize=chunksize
        )
        for out in tqdm(outputs, total=len(conformers), desc="Processing conformers"):
            if out is None:
                n_failed += 1
            else:
                mol_bytes_list.append(out)

    return mol_bytes_list, n_failed


class GEOM(Dataset):
    """GEOM dataset with pickle-bytes serialization for fast loading.

    Raw pickle files containing (smiles, list of conformers) tuples are
    processed once via ``process()`` or a standalone preprocessing script.
    Each conformer is serialized to compact pickle bytes (via ``mol_to_bytes``)
    and stored in a single ``.pkl`` file.  Subsequent loads deserialize only
    the byte-string list (no tensor reconstruction), and individual molecules
    are reconstructed lazily in ``__getitem__``.
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
        self.processed_dir = os.path.join(root, "processed")

        os.makedirs(self.processed_dir, exist_ok=True)

        self.processed_file = os.path.join(self.processed_dir, f"{split}_data.pkl")

        if not os.path.exists(self.processed_file):
            self.process()

        self._mol_bytes = load_dataset_bytes(self.processed_file)

    def process(self):
        """Process raw pickle files and save as serialized molecule bytes."""
        raw_file = os.path.join(self.raw_dir, f"{self.split}_data.pickle")

        if not os.path.exists(raw_file):
            raise FileNotFoundError(f"Raw data file not found: {raw_file}")

        print(f"Loading raw data from {raw_file}...")
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
        print(f"Processing {len(all_conformers)} conformers with multiprocessing...")
        mol_bytes_list, n_failed = process_conformers_parallel(all_conformers)

        print(f"Successfully processed: {len(mol_bytes_list)} molecules")
        print(f"Failed: {n_failed} molecules")

        del all_conformers

        print(f"Saving processed data to {self.processed_file}...")
        save_dataset_bytes(mol_bytes_list, self.processed_file)

    def __len__(self):
        return len(self._mol_bytes)

    def __getitem__(self, index):
        return mol_from_bytes(self._mol_bytes[index])


class FlowMatchingGEOMDataset(GEOM):
    def __init__(
        self,
        root,
        vocab: Vocab,
        distributions: Distributions,
        transform=None,
        pre_transform=None,
        rotate=False,
        split="train",
    ):
        super().__init__(root, split, transform, pre_transform)

        self.vocab = vocab
        self.distributions = distributions

        self.rotate = rotate

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
