from torch.utils.data import Dataset
import pickle
import torch
from torch_geometric.data import Data
from rdkit import Chem
from tqdm import tqdm
import os

from chemflow.utils import (
    edge_types_to_triu_entries,
    edge_types_to_symmetric,
    z_to_atom_types,
    token_to_index,
)

from chemflow.dataset.molecule_data import MoleculeData
from chemflow.rdkit import mol_is_valid, sanitize_mol_correctly, BOND_IDX_MAP

from chemflow.dataset.vocab import Vocab, Distributions


def process_one_conformer(mol: Chem.Mol):
    """
    Process a single conformer (molecule) and extract features.
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

        # Extract atomic numbers and charges
        charges = []
        atomic_number = []
        for atom in mol.GetAtoms():
            atomic_number.append(atom.GetAtomicNum())
            charges.append(atom.GetFormalCharge())

        z = torch.tensor(atomic_number, dtype=torch.long)
        charges = torch.tensor(charges, dtype=torch.int64)

        # Extract edges
        rows, cols, edge_types = [], [], []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            rows += [start, end]
            cols += [end, start]
            edge_types += 2 * [BOND_IDX_MAP[bond.GetBondType()]]

        if len(rows) == 0:
            # Handle molecules with no bonds
            edge_index = torch.tensor([[], []], dtype=torch.long)
            edge_type = torch.tensor([], dtype=torch.long)
        else:
            edge_index = torch.tensor([rows, cols], dtype=torch.long)
            edge_type = torch.tensor(edge_types, dtype=torch.long)

            # Sort edges
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


class GEOM(Dataset):
    """
    GEOM dataset with parallel preprocessing.
    Processes pickle/parquet files containing (smiles, list of conformers) tuples.
    Each conformer becomes a separate entry in the dataset.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        vocab: Vocab = None,
        distributions: Distributions = None,
    ):
        """
        Initialize GEOM dataset.

        Args:
            root: Root directory containing raw data files
            split: Dataset split ('train', 'val', 'test')
            vocab: Vocab object for token mapping
            distributions: Distributions object
        """
        self.root = root
        self.split = split
        self.vocab = vocab
        self.distributions = distributions

        self.raw_dir = os.path.join(root, "raw")
        self.processed_dir = os.path.join(root, "processed")

        # Create processed directory if it doesn't exist
        os.makedirs(self.processed_dir, exist_ok=True)

        self.processed_file = os.path.join(self.processed_dir, f"{split}_data.pt")

        # Process data if not already processed
        if not os.path.exists(self.processed_file):
            self.process()

        # Load processed data
        self.data_list = torch.load(self.processed_file)

    def process(self):
        """
        Process raw pickle files serially.
        Each conformer from the pickle file becomes a separate Data object.
        """
        # Load raw data
        raw_file = os.path.join(self.raw_dir, f"{self.split}_data.pickle")

        if not os.path.exists(raw_file):
            raise FileNotFoundError(f"Raw data file not found: {raw_file}")

        print(f"Loading raw data from {raw_file}...")
        with open(raw_file, "rb") as f:
            raw_data = pickle.load(f)

        # Flatten: create list of all conformers with their SMILES
        print("Preparing conformers for processing...")
        all_conformers = []
        all_smiles_list = []

        for smiles, conformers_list in tqdm(raw_data, desc="Preparing data"):
            for conformer in conformers_list:
                all_conformers.append(conformer)
                all_smiles_list.append(smiles)

        # Clean up raw data to free memory
        del raw_data

        print(f"Total conformers to process: {len(all_conformers)}")

        # Process serially
        print(f"Processing {len(all_conformers)} conformers...")
        data_list = []
        for conformer in tqdm(all_conformers, desc="Processing conformers"):
            data = process_one_conformer(conformer)
            if data is not None:
                data_list.append(data)

        # Clean up to free memory
        del all_conformers
        del all_smiles_list

        print(f"Successfully processed: {len(data_list)} molecules")
        print(f"Failed: {len(all_conformers) - len(data_list)} molecules")

        # Save processed data
        print(f"Saving processed data to {self.processed_file}...")
        torch.save(data_list, self.processed_file)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data = self.data_list[index]

        # If vocab and distributions are provided, transform to MoleculeData
        if self.vocab is not None and self.distributions is not None:
            return self._transform_to_molecule_data(data)

        return data

class FlowMatchingGEOMDataset(GEOM):
    def __init__(
        self,
        root,
        vocab: Vocab,
        distributions: Distributions,
        transform=None,
        pre_transform=None,
    ):
        super().__init__(root, transform, pre_transform)

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
    