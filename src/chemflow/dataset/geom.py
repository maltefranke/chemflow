from torch.utils.data import Dataset
import pickle
import torch

from rdkit import Chem
from torch_geometric.data import Data
from chemflow.dataset.molecule_data import MoleculeData
from chemflow.rdkit import mol_is_valid, sanitize_mol_correctly, BOND_IDX_MAP


def process_one_mol(mol: Chem.Mol):
    if mol is None or not mol_is_valid(mol):
        return None

    mol = sanitize_mol_correctly(mol)
    if mol is None:
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

    rows, cols, edge_types = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        rows += [start, end]
        cols += [end, start]
        edge_types += 2 * [BOND_IDX_MAP[bond.GetBondType()]]

    edge_index = torch.tensor([rows, cols], dtype=torch.long)
    edge_type = torch.tensor(edge_types, dtype=torch.long)

    charges = torch.tensor(charges, dtype=torch.int64)

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


class GEOM(Dataset):
    """
    GEOM dataset. Adapted from FlowMol3 preprocessing code.
    """

    def __init__(self, data_file):
        self.data_file = data_file

        self.process()

    def process(self):
        with open(self.data_file, "rb") as f:
            raw_data = pickle.load(f)

        all_molecules = []
        all_smiles = []
        for molecule_chunk in raw_data:
            all_smiles.append(molecule_chunk[0])

            n_confs_this_mol = len(molecule_chunk[1])

            for conformer in molecule_chunk[1][:n_confs_this_mol]:
                all_molecules.append(conformer)

        # clean up memory
        del raw_data

        all_positions = []
        all_atom_types = []
        all_atom_charges = []
        all_bond_types = []
        all_bond_idxs = []

        failed_molecules = 0
        for molecule_chunk in tqdm_iterator:
            batch_data: BatchMoleculeData = mol_featurizer.featurize_molecules(
                molecule_chunk
            )

            num_failed = len(batch_data.failed_idxs)

            failed_molecules += num_failed
            failed_molecules_bar.update(num_failed)
            total_molecules_bar.update(batch_data.n_mols)
            for k, v in batch_data.failure_counts.items():
                failure_counts[k] += v

            all_positions.extend(batch_data.positions)
            all_atom_types.extend(batch_data.atom_types)
            all_atom_charges.extend(batch_data.atom_charges)
            all_bond_types.extend(batch_data.bond_types)
            all_bond_idxs.extend(batch_data.bond_idxs)

        # create a dictionary to store all the data
        data_dict = {
            "smiles": all_smiles,
            "positions": all_positions,
            "atom_types": all_atom_types,
            "atom_charges": all_atom_charges,
            "bond_types": all_bond_types,
            "bond_idxs": all_bond_idxs,
            "node_idx_array": node_idx_array,
            "edge_idx_array": edge_idx_array,
        }

        # save the data
        torch.save(data_dict, output_file)

    def __getitem__(self, index):
        data = MoleculeData(
            x=positions,
            a=atom_types,
            e=bond_types,
            edge_index=bond_idxs,
            c=atom_charges,
        )
        return data
