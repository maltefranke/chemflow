from torch_geometric.datasets.qm9 import QM9, conversion
import torch

from chemflow.utils.utils import (
    edge_types_to_triu_entries,
    edge_types_to_symmetric,
    z_to_atom_types,
    token_to_index,
)

import sys
import os
from tqdm import tqdm

from rdkit import Chem, RDLogger
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem.rdchem import HybridizationType

RDLogger.DisableLog("rdApp.*")  # type: ignore[attr-defined]

from torch_geometric.data import Data
from torch_geometric.io import fs
from torch_geometric.utils import one_hot, scatter
from torch_geometric.data import InMemoryDataset, download_url, extract_zip

from chemflow.dataset.molecule_data import MoleculeData
from chemflow.dataset.vocab import Vocab, Distributions
from chemflow.utils.rdkit import mol_is_valid, sanitize_mol_correctly, BOND_IDX_MAP, smiles_from_mol
from scipy.spatial.transform import Rotation

# Maps QM9 property names to their column index in the y tensor after
# the rearrangement `y = torch.cat([y[:, 3:], y[:, :3]], dim=-1)` applied
# during processing.  The raw CSV columns (A, B, C) are moved to the end, so
# the first 16 entries are the standard QM9 molecular-property targets:
QM9_PROPERTY_NAMES: dict[str, int] = {
    "mu": 0,    # dipole moment (D)
    "alpha": 1, # isotropic polarizability (a0^3)
    "homo": 2,  # HOMO energy (eV)
    "lumo": 3,  # LUMO energy (eV)
    "gap": 4,   # HOMO-LUMO gap (eV)
    "r2": 5,    # electronic spatial extent (a0^2)
    "zpve": 6,  # zero-point vibrational energy (kcal/mol)
    "u0": 7,    # internal energy at 0 K (kcal/mol)
    "u": 8,     # internal energy at 298.15 K (kcal/mol)
    "h": 9,     # enthalpy at 298.15 K (kcal/mol)
    "g": 10,    # free energy at 298.15 K (kcal/mol)
    "cv": 11,   # heat capacity at 298.15 K (cal/mol/K)
}


class RevisedQM9(InMemoryDataset):
    """
    QM9 dataset from PropMolFlow which fixes some major issues with the original QM9 dataset.
    """

    mols_url = "https://zenodo.org/records/15700961/files/all_fixed_gdb9.zip"
    props_url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/molnet_publish/qm9.zip"
    split_to_index = {"train": 0, "val": 1, "test": 2}

    @property
    def raw_file_names(self) -> list[str]:
        return ["all_fixed_gdb9.sdf", "gdb9.sdf.csv"]

    @property
    def processed_file_names(self) -> list[str]:
        return ["train.pt", "val.pt", "test.pt"]

    def load(self, split: str) -> None:
        if split not in self.split_to_index:
            raise ValueError(
                f"Invalid split '{split}'. Expected one of {tuple(self.split_to_index)}."
            )
        self._split = split
        super().load(self.processed_paths[self.split_to_index[split]])

    def download(self) -> None:
        file_path = download_url(self.mols_url, self.raw_dir)
        extract_zip(file_path, self.raw_dir)
        os.unlink(file_path)

        file_path = download_url(self.props_url, self.raw_dir)
        extract_zip(file_path, self.raw_dir)

        os.unlink(file_path)
        # also delete the extra data file in the zip
        os.unlink(os.path.join(self.raw_dir, "gdb9.sdf"))

    def process(self) -> None:

        types = {"H": 0, "C": 1, "N": 2, "O": 3, "F": 4}
        bonds = BOND_IDX_MAP

        with open(self.raw_paths[1]) as f:
            target = [
                [float(x) for x in line.split(",")[1:20]]
                for line in f.read().split("\n")[1:-1]
            ]
            y = torch.tensor(target, dtype=torch.float)
            y = torch.cat([y[:, 3:], y[:, :3]], dim=-1)
            y = y * conversion.view(1, -1)

        suppl = Chem.SDMolSupplier(self.raw_paths[0], removeHs=False, sanitize=False)

        errors = 0
        skipped = 0

        data_list = []
        for i, mol in enumerate(tqdm(suppl)):
            # if i in skip:
            #    continue
            if mol is None or not mol_is_valid(mol):
                errors += 1
                continue

            mol = sanitize_mol_correctly(mol)
            if mol is None:
                errors += 1
                continue

            N = mol.GetNumAtoms()

            conf = mol.GetConformer()
            pos = conf.GetPositions()
            pos = torch.tensor(pos, dtype=torch.float)

            charges = []

            type_idx = []
            atomic_number = []
            aromatic = []
            sp = []
            sp2 = []
            sp3 = []
            num_hs = []
            for atom in mol.GetAtoms():
                type_idx.append(types[atom.GetSymbol()])
                atomic_number.append(atom.GetAtomicNum())
                aromatic.append(1 if atom.GetIsAromatic() else 0)
                hybridization = atom.GetHybridization()
                sp.append(1 if hybridization == HybridizationType.SP else 0)
                sp2.append(1 if hybridization == HybridizationType.SP2 else 0)
                sp3.append(1 if hybridization == HybridizationType.SP3 else 0)
                charges.append(atom.GetFormalCharge())

            z = torch.tensor(atomic_number, dtype=torch.long)

            rows, cols, edge_types = [], [], []
            for bond in mol.GetBonds():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                rows += [start, end]
                cols += [end, start]
                edge_types += 2 * [bonds[bond.GetBondType()]]

            edge_index = torch.tensor([rows, cols], dtype=torch.long)
            edge_type = torch.tensor(edge_types, dtype=torch.long)
            # edge_attr = one_hot(edge_type, num_classes=len(bonds))

            perm = (edge_index[0] * N + edge_index[1]).argsort()
            edge_index = edge_index[:, perm]
            edge_type = edge_type[perm]
            # edge_attr = edge_attr[perm]

            row, col = edge_index
            hs = (z == 1).to(torch.float)
            num_hs = scatter(hs[row], col, dim_size=N, reduce="sum").tolist()

            x1 = one_hot(torch.tensor(type_idx), num_classes=len(types))
            x2 = (
                torch.tensor(
                    [atomic_number, aromatic, sp, sp2, sp3, num_hs], dtype=torch.float
                )
                .t()
                .contiguous()
            )
            x = torch.cat([x1, x2], dim=-1)

            charges = torch.tensor(charges, dtype=torch.int64)

            name = mol.GetProp("_Name")
            smiles = smiles_from_mol(mol, canonical=True) or ""

            # TODO exchange with our own MolData object for consistency
            data = Data(
                x=x,
                z=z,
                pos=pos,
                charges=charges,
                edge_index=edge_index,
                smiles=smiles,
                edge_attr=edge_type,
                y=y[i].unsqueeze(0),
                name=name,
                idx=i,
            )

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        n_train = 100_000
        n_val = 20_000

        self.save(data_list[:n_train], self.processed_paths[0])
        self.save(data_list[n_train : n_train + n_val], self.processed_paths[1])
        self.save(data_list[n_train + n_val :], self.processed_paths[2])

        for split_name, split_data in [
            ("train", data_list[:n_train]),
            ("val", data_list[n_train : n_train + n_val]),
            ("test", data_list[n_train + n_val :]),
        ]:
            smiles_path = os.path.join(self.processed_dir, f"{split_name}_smiles.txt")
            unique_smiles = sorted(set(d.smiles for d in split_data))
            with open(smiles_path, "w") as f:
                f.write("\n".join(unique_smiles))

        print(f"Errors: {errors}")
        print(f"Skipped: {skipped}")

    def get_all_smiles(self) -> list[str]:
        smiles_path = os.path.join(self.processed_dir, f"{self._split}_smiles.txt")
        with open(smiles_path) as f:
            return f.read().splitlines()


class FlowMatchingQM9Dataset(RevisedQM9):
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
        super().__init__(root, transform, pre_transform)
        self.load(split)

        self.vocab = vocab
        self.distributions = distributions

        self.rotate = rotate

    def __getitem__(self, index):
        data = super().__getitem__(index)

        # remove center of mass
        coord = data.pos - data.pos.mean(dim=0)

        if self.rotate:
            # do a random rotation
            rotation = Rotation.random(1)
            coord = coord @ rotation.as_matrix()[0]
            coord = coord.to(dtype=data.pos.dtype)

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

        mol = MoleculeData(
            x=coord, a=atom_types, e=edge_types, c=charges, edge_index=data.edge_index
        )

        # Carry through molecular properties if available (shape: [1, num_properties])
        # Keep 2D so PyG batching concatenates along dim 0 → [batch_size, num_properties]
        if hasattr(data, "y") and data.y is not None:
            mol.y = data.y if data.y.dim() == 2 else data.y.unsqueeze(0)

        return mol


if __name__ == "__main__":
    qm9 = FlowMatchingQM9Dataset(
        root="/cluster/project/krause/frankem/chemflow/data/qm9"
    )
    print(qm9[0])
