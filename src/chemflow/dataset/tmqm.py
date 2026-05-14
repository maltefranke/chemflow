"""tmQM dataset — transition-metal complexes, pointcloud-only.

Source: https://github.com/uiocompcat/tmQM. ~108k XTB-optimized structures with
DFT-computed properties. We treat tmQM as pointcloud-only: no formal charges
(the .q file is *natural* atomic charges, continuous floats), no bond types
(the .BO file is Wiberg bond orders, continuous). SMILES are Hückel-derived
and carried through for downstream conditioning but never used as supervision.
"""

import gzip
import os
import random
import shutil

import torch
from rdkit import Chem, RDLogger
from torch_geometric.data import Data, InMemoryDataset, download_url
from tqdm import tqdm

from chemflow.dataset.molecule_data import MoleculeData
from chemflow.dataset.representation import Capabilities
from chemflow.dataset.vocab import Distributions, Vocab
from chemflow.utils.utils import token_to_index, z_to_atom_types

RDLogger.DisableLog("rdApp.*")  # type: ignore[attr-defined]

# DFT/xTB property columns kept from tmQM_y.csv (order is preserved as mol.y).
TMQM_PROPERTY_NAMES: dict[str, int] = {
    "Electronic_E": 0,
    "Dispersion_E": 1,
    "Dipole_M": 2,
    "Metal_q": 3,
    "HL_Gap": 4,
    "HOMO_Energy": 5,
    "LUMO_Energy": 6,
    "Polarizability": 7,
}


class TMQM(InMemoryDataset):
    """tmQM 2024 release. Pointcloud only — no edge_attr, no per-atom charges."""

    base_url = "https://raw.githubusercontent.com/uiocompcat/tmQM/master/tmQM"
    raw_files = [
        "tmQM_X1.xyz.gz",
        "tmQM_X2.xyz.gz",
        "tmQM_X3.xyz.gz",
        "tmQM_y.csv",
    ]
    split_to_index = {"train": 0, "val": 1, "test": 2}

    # Drop oversized complexes — the long tail goes well past this but the bulk
    # of the dataset is <120 atoms.
    N_ATOM_CAP = 120
    SPLIT_RATIO = (0.9, 0.05, 0.05)
    SHUFFLE_SEED = 42

    @property
    def raw_file_names(self) -> list[str]:
        # Post-download: gzipped .xyz files are decompressed in-place.
        return ["tmQM_X1.xyz", "tmQM_X2.xyz", "tmQM_X3.xyz", "tmQM_y.csv"]

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
        for fname in self.raw_files:
            target = os.path.join(self.raw_dir, fname)
            decompressed = target[:-3] if fname.endswith(".gz") else target
            if os.path.exists(decompressed):
                continue
            if not os.path.exists(target):
                download_url(f"{self.base_url}/{fname}", self.raw_dir)
            if fname.endswith(".gz"):
                with gzip.open(target, "rb") as fin, open(decompressed, "wb") as fout:
                    shutil.copyfileobj(fin, fout)
                os.unlink(target)

    def _load_properties(self) -> dict[str, tuple[torch.Tensor, str]]:
        """Parse tmQM_y.csv → {CSD_code: (y_tensor[8], smiles)}."""
        csv_path = os.path.join(self.raw_dir, "tmQM_y.csv")
        out: dict[str, tuple[torch.Tensor, str]] = {}
        with open(csv_path) as f:
            header = f.readline().strip().split(";")
            col_idx = {name: header.index(name) for name in TMQM_PROPERTY_NAMES}
            smiles_idx = header.index("SMILES")
            code_idx = header.index("CSD_code")
            for line in f:
                parts = line.rstrip("\n").split(";")
                if len(parts) < len(header):
                    continue
                code = parts[code_idx]
                try:
                    vals = [float(parts[col_idx[n]]) for n in TMQM_PROPERTY_NAMES]
                except ValueError:
                    continue
                y = torch.tensor(vals, dtype=torch.float).unsqueeze(0)
                smiles = parts[smiles_idx]
                out[code] = (y, smiles)
        return out

    def _iter_xyz_blocks(self, path: str):
        """Yield (csd_code, z_tensor, pos_tensor) for each block in a tmQM xyz file."""
        with open(path) as f:
            while True:
                header = f.readline()
                if not header:
                    return
                n = int(header.strip())
                comment = f.readline().strip()
                # comment format: "CSD_code = XXX | q = .. | S = .. | ..."
                csd_code = comment.split("|", 1)[0].split("=", 1)[1].strip()
                z_list = []
                pos_list = []
                for _ in range(n):
                    tok = f.readline().split()
                    z_list.append(Chem.GetPeriodicTable().GetAtomicNumber(tok[0]))
                    pos_list.append([float(tok[1]), float(tok[2]), float(tok[3])])
                # Trailing blank line between blocks.
                f.readline()
                z = torch.tensor(z_list, dtype=torch.long)
                pos = torch.tensor(pos_list, dtype=torch.float)
                yield csd_code, z, pos

    def process(self) -> None:
        props = self._load_properties()

        data_list: list[Data] = []
        for fname in ["tmQM_X1.xyz", "tmQM_X2.xyz", "tmQM_X3.xyz"]:
            path = os.path.join(self.raw_dir, fname)
            for csd_code, z, pos in tqdm(self._iter_xyz_blocks(path), desc=fname):
                if z.numel() > self.N_ATOM_CAP:
                    continue
                if csd_code not in props:
                    continue
                y, smiles = props[csd_code]
                data = Data(
                    z=z,
                    pos=pos,
                    y=y,
                    smiles=smiles,
                    name=csd_code,
                    idx=len(data_list),
                )
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                data_list.append(data)

        rng = random.Random(self.SHUFFLE_SEED)
        rng.shuffle(data_list)

        n = len(data_list)
        n_train = int(self.SPLIT_RATIO[0] * n)
        n_val = int(self.SPLIT_RATIO[1] * n)
        splits = {
            "train": data_list[:n_train],
            "val": data_list[n_train : n_train + n_val],
            "test": data_list[n_train + n_val :],
        }

        for split_name, split_data in splits.items():
            self.save(split_data, self.processed_paths[self.split_to_index[split_name]])
            smiles_path = os.path.join(self.processed_dir, f"{split_name}_smiles.txt")
            unique = sorted({d.smiles for d in split_data if d.smiles})
            with open(smiles_path, "w") as f:
                f.write("\n".join(unique))

    def get_all_smiles(self) -> list[str]:
        smiles_path = os.path.join(self.processed_dir, f"{self._split}_smiles.txt")
        with open(smiles_path) as f:
            return f.read().splitlines()


class FlowMatchingTMQMDataset(TMQM):
    CAPABILITIES = Capabilities(provides_charges=False, provides_topology=False)

    def __init__(
        self,
        root,
        vocab: Vocab,
        distributions: Distributions,
        transform=None,
        pre_transform=None,
        split="train",
    ):
        super().__init__(root, transform, pre_transform)
        self.load(split)

        self.vocab = vocab
        self.distributions = distributions

    def __getitem__(self, index):
        data = super().__getitem__(index)

        coord = data.pos - data.pos.mean(dim=0)
        if self.distributions.coordinate_std is not None:
            coord = coord / self.distributions.coordinate_std

        atom_types = z_to_atom_types(data.z.tolist())
        atom_types = torch.tensor(
            [token_to_index(self.vocab.atom_tokens, t) for t in atom_types],
            dtype=torch.long,
        )

        # c, e, edge_index left as None — the wrapper's projection fills them
        # canonically (neutral charge, fully-connected, <NO_BOND>).
        mol = MoleculeData(
            x=coord, a=atom_types, c=None, e=None, edge_index=None,
        )

        if hasattr(data, "y") and data.y is not None:
            mol.y = data.y if data.y.dim() == 2 else data.y.unsqueeze(0)

        if hasattr(data, "smiles"):
            mol.smiles = data.smiles

        return mol
