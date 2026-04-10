"""Generate {split}_smiles.txt files for already-processed datasets.

Run once per dataset whose processed directory predates the SMILES file feature.

Usage:
    python scripts/generate_smiles_files.py data=qm9
    python scripts/generate_smiles_files.py data=geom
"""

import os

import hydra
from omegaconf import DictConfig, OmegaConf
from rdkit import Chem
from tqdm import tqdm

from chemflow.utils.rdkit import smiles_from_mol

OmegaConf.register_new_resolver("oc.eval", eval)
OmegaConf.register_new_resolver("len", lambda x: len(x))
OmegaConf.register_new_resolver("if", lambda cond, t, f: t if cond else f)
OmegaConf.register_new_resolver("eq", lambda x, y: x == y)


def write_smiles_file(smiles: list[str], path: str) -> None:
    unique = sorted(set(smiles))
    with open(path, "w") as f:
        f.write("\n".join(unique))
    print(f"Wrote {len(unique)} unique SMILES to {path}")


def generate_qm9(root: str) -> None:
    from chemflow.dataset.qm9 import RevisedQM9

    for split in ("train", "val", "test"):
        smiles_path = os.path.join(root, "processed", f"{split}_smiles.txt")

        ds = RevisedQM9(root)
        ds.load(split)
        raw = [ds.get(i).smiles for i in tqdm(range(len(ds)), desc=split)]
        smiles = [
            s for s in (
                smiles_from_mol(Chem.MolFromSmiles(smi), canonical=True)
                for smi in raw if smi
            ) if s is not None
        ]
        write_smiles_file(smiles, smiles_path)


def generate_geom(root: str) -> None:
    from chemflow.dataset.geom import GEOM

    for split in ("train", "val", "test"):
        smiles_path = os.path.join(root, "processed", f"{split}_smiles.txt")

        ds = GEOM(root, split)
        raw = [ds.get(i).smiles for i in tqdm(range(len(ds)), desc=split)]
        smiles = [
            s for s in (
                smiles_from_mol(Chem.MolFromSmiles(smi), canonical=True)
                for smi in raw if smi
            ) if s is not None
        ]
        write_smiles_file(smiles, smiles_path)


@hydra.main(config_path="../configs", config_name="default", version_base="1.1")
def main(cfg: DictConfig) -> None:
    OmegaConf.set_struct(cfg, False)
    root = cfg.data.preprocessing.root

    target = cfg.data.datamodule.datasets.train._target_
    if "qm9" in target.lower():
        generate_qm9(root)
    elif "geom" in target.lower():
        generate_geom(root)
    else:
        raise ValueError(f"Unknown dataset target: {target}")


if __name__ == "__main__":
    main()
