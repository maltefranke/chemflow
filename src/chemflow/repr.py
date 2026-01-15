from rdkit import Chem
from rdkit.Chem.rdDetermineBonds import DetermineBonds
from chemflow.utils import index_to_token
import torch
import numpy as np


def tensors_to_rdkit_mol_old(
    atom_types: torch.Tensor,
    coords: torch.Tensor,
    token_list: list[str],
    infer_bonds: bool = True,
):
    tokens = [index_to_token(token_list, index) for index in atom_types]
    coords = coords.detach().numpy()
    mol = Chem.EditableMol(Chem.Mol())

    for idx, atomic in enumerate(tokens):
        atom = Chem.Atom(atomic)
        mol.AddAtom(atom)
    mol = mol.GetMol()
    # Add 3D coords
    conf = Chem.Conformer(len(tokens))
    for i, xyz in enumerate(coords):
        conf.SetAtomPosition(i, tuple(map(float, xyz)))
    mol.AddConformer(conf, assignId=True)
    try:
        for atom in mol.GetAtoms():
            atom.UpdatePropertyCache(strict=False)
    except Exception:
        return None
    if infer_bonds:
        DetermineBonds(mol)
    return mol


def tensors_to_rdkit_mol(
    atom_types: list[str],
    coords: np.ndarray,
    charge_types: list[int],
    edge_types: list[Chem.BondType],
    edge_index: list[tuple[int, int]],
    sanitize: bool = True,
):
    mol = Chem.EditableMol(Chem.Mol())

    # Add atoms with formal charges
    for idx, atomic in enumerate(atom_types):
        atom = Chem.Atom(atomic)
        charge = charge_types[idx]
        atom.SetFormalCharge(charge)
        mol.AddAtom(atom)

    mol = mol.GetMol()

    # Add 3D coords
    conf = Chem.Conformer(len(atom_types))

    for i, xyz in enumerate(coords):
        conf.SetAtomPosition(i, tuple(map(float, xyz)))

    mol.AddConformer(conf, assignId=True)

    mol = Chem.EditableMol(mol)
    for edge, edge_type in zip(edge_index, edge_types):
        start, end = edge
        mol.AddBond(start, end, edge_type)

    try:
        mol = mol.GetMol()
        for atom in mol.GetAtoms():
            atom.UpdatePropertyCache(strict=False)
    except Exception as e:
        print(e)
        return None

    if sanitize:
        try:
            Chem.SanitizeMol(mol)
        except Exception as e:
            print(e)
            return None

    return mol
