from rdkit import Chem
from rdkit.Chem.rdDetermineBonds import DetermineBonds
from chemflow.utils import index_to_token
import torch

def tensors_to_rdkit_mol(atom_types: torch.Tensor, coords: torch.Tensor, token_list: list[str], infer_bonds: bool = True):
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
