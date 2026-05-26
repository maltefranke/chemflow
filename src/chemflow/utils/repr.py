import numpy as np
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")


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
        # DetermineBonds(mol)
        for atom in mol.GetAtoms():
            atom.UpdatePropertyCache(strict=False)
    except Exception:
        return None

    if sanitize:
        err = Chem.SanitizeMol(mol, catchErrors=True)
        if err:
            return None

        try:
            for atom in mol.GetAtoms():
                atom.SetNoImplicit(True) # Ensure no "ghost" Hs are expected
                atom.UpdatePropertyCache()
        except Exception:
            return None

    return mol
