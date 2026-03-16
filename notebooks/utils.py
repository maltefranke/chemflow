import torch
import os
from chemflow.utils.utils import index_to_token

from chemflow.utils.utils import EdgeAligner

import py3Dmol
from rdkit import Chem

from chemflow.utils.rdkit import IDX_BOND_MAP as bond_mapping

_KEKULIZE_EXCEPTIONS = tuple(
    exc
    for exc in (
        getattr(Chem, "AtomKekulizeException", None),
        getattr(Chem, "KekulizeException", None),
    )
    if exc is not None
)


def _mol_to_mol_block_with_single_bond_fallback(mol):
    """Export a MolBlock, downgrading aromatic bonds to single on kekulize errors."""
    try:
        return Chem.MolToMolBlock(mol)
    except _KEKULIZE_EXCEPTIONS:
        fallback_mol = Chem.RWMol(mol)

        for atom in fallback_mol.GetAtoms():
            atom.SetIsAromatic(False)

        for bond in fallback_mol.GetBonds():
            if bond.GetBondType() == Chem.BondType.AROMATIC or bond.GetIsAromatic():
                bond.SetBondType(Chem.BondType.SINGLE)
            bond.SetIsAromatic(False)

        return Chem.MolToMolBlock(fallback_mol, kekulize=False)


def load_mols(root_path: str, filename: str):
    path = os.path.join(root_path, filename)
    mols = torch.load(path, weights_only=False)
    return mols


def load_tokens(root_path: str):
    atom_tokens = os.path.join(root_path, "atom_tokens.txt")
    edge_tokens = os.path.join(root_path, "edge_tokens.txt")
    charge_tokens = os.path.join(root_path, "charge_tokens.txt")

    atom_tokens = open(atom_tokens).read().splitlines()
    edge_tokens = open(edge_tokens).read().splitlines()
    charge_tokens = open(charge_tokens).read().splitlines()

    distributions_path = os.path.join(root_path, "distributions.pt")
    distributions = torch.load(distributions_path)

    return atom_tokens, edge_tokens, charge_tokens, distributions


def process_mol(mol, atom_tokens, charge_tokens, edge_tokens):
    edge_aligner = EdgeAligner()

    x = mol.x
    a = mol.a.tolist()
    a = [index_to_token(atom_tokens, i) for i in a]

    c = mol.c.tolist()
    c = [index_to_token(charge_tokens, i) for i in c]

    edge_infos = edge_aligner.align_edges(
        source_group=(mol.edge_index, [mol.e]),
    )
    edge_index_triu, e_pred_triu = (
        edge_infos["edge_index"],
        edge_infos["edge_attr"][0],
    )

    edge_index = edge_index_triu.T.tolist()
    e = e_pred_triu.tolist()
    e = [index_to_token(edge_tokens, i) for i in e]

    e_sanitized = []
    edge_index_sanitized = []
    # remove edges with <NO_BOND> or <MASK>
    for i, edge in enumerate(e):
        if edge == "<NO_BOND>" or edge == "<MASK>":
            continue
        e_sanitized.append(int(edge))
        edge_index_sanitized.append(edge_index[i])

    return {
        "atoms": a,
        "pos": x,
        "edges": edge_index_sanitized,
        "edge_types": e_sanitized,
        "charges": c,
    }


def visualize_single_mol(mol_data, width=800, height=400):
    """
    Visualizes a single molecule.

    Args:
        mol_data (dict): A dictionary representing one molecule/frame. Must have:
            - 'atoms': List of symbols (e.g., ['C', 'H'])
            - 'charges': List of formal charges
            - 'pos': List of [x, y, z] coordinates
            - 'edges': List of (start_idx, end_idx) tuples
            - 'edge_types': List of bond orders (1, 2, 1.5, etc.)
    """

    mol = Chem.RWMol()

    # A. Add Atoms and Charges
    for i, symbol in enumerate(mol_data["atoms"]):
        atom = Chem.Atom(symbol)
        atom.SetFormalCharge(int(mol_data["charges"][i]))
        mol.AddAtom(atom)

    # B. Add Bonds
    for (src, dst), b_type in zip(mol_data["edges"], mol_data["edge_types"]):
        rdkit_type = bond_mapping.get(b_type, Chem.BondType.SINGLE)
        mol.AddBond(int(src), int(dst), rdkit_type)

    # C. Set Positions (Conformer)
    conf = Chem.Conformer(len(mol_data["atoms"]))
    for i, (x, y, z) in enumerate(mol_data["pos"]):
        conf.SetAtomPosition(i, (float(x), float(y), float(z)))
    mol.AddConformer(conf)

    # D. Convert to single MolBlock
    mol_block = _mol_to_mol_block_with_single_bond_fallback(mol)

    # E. Visualize with py3Dmol
    view = py3Dmol.view(width=width, height=height)

    # Use addModel instead of addModelsAsFrames for a single structure
    view.addModel(mol_block, "sdf")

    # Styling
    view.setStyle({"stick": {"radius": 0.15}, "sphere": {"scale": 0.2}})
    view.zoomTo()

    return view


def visualize_variable_topology(trajectory_frames, width=800, height=400, interval=25):
    """
    Visualizes a trajectory where atom counts and types change.

    Args:
        trajectory_frames (list): A list of dicts. Each dict must have:
            - 'atoms': List of symbols (e.g., ['C', 'H'])
            - 'pos': List of [x, y, z] coordinates
            - 'edges': List of (start_idx, end_idx) tuples
            - 'edge_types': List of bond orders (1, 2, 1.5, etc.)
    """

    combined_sdf_string = ""

    # 1. Iterate over every frame to build distinct molecules
    for frame in trajectory_frames:
        mol = Chem.RWMol()

        # A. Add Atoms
        for i, symbol in enumerate(frame["atoms"]):
            atom = Chem.Atom(symbol)
            atom.SetFormalCharge(int(frame["charges"][i]))
            mol.AddAtom(atom)

        # B. Add Bonds
        # We loop through edges and edge_types simultaneously
        for (src, dst), b_type in zip(frame["edges"], frame["edge_types"]):
            rdkit_type = bond_mapping.get(b_type, Chem.BondType.SINGLE)
            mol.AddBond(int(src), int(dst), rdkit_type)

        # C. Set Positions (Conformer)
        conf = Chem.Conformer(len(frame["atoms"]))
        for i, (x, y, z) in enumerate(frame["pos"]):
            conf.SetAtomPosition(i, (float(x), float(y), float(z)))

        mol.AddConformer(conf)

        # D. Convert to SDF Block and append to master string
        # '$$$$' is the delimiter between molecules in an SDF file
        mol_block = _mol_to_mol_block_with_single_bond_fallback(mol)
        combined_sdf_string += mol_block + "$$$$\n"

    # 2. Visualize with py3Dmol
    view = py3Dmol.view(width=width, height=height)

    # Load the multi-molecule string
    view.addModelsAsFrames(combined_sdf_string, "sdf")

    # 3. Styling
    view.setStyle({"stick": {"radius": 0.15}, "sphere": {"scale": 0.2}})

    # 4. Animation Settings
    # 'interval' controls speed (ms per frame)
    view.animate({"loop": "forward", "interval": interval})
    view.zoomTo()

    return view
