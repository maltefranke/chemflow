"""A dataset for editing molecules."""

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import rdFMCS


def sort_by_scaffold(data_list):
    scaffold_groups = {}

    for idx, data in enumerate(data_list):
        smiles = data.smiles

        scaffold = Chem.MolFromSmiles(smiles)

        # get Bemis-Murcko scaffold
        scaffold = MurckoScaffold.GetScaffoldForMol(scaffold)

        if scaffold not in scaffold_groups:
            scaffold_groups[scaffold] = []

        scaffold_groups[scaffold].append(idx)

    return scaffold_groups


def get_mcs_atom_mapping(smiles1, smiles2):
    """
    Finds the Maximum Common Substructure between two SMILES strings
    and returns the atom mapping as a list of index pairs.
    """
    # 1. Convert SMILES to RDKit molecule objects
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)

    mol1 = Chem.AddHs(mol1)
    mol2 = Chem.AddHs(mol2)

    if not mol1 or not mol2:
        raise ValueError("One or both SMILES strings are invalid.")

    # 2. Find the Maximum Common Substructure (MCS)
    # Default parameters match exact atom types and bond types.
    mcs_result = rdFMCS.FindMCS([mol1, mol2])

    if not mcs_result.smartsString:
        return []  # No common structure found

    # 3. Create a query molecule from the MCS SMARTS string
    mcs_query = Chem.MolFromSmarts(mcs_result.smartsString)

    # 4. Find the matching atom indices in both molecules
    # GetSubstructMatch returns a tuple of atom indices.
    # The i-th index in the tuple corresponds to the i-th atom in the mcs_query.
    match1 = mol1.GetSubstructMatch(mcs_query)
    match2 = mol2.GetSubstructMatch(mcs_query)

    # 5. Zip the matches together to pair the corresponding atoms
    # This automatically excludes any atoms not present in the MCS.
    atom_mapping = list(zip(match1, match2))

    return atom_mapping
