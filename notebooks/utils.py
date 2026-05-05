import torch
import os
from chemflow.utils.utils import index_to_token

from chemflow.utils.utils import EdgeAligner

import py3Dmol
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D

from chemflow.utils.rdkit_utils import IDX_BOND_MAP as bond_mapping

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


def visualize_mols_side_by_side(mol_data_list, width=800, height=400, gap=4.0):
    """
    One static py3Dmol scene with multiple molecules laid out along +X (no animation).

    Args:
        mol_data_list: List of dicts in the same format as ``process_mol`` / ``visualize_single_mol``.
        gap: Extra spacing (Angstrom) between molecular bounding boxes along X.
    """
    if not mol_data_list:
        raise ValueError("mol_data_list is empty")

    view = py3Dmol.view(width=width, height=height)
    x_cursor = 0.0

    for mol_data in mol_data_list:
        xs = [float(p[0]) for p in mol_data["pos"]]
        x_min, x_max = min(xs), max(xs)
        width_mol = max(x_max - x_min, 1e-3)
        dx = x_cursor - x_min

        mol = Chem.RWMol()

        for i, symbol in enumerate(mol_data["atoms"]):
            atom = Chem.Atom(symbol)
            atom.SetFormalCharge(int(mol_data["charges"][i]))
            mol.AddAtom(atom)

        for (src, dst), b_type in zip(mol_data["edges"], mol_data["edge_types"]):
            rdkit_type = bond_mapping.get(b_type, Chem.BondType.SINGLE)
            mol.AddBond(int(src), int(dst), rdkit_type)

        conf = Chem.Conformer(len(mol_data["atoms"]))
        for i, (x, y, z) in enumerate(mol_data["pos"]):
            conf.SetAtomPosition(i, (float(x) + dx, float(y), float(z)))
        mol.AddConformer(conf)

        mol_block = _mol_to_mol_block_with_single_bond_fallback(mol)
        view.addModel(mol_block, "sdf")
        x_cursor += width_mol + gap

    view.setStyle({"stick": {"radius": 0.15}, "sphere": {"scale": 0.2}})
    view.zoomTo()

    return view


def visualize_mols_grid(mol_data_list, width=800, height=600, num_cols=8, gap=4.0):
    """
    One static py3Dmol scene with multiple molecules laid out in an XY grid.

    Args:
        mol_data_list: List of dicts in the same format as ``process_mol`` / ``visualize_single_mol``.
        num_cols: Number of molecules per row.
        gap: Extra spacing (Angstrom) between grid cells along X and Y.
    """
    if not mol_data_list:
        raise ValueError("mol_data_list is empty")
    if num_cols < 1:
        raise ValueError("num_cols must be >= 1")

    bounds = []
    for mol_data in mol_data_list:
        xs = [float(p[0]) for p in mol_data["pos"]]
        ys = [float(p[1]) for p in mol_data["pos"]]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        bounds.append((x_min, x_max, y_min, y_max))

    cell_width = max(max(x_max - x_min, 1e-3) for x_min, x_max, _, _ in bounds) + gap
    cell_height = max(max(y_max - y_min, 1e-3) for _, _, y_min, y_max in bounds) + gap

    view = py3Dmol.view(width=width, height=height)

    for idx, (mol_data, (x_min, x_max, y_min, y_max)) in enumerate(zip(mol_data_list, bounds)):
        row, col = divmod(idx, num_cols)
        target_x = col * cell_width
        target_y = -row * cell_height
        dx = target_x - (x_min + x_max) / 2
        dy = target_y - (y_min + y_max) / 2

        mol = Chem.RWMol()

        for i, symbol in enumerate(mol_data["atoms"]):
            atom = Chem.Atom(symbol)
            atom.SetFormalCharge(int(mol_data["charges"][i]))
            mol.AddAtom(atom)

        for (src, dst), b_type in zip(mol_data["edges"], mol_data["edge_types"]):
            rdkit_type = bond_mapping.get(b_type, Chem.BondType.SINGLE)
            mol.AddBond(int(src), int(dst), rdkit_type)

        conf = Chem.Conformer(len(mol_data["atoms"]))
        for i, (x, y, z) in enumerate(mol_data["pos"]):
            conf.SetAtomPosition(i, (float(x) + dx, float(y) + dy, float(z)))
        mol.AddConformer(conf)

        mol_block = _mol_to_mol_block_with_single_bond_fallback(mol)
        view.addModel(mol_block, "sdf")

    view.setStyle({"stick": {"radius": 0.15}, "sphere": {"scale": 0.2}})
    view.zoomTo()

    return view


def _frame_to_mol(frame):
    """Build an RDKit RWMol (with 3D conformer) from a single trajectory frame."""
    mol = Chem.RWMol()

    for i, symbol in enumerate(frame["atoms"]):
        atom = Chem.Atom(symbol)
        atom.SetFormalCharge(int(frame["charges"][i]))
        mol.AddAtom(atom)

    for (src, dst), b_type in zip(frame["edges"], frame["edge_types"]):
        rdkit_type = bond_mapping.get(b_type, Chem.BondType.SINGLE)
        mol.AddBond(int(src), int(dst), rdkit_type)

    conf = Chem.Conformer(len(frame["atoms"]))
    for i, (x, y, z) in enumerate(frame["pos"]):
        conf.SetAtomPosition(i, (float(x), float(y), float(z)))
    mol.AddConformer(conf)

    return mol


def _frame_to_mol_block(frame):
    """Build an SDF MolBlock string for a single trajectory frame."""
    return _mol_to_mol_block_with_single_bond_fallback(_frame_to_mol(frame))


def visualize_variable_topology(
    trajectory_frames,
    width=800,
    height=400,
    interval=25,
    slider=False,
):
    """
    Visualizes a trajectory where atom counts and types change.

    Args:
        trajectory_frames (list): A list of dicts. Each dict must have:
            - 'atoms': List of symbols (e.g., ['C', 'H'])
            - 'pos': List of [x, y, z] coordinates
            - 'edges': List of (start_idx, end_idx) tuples
            - 'edge_types': List of bond orders (1, 2, 1.5, etc.)
        width (int): Viewer width in pixels.
        height (int): Viewer height in pixels.
        interval (int): Animation speed in ms/frame (ignored when slider=True).
        slider (bool): If True, renders an interactive ipywidgets slider to
            scrub through frames instead of auto-playing the animation.
            Returns the resulting ipywidgets widget. Requires ipywidgets and
            a Jupyter-like frontend.
    """

    mol_blocks = [_frame_to_mol_block(frame) for frame in trajectory_frames]

    if slider:
        import json
        import uuid

        from IPython.display import HTML, display

        combined_sdf_string = "".join(block + "$$$$\n" for block in mol_blocks)
        num_frames = len(mol_blocks)
        uid = uuid.uuid4().hex[:8]
        sdf_json = json.dumps(combined_sdf_string)

        html_str = f"""
<div style="font-family: sans-serif;">
  <div id="viewer_{uid}"
       style="width: {width}px; height: {height}px; position: relative;"></div>
  <div style="width: {width}px; display: flex; align-items: center; gap: 8px; margin-top: 6px;">
    <span>Frame:
      <span id="label_{uid}" style="display:inline-block; min-width: 3em;">0</span>
      / {max(num_frames - 1, 0)}
    </span>
    <input type="range" id="slider_{uid}"
           min="0" max="{max(num_frames - 1, 0)}" value="0" step="1"
           style="flex: 1;">
  </div>
</div>
<script>
(function() {{
  function initViewer() {{
    var viewer = $3Dmol.createViewer("viewer_{uid}", {{backgroundColor: "white"}});
    var sdfData = {sdf_json};
    viewer.addModelsAsFrames(sdfData, "sdf");
    viewer.setStyle({{}}, {{stick: {{radius: 0.15}}, sphere: {{scale: 0.2}}}});
    viewer.zoomTo();
    viewer.setFrame(0);
    viewer.render();
    document.getElementById("slider_{uid}").addEventListener("input", function(e) {{
      var idx = parseInt(e.target.value);
      document.getElementById("label_{uid}").textContent = idx;
      viewer.setFrame(idx);
      viewer.render();
    }});
  }}
  if (typeof $3Dmol !== "undefined") {{
    initViewer();
  }} else {{
    var s = document.createElement("script");
    s.src = "https://3Dmol.csb.pitt.edu/build/3Dmol-min.js";
    s.onload = initViewer;
    document.head.appendChild(s);
  }}
}})();
</script>
"""

        class _SliderView:
            """Thin wrapper so callers can use `.show()` like a py3Dmol view."""

            def __init__(self, html_body):
                self._html = html_body

            def show(self):
                display(HTML(self._html))

            def _ipython_display_(self):
                display(HTML(self._html))

        return _SliderView(html_str)

    combined_sdf_string = "".join(block + "$$$$\n" for block in mol_blocks)

    view = py3Dmol.view(width=width, height=height)
    view.addModelsAsFrames(combined_sdf_string, "sdf")
    view.setStyle({"stick": {"radius": 0.15}, "sphere": {"scale": 0.2}})
    view.animate({"loop": "forward", "interval": interval})
    view.zoomTo()

    return view
