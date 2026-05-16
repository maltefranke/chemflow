import io
import os
from pathlib import Path

import numpy as np
import torch
from chemflow.utils.utils import index_to_token

from chemflow.utils.utils import EdgeAligner

import py3Dmol
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D

from chemflow.utils.rdkit_utils import IDX_BOND_MAP as bond_mapping
from chemflow.dataset.molecule_data import IDX_BOND_MAP as MD_IDX_BOND_MAP
from chemflow.utils.repr import tensors_to_rdkit_mol

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


# ---------------------------------------------------------------------------
# Trajectory analysis: per-step n_atoms, ins/del classification, file loading
# ---------------------------------------------------------------------------


def load_trajectory_pt(path):
    """Load a trajectory .pt produced by eval_natoms_cfg.py / generate_uncond.py.

    Returns the dict as-saved (with ``valid_trajectories`` /
    ``invalid_trajectories`` etc.). Uses ``weights_only=False`` because the
    payload contains ``MoleculeData`` objects.
    """
    return torch.load(str(path), weights_only=False, map_location="cpu")


def natoms_per_step(traj) -> list[int]:
    """Per-step atom count for a single trajectory (list of MoleculeData)."""
    return [int(getattr(f, "num_nodes", f.a.shape[0])) for f in traj]


def classify_step_edits(natoms: list[int]) -> list[str]:
    """Per-step edit kind: ``"start"`` for step 0, then ``"ins"``/``"del"``/``"same"``."""
    out: list[str] = []
    for i, n in enumerate(natoms):
        if i == 0:
            out.append("start")
        elif n > natoms[i - 1]:
            out.append("ins")
        elif n < natoms[i - 1]:
            out.append("del")
        else:
            out.append("same")
    return out


def find_trajectories_with_ins_and_del(
    trajectories,
    min_ins: int = 1,
    min_del: int = 1,
) -> list[tuple[int, list[int], int, int]]:
    """Locate trajectories whose atom count both increases and decreases.

    Returns list of ``(idx, natoms, n_ins, n_del)`` tuples for trajectories
    satisfying both ``min_ins`` and ``min_del``.
    """
    hits: list[tuple[int, list[int], int, int]] = []
    for idx, traj in enumerate(trajectories):
        if not isinstance(traj, (list, tuple)) or len(traj) < 2:
            continue
        natoms = natoms_per_step(traj)
        n_ins = sum(1 for a, b in zip(natoms[:-1], natoms[1:]) if b > a)
        n_del = sum(1 for a, b in zip(natoms[:-1], natoms[1:]) if b < a)
        if n_ins >= min_ins and n_del >= min_del:
            hits.append((idx, natoms, n_ins, n_del))
    return hits


# ---------------------------------------------------------------------------
# RDKit conversion for intermediate trajectory frames (no sanitization)
# ---------------------------------------------------------------------------


def molecule_data_to_rdkit_unsanitized(
    frame,
    atom_tokens: list[str],
    edge_tokens: list[str],
    charge_tokens: list[str],
):
    """Build an RDKit mol from a ``MoleculeData`` without sanitization.

    Mirrors ``MoleculeData.to_rdkit_mol`` but skips ``Chem.SanitizeMol`` so
    intermediate trajectory frames (which often have non-physical valences)
    survive instead of being dropped.
    """
    a = frame.a.detach().cpu().numpy()
    x = frame.x.detach().cpu().numpy()
    c = frame.c.detach().cpu().numpy()
    atom_syms = [index_to_token(atom_tokens, int(i)) for i in a]
    charges = [int(index_to_token(charge_tokens, int(i))) for i in c]

    if frame.num_nodes <= 1:
        edge_types: list = []
        edge_index_list: list = []
    else:
        e_triu, edge_index_triu = frame.get_e_triu()
        e_arr = e_triu.detach().cpu().numpy()
        edge_index = edge_index_triu.detach().cpu().numpy().T.tolist()
        edge_tok_list = [index_to_token(edge_tokens, int(i)) for i in e_arr]
        edge_types = []
        edge_index_list = []
        for edge, et in zip(edge_index, edge_tok_list):
            if et in ("<NO_BOND>", "<MASK>"):
                continue
            if et not in MD_IDX_BOND_MAP:
                continue
            edge_types.append(MD_IDX_BOND_MAP[et])
            edge_index_list.append(edge)

    return tensors_to_rdkit_mol(
        atom_syms, x, charges, edge_types, edge_index_list, sanitize=False
    )


# ---------------------------------------------------------------------------
# GIF rendering helpers
# ---------------------------------------------------------------------------


DEFAULT_LINE_COLOR = "#6A00F4"


def composite_mol_on_plot(
    plot_img,
    mol_img,
    frac_w: float = 0.50,
    bottom_margin_frac: float = 0.05,
):
    """Paste ``mol_img`` (RGBA) onto ``plot_img`` (RGBA) at bottom-center."""
    from PIL import Image  # local import: heavy module, only needed for GIFs

    target_w = int(plot_img.width * frac_w)
    aspect = mol_img.width / mol_img.height
    target_h = int(target_w / aspect)
    mol_resized = mol_img.resize((target_w, target_h), Image.LANCZOS)
    x = (plot_img.width - target_w) // 2
    y = plot_img.height - target_h - int(plot_img.height * bottom_margin_frac)
    out = plot_img.copy()
    out.paste(mol_resized, (x, y), mol_resized)
    return out


def render_natoms_plot_frames(
    natoms_trace,
    plot_idxs,
    *,
    t_axis=None,
    target_n=None,
    scale=None,
    ylim=None,
    line_color: str = DEFAULT_LINE_COLOR,
    figsize=(5.5, 4.0),
    dpi: int = 120,
    title_fmt: str | None = None,
):
    """Render a per-step n_atoms plot to a list of PIL RGBA frames.

    Re-uses a single matplotlib figure (much faster than creating one per
    frame). ``natoms_trace`` is the full per-step series; ``plot_idxs`` is
    the subsample of step indices that should produce frames.
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    from PIL import Image

    natoms = np.asarray(natoms_trace, dtype=float)
    n_steps = len(natoms)
    if t_axis is None:
        t_axis = np.arange(n_steps) / max(n_steps - 1, 1)
    t_axis = np.asarray(t_axis, dtype=float)

    if ylim is None:
        clean = natoms[~np.isnan(natoms)]
        if clean.size == 0:
            ylo, yhi = 0, 1
        else:
            ylo = int(clean.min()) - 2
            yhi = int(max(clean.max(), target_n if target_n is not None else clean.max())) + 2
        ylim = (ylo, yhi)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.plot(
        t_axis, natoms, color=line_color, linewidth=1.0, alpha=0.18,
        drawstyle="steps-post",
    )
    (line,) = ax.plot(
        [], [], color=line_color, linewidth=1.8, drawstyle="steps-post",
        label=r"$n_\mathrm{atoms}$",
    )
    (marker,) = ax.plot(
        [], [], "o", color=line_color, markersize=6,
        markeredgecolor="black", markeredgewidth=0.6,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(*ylim)
    ax.set_xlabel(r"integration time $t$")
    ax.set_ylabel(r"$n_\mathrm{atoms}$")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(fontsize=8, loc="lower right", frameon=False)
    title = ax.set_title("")
    fig.tight_layout()
    fig.subplots_adjust(top=0.90)

    if title_fmt is None:
        if target_n is not None and scale is not None:
            title_fmt = "scale={scale}, target={target_n}, t={t:.2f}"
        elif target_n is not None:
            title_fmt = "target={target_n}, t={t:.2f}"
        elif scale is not None:
            title_fmt = "scale={scale}, t={t:.2f}"
        else:
            title_fmt = "t={t:.2f}"

    frames = []
    for raw_idx in plot_idxs:
        line.set_data(t_axis[: raw_idx + 1], natoms[: raw_idx + 1])
        cur = natoms[raw_idx]
        if np.isnan(cur):
            marker.set_data([], [])
        else:
            marker.set_data([t_axis[raw_idx]], [cur])
        title.set_text(
            title_fmt.format(scale=scale, target_n=target_n, t=t_axis[raw_idx])
        )
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        frames.append(Image.open(buf).convert("RGBA"))
    plt.close(fig)
    return frames


def per_frame_durations_with_fast_range(
    t_axis,
    plot_idxs,
    base_fps: int,
    fast_range: tuple[float, float] | None = None,
    fast_factor: float = 1.0,
) -> list[int]:
    """Per-frame GIF durations (ms) with optional speed-up over a t window.

    Frames whose ``t_axis[plot_idxs[i]]`` falls inside ``fast_range`` get
    ``base_dur / fast_factor`` ms; the rest get ``base_dur = 1000 / base_fps``.
    """
    base_dur = max(1, int(1000 / max(base_fps, 1)))
    if fast_range is None or fast_factor <= 1.0:
        return [base_dur] * len(plot_idxs)
    fast_dur = max(1, int(base_dur / fast_factor))
    lo, hi = fast_range
    return [
        fast_dur if lo <= t_axis[idx] <= hi else base_dur for idx in plot_idxs
    ]


def map_images_to_steps(n_steps: int, n_imgs: int) -> list[int]:
    """Map a sequence of ``n_imgs`` images to step indices in a length-``n_steps`` trace."""
    if n_imgs <= 0:
        return []
    if n_steps % n_imgs == 0:
        stride = n_steps // n_imgs
        return list(range(0, n_steps, stride))
    return [round(i * (n_steps - 1) / max(n_imgs - 1, 1)) for i in range(n_imgs)]


def write_gif(out_path, frames, durations_ms, loop: int = 0, optimize: bool = False):
    """Write a list of PIL frames to a GIF (per-frame durations supported)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        out_path, save_all=True, append_images=frames[1:],
        duration=durations_ms, loop=loop, optimize=optimize,
    )
    return out_path
