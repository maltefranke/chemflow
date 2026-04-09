"""Visualise the predicted insertion density around a perturbed molecule.

Usage (standalone — saves .cube + .xyz + _gmm.pt files):
    python scripts/probability_hull.py --ckpt <path> --output_dir hulls/

Workflow:
    1. Load QM9 val data and vocab/distributions via Hydra preprocessing.
    2. Pick a few molecules, remove one atom each ("perturb").
    3. Build a MoleculeBatch at t = 1 - eps and run the model forward.
    4. Read the GMM + insertion-rate predictions.
    5. Evaluate the GMM density on a 3D voxel grid.
    6. Save .cube (density), .xyz (molecule), and _gmm.pt (raw GMM modes)
       files to disk.
    7. In Jupyter: load _gmm.pt, filter top-N modes, visualise with py3Dmol.
"""

import os
import sys
from copy import deepcopy

_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), ".."),
)
for _p in [_PROJECT_ROOT, os.path.join(_PROJECT_ROOT, "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

from chemflow.dataset.molecule_data import MoleculeBatch, MoleculeData
from chemflow.utils.utils import index_to_token, init_uniform_prior

DEVICE = "cpu"

OmegaConf.register_new_resolver("oc.eval", eval, replace=True)
OmegaConf.register_new_resolver(
    "len",
    lambda x: len(x),
    replace=True,
)
OmegaConf.register_new_resolver(
    "if",
    lambda cond, t, f: t if cond else f,
    replace=True,
)
OmegaConf.register_new_resolver(
    "eq",
    lambda x, y: x == y,
    replace=True,
)


# ── 1. Perturbation ──────────────────────────────────────────────────────


def perturb_molecule(
    mol: MoleculeData,
) -> tuple[MoleculeData, int]:
    """Remove one random atom from *mol*.

    Returns:
        (perturbed_mol, removed_index)
    """
    N = mol.num_nodes
    if N <= 1:
        raise ValueError("Cannot perturb a molecule with <= 1 atom")

    idx = torch.randint(0, N, (1,)).item()

    keep_mask = torch.ones(N, dtype=torch.bool)
    keep_mask[idx] = False
    keep_indices = torch.where(keep_mask)[0]

    new_x = mol.x[keep_indices]
    new_a = mol.a[keep_indices]
    new_c = mol.c[keep_indices]

    mapping = torch.full((N,), -1, dtype=torch.long)
    mapping[keep_indices] = torch.arange(len(keep_indices))

    if mol.edge_index is not None and mol.edge_index.numel() > 0:
        edge_mask = keep_mask[mol.edge_index[0]] & keep_mask[mol.edge_index[1]]
        new_edge_index = mapping[mol.edge_index[:, edge_mask]]
        new_e = mol.e[edge_mask] if mol.e is not None else None
    else:
        new_edge_index = torch.empty((2, 0), dtype=torch.long)
        new_e = torch.empty(0, dtype=torch.long)

    return MoleculeData(
        x=new_x,
        a=new_a,
        c=new_c,
        e=new_e,
        edge_index=new_edge_index,
    ), idx


# ── 2. Density evaluation ────────────────────────────────────────────────


def compute_probability_grid(
    mu,
    sigma,
    pi,
    lambda_ins,
    grid_res=0.5,
    margin=3.0,
):
    """Evaluate the GMM density on a 3D voxel grid.

    Args:
        mu:         [N, K, 3]  component means
        sigma:      [N, K]     component std-devs
        pi:         [N, K]     component mixing weights
        lambda_ins: [N, 1]     predicted insertion rate per node
        grid_res:   voxel edge length
        margin:     bounding-box padding
    """
    N, K, _ = mu.shape
    device = mu.device

    min_coords = mu.view(-1, 3).min(dim=0)[0] - margin
    max_coords = mu.view(-1, 3).max(dim=0)[0] + margin

    x_r = torch.arange(
        min_coords[0],
        max_coords[0],
        grid_res,
        device=device,
    )
    y_r = torch.arange(
        min_coords[1],
        max_coords[1],
        grid_res,
        device=device,
    )
    z_r = torch.arange(
        min_coords[2],
        max_coords[2],
        grid_res,
        device=device,
    )

    gx, gy, gz = torch.meshgrid(x_r, y_r, z_r, indexing="ij")
    grid_pts = torch.stack(
        [gx.flatten(), gy.flatten(), gz.flatten()],
        dim=1,
    )

    mu_flat = mu.view(N * K, 3)
    sigma_flat = sigma.view(N * K)
    weights_flat = (lambda_ins * pi).view(N * K)

    dist_sq = torch.cdist(grid_pts, mu_flat, p=2).pow(2)
    var = (sigma_flat**2).unsqueeze(0)
    norm = (2 * np.pi * var) ** (-1.5)
    pdf_vals = norm * torch.exp(-0.5 * dist_sq / var)

    grid_density = torch.sum(
        pdf_vals * weights_flat.unsqueeze(0),
        dim=1,
    )
    density_3d = grid_density.view(len(x_r), len(y_r), len(z_r))

    return density_3d.cpu().numpy(), (
        min_coords.cpu().numpy(),
        max_coords.cpu().numpy(),
    )


# ── 3. File I/O (.cube + .xyz) ───────────────────────────────────────────

_ATOMIC_NUMBERS = {
    "H": 1,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "S": 16,
    "Cl": 17,
    "Br": 35,
}

_BOHR = 1.8897259886  # Angstrom -> Bohr


def write_cube(
    path,
    density_3d,
    origin,
    grid_res,
    atom_symbols,
    coords,
):
    """Write a Gaussian .cube file (density in Angstrom, stored as Bohr)."""
    nx, ny, nz = density_3d.shape
    n_atoms = len(atom_symbols)

    with open(path, "w") as f:
        f.write("Probability hull density\n")
        f.write("Generated by chemflow\n")
        f.write(
            f"{n_atoms:5d}"
            f" {origin[0] * _BOHR:12.6f}"
            f" {origin[1] * _BOHR:12.6f}"
            f" {origin[2] * _BOHR:12.6f}\n"
        )
        f.write(f"{nx:5d} {grid_res * _BOHR:12.6f} {0.0:12.6f} {0.0:12.6f}\n")
        f.write(f"{ny:5d} {0.0:12.6f} {grid_res * _BOHR:12.6f} {0.0:12.6f}\n")
        f.write(f"{nz:5d} {0.0:12.6f} {0.0:12.6f} {grid_res * _BOHR:12.6f}\n")

        for i in range(n_atoms):
            z_num = _ATOMIC_NUMBERS.get(atom_symbols[i], 0)
            cx, cy, cz = coords[i] * _BOHR
            f.write(f"{z_num:5d} {0.0:12.6f} {cx:12.6f} {cy:12.6f} {cz:12.6f}\n")

        for ix in range(nx):
            for iy in range(ny):
                row = density_3d[ix, iy, :]
                for iz in range(0, nz, 6):
                    chunk = row[iz : iz + 6]
                    f.write(" ".join(f"{v:13.5E}" for v in chunk) + "\n")


def write_xyz(path, atom_symbols, coords):
    """Write an .xyz file (coordinates in Angstrom)."""
    with open(path, "w") as f:
        f.write(f"{len(atom_symbols)}\n")
        f.write("Generated by chemflow\n")
        for sym, (cx, cy, cz) in zip(
            atom_symbols,
            coords,
            strict=True,
        ):
            f.write(f"{sym}  {cx:.6f}  {cy:.6f}  {cz:.6f}\n")


def save_gmm_modes(
    path,
    mu,
    sigma,
    pi,
    lambda_ins,
    coord_std,
    atom_symbols,
    atom_coords,
    removed_atom_coord,
    removed_atom_symbol,
):
    """Save flattened GMM modes, molecule info, and ground truth to a .pt file.

    Saved tensors (all in real / Angstrom space):
        mu_real              : [N*K, 3]  mode centres
        sigma_real           : [N*K]     mode std-devs
        weights              : [N*K]     effective weight  (lambda_ins * pi)
        atom_coords          : [M, 3]    molecule atom positions
        atom_symbols         : list[str] element symbols
        removed_atom_coord   : [3]       ground-truth removed atom position
        removed_atom_symbol  : str       ground-truth removed atom element
    """
    N, K, _ = mu.shape
    mu_real = mu.view(N * K, 3).cpu() * coord_std
    sigma_real = sigma.view(N * K).cpu() * coord_std
    weights = (lambda_ins * pi).view(N * K).cpu()

    torch.save(
        {
            "mu": mu_real,
            "sigma": sigma_real,
            "weights": weights,
            "atom_coords": torch.as_tensor(atom_coords, dtype=torch.float32),
            "atom_symbols": list(atom_symbols),
            "removed_atom_coord": torch.as_tensor(
                removed_atom_coord,
                dtype=torch.float32,
            ),
            "removed_atom_symbol": removed_atom_symbol,
        },
        path,
    )


# ── 5. Model loading & inference helpers ──────────────────────────────────


def load_config_and_preprocessing(
    config_path="../configs",
    config_name="default",
):
    """Use Hydra compose API to load the config + preprocessing."""
    rel = os.path.relpath(config_path, os.getcwd())
    with hydra.initialize(config_path=rel, version_base="1.1"):
        cfg = hydra.compose(config_name=config_name)

    OmegaConf.set_struct(cfg, False)

    preprocessing = hydra.utils.instantiate(cfg.data.preprocessing)
    vocab = preprocessing.vocab
    distributions = preprocessing.distributions
    token_prior = init_uniform_prior(distributions)

    cfg.data.vocab = vocab
    return cfg, vocab, distributions, token_prior


def load_model(cfg, vocab, distributions, token_prior, ckpt_path):
    """Instantiate the LightningModule and load a checkpoint."""
    from chemflow.dataset.vocab import setup_token_weights

    loss_weight_distributions = deepcopy(distributions)

    tw = cfg.model.token_weighting
    (
        atom_type_weights,
        edge_token_weights,
        charge_token_weights,
    ) = setup_token_weights(
        vocab=vocab,
        distributions=loss_weight_distributions,
        weight_alpha=tw.weight_alpha,
        type_loss_token_weights=tw.type_loss_token_weights,
    )

    module = hydra.utils.instantiate(
        cfg.model.module,
        _recursive_=False,
        distributions=token_prior,
        loss_weight_distributions=loss_weight_distributions,
        atom_type_weights=atom_type_weights,
        edge_token_weights=edge_token_weights,
        charge_token_weights=charge_token_weights,
    )

    ckpt = torch.load(
        ckpt_path,
        map_location="cpu",
        weights_only=False,
    )
    state_dict = ckpt["state_dict"]
    state_dict = {k.replace("._orig_mod", ""): v for k, v in state_dict.items()}
    module.load_state_dict(state_dict)
    module.eval()
    return module


def load_val_molecules(
    cfg,
    vocab,
    distributions,
    n_mols=4,
):
    """Instantiate QM9 val dataset and return *n_mols* items."""
    from chemflow.dataset.qm9 import FlowMatchingQM9Dataset

    val_cfg = cfg.data.datamodule.datasets.val
    if OmegaConf.is_list(val_cfg) or isinstance(val_cfg, (list, tuple)):
        val_cfg = val_cfg[0]

    val_cfg = OmegaConf.to_container(val_cfg, resolve=True)
    root = val_cfg["root"]

    ds = FlowMatchingQM9Dataset(
        root=root,
        vocab=vocab,
        distributions=distributions,
        split="val",
        rotate=False,
    )
    indices = list(range(min(n_mols, len(ds))))
    return [ds[i] for i in indices]


@torch.no_grad()
def predict_insertion_density(
    module,
    mol_t_batch,
    t_value=0.65,
):
    """Run model forward and return GMM params + insertion rates."""
    model = module._get_model()
    model.eval()

    bs = mol_t_batch.num_graphs
    t = torch.full((bs,), t_value, device=DEVICE)

    preds = model(
        mol_t_batch,
        t.view(-1, 1),
        cfg_inputs={},
    )
    return preds["gmm_head"], preds["ins_rate_head"]


# ── 6. End-to-end pipeline (saves to disk) ────────────────────────────────


def run_probability_hull(
    ckpt_path: str,
    n_mols: int = 4,
    grid_res: float = 0.3,
    margin: float = 3.0,
    output_dir: str = "hulls",
):
    """Load data, perturb, predict, and save .cube + .xyz + _gmm.pt files.

    Creates ``output_dir/mol_<i>.{cube,xyz,_gmm.pt}`` for each molecule.
    The ``_gmm.pt`` file contains the flattened GMM modes (mu, sigma,
    effective weights) for downstream visualisation in Jupyter.
    """
    os.makedirs(output_dir, exist_ok=True)

    cfg, vocab, distributions, token_prior = load_config_and_preprocessing()
    module = load_model(
        cfg,
        vocab,
        distributions,
        token_prior,
        ckpt_path,
    )
    module.to(DEVICE)

    molecules = load_val_molecules(
        cfg,
        vocab,
        distributions,
        n_mols=n_mols,
    )

    perturbed = []
    removed_indices = []
    for mol in molecules:
        p, idx = perturb_molecule(mol)
        perturbed.append(p)
        removed_indices.append(idx)

    mol_t_batch = MoleculeBatch.from_data_list(perturbed)
    mol_t_batch = mol_t_batch.to(DEVICE)

    gmm, ins_rate = predict_insertion_density(module, mol_t_batch)

    coord_std = distributions.coordinate_std

    data_list = mol_t_batch.to_data_list()
    node_offset = 0
    saved = []
    for i, mol_data in enumerate(data_list):
        n_nodes = mol_data.num_nodes
        sl = slice(node_offset, node_offset + n_nodes)

        mu_i = gmm["mu"][sl]
        sigma_i = gmm["sigma"][sl]
        pi_i = gmm["pi"][sl]
        rate_i = ins_rate[sl]
        if rate_i.ndim == 1:
            rate_i = rate_i.unsqueeze(-1)

        density_3d, (origin, _) = compute_probability_grid(
            mu_i,
            sigma_i,
            pi_i,
            rate_i,
            grid_res=grid_res,
            margin=margin,
        )

        coords_real = mol_data.x.cpu().numpy() * coord_std
        atom_symbols = [
            index_to_token(vocab.atom_tokens, int(a_idx))
            for a_idx in mol_data.a.cpu().numpy()
        ]
        origin_real = origin * coord_std
        grid_res_real = grid_res * coord_std

        xyz_path = os.path.join(output_dir, f"mol_{i}.xyz")
        cube_path = os.path.join(output_dir, f"mol_{i}.cube")
        gmm_path = os.path.join(output_dir, f"mol_{i}_gmm.pt")

        write_xyz(xyz_path, atom_symbols, coords_real)
        write_cube(
            cube_path,
            density_3d,
            origin_real,
            grid_res_real,
            atom_symbols,
            coords_real,
        )
        orig_mol = molecules[i]
        rm_idx = removed_indices[i]
        removed_coord = orig_mol.x[rm_idx].cpu().numpy() * coord_std
        removed_symbol = index_to_token(
            vocab.atom_tokens,
            int(orig_mol.a[rm_idx].item()),
        )

        save_gmm_modes(
            gmm_path,
            mu_i,
            sigma_i,
            pi_i,
            rate_i,
            coord_std,
            atom_symbols,
            coords_real,
            removed_coord,
            removed_symbol,
        )

        saved.append((xyz_path, cube_path, gmm_path))
        print(
            f"[{i}] {n_nodes} atoms "
            f"(removed #{removed_indices[i]}) -> "
            f"{xyz_path}, {cube_path}, {gmm_path}"
        )

        node_offset += n_nodes

    return saved


# ── 7. CLI entry point ───────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Probability hull: save .cube + .xyz",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--n_mols",
        type=int,
        default=4,
        help="Number of val molecules",
    )
    parser.add_argument(
        "--grid_res",
        type=float,
        default=0.3,
        help="Voxel resolution (Angstrom)",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=3.0,
        help="Bounding-box margin (Angstrom)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="hulls",
        help="Output directory for .cube + .xyz files",
    )
    args = parser.parse_args()

    run_probability_hull(
        ckpt_path=args.ckpt,
        n_mols=args.n_mols,
        grid_res=args.grid_res,
        margin=args.margin,
        output_dir=args.output_dir,
    )
