"""Species-conditional insertion-density probe on test/val molecules.

For each dataset molecule, we forward the model at a (late) time t and
read the per-node, per-component GMM head plus the per-node insertion
rate.  Each Gaussian component carries its own predicted *atom-type* and
*charge* distribution, which lets us decompose the predicted insertion
density by atomic species (e.g. H+, C, O, ...). This is conceptually
analogous to Fukui-style site reactivity indices.

For every molecule we save raw per-component outputs so any downstream
species query can be evaluated in a notebook without re-running the
model.  Optionally we also save a 3D voxel density per requested species
as a Gaussian .cube file.

Per-atom (existing-node) reactivity scalars for species (a*, c*):

    F_n^{a*,c*}        = lambda_ins[n] * sum_k pi[n,k]
                            * a_probs[n,k,a*] * c_probs[n,k,c*]
    F_n^{a*,c*} (rate-free)
                       = sum_k pi[n,k]
                            * a_probs[n,k,a*] * c_probs[n,k,c*]

Continuous density (in real Angstrom space):

    rho^{a*,c*}(r) = sum_{n,k} w_{n,k}^{a*,c*}
                       * N(r; mu_{n,k}, sigma_{n,k})

Usage:
    python scripts/probability_hull_species.py \\
        --ckpt outputs/.../epoch=499.ckpt \\
        --output_dir hulls/species \\
        --n_mols 8 --t 0.99 \\
        --species "H:1,H:0,C:0,N:0,O:0,F:0" \\
        --write_cubes
"""

import os
import sys

_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), ".."),
)
for _p in [_PROJECT_ROOT, os.path.join(_PROJECT_ROOT, "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import torch.nn.functional as F

from scripts.probability_hull import (
    DEVICE,
    compute_probability_grid,
    load_config_and_preprocessing,
    load_val_molecules,
    write_cube,
    write_xyz,
)
from chemflow.dataset.molecule_data import MoleculeBatch
from chemflow.dataset.qm9 import FlowMatchingQM9Dataset
from chemflow.dataset.vocab import setup_token_weights
from chemflow.utils.metrics import init_metrics
from chemflow.utils.utils import index_to_token, token_to_index

import hydra
from copy import deepcopy
from omegaconf import OmegaConf


def load_model(cfg, vocab, distributions, token_prior, ckpt_path):
    """Like ``probability_hull.load_model`` but also passes the metric
    collections that the LightningModule constructor requires.
    """
    loss_weight_distributions = deepcopy(distributions)
    tw = cfg.model.token_weighting
    atom_type_weights, edge_token_weights, charge_token_weights = setup_token_weights(
        vocab=vocab,
        distributions=loss_weight_distributions,
        weight_alpha=tw.weight_alpha,
        type_loss_token_weights=tw.type_loss_token_weights,
    )

    metrics, stability_metrics, distribution_metrics, batch_metrics = init_metrics(
        train_smiles=None,
        target_n_atoms_distribution=distributions.n_atoms_distribution,
        atom_type_distribution=distributions.atom_type_distribution,
        edge_type_distribution=distributions.edge_type_distribution,
        charge_type_distribution=distributions.charge_type_distribution,
        atom_tokens=vocab.atom_tokens,
        edge_tokens=vocab.edge_tokens,
        charge_tokens=vocab.charge_tokens,
        allow_charged=True,
    )

    module = hydra.utils.instantiate(
        cfg.model.module,
        _recursive_=False,
        distributions=token_prior,
        loss_weight_distributions=loss_weight_distributions,
        atom_type_weights=atom_type_weights,
        edge_token_weights=edge_token_weights,
        charge_token_weights=charge_token_weights,
        metrics=metrics,
        stability_metrics=stability_metrics,
        distribution_metrics=distribution_metrics,
        batch_metrics=batch_metrics,
    )

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"]
    state_dict = {k.replace("._orig_mod", ""): v for k, v in state_dict.items()}
    module.load_state_dict(state_dict)
    module.eval()
    return module


# ── 1. Species query parsing ─────────────────────────────────────────────


def parse_species_query(query: str, vocab):
    """Parse "H:1" / "C" / "N:-1" into (atom_idx, charge_idx | None, label).

    A missing charge means "marginalise over all charges".
    """
    if ":" in query:
        atom_str, charge_str = query.split(":", 1)
    else:
        atom_str, charge_str = query, None

    atom_str = atom_str.strip()
    a_idx = token_to_index(vocab.atom_tokens, atom_str)

    if charge_str is None or charge_str == "":
        return a_idx, None, atom_str

    charge_str = charge_str.strip()
    c_idx = token_to_index(vocab.charge_tokens, charge_str)

    sign = ""
    try:
        c_int = int(charge_str)
        if c_int > 0:
            sign = "+" * c_int
        elif c_int < 0:
            sign = "-" * (-c_int)
    except ValueError:
        sign = f":{charge_str}"

    label = f"{atom_str}{sign}" if sign else f"{atom_str}:{charge_str}"
    return a_idx, c_idx, label


def parse_species_list(spec: str, vocab):
    """Comma-separated species like "H:1,H:0,C:0" -> list of (a, c, label)."""
    if not spec:
        # Sensible default: every atom token, neutral charge, plus H+ and OH-
        defaults = []
        for atom_tok in vocab.atom_tokens:
            defaults.append(f"{atom_tok}:0")
        if "1" in vocab.charge_tokens and "H" in vocab.atom_tokens:
            defaults.append("H:1")
        if "-1" in vocab.charge_tokens and "O" in vocab.atom_tokens:
            defaults.append("O:-1")
        spec = ",".join(defaults)

    return [parse_species_query(q, vocab) for q in spec.split(",") if q.strip()]


# ── 2. Model forward at parameterized t ──────────────────────────────────


@torch.no_grad()
def run_inference(module, mol_batch, t_value: float):
    """Forward at a single scalar time t (no CFG, no perturbation)."""
    model = module._get_model()
    model.eval()

    bs = mol_batch.num_graphs
    t = torch.full((bs,), float(t_value), device=DEVICE)

    preds = model(mol_batch, t.view(-1, 1), overrides={}, drop_masks=None)

    gmm = preds["gmm_head"]
    # forward() does not apply post-head activations (apply_activations is
    # called by CFGGuidance.guided_predict). We invoke softplus on the rate
    # head ourselves.
    ins_rate = F.softplus(preds["ins_rate_head"])
    if ins_rate.ndim > 1:
        ins_rate = ins_rate.squeeze(-1)

    return gmm, ins_rate


# ── 3. Species-conditional weights ───────────────────────────────────────


def species_weights(
    pi,            # [N, K]
    a_probs,       # [N, K, N_a]
    c_probs,       # [N, K, N_c]
    lambda_ins,    # [N]
    a_idx: int,
    c_idx: int | None,
):
    """Component weights w_{n,k}^{a*,c*} (rate-weighted and rate-free)."""
    a_term = a_probs[:, :, a_idx]
    if c_idx is None:
        c_term = torch.ones_like(a_term)  # marginalise over charge
    else:
        c_term = c_probs[:, :, c_idx]

    base = pi * a_term * c_term                           # [N, K]
    weighted = base * lambda_ins.unsqueeze(-1)            # [N, K]
    return weighted, base


def fukui_indices(weighted, base):
    """Per-spawn-node scalars: F_n = sum_k weight_{n,k}."""
    return weighted.sum(dim=-1), base.sum(dim=-1)


# ── 4. Per-molecule extraction ───────────────────────────────────────────


def extract_molecule_record(
    mol_data,
    gmm,
    ins_rate,
    species_list,
    coord_std,
    vocab,
    node_offset: int,
):
    """Slice the global GMM tensors to one molecule and compute species data.

    All spatial quantities are returned in real Angstrom space.
    """
    n_nodes = mol_data.num_nodes
    sl = slice(node_offset, node_offset + n_nodes)

    mu_norm = gmm["mu"][sl].detach().cpu()        # [N, K, 3]  (normalised)
    sigma_norm = gmm["sigma"][sl].detach().cpu()  # [N, K]
    pi = gmm["pi"][sl].detach().cpu()             # [N, K]
    a_probs = gmm["a_probs"][sl].detach().cpu()   # [N, K, N_a]
    c_probs = gmm["c_probs"][sl].detach().cpu()   # [N, K, N_c]
    lam = ins_rate[sl].detach().cpu()             # [N]

    coord_std_cpu = (
        coord_std.detach().cpu()
        if torch.is_tensor(coord_std)
        else torch.tensor(float(coord_std))
    )

    mu_real = mu_norm * coord_std_cpu
    sigma_real = sigma_norm * coord_std_cpu

    atom_coords = mol_data.x.detach().cpu() * coord_std_cpu
    atom_symbols = [
        index_to_token(vocab.atom_tokens, int(a))
        for a in mol_data.a.detach().cpu().numpy()
    ]
    atom_charges = [
        index_to_token(vocab.charge_tokens, int(c))
        for c in mol_data.c.detach().cpu().numpy()
    ]

    species_data = {}
    for a_idx, c_idx, label in species_list:
        weighted, base = species_weights(pi, a_probs, c_probs, lam, a_idx, c_idx)
        F_w, F_b = fukui_indices(weighted, base)

        # Best component per node for this species (site selectivity)
        best_k = base.argmax(dim=-1)              # [N]

        species_data[label] = {
            "atom_idx": int(a_idx),
            "charge_idx": (None if c_idx is None else int(c_idx)),
            "fukui_rate_weighted": F_w.float(),   # [N]
            "fukui_rate_free": F_b.float(),       # [N]
            "weighted_components": weighted.float(),  # [N, K]
            "base_components": base.float(),          # [N, K]
            "best_k": best_k.long(),              # [N]
        }

    return {
        "n_nodes": n_nodes,
        "atom_coords": atom_coords.float(),
        "atom_symbols": atom_symbols,
        "atom_charges": atom_charges,
        "mu": mu_real.float(),                # [N, K, 3]
        "sigma": sigma_real.float(),          # [N, K]
        "pi": pi.float(),                     # [N, K]
        "a_probs": a_probs.float(),           # [N, K, N_a]
        "c_probs": c_probs.float(),           # [N, K, N_c]
        "lambda_ins": lam.float(),            # [N]
        "species": species_data,
    }


# ── 5. Optional cube grid per species ────────────────────────────────────


def species_density_grid(
    record,
    a_idx: int,
    c_idx: int | None,
    grid_res: float,
    margin: float,
):
    """Compute a per-species 3D density grid in real Angstrom space.

    Reuses ``compute_probability_grid`` which already returns a grid in
    the units of its inputs.  We pass real-space mu/sigma so the grid is
    real-space too.
    """
    mu = record["mu"]                               # [N, K, 3]  Angstrom
    sigma = record["sigma"]                         # [N, K]
    pi = record["pi"]
    a_probs = record["a_probs"]
    c_probs = record["c_probs"]
    lam = record["lambda_ins"]

    a_term = a_probs[:, :, a_idx]
    c_term = (
        torch.ones_like(a_term) if c_idx is None else c_probs[:, :, c_idx]
    )

    density, (origin, _) = compute_probability_grid(
        mu, sigma, pi * a_term * c_term, lam.unsqueeze(-1),
        grid_res=grid_res, margin=margin,
    )
    return density, origin


# ── 6. End-to-end pipeline ───────────────────────────────────────────────


def load_molecules(cfg, vocab, distributions, split: str, n_mols: int):
    """Load molecules from a configurable split (val/test)."""
    if split == "val":
        return load_val_molecules(cfg, vocab, distributions, n_mols=n_mols)

    val_cfg = cfg.data.datamodule.datasets.val
    if OmegaConf.is_list(val_cfg) or isinstance(val_cfg, (list, tuple)):
        val_cfg = val_cfg[0]
    val_cfg = OmegaConf.to_container(val_cfg, resolve=True)
    root = val_cfg["root"]

    ds = FlowMatchingQM9Dataset(
        root=root, vocab=vocab, distributions=distributions,
        split=split, rotate=False,
    )
    indices = list(range(min(n_mols, len(ds))))
    return [ds[i] for i in indices]


def run_species_probe(
    ckpt_path: str,
    output_dir: str = "hulls/species",
    n_mols: int = 8,
    split: str = "val",
    t_value: float = 0.99,
    species_spec: str = "",
    grid_res: float = 0.3,
    margin: float = 3.0,
    write_cubes: bool = False,
    seed: int = 0,
):
    os.makedirs(output_dir, exist_ok=True)
    torch.manual_seed(seed)

    cfg, vocab, distributions, token_prior = load_config_and_preprocessing()
    module = load_model(cfg, vocab, distributions, token_prior, ckpt_path)
    module.to(DEVICE)
    module.eval()

    species_list = parse_species_list(species_spec, vocab)
    print(f"Querying {len(species_list)} species: "
          f"{[lbl for _, _, lbl in species_list]}")

    molecules = load_molecules(cfg, vocab, distributions, split, n_mols)
    mol_batch = MoleculeBatch.from_data_list(molecules).to(DEVICE)
    _ = mol_batch.remove_com()

    gmm, lam = run_inference(module, mol_batch, t_value)

    coord_std = distributions.coordinate_std
    data_list = mol_batch.to_data_list()

    summary = {
        "t_value": float(t_value),
        "split": split,
        "atom_tokens": list(vocab.atom_tokens),
        "charge_tokens": list(vocab.charge_tokens),
        "species_labels": [lbl for _, _, lbl in species_list],
        "species_atom_idx": [int(a) for a, _, _ in species_list],
        "species_charge_idx": [None if c is None else int(c)
                                for _, c, _ in species_list],
        "molecules": [],
    }

    node_offset = 0
    for i, mol_data in enumerate(data_list):
        record = extract_molecule_record(
            mol_data, gmm, lam, species_list,
            coord_std=coord_std, vocab=vocab, node_offset=node_offset,
        )
        record["index"] = i

        out_pt = os.path.join(output_dir, f"mol_{i:03d}.pt")
        torch.save(record, out_pt)

        xyz_path = os.path.join(output_dir, f"mol_{i:03d}.xyz")
        write_xyz(xyz_path, record["atom_symbols"], record["atom_coords"].numpy())

        if write_cubes:
            for a_idx, c_idx, label in species_list:
                density, origin = species_density_grid(
                    record, a_idx, c_idx, grid_res, margin,
                )
                cube_path = os.path.join(
                    output_dir, f"mol_{i:03d}_{label.replace(':', '_')}.cube",
                )
                # NB: density grid is already in real space (mu was in Angstrom),
                # so origin and grid_res are in Angstrom already.
                write_cube(
                    cube_path,
                    density,
                    origin,
                    grid_res,
                    record["atom_symbols"],
                    record["atom_coords"].numpy(),
                )

        summary["molecules"].append({
            "index": i,
            "n_nodes": record["n_nodes"],
            "pt_path": out_pt,
            "xyz_path": xyz_path,
            "atom_symbols": list(record["atom_symbols"]),
            "atom_charges": list(record["atom_charges"]),
            "fukui_rate_weighted": {
                lbl: record["species"][lbl]["fukui_rate_weighted"].tolist()
                for _, _, lbl in species_list
            },
            "fukui_rate_free": {
                lbl: record["species"][lbl]["fukui_rate_free"].tolist()
                for _, _, lbl in species_list
            },
            "lambda_ins": record["lambda_ins"].tolist(),
        })

        print(
            f"[{i}] {record['n_nodes']} atoms -> {out_pt} "
            f"(lambda_ins range "
            f"[{record['lambda_ins'].min():.4f}, "
            f"{record['lambda_ins'].max():.4f}])"
        )

        node_offset += mol_data.num_nodes

    summary_path = os.path.join(output_dir, "summary.pt")
    torch.save(summary, summary_path)
    print(f"Saved {len(data_list)} molecules + summary -> {summary_path}")
    return summary_path


# ── 7. CLI ───────────────────────────────────────────────────────────────


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Species-conditional insertion-density probe (Fukui-like).",
    )
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="hulls/species")
    parser.add_argument("--n_mols", type=int, default=8,
                        help="Number of molecules to probe from --split.")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "val", "test"])
    parser.add_argument("--t", type=float, default=0.99,
                        help="Late-time evaluation point (in [0, 1)).")
    parser.add_argument(
        "--species", type=str, default="",
        help='Comma list like "H:1,H:0,C:0,N:0,O:0,F:0". Empty = sensible default.',
    )
    parser.add_argument("--grid_res", type=float, default=0.3)
    parser.add_argument("--margin", type=float, default=3.0)
    parser.add_argument("--write_cubes", action="store_true",
                        help="Also save per-species 3D voxel grids as .cube.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    run_species_probe(
        ckpt_path=args.ckpt,
        output_dir=args.output_dir,
        n_mols=args.n_mols,
        split=args.split,
        t_value=args.t,
        species_spec=args.species,
        grid_res=args.grid_res,
        margin=args.margin,
        write_cubes=args.write_cubes,
        seed=args.seed,
    )
