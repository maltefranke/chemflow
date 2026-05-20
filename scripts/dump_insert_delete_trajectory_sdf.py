"""Dump SDF files for trajectories containing both insertion and deletion edits.

Loads a trajectory .pt file produced by ``eval_natoms_cfg.py`` (which holds
``valid_trajectories`` / ``invalid_trajectories`` as lists of per-step
``MoleculeData`` frames), finds trajectories where the atom count both
increases and decreases at some point, and writes one SDF per such trajectory
with one molecule entry per integration step.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from rdkit import Chem, RDLogger

from chemflow.dataset.molecule_data import IDX_BOND_MAP
from chemflow.utils.repr import tensors_to_rdkit_mol
from chemflow.utils.utils import index_to_token

RDLogger.DisableLog("rdApp.*")


def _read_tokens(tokens_dir: Path) -> tuple[list[str], list[str], list[str]]:
    def _read(name: str) -> list[str]:
        with open(tokens_dir / name) as f:
            return f.read().splitlines()

    return _read("atom_tokens.txt"), _read("edge_tokens.txt"), _read("charge_tokens.txt")


def _natoms_per_step(traj) -> list[int]:
    return [int(getattr(frame, "num_nodes", frame.a.shape[0])) for frame in traj]


def _has_insert_and_delete(natoms: list[int]) -> bool:
    has_ins = any(b > a for a, b in zip(natoms[:-1], natoms[1:]))
    has_del = any(b < a for a, b in zip(natoms[:-1], natoms[1:]))
    return has_ins and has_del


_KEKULIZE_EXC = tuple(
    e for e in (getattr(Chem, "AtomKekulizeException", None),
                getattr(Chem, "KekulizeException", None)) if e is not None
)


def _frame_to_mol(frame, atom_tokens, edge_tokens, charge_tokens):
    """Build an RDKit mol *without* sanitization so intermediate frames survive."""
    try:
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
                if et == "<NO_BOND>" or et == "<MASK>":
                    continue
                if et not in IDX_BOND_MAP:
                    continue
                edge_types.append(IDX_BOND_MAP[et])
                edge_index_list.append(edge)

        return tensors_to_rdkit_mol(
            atom_syms, x, charges, edge_types, edge_index_list, sanitize=False
        )
    except Exception as e:
        print(f"    frame->mol failed: {e}")
        return None


def _write_frame(writer: Chem.SDWriter, mol, name: str, step: int, n_atoms: int,
                 edit_kind: str) -> bool:
    if mol is None:
        return False
    mol.SetProp("_Name", name)
    mol.SetProp("step", str(step))
    mol.SetProp("n_atoms", str(n_atoms))
    mol.SetProp("edit_from_prev", edit_kind)
    try:
        writer.write(mol)
        return True
    except _KEKULIZE_EXC:
        rw = Chem.RWMol(mol)
        for atom in rw.GetAtoms():
            atom.SetIsAromatic(False)
        for bond in rw.GetBonds():
            if bond.GetBondType() == Chem.BondType.AROMATIC or bond.GetIsAromatic():
                bond.SetBondType(Chem.BondType.SINGLE)
            bond.SetIsAromatic(False)
        rw.SetProp("_Name", name)
        rw.SetProp("step", str(step))
        rw.SetProp("n_atoms", str(n_atoms))
        rw.SetProp("edit_from_prev", edit_kind)
        try:
            writer.write(rw)
            return True
        except Exception as e:
            print(f"    skipping {name}: {e}")
            return False
    except Exception as e:
        print(f"    skipping {name}: {e}")
        return False


def _edit_kind(prev_n: int, cur_n: int) -> str:
    if cur_n > prev_n:
        return "ins"
    if cur_n < prev_n:
        return "del"
    return "same"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--trajectories",
        default="/cluster/project/krause/frankem/chemflow/eval_outputs/"
                "natoms_cfg_extrapolate/trajectories/scale=1.0_target=18.pt",
    )
    p.add_argument(
        "--tokens-dir",
        default="/cluster/project/krause/frankem/chemflow/data/qm9/processed",
    )
    p.add_argument(
        "--output-dir",
        default="/cluster/project/krause/frankem/chemflow/eval_outputs/"
                "natoms_cfg_extrapolate/insert_delete_sdf",
    )
    p.add_argument("--max-trajectories", type=int, default=5,
                   help="How many matching trajectories to dump.")
    p.add_argument("--include", choices=("valid", "invalid", "both"), default="both")
    args = p.parse_args()

    traj_path = Path(args.trajectories)
    tokens_dir = Path(args.tokens_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    atom_tokens, edge_tokens, charge_tokens = _read_tokens(tokens_dir)

    print(f"Loading {traj_path}")
    obj = torch.load(traj_path, weights_only=False, map_location="cpu")

    target = obj.get("target_n_atoms")
    scale = obj.get("guidance_scale")
    print(f"  guidance_scale={scale}  target_n_atoms={target}")

    buckets: list[tuple[str, list]] = []
    if args.include in ("valid", "both"):
        buckets.append(("valid", obj.get("valid_trajectories", []) or []))
    if args.include in ("invalid", "both"):
        buckets.append(("invalid", obj.get("invalid_trajectories", []) or []))

    # Find candidates first so we can pick a balanced sample.
    candidates: list[tuple[str, int, list[int], list]] = []
    for bucket_name, trajs in buckets:
        for idx, traj in enumerate(trajs):
            if not isinstance(traj, (list, tuple)) or len(traj) < 2:
                continue
            natoms = _natoms_per_step(traj)
            if _has_insert_and_delete(natoms):
                candidates.append((bucket_name, idx, natoms, traj))

    print(f"  trajectories with both insertion and deletion: {len(candidates)}")
    if not candidates:
        print("No matching trajectory found; nothing to write.")
        return

    n_to_write = min(args.max_trajectories, len(candidates))
    print(f"  writing {n_to_write} SDF files to {out_dir}")

    for bucket_name, idx, natoms, traj in candidates[:n_to_write]:
        n_ins = sum(1 for a, b in zip(natoms[:-1], natoms[1:]) if b > a)
        n_del = sum(1 for a, b in zip(natoms[:-1], natoms[1:]) if b < a)
        out_path = out_dir / (
            f"scale={scale}_target={target}_{bucket_name}_idx={idx:04d}"
            f"_ins={n_ins}_del={n_del}.sdf"
        )

        writer = Chem.SDWriter(str(out_path))
        writer.SetKekulize(False)
        n_written = 0
        try:
            prev_n = natoms[0]
            for step, frame in enumerate(traj):
                cur_n = natoms[step]
                kind = "start" if step == 0 else _edit_kind(prev_n, cur_n)
                mol = _frame_to_mol(frame, atom_tokens, edge_tokens, charge_tokens)
                if _write_frame(
                    writer,
                    mol,
                    name=f"{bucket_name}_{idx:04d}_step{step:03d}",
                    step=step,
                    n_atoms=cur_n,
                    edit_kind=kind,
                ):
                    n_written += 1
                prev_n = cur_n
        finally:
            writer.close()

        print(
            f"    {out_path.name}: {n_written}/{len(traj)} frames written "
            f"(n_atoms range {min(natoms)}..{max(natoms)}, "
            f"ins={n_ins}, del={n_del})"
        )


if __name__ == "__main__":
    main()
