"""Standalone test for MCS-guided smooth interpolation between two molecules.

Creates MoleculeData directly from SMILES (no Hydra / dataset dependency),
runs the MCS-based assignment, and builds a smooth trajectory.
"""

import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.utils import to_dense_adj

from chemflow.dataset.data_utils import get_mcs_atom_mapping
from chemflow.dataset.molecule_data import (
    AugmentedMoleculeData,
    MoleculeData,
    filter_nodes,
)
from chemflow.dataset.vocab import Distributions, Vocab
from chemflow.flow_matching.assignment import mcs_based_assignment_single
from chemflow.flow_matching.interpolation import Interpolator
from chemflow.flow_matching.schedules import SmoothstepSchedule
from chemflow.utils.utils import build_fully_connected_edge_index

torch.set_float32_matmul_precision("medium")
torch.manual_seed(42)


# ---------------------------------------------------------------------------
# Molecule construction helpers
# ---------------------------------------------------------------------------

BOND_TYPE_TO_TOKEN = {1.0: "1", 2.0: "2", 3.0: "3", 1.5: "4"}


def smiles_to_molecule_data(smiles: str, vocab: Vocab) -> MoleculeData:
    """Build a MoleculeData from a SMILES string.

    Atom ordering matches ``Chem.AddHs(Chem.MolFromSmiles(smiles))`` so that
    MCS atom indices are directly usable.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol)

    conf = mol.GetConformer()
    positions = torch.tensor(
        [list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())],
        dtype=torch.float32,
    )
    positions = positions - positions.mean(dim=0)

    atom_types = torch.tensor(
        [vocab.atom_tokens.index(a.GetSymbol()) for a in mol.GetAtoms()],
        dtype=torch.long,
    )

    charges = torch.tensor(
        [vocab.charge_tokens.index(str(a.GetFormalCharge())) for a in mol.GetAtoms()],
        dtype=torch.long,
    )

    n_atoms = mol.GetNumAtoms()
    edge_index = build_fully_connected_edge_index(n_atoms)

    edge_types = []
    for k in range(edge_index.shape[1]):
        src, dst = edge_index[0, k].item(), edge_index[1, k].item()
        bond = mol.GetBondBetweenAtoms(src, dst)
        if bond is None:
            edge_types.append(vocab.edge_tokens.index("<NO_BOND>"))
        else:
            bt = bond.GetBondTypeAsDouble()
            token = BOND_TYPE_TO_TOKEN.get(bt, "<NO_BOND>")
            edge_types.append(vocab.edge_tokens.index(token))
    edge_types = torch.tensor(edge_types, dtype=torch.long)

    return MoleculeData(
        x=positions, a=atom_types, e=edge_types, c=charges, edge_index=edge_index
    )


# ---------------------------------------------------------------------------
# Smooth-trajectory helpers (mirrors scripts/interpolation_test.py)
# ---------------------------------------------------------------------------


def pre_sample_events_mcs(
    interpolator: Interpolator,
    sample_mol: MoleculeData,
    target_mol: MoleculeData,
    smiles_sample: str,
    smiles_target: str,
    device: str,
):
    """Pre-sample all stochastic events using MCS-based OT alignment."""
    sample, target = mcs_based_assignment_single(
        sample_mol,
        target_mol,
        smiles_sample,
        smiles_target,
        c_move=interpolator.c_move,
        c_sub=interpolator.c_sub,
        c_ins=interpolator.c_ins,
        c_del=interpolator.c_del,
        optimal_transport=interpolator.optimal_transport,
    )

    N = sample.x.shape[0]

    is_sub = ((~sample.is_auxiliary) & (~target.is_auxiliary)).squeeze()
    is_del = ((~sample.is_auxiliary) & (target.is_auxiliary)).squeeze()
    is_ins = ((sample.is_auxiliary) & (~target.is_auxiliary)).squeeze()

    tau_del = torch.rand(N, device=device)
    tau_ins = torch.rand(N, device=device)

    rand_disc_a = torch.rand(N, device=device)
    rand_disc_c = torch.rand(N, device=device)

    triu_rows, triu_cols = torch.triu_indices(N, N, offset=1, device=device)
    n_triu = triu_rows.shape[0]
    rand_disc_e = torch.rand(n_triu, device=device)

    ins_indices = torch.where(is_ins)[0]
    n_ins = ins_indices.shape[0]

    ins_noise = (
        torch.randn(n_ins, sample.x.shape[-1], device=device)
        * interpolator.ins_noise_scale
    )
    if n_ins > 0:
        a_ins_rand = interpolator._cat_atom.sample((n_ins,)).to(device)
        c_ins_rand = interpolator._cat_charge.sample((n_ins,)).to(device)
    else:
        a_ins_rand = torch.empty((0,), dtype=sample.a.dtype, device=device)
        c_ins_rand = torch.empty((0,), dtype=sample.c.dtype, device=device)

    ins_edge_potential_mask = is_ins[triu_rows] | is_ins[triu_cols]
    n_ins_edges = ins_edge_potential_mask.sum().item()
    if n_ins_edges > 0:
        rand_edges_ins = interpolator._cat_edge.sample((n_ins_edges,)).to(device)
    else:
        rand_edges_ins = torch.empty((0,), dtype=torch.long, device=device)

    return {
        "sample": sample,
        "target": target,
        "N": N,
        "is_sub": is_sub,
        "is_del": is_del,
        "is_ins": is_ins,
        "ins_indices": ins_indices,
        "tau_del": tau_del,
        "tau_ins": tau_ins,
        "rand_disc_a": rand_disc_a,
        "rand_disc_c": rand_disc_c,
        "rand_disc_e": rand_disc_e,
        "triu_rows": triu_rows,
        "triu_cols": triu_cols,
        "ins_noise": ins_noise,
        "a_ins_rand": a_ins_rand,
        "c_ins_rand": c_ins_rand,
        "ins_edge_potential_mask": ins_edge_potential_mask,
        "rand_edges_ins": rand_edges_ins,
    }


def compute_state_at_t(
    interpolator: Interpolator, events: dict, t_scalar: float
) -> AugmentedMoleculeData:
    """Deterministically compute the interpolated state at time *t*."""
    sample = events["sample"].clone()
    target = events["target"].clone()

    N = events["N"]
    device = sample.x.device
    t_i = torch.tensor(t_scalar, device=device)

    is_sub = events["is_sub"]
    is_del = events["is_del"]
    is_ins = events["is_ins"]
    ins_indices = events["ins_indices"]
    triu_rows = events["triu_rows"]
    triu_cols = events["triu_cols"]

    t_del = interpolator.del_schedule.kappa_t(t_i)
    t_ins = interpolator.ins_schedule.kappa_t(t_i)
    t_kappa = interpolator.sub_schedule.kappa_t(t_i)
    t_kappa_e = interpolator.sub_e_schedule.kappa_t(t_i)

    mask_keep_del = is_del & (t_del < events["tau_del"])
    mask_keep_ins = is_ins & (t_ins > events["tau_ins"])
    mask_exists = is_sub | mask_keep_del | mask_keep_ins

    if mask_keep_del.any():
        target.x[mask_keep_del] = sample.x[mask_keep_del]
        target.a[mask_keep_del] = sample.a[mask_keep_del]
        target.c[mask_keep_del] = sample.c[mask_keep_del]

    if mask_keep_ins.any():
        local_ins_mask = mask_keep_ins[ins_indices]
        sample.x[mask_keep_ins] = (
            target.x[mask_keep_ins] + events["ins_noise"][local_ins_mask]
        )
        sample.a[mask_keep_ins] = events["a_ins_rand"][local_ins_mask]
        sample.c[mask_keep_ins] = events["c_ins_rand"][local_ins_mask]

    def _extract_triu_feats(data_obj):
        attr = data_obj.e
        if attr is None or attr.numel() == 0:
            return torch.zeros(triu_rows.shape[0], device=device, dtype=torch.long)
        if attr.dim() == 1:
            attr = attr.unsqueeze(-1)
        dense = to_dense_adj(data_obj.edge_index, edge_attr=attr, max_num_nodes=N)[0]
        return dense[triu_rows, triu_cols].squeeze()

    e0_triu = _extract_triu_feats(sample)
    e1_triu = _extract_triu_feats(target)

    edge_has_del = mask_keep_del[triu_rows] | mask_keep_del[triu_cols]
    edge_has_ins = mask_keep_ins[triu_rows] | mask_keep_ins[triu_cols]

    edge_mask_del = edge_has_del & ~edge_has_ins
    if edge_mask_del.any():
        e1_triu[edge_mask_del] = e0_triu[edge_mask_del]

    ins_edge_potential_mask = events["ins_edge_potential_mask"]
    edge_mask_ins = ins_edge_potential_mask & edge_has_ins & ~edge_has_del
    if edge_mask_ins.any():
        pot_indices = torch.where(ins_edge_potential_mask)[0]
        cur_indices = torch.where(edge_mask_ins)[0]
        local_idx = torch.searchsorted(pot_indices, cur_indices)
        e0_triu[edge_mask_ins] = events["rand_edges_ins"][local_idx].to(
            dtype=e0_triu.dtype
        )

    mask_a_keep = events["rand_disc_a"] > t_kappa
    a_t = target.a.clone()
    a_t[mask_a_keep] = sample.a[mask_a_keep]

    mask_c_keep = events["rand_disc_c"] > t_kappa
    c_t = target.c.clone()
    c_t[mask_c_keep] = sample.c[mask_c_keep]

    mask_e_keep = events["rand_disc_e"] > t_kappa_e
    e_t = e1_triu.clone()
    e_t[mask_e_keep] = e0_triu[mask_e_keep]

    x_t = sample.x * (1 - t_i) + target.x * t_i

    triu_edge_index = torch.stack([triu_rows, triu_cols], dim=0)
    interp_state = AugmentedMoleculeData(
        x=x_t,
        a=a_t,
        c=c_t,
        e=e_t,
        edge_index=triu_edge_index,
        is_auxiliary=sample.is_auxiliary | target.is_auxiliary,
    )

    full_idx, full_attrs = interpolator.edge_aligner.symmetrize_edges(
        interp_state.edge_index, [interp_state.e]
    )
    interp_state.edge_index = full_idx
    interp_state.e = full_attrs[0]

    interp_state = filter_nodes(interp_state, mask_exists.squeeze())

    x_mean = interp_state.x.mean(dim=0)
    interp_state.x = interp_state.x - x_mean

    return interp_state


def smooth_trajectory_mcs(
    interpolator: Interpolator,
    sample_mol: MoleculeData,
    target_mol: MoleculeData,
    smiles_sample: str,
    smiles_target: str,
    time_points: torch.Tensor,
    device: str,
):
    """Build a smooth interpolation trajectory using MCS-based assignment."""
    events = pre_sample_events_mcs(
        interpolator, sample_mol, target_mol, smiles_sample, smiles_target, device
    )
    trajectory = []
    for t in time_points:
        state = compute_state_at_t(interpolator, events, t.item())
        trajectory.append(state)
    return trajectory


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    vocab = Vocab(
        atom_tokens=["C", "F", "H", "N", "O"],
        edge_tokens=["<NO_BOND>", "1", "2", "3", "4"],
        charge_tokens=["-1", "0", "1"],
    )

    n_atom = len(vocab.atom_tokens)
    n_edge = len(vocab.edge_tokens)
    n_charge = len(vocab.charge_tokens)
    distributions = Distributions(
        atom_type_distribution=torch.ones(n_atom) / n_atom,
        edge_type_distribution=torch.ones(n_edge) / n_edge,
        charge_type_distribution=torch.ones(n_charge) / n_charge,
        n_atoms_distribution=torch.ones(30) / 30,
        coordinate_std=torch.tensor(1.0),
    )

    smi1 = "CCCC1=NN(C)C2=C1NC(=NC2=O)C3=C(OCC)C=CC(=C3)C(=O)N4CCN(C)CC4"
    smi2 = "CCCC1=NC(=C2N1N=C(NC2=O)C3=C(OCC)C=CC(=C3)C(=O)N4CCN(CC)CC4)C"

    mol1 = smiles_to_molecule_data(smi1, vocab)
    mol2 = smiles_to_molecule_data(smi2, vocab)

    print(f"Molecule 1 ({smi1}): {mol1.x.shape[0]} atoms")
    print(f"Molecule 2 ({smi2}): {mol2.x.shape[0]} atoms")

    mcs = get_mcs_atom_mapping(smi1, smi2)
    print(f"MCS matched pairs: {len(mcs)}")
    for i_s, i_t in mcs:
        sym_s = Chem.AddHs(Chem.MolFromSmiles(smi1)).GetAtomWithIdx(i_s).GetSymbol()
        sym_t = Chem.AddHs(Chem.MolFromSmiles(smi2)).GetAtomWithIdx(i_t).GetSymbol()
        print(f"  sample atom {i_s} ({sym_s}) <-> target atom {i_t} ({sym_t})")

    interpolator = Interpolator(
        vocab=vocab,
        distributions=distributions,
        ins_noise_scale=0.25,
        ins_schedule=SmoothstepSchedule(shift=0.65),
        del_schedule=SmoothstepSchedule(shift=0.65),
        sub_schedule=SmoothstepSchedule(shift=1.5),
        sub_e_schedule=SmoothstepSchedule(shift=1.5),
        c_del=0.01,
        c_ins=0.01,
        c_sub=0.01,
        c_move=10.0,
    )

    device = "cpu"
    num_steps = 100
    time_points = torch.linspace(0, 1, num_steps + 1, device=device)

    print("\nRunning MCS-based smooth interpolation...")
    trajectory = smooth_trajectory_mcs(
        interpolator, mol1, mol2, smi1, smi2, time_points, device
    )

    print(f"Generated {len(trajectory)} frames")
    print(f"  t=0.0 : {trajectory[0].x.shape[0]} atoms")
    print(f"  t=0.5 : {trajectory[len(trajectory) // 2].x.shape[0]} atoms")
    print(f"  t=1.0 : {trajectory[-1].x.shape[0]} atoms")

    torch.save([trajectory], "mcs_results.pt")
    torch.save([mol2], "mcs_ground_truth.pt")
    print("\nSaved mcs_results.pt and mcs_ground_truth.pt")


if __name__ == "__main__":
    main()
