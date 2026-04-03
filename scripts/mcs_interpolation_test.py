import os
import random
from pathlib import Path

import hydra
import omegaconf
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from rdkit import Chem
from torch_geometric.utils import to_dense_adj

from chemflow.dataset.data_utils import select_scaffold_pairs_by_neighbor_count
from chemflow.dataset.molecule_data import (
    AugmentedMoleculeData,
    MoleculeData,
    filter_nodes,
)
from chemflow.flow_matching.assignment import (
    mcs_based_assignment_single,
    substituent_based_assignment_single,
)
from chemflow.flow_matching.interpolation import Interpolator
from chemflow.flow_matching.schedules import SmoothstepSchedule
from chemflow.utils.utils import init_uniform_prior

# resolvers for more complex config expressions
OmegaConf.register_new_resolver("oc.eval", eval)
OmegaConf.register_new_resolver("len", lambda x: len(x))
OmegaConf.register_new_resolver("if", lambda cond, t, f: t if cond else f)
OmegaConf.register_new_resolver("eq", lambda x, y: x == y)

torch.set_float32_matmul_precision("medium")

pl.seed_everything(42)


def compute_scaffold_pairs(
    src_matches: list,
    tgt_matches: list,
    src_dec: torch.Tensor,
    tgt_dec: torch.Tensor,
) -> list:
    """Pick the scaffold automorphism pair by decoration-count L1 score.

    Returns a list of (src_atom_idx, tgt_atom_idx) pairs, or [] if no matches.
    """
    return select_scaffold_pairs_by_neighbor_count(
        src_dec, src_matches, tgt_dec, tgt_matches
    )


def pre_sample_events_mcs(
    interpolator: Interpolator,
    sample_mol: MoleculeData,
    target_mol: MoleculeData,
    scaffold_pairs: list,
    device: str,
    scaffold_substituents: tuple | None = None,
):
    """Pre-sample all stochastic events using scaffold-based OT alignment."""
    if scaffold_substituents is not None and scaffold_pairs:
        src_subs, tgt_subs = scaffold_substituents
        sample, target = substituent_based_assignment_single(
            sample_mol,
            target_mol,
            fixed_pairs=scaffold_pairs,
            src_subs=src_subs,
            tgt_subs=tgt_subs,
            c_move=interpolator.c_move,
            c_sub=interpolator.c_sub,
            c_ins=interpolator.c_ins,
            c_del=interpolator.c_del,
            optimal_transport=interpolator.optimal_transport,
        )
    else:
        sample, target = mcs_based_assignment_single(
            sample_mol,
            target_mol,
            fixed_pairs=scaffold_pairs if scaffold_pairs else None,
            c_move=interpolator.c_move,
            c_sub=interpolator.c_sub,
            c_ins=interpolator.c_ins,
            c_del=interpolator.c_del,
            optimal_transport=interpolator.optimal_transport,
        )

    N = sample.x.shape[0]

    is_sub = ((~sample.is_auxiliary) & (~target.is_auxiliary)).squeeze()  # Real → Real
    is_del = ((~sample.is_auxiliary) & (target.is_auxiliary)).squeeze()  # Real → Dummy
    is_ins = ((sample.is_auxiliary) & (~target.is_auxiliary)).squeeze()  # Dummy → Real

    # ---- node-level event times (uniform draws, fixed for the whole trajectory) ----
    tau_del = torch.rand(N, device=device)
    tau_ins = torch.rand(N, device=device)

    # ---- discrete substitution thresholds (one per node / per edge) ----
    rand_disc_a = torch.rand(N, device=device)
    rand_disc_c = torch.rand(N, device=device)

    # Upper-triangular edge pairs (dense edge representation used throughout)
    triu_rows, triu_cols = torch.triu_indices(N, N, offset=1, device=device)
    n_triu = triu_rows.shape[0]
    rand_disc_e = torch.rand(n_triu, device=device)

    # ---- insertion node noise / identity, sampled once ----
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

    # ---- random edge types for edges incident to any insertion node, sampled once ----
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
    """
    Deterministically compute the interpolated state at a given time t using
    the pre-sampled events dict returned by pre_sample_events_mcs.
    """
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

    # Schedule values at the current time
    t_del = interpolator.del_schedule.kappa_t(t_i)
    t_ins = interpolator.ins_schedule.kappa_t(t_i)
    t_kappa = interpolator.sub_schedule.kappa_t(t_i)
    t_kappa_e = interpolator.sub_e_schedule.kappa_t(t_i)

    # ---- Node existence masks (deterministic, based on pre-sampled tau) ----
    mask_keep_del = is_del & (t_del < events["tau_del"])
    mask_keep_ins = is_ins & (t_ins > events["tau_ins"])
    mask_exists = is_sub | mask_keep_del | mask_keep_ins

    # ---- A. Deletions: freeze target coordinates/types at source ----
    if mask_keep_del.any():
        target.x[mask_keep_del] = sample.x[mask_keep_del]
        target.a[mask_keep_del] = sample.a[mask_keep_del]
        target.c[mask_keep_del] = sample.c[mask_keep_del]

    # ---- B. Insertions: jump to target with pre-sampled noise/types ----
    if mask_keep_ins.any():
        local_ins_mask = mask_keep_ins[ins_indices]
        sample.x[mask_keep_ins] = (
            target.x[mask_keep_ins] + events["ins_noise"][local_ins_mask]
        )
        sample.a[mask_keep_ins] = events["a_ins_rand"][local_ins_mask]
        sample.c[mask_keep_ins] = events["c_ins_rand"][local_ins_mask]

    # ---- C. Edge handling (dense upper-triangular representation) ----
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

    # ---- D. Discrete substitution with pre-sampled thresholds ----
    mask_a_keep = events["rand_disc_a"] > t_kappa
    a_t = target.a.clone()
    a_t[mask_a_keep] = sample.a[mask_a_keep]

    mask_c_keep = events["rand_disc_c"] > t_kappa
    c_t = target.c.clone()
    c_t[mask_c_keep] = sample.c[mask_c_keep]

    mask_e_keep = events["rand_disc_e"] > t_kappa_e
    e_t = e1_triu.clone()
    e_t[mask_e_keep] = e0_triu[mask_e_keep]

    # ---- E. Continuous interpolation of coordinates ----
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

    # ---- F. Symmetrise edges ----
    full_idx, full_attrs = interpolator.edge_aligner.symmetrize_edges(
        interp_state.edge_index, [interp_state.e]
    )
    interp_state.edge_index = full_idx
    interp_state.e = full_attrs[0]

    # ---- G. Filter to existing nodes only ----
    interp_state = filter_nodes(interp_state, mask_exists.squeeze())
    interp_state.scaffold_mask = is_sub[mask_exists.squeeze()]

    # ---- H. Centre coordinates ----
    x_mean = interp_state.x.mean(dim=0)
    interp_state.x = interp_state.x - x_mean

    return interp_state


def smooth_trajectory_mcs(
    interpolator: Interpolator,
    sample_mol: MoleculeData,
    target_mol: MoleculeData,
    scaffold_pairs: list,
    time_points: torch.Tensor,
    device: str,
    scaffold_substituents: tuple | None = None,
):
    """Build a smooth interpolation trajectory using scaffold-based assignment."""
    events = pre_sample_events_mcs(
        interpolator, sample_mol, target_mol, scaffold_pairs, device,
        scaffold_substituents=scaffold_substituents,
    )
    trajectory = []
    for t in time_points:
        state = compute_state_at_t(interpolator, events, t.item())
        trajectory.append(state)
    return trajectory


def run(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)
    os.environ.setdefault("PROJECT_ROOT", str(Path(__file__).parent.parent))

    hydra.utils.log.info("Instantiating preprocessing...")
    preprocessing = hydra.utils.instantiate(cfg.data.preprocessing)

    vocab = preprocessing.vocab
    distributions = preprocessing.distributions
    token_prior_distribution = init_uniform_prior(distributions)

    cfg.data.vocab = vocab

    hydra.utils.log.info(
        f"Preprocessing complete.\n"
        f"Found {len(vocab.atom_tokens)} atom tokens: {vocab.atom_tokens}\n"
        f"Found {len(vocab.edge_tokens)} edge tokens: {vocab.edge_tokens}\n"
        f"Found {len(vocab.charge_tokens)} charge tokens: {vocab.charge_tokens}"
    )
    hydra.utils.log.info("Distributions computed from training dataset.")

    hydra.utils.log.info(f"Instantiating <{cfg.data.datamodule._target_}>")
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(
        cfg.data.datamodule,
        _recursive_=False,
        vocab=vocab,
        distributions=token_prior_distribution,
    )
    datamodule.setup()

    val_dataset = datamodule.val_dataloader()[0].dataset

    device = "cuda" if torch.cuda.is_available() else "cpu"

    interpolator = Interpolator(
        vocab=vocab,
        distributions=token_prior_distribution,
        ins_noise_scale=0.25,
        ins_schedule=SmoothstepSchedule(shift=0.65),
        del_schedule=SmoothstepSchedule(shift=0.65),
        sub_schedule=SmoothstepSchedule(shift=1.5),
        sub_e_schedule=SmoothstepSchedule(shift=1.5),
        c_del=0.0,
        c_ins=1e8,
        c_sub=0.0,
        c_move=1.0, # originally; #1e10 to effectively disable moves
    )

    integrator = hydra.utils.instantiate(
        cfg.model.integrator,
        vocab=vocab,
        distributions=token_prior_distribution,
    )

    num_steps = 100
    if integrator.time_strategy == "linear":
        time_points = torch.linspace(0, 1, num_steps + 1, device=device)
    elif integrator.time_strategy == "log":
        start_log = torch.log10(torch.tensor(0.01, device=device))
        end_log = torch.log10(torch.tensor(1.0, device=device))
        time_points = 1 - torch.logspace(
            start_log, end_log, num_steps + 1, device=device
        )
        time_points = torch.flip(time_points, dims=[0])
    else:
        raise ValueError(f"Invalid time strategy: {integrator.time_strategy}")

    base_dataset = val_dataset.base_dataset
    assignment_method = cfg.data.datamodule.wrapper.get(
        "assignment_method", "mcs_constrained"
    )
    hydra.utils.log.info(f"Using assignment method: {assignment_method}")

    results = []
    targets_single = []

    for i in range(32):
        base_idx = val_dataset._filtered_indices[i]
        gid = int(val_dataset._mol_to_group[base_idx])
        candidates = [j for j in val_dataset._groups[gid] if j != base_idx]
        source_idx = random.choice(candidates)

        target = base_dataset[base_idx]
        smiles_target = base_dataset.get(base_idx).smiles
        source = base_dataset[source_idx]
        smiles_source = base_dataset.get(source_idx).smiles

        src_matches = val_dataset._scaffold_atom_indices[source_idx]
        tgt_matches = val_dataset._scaffold_atom_indices[base_idx]
        print(f"Source scaffold matches: {src_matches}")
        print(f"Target scaffold matches: {tgt_matches}")
        hydra.utils.log.info(
            f"Found {src_matches} scaffold matches for source and {tgt_matches} for target."
        )
        scaffold_pairs = compute_scaffold_pairs(
            src_matches, tgt_matches,
            src_dec=val_dataset._scaffold_decoration_counts[source_idx],
            tgt_dec=val_dataset._scaffold_decoration_counts[base_idx],
        )

        if assignment_method == "substituent" and scaffold_pairs:
            scaffold_substituents = (
                val_dataset._scaffold_substituents[source_idx],
                val_dataset._scaffold_substituents[base_idx],
            )
        else:
            scaffold_substituents = None

        src_mol = Chem.MolFromSmiles(smiles_source)
        tgt_mol = Chem.MolFromSmiles(smiles_target)
        for atom in src_mol.GetAtoms():
            print(atom.GetIdx(), atom.GetSymbol())
        for atom in tgt_mol.GetAtoms():
            print(atom.GetIdx(), atom.GetSymbol())
        hydra.utils.log.info(
            f"Processing pair {i}: {smiles_source} → {smiles_target}\n"
            f"  scaffold_pairs ({len(scaffold_pairs)}): {scaffold_pairs}\n"
            f"  source (CDK): {Chem.MolToSmiles(src_mol)}\n"
            f"  target (CDK): {Chem.MolToSmiles(tgt_mol)}"
        )

        trajectory = smooth_trajectory_mcs(
            interpolator,
            source.to(device),
            target.to(device),
            scaffold_pairs,
            time_points,
            device,
            scaffold_substituents=scaffold_substituents,
        )
        results.append(trajectory)
        targets_single.append(target)

    torch.save(results, "results.pt")
    torch.save(targets_single, "ground_truth.pt")


@hydra.main(
    config_path="../configs",
    config_name="default",
    version_base="1.1",
)
def main(cfg: omegaconf.DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
