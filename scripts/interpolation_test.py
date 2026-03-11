import hydra
import omegaconf
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from omegaconf import DictConfig
from omegaconf import OmegaConf
from torch_geometric.utils import to_dense_adj

from chemflow.flow_matching.interpolation import Interpolator
from chemflow.flow_matching.assignment import partial_optimal_transport_single
from chemflow.dataset.molecule_data import (
    AugmentedMoleculeData,
    MoleculeData,
    MoleculeBatch,
    filter_nodes,
)
import torch.nn.functional as F

from chemflow.flow_matching.schedules import FastPowerSchedule
from chemflow.flow_matching.schedules import SmoothstepSchedule

# resolvers for more complex config expressions
OmegaConf.register_new_resolver("oc.eval", eval)
OmegaConf.register_new_resolver("len", lambda x: len(x))
OmegaConf.register_new_resolver("if", lambda cond, t, f: t if cond else f)
OmegaConf.register_new_resolver("eq", lambda x, y: x == y)

torch.set_float32_matmul_precision("medium")

pl.seed_everything(42)


def pre_sample_events(
    interpolator: Interpolator,
    sample_mol: MoleculeData,
    target_mol: MoleculeData,
    device: str,
):
    """
    Perform OT alignment once and pre-sample ALL stochastic quantities.

    By drawing every random variable here (rather than inside each
    interpolate_single call), we guarantee that the resulting trajectory is
    a smooth, monotone path:
      - each deletion node disappears exactly once (when kappa_del(t) crosses tau_del[i]),
      - each insertion node appears exactly once (when kappa_ins(t) crosses tau_ins[i]),
      - each discrete token flips from source to target exactly once
        (when kappa_sub(t) crosses rand_disc[i]).

    Returns a dict of pre-sampled data to be consumed by compute_state_at_t.
    """
    # OT alignment — done once for the whole trajectory
    sample, target = partial_optimal_transport_single(
        sample_mol,
        target_mol,
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
    tau_del = torch.rand(N, device=device)  # deletion threshold per node
    tau_ins = torch.rand(N, device=device)  # insertion threshold per node

    # ---- discrete substitution thresholds (one per node / per edge) ----
    # Node i's atom type flips from source → target the first t where kappa_sub(t) ≥ rand_disc_a[i]
    rand_disc_a = torch.rand(N, device=device)
    rand_disc_c = torch.rand(N, device=device)

    # Upper-triangular edge pairs (dense edge representation used throughout)
    triu_rows, triu_cols = torch.triu_indices(N, N, offset=1, device=device)
    n_triu = triu_rows.shape[0]
    rand_disc_e = torch.rand(n_triu, device=device)

    # ---- insertion node noise / identity, sampled once ----
    ins_indices = torch.where(is_ins)[0]  # original indices of all potential ins nodes
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
    # These are used to initialise the source-side edges of insertion nodes,
    # and remain fixed so the edge types are consistent across time steps.
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
    the pre-sampled events dict returned by pre_sample_events.

    The logic mirrors Interpolator.interpolate_single but replaces every
    torch.rand call with the pre-sampled values, ensuring a smooth trajectory.
    """
    # Work on fresh copies so we never mutate the pre-sampled reference data
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
        # Map the global insertion mask to a local index into ins_noise / a_ins_rand / c_ins_rand
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

    # Use mutually exclusive edge classes to avoid double-editing del-ins edges.
    edge_has_del = mask_keep_del[triu_rows] | mask_keep_del[triu_cols]
    edge_has_ins = mask_keep_ins[triu_rows] | mask_keep_ins[triu_cols]

    # Deletion-only edges: freeze at source
    edge_mask_del = edge_has_del & ~edge_has_ins
    if edge_mask_del.any():
        e1_triu[edge_mask_del] = e0_triu[edge_mask_del]

    # Insertion edges: use pre-sampled random types (consistent across time)
    ins_edge_potential_mask = events["ins_edge_potential_mask"]
    edge_mask_ins = ins_edge_potential_mask & edge_has_ins & ~edge_has_del
    if edge_mask_ins.any():
        # Map active ins edges to their index within the potential-ins-edge set
        pot_indices = torch.where(ins_edge_potential_mask)[0]
        cur_indices = torch.where(edge_mask_ins)[0]
        local_idx = torch.searchsorted(pot_indices, cur_indices)
        e0_triu[edge_mask_ins] = events["rand_edges_ins"][local_idx].to(
            dtype=e0_triu.dtype
        )

    # ---- D. Discrete substitution with pre-sampled thresholds ----
    # Atom types: node i stays at source while rand_disc_a[i] > kappa_sub(t)
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

    # ---- H. Centre coordinates ----
    x_mean = interp_state.x.mean(dim=0)
    interp_state.x = interp_state.x - x_mean

    return interp_state


def smooth_trajectory(
    interpolator: Interpolator,
    sample_mol: MoleculeData,
    target_mol: MoleculeData,
    time_points: torch.Tensor,
    device: str,
):
    """
    Build a smooth interpolation trajectory for a single molecule pair.

    All stochastic events (deletion/insertion times, discrete token flip
    times, insertion noise) are sampled once in pre_sample_events, and
    then each frame is computed deterministically from those fixed draws.
    """
    events = pre_sample_events(interpolator, sample_mol, target_mol, device)
    trajectory = []
    for t in time_points:
        state = compute_state_at_t(interpolator, events, t.item())
        trajectory.append(state)
    return trajectory


def run(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)

    hydra.utils.log.info("Instantiating preprocessing...")
    preprocessing = hydra.utils.instantiate(cfg.data.preprocessing)

    vocab = preprocessing.vocab
    distributions = preprocessing.distributions

    cfg.data.vocab = vocab
    print(distributions)

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
        distributions=distributions,
    )
    datamodule.setup()

    samples_batched, targets_batched = next(iter(datamodule.val_dataloader()[0]))

    samples_list = samples_batched.to_data_list()[:32]
    targets_list = targets_batched.to_data_list()[:32]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    interpolator = Interpolator(
        vocab=vocab,
        distributions=distributions,
        ins_noise_scale=0.25,
        ins_schedule=SmoothstepSchedule(shift=0.65),
        del_schedule=SmoothstepSchedule(shift=0.65),
        sub_schedule=SmoothstepSchedule(shift=1.0),
        sub_e_schedule=SmoothstepSchedule(shift=1.0),
        c_del=0.0,
        c_ins=1e8,
        c_sub=0.0,
        c_move=1.0,
    )

    integrator = hydra.utils.instantiate(
        cfg.model.integrator,
        vocab=vocab,
        distributions=distributions,
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

    # Exclude the endpoint t=1 to stay in [0, 1)
    # time_points = time_points[:-1]

    results = []
    targets_single = []

    for sample_single, target_single in zip(samples_list, targets_list):
        trajectory = smooth_trajectory(
            interpolator,
            sample_single,
            target_single,
            time_points,
            device,
        )
        results.append(trajectory)
        targets_single.append(target_single)

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
