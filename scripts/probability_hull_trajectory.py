"""Time-dependent GMM density tracking during generation from scratch.

Samples a random prior graph at t=0, runs the standard integration loop
to t=1, and records the GMM predictions (mu, sigma, pi, insertion rate)
at every time step. Saves a single .pt trajectory file for downstream
animation in Jupyter with py3Dmol.

Usage:
    python scripts/probability_hull_trajectory.py \
        --ckpt outputs/epoch=499-step=9500.ckpt \
        --output_dir hulls/trajectory \
        --num_steps 50 \
        --target_n_atoms 9
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
    load_config_and_preprocessing,
    load_model,
)
from chemflow.dataset.molecule_data import MoleculeBatch
from chemflow.flow_matching.sampling import sample_prior_graph
from chemflow.utils.utils import index_to_token


def _save_frame(
    gmm,
    ins_rate,
    mol_t,
    coord_std,
    vocab,
):
    """Extract GMM data for a single-molecule batch."""
    n_nodes = mol_t.num_nodes

    mu_i = gmm["mu"][:n_nodes]          # [N, K, 3]
    sigma_i = gmm["sigma"][:n_nodes]    # [N, K]
    pi_i = gmm["pi"][:n_nodes]          # [N, K]
    rate_i = ins_rate[:n_nodes]          # [N] or [N, 1]
    if rate_i.ndim == 1:
        rate_i = rate_i.unsqueeze(-1)

    N, K, _ = mu_i.shape
    mu_real = mu_i.reshape(N * K, 3).cpu() * coord_std
    sigma_real = sigma_i.reshape(N * K).cpu() * coord_std
    weights = (rate_i * pi_i).reshape(N * K).cpu()

    coords_real = mol_t.x.cpu().numpy() * coord_std
    atom_symbols = [
        index_to_token(vocab.atom_tokens, int(a))
        for a in mol_t.a.cpu().numpy()
    ]

    return {
        "mu": mu_real,
        "sigma": sigma_real,
        "weights": weights,
        "atom_coords": torch.as_tensor(coords_real, dtype=torch.float32),
        "atom_symbols": atom_symbols,
    }


@torch.no_grad()
def run_trajectory(
    ckpt_path: str,
    num_steps: int = 50,
    target_n_atoms: int | None = None,
    output_dir: str = "hulls/trajectory",
):
    """Generate from noise (t=0 -> t=1), saving GMM predictions per step.

    Saves ``output_dir/gen_traj.pt`` containing:
        frames    : list[dict]  one entry per time step
        num_steps : int
    """
    os.makedirs(output_dir, exist_ok=True)

    cfg, vocab, distributions, token_prior = load_config_and_preprocessing()
    module = load_model(cfg, vocab, distributions, token_prior, ckpt_path)
    module.to(DEVICE)
    module.eval()

    coord_std = distributions.coordinate_std
    n_atoms_strategy = module.n_atoms_strategy

    prior = sample_prior_graph(distributions, n_atoms=target_n_atoms)
    mol_t = MoleculeBatch.from_data_list([prior]).to(DEVICE)
    _ = mol_t.remove_com()

    model = module._get_model()
    model.eval()

    guidance = module.cfg_guidance
    integrator = module.integrator
    step_sizes = integrator.get_time_steps(num_steps)

    t = torch.zeros(1, device=DEVICE)
    preds = None

    overrides: dict = {}
    if target_n_atoms is not None and guidance.has_signal("n_atoms"):
        overrides["n_atoms"] = torch.tensor([target_n_atoms], device=DEVICE)

    frames = []

    for step_i, step_size in enumerate(step_sizes):
        batch_id = mol_t.batch
        prev_preds = preds

        preds = guidance.guided_predict(
            model, mol_t, t, prev_preds, overrides,
        )

        gmm_pred = preds["gmm_head"]

        # ins_rate_head already has F.softplus applied inside model.forward()
        ins_rate_pred = preds["ins_rate_head"]
        if ins_rate_pred.ndim > 1:
            ins_rate_pred = ins_rate_pred.squeeze(-1)

        frame = _save_frame(
            gmm_pred, ins_rate_pred, mol_t,
            coord_std=coord_std, vocab=vocab,
        )
        frame["t"] = float(t[0].item())
        frames.append(frame)

        # --- replicate sample() integration logic exactly ---
        x1_pred = preds["pos_head"]

        a_pred = preds["atom_type_head"]
        a_pred = F.softmax(a_pred, dim=-1)
        a_pred = torch.distributions.Categorical(probs=a_pred).sample()

        c_pred = preds["charge_head"]
        c_pred = F.softmax(c_pred, dim=-1)
        c_pred = torch.distributions.Categorical(probs=c_pred).sample()

        e_pred = preds["edge_type_head"]
        e_pred = F.softmax(e_pred, dim=-1)
        e_pred = torch.distributions.Categorical(probs=e_pred).sample()

        mol_1_pred = MoleculeBatch(
            x=x1_pred,
            a=a_pred,
            c=c_pred,
            e=e_pred,
            edge_index=mol_t.edge_index.clone(),
            batch=batch_id.clone(),
        )

        do_sub_a_logits = preds["do_sub_a_head"]
        do_sub_e_logits = preds["do_sub_e_head"]
        do_del_logits = preds["do_del_head"]

        if do_sub_a_logits.ndim > 1:
            do_sub_a_logits = do_sub_a_logits.squeeze(-1)
        if do_sub_e_logits.ndim > 1:
            do_sub_e_logits = do_sub_e_logits.squeeze(-1)
        if do_del_logits.ndim > 1:
            do_del_logits = do_del_logits.squeeze(-1)

        do_sub_a_probs = torch.sigmoid(do_sub_a_logits)
        do_sub_e_probs = torch.sigmoid(do_sub_e_logits)
        do_del_probs = torch.sigmoid(do_del_logits)

        num_ins_pred = ins_rate_pred

        if n_atoms_strategy == "fixed":
            num_ins_pred = torch.zeros_like(num_ins_pred)
            do_del_probs = torch.zeros_like(do_del_probs)

        ins_edge_head = getattr(model, "ins_edge_head", None)

        mol_t = integrator.integrate_step_gnn(
            mol_t=mol_t.clone(),
            mol_1_pred=mol_1_pred.clone(),
            do_sub_a_probs=do_sub_a_probs,
            do_sub_e_probs=do_sub_e_probs,
            do_del_probs=do_del_probs,
            num_ins_pred=num_ins_pred,
            ins_gmm_preds=gmm_pred,
            t=t,
            dt=step_size,
            h_latent=preds.get("h_latent"),
            ins_edge_head=ins_edge_head,
        )
        _ = mol_t.remove_com()
        t = t + step_size

        if (step_i + 1) % 10 == 0 or step_i == 0:
            print(
                f"  step {step_i + 1}/{len(step_sizes)}  "
                f"t={float(t[0]):.4f}  "
                f"n_atoms={mol_t.num_nodes}"
            )

    out_path = os.path.join(output_dir, "gen_traj.pt")
    torch.save(
        {
            "frames": frames,
            "num_steps": len(frames),
        },
        out_path,
    )
    print(f"Saved {len(frames)} frames -> {out_path}")
    return out_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="GMM density trajectory: generation from scratch",
    )
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument(
        "--target_n_atoms", type=int, default=None,
        help="Target atom count (sampled from distribution if not set)",
    )
    parser.add_argument("--output_dir", type=str, default="hulls/trajectory")
    args = parser.parse_args()

    run_trajectory(
        ckpt_path=args.ckpt,
        num_steps=args.num_steps,
        target_n_atoms=args.target_n_atoms,
        output_dir=args.output_dir,
    )
