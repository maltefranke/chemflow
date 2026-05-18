import torch
import torch.nn.functional as F

from chemflow.model.gmm import (
    sample_from_typed_gmm,
)
from chemflow.utils.utils import (
    EDGE_ALIGNER,
)

from chemflow.dataset.molecule_data import (
    MoleculeData,
    PointCloud,
    MoleculeBatch,
    join_molecules_with_atoms,
    join_molecules_with_predicted_edges,
    filter_nodes,
)
from chemflow.dataset.vocab import Vocab, Distributions
from chemflow.flow_matching.schedules import (
    FastPowerSchedule,
    KappaSchedule,
    LinearSchedule,
)


class RateIntegrator:
    """
    Integrator for flow matching with discrete and continuous updates.

    Handles integration steps for molecular generation using rate-based processes
    for substitutions, deletions, and insertions. Supports both typed and untyped
    categorical strategies and flexible/fixed atom count strategies.

    Args:
        vocab: Vocabulary containing atom, edge, and charge tokens
        distributions: Distribution statistics for the dataset
        gmm_params: Parameters for Gaussian Mixture Model used in insertions
        n_atoms_strategy: Strategy for number of atoms ("flexible" or "fixed")
        num_integration_steps: Number of integration steps from t=0 to t=1
        time_strategy: Time scheduling strategy ("linear" or "log")
        device: Device to run computations on
    """

    def __init__(
        self,
        vocab: Vocab,
        distributions: Distributions,
        gmm_params,
        n_atoms_strategy="flexible",
        num_integration_steps=100,
        time_strategy="log",
        ins_noise_scale=0.01,
        del_schedule: KappaSchedule | None = None,
        ins_schedule: KappaSchedule | None = None,
        sub_schedule: KappaSchedule | None = None,
        sub_e_schedule: KappaSchedule | None = None,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.vocab = vocab
        self.distributions = distributions
        self.gmm_params = gmm_params
        self.ins_noise_scale = ins_noise_scale
        self.n_atoms_strategy = n_atoms_strategy
        self.time_strategy = time_strategy
        self.num_integration_steps = num_integration_steps
        self.device = device

        if del_schedule is None:
            self.del_schedule = FastPowerSchedule(beta=2.5)
        else:
            self.del_schedule = del_schedule

        if ins_schedule is None:
            self.ins_schedule = FastPowerSchedule(beta=2.5)
        else:
            self.ins_schedule = ins_schedule

        if sub_schedule is None:
            self.sub_schedule = LinearSchedule()
        else:
            self.sub_schedule = sub_schedule

        if sub_e_schedule is None:
            self.sub_e_schedule = self.sub_schedule
        else:
            self.sub_e_schedule = sub_e_schedule

        self.edge_aligner = EDGE_ALIGNER

        n_atoms_distribution = self.distributions.n_atoms_distribution
        # NOTE true max_atoms would be 2* max(n_atoms_distr) to account for ins / del,
        # but we add a buffer instead to be safe and avoid OOM errors
        self.max_atoms = len(n_atoms_distribution) + 10  # add some buffer

        self._cat_atom = torch.distributions.Categorical(
            probs=distributions.atom_type_distribution.to(device)
        )
        self._cat_charge = torch.distributions.Categorical(
            probs=distributions.charge_type_distribution.to(device)
        )
        self._cat_edge = torch.distributions.Categorical(
            probs=distributions.edge_type_distribution.to(device)
        )
        self._cat_device = (
            torch.device(device) if not isinstance(device, torch.device) else device
        )

    def _distr_to_device(self, device: torch.device):
        # Keep all base categorical distributions on the current batch device.
        # This avoids mixed-device ops in insertion/substitution branches.
        # Rebuilding Categoricals allocates a new logits tensor each call, so
        # short-circuit when the device hasn't changed to avoid per-step churn.
        device = (
            torch.device(device) if not isinstance(device, torch.device) else device
        )
        if getattr(self, "_cat_device", None) == device:
            return
        self._cat_atom = torch.distributions.Categorical(
            probs=self._cat_atom.probs.to(device)
        )
        self._cat_charge = torch.distributions.Categorical(
            probs=self._cat_charge.probs.to(device)
        )
        self._cat_edge = torch.distributions.Categorical(
            probs=self._cat_edge.probs.to(device)
        )
        self._cat_device = device

    def get_time_steps(self, num_steps: int | None = None) -> list[float]:
        if num_steps is None:
            num_steps = self.num_integration_steps

        if self.time_strategy == "linear":
            time_points = torch.linspace(0, 1, num_steps + 1).tolist()

        elif self.time_strategy == "log":
            time_points = (1 - torch.logspace(0, -3, num_steps)).tolist()

        else:
            raise ValueError(f"Invalid time strategy: {self.time_strategy}")

        step_sizes = [t1 - t0 for t0, t1 in zip(time_points[:-1], time_points[1:])]

        return step_sizes

    def sample_insertions(
        self,
        ins_gmm_dict: dict,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample insertions from GMM (works with batch_ids).

        Args:
            ins_gmm_dict: dict - GMM parameters per graph
            t: Time tensor

        Returns:
            new_atoms: PointCloud - new atoms
        """
        # We will only do one insertion per provided GMM at a time
        num_ins = 1

        K = self.gmm_params.K
        D = self.gmm_params.D
        A = len(self.vocab.atom_tokens)
        C = len(self.vocab.charge_tokens)

        # Sample from GMM for this graph
        # Keep batch dimension
        sampled_x, sampled_a, sampled_c = sample_from_typed_gmm(
            ins_gmm_dict, num_ins, K, D, A, C
        )

        new_atoms = PointCloud(
            x=sampled_x.view(-1, self.gmm_params.D),
            a=sampled_a.view(-1),
            c=sampled_c.view(-1),
        )

        return new_atoms

    def integrate_step_gnn(
        self,
        mol_t: MoleculeData,
        mol_1_pred: MoleculeData,
        do_sub_a_probs: torch.Tensor,
        do_sub_e_probs: torch.Tensor,
        do_del_probs: torch.Tensor,
        num_ins_pred: torch.Tensor,
        ins_gmm_preds: torch.Tensor,
        t: torch.Tensor,
        dt: float,
        eps: float = 1e-6,
        h_latent: torch.Tensor = None,
        ins_edge_head=None,
    ) -> MoleculeBatch:
        """
        Integrate one step of the stochastic process for GNN models.

        Works with variable-sized graphs using batch_ids.
        All tensors are flat (no padding).

        Args:
            mol_t: Current molecule state at time t
            mol_1_pred: Predicted molecule state at time t=1
            do_sub_a_probs: Shape (N,) - probabilities for atom type substitution decisions (scalar per node)
            do_sub_e_probs: Shape (E,) - probabilities for edge type substitution decisions (scalar per edge)
            do_del_probs: Shape (N,) - probabilities for deletion decisions (scalar per node)
            num_ins_pred: Shape (N,) - predicted insertion count per node (Poisson rate, includes 0-count nodes)
            ins_gmm_preds: Dict with GMM parameters (mu, sigma, pi, a_probs, c_probs) for each node
            t: Shape (num_graphs,) - current time for each graph in the batch
            dt: Time step size
            eps: Small epsilon value for numerical stability (default: 1e-6)
            h_latent: Shape (N, hidden_dim) - latent node features for edge prediction (optional)
            ins_edge_head: InsertionEdgeHead instance for predicting edges (optional)

        Returns:
            mol_t_final: MoleculeBatch - updated molecule after one integration step
        """
        # Keep all base categorical distributions on the current batch device.
        # This avoids mixed-device ops in insertion/substitution branches.
        active_device = mol_t.x.device
        self._distr_to_device(active_device)

        x_t, a_t, _, e_t, edge_index, batch_id = mol_t.unpack()
        x_1, a_1, c_1, e_1, edge_index_1, _ = mol_1_pred.unpack()

        """FOR NOISING"""
        gamma = torch.clamp(1.0 - t**2, min=0.0)
        noise_scale = 0.0
        noise_weight = gamma * noise_scale
        noise_weight_node = noise_weight[batch_id]

        # Rate for movement
        # TODO if we change the position interpolation, need to change this
        move_rate = 1 / (1 - t).clamp(min=eps, max=1.0 - eps)
        move_rate_node = move_rate[batch_id]

        # Rate for atom-level substitution
        sub_rate = self.sub_schedule.rate(t)
        sub_rate_node = sub_rate[batch_id]

        # Rate for edge-level substitution (separate schedule)
        sub_e_rate = self.sub_e_schedule.rate(t)

        # Rate for insertion
        ins_rate = self.ins_schedule.rate(t)
        ins_rate_node = ins_rate[batch_id]

        # Rate for deletion
        del_rate = self.del_schedule.rate(t)
        del_rate_node = del_rate[batch_id]

        # 1. Update positions (Euler-Maruyama scheme)
        x_t_original = x_t

        velocity = (x_1 - x_t) * move_rate_node.view(-1, 1)
        # Scaffold preservation: when mol_t carries a scaffold_mask, freeze the
        # positions of scaffold atoms by zeroing their velocity. Edit-channel
        # masking (do_del / do_sub_a / scaffold-scaffold do_sub_e) is done by
        # the caller in lightning_module.sample(), so insertions seeded *from*
        # scaffold atoms (decoration growth) remain unaffected.
        scaffold_mask = getattr(mol_t, "scaffold_mask", None)
        if scaffold_mask is not None:
            # this seemed to work better
            # velocity[scaffold_mask.bool()] = 0.0
            pass
        x_t = x_t + velocity * dt

        """INSERTION"""
        # Single Poisson over all nodes (including zero-count nodes).
        # No binary gate: num_ins_pred is the Poisson rate for every node.
        num_graphs = t.shape[0]

        expected_num_ins = (num_ins_pred + noise_weight_node) * ins_rate_node * dt
        # NOTE in OneFlow, instead of sampling Poisson, they sample Bernoulli
        do_ins = torch.rand_like(expected_num_ins) < expected_num_ins

        # Fail-safe: ensure that the number of atoms per graph won't be greater than the max number of atoms
        if do_ins.any():
            n_atoms_per_graph = torch.bincount(batch_id, minlength=num_graphs)
            n_ins_per_graph = torch.bincount(batch_id[do_ins], minlength=num_graphs)
            overflow = (n_atoms_per_graph + n_ins_per_graph - self.max_atoms).clamp(
                min=0
            )
            if overflow.any():
                overflow_graphs = (
                    torch.nonzero(overflow, as_tuple=False).flatten().tolist()
                )
                for g in overflow_graphs:
                    n_to_remove = int(overflow[g].item())
                    removed_indices = torch.where((batch_id == g) & do_ins)[0]
                    removed_indices = removed_indices[
                        torch.randperm(removed_indices.shape[0], device=do_ins.device)
                    ]
                    do_ins[removed_indices[:n_to_remove]] = False

        """NODE DELETION or SUBSTITUTION"""
        # 1. Scale probabilities by time (converting prob -> rate * dt)
        p_sub_scaled = (
            (do_sub_a_probs.view(-1) + noise_weight_node) * sub_rate_node * dt
        )
        p_del_scaled = (do_del_probs.view(-1) + noise_weight_node) * del_rate_node * dt

        # 2. Sum rates first (Paper Formulation)
        # "probability of ANY edit is h(lambda_sub + lambda_del)"
        p_any_edit = p_sub_scaled + p_del_scaled

        # 3. Sample "Does an edit happen?" (1 Random Sample)
        do_edit = torch.rand_like(p_any_edit) < p_any_edit

        # 4. If edit happens, decide Which Type (1 Random Sample)
        # We only need to compute this splitting logic where edits actually happen,
        # but doing it globally and masking is usually faster than indexing on GPUs
        # due to memory fragmentation.
        prob_cond_del = p_del_scaled / (p_any_edit + 1e-8)
        is_deletion_type = torch.rand_like(prob_cond_del) < prob_cond_del

        # 5. Assign Final Masks
        # An edit occurs AND it is a deletion type
        do_del = do_edit & is_deletion_type

        # An edit occurs AND it is NOT a deletion type (therefore substitution)
        do_sub_a = do_edit & (~is_deletion_type)

        # 6. Apply
        # a_1 is your predicted token values from Step 2
        a_t[do_sub_a] = a_1[do_sub_a]

        """EDGE SUBSTITUTION"""
        # Get current edge types (already indices)

        # we will deal with only the triu edge types and symmetrize later
        edge_infos = self.edge_aligner.align_edges(
            source_group=(edge_index, [e_t, do_sub_e_probs]),
            target_group=(edge_index_1, [e_1]),
        )
        e_triu, do_sub_e_probs_triu, e_1_triu = edge_infos["edge_attr"]
        edge_index_triu, _ = edge_infos["edge_index"]

        # Use probabilities directly (sigmoid already applied in sample())
        p_sub_e = do_sub_e_probs_triu.view(-1)

        # Get batch_id for edges from the source nodes of the edges
        # edge_index_triu[0] gives the source node indices for each edge
        batch_id_edge = batch_id[edge_index_triu[0]]

        noise_weight_edge = noise_weight[batch_id_edge]

        # Calculate rate for edges using the separate edge substitution schedule
        sub_e_rate_triu = sub_e_rate[batch_id_edge]
        p_mod_e = (p_sub_e + noise_weight_edge) * sub_e_rate_triu * dt
        p_mod_e = p_mod_e.view(-1)
        do_sub_e = torch.rand_like(p_mod_e) < p_mod_e
        e_triu[do_sub_e] = e_1_triu[do_sub_e]

        # Finally, symmetrize the edge types
        edge_index, e_attrs = self.edge_aligner.symmetrize_edges(
            edge_index_triu, [e_triu]
        )
        e_t = e_attrs[0]

        mol = MoleculeBatch(
            x=x_t,
            a=a_t,
            c=c_1,  # always take the predicted charge, c not interpolated
            e=e_t,
            edge_index=edge_index,
            batch=batch_id,
        )

        # Carry scaffold_mask through so the deletion/insertion stages below
        # (and any downstream filter_nodes / join / sort_nodes_by_batch) can
        # keep it aligned with the evolving node set.
        if scaffold_mask is not None:
            mol.scaffold_mask = scaffold_mask

        if self.n_atoms_strategy != "fixed":
            # Build index mapping: original_idx -> post_deletion_idx (or -1 if deleted)
            N_original = mol.num_nodes
            keep_mask = ~do_del

            # Fail-safe: prevent deletion of entire sample (keep at least 2 nodes per batch_id)
            n_kept_per_graph = torch.bincount(batch_id[keep_mask], minlength=num_graphs)
            under = (2 - n_kept_per_graph).clamp(min=0)
            if under.any():
                under_graphs = torch.nonzero(under, as_tuple=False).flatten().tolist()
                for g in under_graphs:
                    n_to_restore = int(under[g].item())
                    deleted_in_graph_idx = torch.where((batch_id == g) & do_del)[0]
                    n_restore = min(n_to_restore, deleted_in_graph_idx.shape[0])
                    if n_restore > 0:
                        keep_mask[deleted_in_graph_idx[:n_restore]] = True

            original_to_postdel = torch.full(
                (N_original,), -1, dtype=torch.long, device=self.device
            )
            kept_indices = torch.where(keep_mask)[0]
            original_to_postdel[kept_indices] = torch.arange(
                len(kept_indices), device=self.device
            )

            # 3. Remove the deleted nodes
            if do_del.any():
                mol = filter_nodes(mol, keep_mask)

            # 4. Add the new insertions
            # Allow insertions from ALL nodes, including deleted ones.
            # The spawn node provides GMM parameters for the new atom's position,
            # but the spawn node itself doesn't need to exist after deletion.
            do_ins_valid = do_ins
            if do_ins_valid.any():
                ins_gmm_dict = {
                    "mu": ins_gmm_preds["mu"][do_ins_valid],
                    "sigma": ins_gmm_preds["sigma"][do_ins_valid],
                    "pi": ins_gmm_preds["pi"][do_ins_valid],
                    "a_probs": ins_gmm_preds["a_probs"][do_ins_valid],
                    "c_probs": ins_gmm_preds["c_probs"][do_ins_valid],
                }

                t_ins = t[batch_id[do_ins_valid]] + dt
                new_atoms = self.sample_insertions(ins_gmm_dict, t_ins)

                new_atoms.batch = batch_id[do_ins_valid]

                # Determine fallback edge distribution
                edge_dist = self.distributions.edge_type_distribution.to(self.device)

                # Build mapping: original spawn index -> new atom index
                # New atoms are ordered by the True values in do_ins_valid
                orig_to_new_atom = torch.full(
                    (N_original,), -1, dtype=torch.long, device=self.device
                )
                spawn_orig_indices = torch.where(do_ins_valid)[0]
                orig_to_new_atom[spawn_orig_indices] = torch.arange(
                    len(spawn_orig_indices), dtype=torch.long, device=self.device
                )

                # Predict edges for insertions if head is available
                ins_edge_logits = None
                ins_edge_new_atom_idx = None
                ins_edge_target_idx = None
                e_t_ins = None

                # ins -> ins predicted edges (between newly inserted atoms)
                e1_ii = None
                e_t_ins_ii = None
                ins_ii_src_local = None
                ins_ii_dst_local = None

                if ins_edge_head is not None and h_latent is not None:
                    # ── ins → existing edges ──────────────────────────────────
                    # Use extrapolated t=1 for edge prediction, then adjust noise level
                    orig_spawn_idx, orig_existing_idx, ins_edge_logits = (
                        ins_edge_head.predict_edges_for_insertion(
                            h=h_latent,
                            x=x_t_original,  # use original frame for edge prediction
                            node_atom_types=a_t,
                            batch=batch_id,
                            insertion_mask=do_ins_valid,
                            ins_x=new_atoms.x,
                            ins_a=new_atoms.a,
                            ins_c=new_atoms.c,
                        )
                    )

                    # sample e1_ins_types
                    ins_edge_probs = F.softmax(ins_edge_logits, dim=-1)
                    e1_ins_types = torch.distributions.Categorical(
                        probs=ins_edge_probs
                    ).sample()
                    e1_ins_types = F.one_hot(
                        e1_ins_types, num_classes=len(self.vocab.edge_tokens)
                    )

                    # Adjust to noise level of next time step t+dt
                    t_ins_orig = t[batch_id[orig_spawn_idx]] + dt
                    kappa_t_e = self.sub_e_schedule.kappa_t(t_ins_orig).unsqueeze(1)

                    prior_edge_probs = self._cat_edge.probs.unsqueeze(0).expand(
                        e1_ins_types.shape[0], -1
                    )
                    e_t_ins = (
                        prior_edge_probs * (1 - kappa_t_e)
                        + e1_ins_types.to(dtype=prior_edge_probs.dtype) * kappa_t_e
                    )
                    # Cast to fp32 + explicit row-normalize: convex combos of bf16
                    # probability tensors can drift outside the [1 - 1e-6, 1 + 1e-6]
                    # simplex tolerance enforced by ``Categorical(validate_args=True)``.
                    e_t_ins = e_t_ins.float()
                    e_t_ins = e_t_ins / e_t_ins.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                    e_t_ins = torch.distributions.Categorical(probs=e_t_ins).sample()

                    if ins_edge_logits is not None and ins_edge_logits.numel() > 0:
                        # Map spawn indices to new atom indices (0, 1, 2, ...)
                        # This works for both deleted and non-deleted spawn nodes
                        ins_edge_new_atom_idx = orig_to_new_atom[orig_spawn_idx]

                        # Map existing-endpoint indices to post-deletion space.
                        # Endpoints must still exist in the post-deletion molecule.
                        ins_edge_target_idx = original_to_postdel[orig_existing_idx]

                        # Filter out edges where:
                        # - Target was deleted (target doesn't exist in mol)
                        # - Spawn wasn't in do_ins_valid (shouldn't happen, but be safe)
                        valid_edges = (ins_edge_target_idx >= 0) & (
                            ins_edge_new_atom_idx >= 0
                        )
                        if valid_edges.any():
                            ins_edge_new_atom_idx = ins_edge_new_atom_idx[valid_edges]
                            ins_edge_target_idx = ins_edge_target_idx[valid_edges]
                            ins_edge_logits = ins_edge_logits[valid_edges]
                            e_t_ins = e_t_ins[valid_edges]
                        else:
                            ins_edge_logits = None
                            e_t_ins = None

                    # ── ins → ins edges (between newly inserted atoms) ────────
                    # Build upper-triangular pairs within the same graph.
                    ii_src_list = []
                    ii_dst_list = []
                    ii_spawn_src_list = []

                    for g in new_atoms.batch.unique():
                        mask_g = new_atoms.batch == g
                        local_idx_g = torch.where(mask_g)[0]
                        if local_idx_g.numel() >= 2:
                            pairs = torch.combinations(local_idx_g, r=2)
                            ii_src_list.append(pairs[:, 0])
                            ii_dst_list.append(pairs[:, 1])
                            ii_spawn_src_list.append(spawn_orig_indices[pairs[:, 0]])

                    if ii_src_list:
                        ii_src = torch.cat(ii_src_list)
                        ii_dst = torch.cat(ii_dst_list)
                        ii_spawn_src = torch.cat(ii_spawn_src_list)

                        ii_logits = ins_edge_head.forward_ins_to_ins(
                            h=h_latent,
                            x=x_t_original,
                            spawn_src_idx=ii_spawn_src,
                            ins_x_src=new_atoms.x[ii_src],
                            ins_a_src=new_atoms.a[ii_src],
                            ins_c_src=new_atoms.c[ii_src],
                            ins_a_dst=new_atoms.a[ii_dst],
                            ins_x_dst=new_atoms.x[ii_dst],
                        )

                        # Pass logits directly: skips the bf16 softmax precision
                        # loss that violates the simplex tolerance.
                        e1_ii = torch.distributions.Categorical(
                            logits=ii_logits.float()
                        ).sample()
                        e1_ii = F.one_hot(
                            e1_ii, num_classes=len(self.vocab.edge_tokens)
                        )

                        # Adjust to noise level t+dt using spawn_src's graph time
                        t_ii = t[batch_id[ii_spawn_src]] + dt
                        kappa_t_ii = self.sub_e_schedule.kappa_t(t_ii).unsqueeze(1)
                        prior_ii = self._cat_edge.probs.unsqueeze(0).expand(
                            e1_ii.shape[0], -1
                        )
                        e_t_ins_ii = (
                            prior_ii * (1 - kappa_t_ii)
                            + e1_ii.to(dtype=prior_ii.dtype) * kappa_t_ii
                        )
                        # e_t_ins_ii = F.softmax(e_t_ins_ii, dim=-1)
                        e_t_ins_ii = torch.distributions.Categorical(
                            probs=e_t_ins_ii
                        ).sample()

                        ins_ii_src_local = ii_src
                        ins_ii_dst_local = ii_dst

                # Check if we have predicted edge logits for insertions
                use_predicted_edges = (
                    ins_edge_logits is not None
                    and ins_edge_new_atom_idx is not None
                    and ins_edge_target_idx is not None
                    and e_t_ins is not None
                    and e_t_ins.numel() > 0
                )

                if use_predicted_edges:
                    # Use predicted edges from InsertionEdgeHead
                    # ins_edge_new_atom_idx: identifies which new atom (0, 1, 2, ...)
                    # ins_edge_target_idx: in post-deletion space (valid indices in mol)
                    mol = join_molecules_with_predicted_edges(
                        mol=mol,
                        new_atoms=new_atoms,
                        e_ins=e_t_ins,
                        spawn_node_idx=ins_edge_new_atom_idx,
                        target_node_idx=ins_edge_target_idx,
                        fallback_edge_dist=edge_dist,
                        e_ins_to_ins=e_t_ins_ii,
                        ins_to_ins_src_idx=ins_ii_src_local,
                        ins_to_ins_dst_idx=ins_ii_dst_local,
                    )
                else:
                    # Fall back to random edge sampling
                    mol = join_molecules_with_atoms(mol, new_atoms, edge_dist)

        # max_seqlen may have shifted from inserts/deletes; recompute once for
        # the next backbone forward (flash-attn varlen needs a Python int).
        mol.max_seqlen = int(torch.bincount(mol.batch).max().item())
        return mol
