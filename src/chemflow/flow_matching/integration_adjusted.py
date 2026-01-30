import torch

from chemflow.flow_matching.gmm import (
    sample_from_typed_gmm,
    sample_from_gmm,
)
from chemflow.utils import (
    token_to_index,
    EdgeAligner,
    validate_no_cross_batch_edges,
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
from chemflow.flow_matching.schedules import FastPowerSchedule, KappaSchedule


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
        cat_strategy: Categorical strategy ("uniform-sample" or "mask")
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
        cat_strategy="uniform-sample",
        n_atoms_strategy="flexible",
        num_integration_steps=100,
        time_strategy="log",
        del_schedule: KappaSchedule | None = None,
        ins_schedule: KappaSchedule | None = None,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.vocab = vocab
        self.distributions = distributions
        self.gmm_params = gmm_params
        self.cat_strategy = cat_strategy
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

        if self.cat_strategy == "mask":
            self.edge_mask_index = token_to_index(self.vocab.edge_tokens, "<MASK>")
            self.atom_mask_index = token_to_index(self.vocab.atom_tokens, "<MASK>")

        self.edge_aligner = EdgeAligner()

    def get_time_steps(self, num_steps: int | None = None) -> list[float]:
        if num_steps is None:
            num_steps = self.num_integration_steps

        if self.time_strategy == "linear":
            time_points = torch.linspace(0, 1, num_steps + 1).tolist()

        elif self.time_strategy == "log":
            # torch requires the log of the start and end points
            start_log = torch.log10(torch.tensor(0.01, device=self.device))
            end_log = torch.log10(torch.tensor(1.0, device=self.device))
            time_points = (
                1 - torch.logspace(start_log, end_log, num_steps + 1)
            ).tolist()
            time_points.reverse()
        else:
            raise ValueError(f"Invalid time strategy: {self.time_strategy}")

        step_sizes = [t1 - t0 for t0, t1 in zip(time_points[:-1], time_points[1:])]

        return step_sizes

    def discrete_integration_step_gnn(
        self,
        class_curr: torch.Tensor,
        class_probs_1: torch.Tensor,
        t: torch.Tensor,
        batch_idx: torch.Tensor,
        dt: float,
        noise_scale: float = 0.0,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Performs a single Euler-Forward integration step for Discrete Flow Matching
        adapted for Graph Neural Networks (stacked node batching).

        Args:
            class_curr (Tensor): Current node token indices. Shape [N, 1] or [N].
            class_probs_1 (Tensor): Predicted target probabilities. Shape [N, Vocab].
            t (Tensor): Current time per graph. Shape [Batch_Size].
            batch_idx (Tensor): Batch index for each node (e.g., from PyG Data.batch). Shape [N].
            dt (float): Integration timestep size.
            noise_scale (float): Additive uniform noise rate.
            mask (Tensor, optional): Boolean mask (True to ignore node). Shape [N, 1] or [N].

        Returns:
            a_next (Tensor): Updated node tokens. Shape [N, 1].
        """
        # 0. Standardization
        # Ensure a_curr is [N, 1] for scatter consistency, but we often need [N] for logic
        if class_curr.ndim == 1:
            class_curr = class_curr.unsqueeze(-1)  # [N, 1]

        # 1. Broadcast Time to Nodes
        # t is [Batch_Size], we need [N, 1] to match a_1_probs
        # t[batch_idx] expands it to [N], then view adds the vocab dim
        times = t[batch_idx].view(-1, 1).clamp(min=1e-5, max=1.0 - 1e-5)

        # 2. Calculate Deterministic Rate: Rate = p_1(y) / (1 - t)
        # rate_scalar will be [N, 1]
        rate_scalar = 1.0 / (1.0 - times)
        rates = class_probs_1 * rate_scalar

        # 3. Add Stochastic Noise (Optional)
        if noise_scale > 0:
            rates = rates + noise_scale

        # 4. Zero out the Diagonal (Self-transitions)
        # rates is [N, Vocab], a_curr is [N, 1]
        # We set the rate of staying (x -> x) to 0 temporarily
        rates.scatter_(-1, class_curr, 0.0)

        # 5. Convert Rates to Probabilities
        step_probs = rates * dt

        # 6. Calculate Probability to Stay
        # P(stay) = 1 - sum(P(jump to y))
        off_diag_sum = step_probs.sum(dim=-1, keepdim=True)
        stay_prob = (1.0 - off_diag_sum).clamp(min=0.0)

        # Insert stay probability back into the current token's position
        step_probs.scatter_(-1, class_curr, stay_prob)

        # 7. Sample Next State
        # Categorical samples from [N, Vocab] -> returns [N]
        class_pred = torch.distributions.Categorical(probs=step_probs).sample()

        # Reshape back to [N, 1] to maintain input shape consistency
        class_pred = class_pred.view(-1, 1)

        # 8. Apply Mask (if provided)
        if mask is not None:
            if mask.ndim == 1:
                mask = mask.unsqueeze(-1)
            class_pred = torch.where(mask, class_curr, class_pred)

        return class_pred

    def sample_insertions(
        self,
        ins_gmm_dict: dict,
        N_a: int,
        N_c: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample birth process for GNN (works with batch_ids).

        Args:
            ins_gmm_dict: dict - GMM parameters per graph
            N_a: Number of atom types
            N_c: Number of charge types

        Returns:
            new_atoms: PointCloud - new atoms
        """
        # We will only do one insertion at a time
        num_ins = 1

        # Sample from GMM for this graph
        # Keep batch dimension
        if self.cat_strategy == "uniform-sample":
            sampled_x, sampled_a, sampled_c = sample_from_typed_gmm(
                ins_gmm_dict, num_ins, self.gmm_params.K, self.gmm_params.D, N_a, N_c
            )
        else:
            sampled_x = sample_from_gmm(
                ins_gmm_dict, num_ins, self.gmm_params.K, self.gmm_params.D
            )
            sampled_a = self.atom_mask_index * torch.ones(
                (num_ins,),
                dtype=torch.long,
                device=self.device,
            )
            sampled_c = self.charge_mask_index * torch.ones(
                (num_ins,),
                dtype=torch.long,
                device=self.device,
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
        do_ins_probs: torch.Tensor,
        num_ins_pred: torch.Tensor,
        ins_gmm_preds: torch.Tensor,
        t: torch.Tensor,
        dt: float,
        eps: float = 1e-6,
        h_latent: torch.Tensor = None,
        ins_edge_head=None,
        global_ins_budget: torch.Tensor = None,
        global_del_budget: torch.Tensor = None,
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
            do_ins_probs: Shape (N,) - probabilities for insertion decisions (scalar per node)
            num_ins_pred: Shape (N,) - predicted number of insertions per node (used to compute insertion probability)
            ins_gmm_preds: Dict with GMM parameters (mu, sigma, pi, a_probs, c_probs) for each node
            t: Shape (num_graphs,) - current time for each graph in the batch
            dt: Time step size
            eps: Small epsilon value for numerical stability (default: 1e-6)
            h_latent: Shape (N, hidden_dim) - latent node features for edge prediction (optional)
            ins_edge_head: InsertionEdgeHead instance for predicting edges (optional)
            global_ins_budget: Shape (num_graphs,) - predicted total number of insertions remaining
                              for each graph (from global_ins_budget_head). If provided, uses
                              velocity-based budget allocation instead of per-node Poisson sampling.
            global_del_budget: Shape (num_graphs,) - predicted total number of deletions remaining
                              for each graph (from global_del_budget_head). If provided, uses
                              velocity-based budget allocation instead of per-node Poisson sampling.

        Returns:
            mol_t_final: MoleculeBatch - updated molecule after one integration step
        """

        x, a, c, e, edge_index, batch_id = mol_t.unpack()
        x_1, a_1, c_1, e_1, edge_index_1, _ = mol_1_pred.unpack()

        rate = 1 / (1 - t).clamp(min=eps, max=1.0 - eps)
        rate_node = rate[batch_id]

        # graph-wise rate for insertion
        ins_rate = self.ins_schedule.rate(t)
        ins_rate_node = ins_rate[batch_id]

        # node-wise rate for deletion
        del_rate = self.del_schedule.rate(t)
        del_rate_node = del_rate[batch_id]

        # 1. Update positions (Euler-Maruyama scheme)

        velocity = (x_1 - x) * rate_node.view(-1, 1)
        x = x + velocity * dt

        # 1. Handle insertions using global budget (velocity-based) or local Poisson
        num_graphs = t.shape[0]

        if global_ins_budget is not None:
            # Use velocity-based budget allocation strategy
            # global_ins_budget: Shape (num_graphs,) - predicted N_missing per graph

            # Expected number of insertions for this specific step
            step_budget_float = (
                global_ins_budget * ins_rate * dt
            )  # Shape: (num_graphs,)

            # Stochastic rounding: e.g., 1.3 inserts -> 30% chance of 2, 70% chance of 1
            floor = torch.floor(step_budget_float)
            prob_ceil = step_budget_float - floor
            extra = (torch.rand_like(prob_ceil) < prob_ceil).float()
            num_inserts_per_graph = (floor + extra).long()  # Shape: (num_graphs,)

            # Now use multinomial sampling to select which nodes spawn insertions
            # Scale local insertion logits (do_ins_probs) to probabilities
            p_ins = do_ins_probs.view(-1)

            # Initialize do_ins mask as all False
            do_ins = torch.zeros_like(p_ins, dtype=torch.bool)

            # For each graph, sample num_inserts_per_graph[g] nodes using multinomial
            for g in range(num_graphs):
                graph_mask = batch_id == g
                n_inserts_g = num_inserts_per_graph[g].item()

                if n_inserts_g > 0 and graph_mask.sum() > 0:
                    # Get insertion probabilities for this graph's nodes
                    graph_probs = p_ins[graph_mask]

                    # Clamp and normalize to ensure valid probabilities
                    graph_probs = graph_probs.clamp(min=eps)

                    # Cap n_inserts to number of available nodes
                    n_inserts_g = min(n_inserts_g, graph_mask.sum().item())

                    if n_inserts_g > 0:
                        # Multinomial sampling without replacement
                        sampled_local_idx = torch.multinomial(
                            graph_probs, n_inserts_g, replacement=False
                        )

                        # Map local indices back to global indices
                        global_indices = torch.where(graph_mask)[0]
                        sampled_global_idx = global_indices[sampled_local_idx]

                        # Set do_ins mask for selected nodes
                        do_ins[sampled_global_idx] = True
        else:
            # Fallback to original per-node Poisson sampling
            # Sample whether to insert with probability h * lambda_ins
            p_ins = do_ins_probs
            p_ins = p_ins.view(-1)
            do_ins = torch.rand_like(p_ins) < p_ins

            # for insertion, we will then sample from the poisson distribution
            expected_num_ins = num_ins_pred * ins_rate_node * dt
            num_ins = torch.poisson(expected_num_ins)
            do_ins = do_ins & (num_ins > 0)

        """DELETION or SUBSTITUTION"""
        # TODO implement multinomial sampling for deletion like above for insertions
        # TODO this should take into account the logic below (conflicts with substitution)

        # 1. Scale probabilities by time (converting prob -> rate * dt)
        # rate_node is 1 / (1 - t)
        p_sub_scaled = do_sub_a_probs.view(-1) * rate_node * dt
        p_del_scaled = do_del_probs.view(-1) * del_rate_node * dt

        # 2. Sum rates first (Paper Formulation)
        # "probability of ANY edit is h(lambda_sub + lambda_del)"
        p_any_edit = p_sub_scaled + p_del_scaled

        # Clamp to 1.0 to prevent numerical explosion near t=1
        # p_any_edit = torch.clamp(p_any_edit, max=1.0)

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
        a[do_sub_a] = a_1[do_sub_a]

        # 2.5. Update edge types
        # Get current edge types (already indices)

        # we will deal with only the triu edge types and symmetrize later
        edge_infos = self.edge_aligner.align_edges(
            source_group=(edge_index, [e, do_sub_e_probs]),
            target_group=(edge_index_1, [e_1]),
        )
        e_triu, do_sub_e_probs_triu, e_1_triu = edge_infos["edge_attr"]
        edge_index_triu, _ = edge_infos["edge_index"]

        # Use probabilities directly (sigmoid already applied in sample())
        p_sub_e = do_sub_e_probs_triu.view(-1)
        do_sub_e = torch.rand_like(p_sub_e) < p_sub_e

        # Get batch_id for edges from the source nodes of the edges
        # edge_index_triu[0] gives the source node indices for each edge
        batch_id_edge = batch_id[edge_index_triu[0]]

        # Calculate rate for edges (same as for nodes)
        rate_edge = rate[batch_id_edge].unsqueeze(-1)
        p_mod_e = rate_edge * dt
        p_mod_e = p_mod_e.view(-1)

        do_mod_e = torch.rand_like(p_mod_e) < p_mod_e

        do_sub_e = do_sub_e & do_mod_e

        e_triu[do_sub_e] = e_1_triu[do_sub_e]

        # Finally, symmetrize the edge types
        edge_index, e_attrs = self.edge_aligner.symmetrize_edges(
            edge_index_triu, [e_triu]
        )
        e = e_attrs[0]

        mol = MoleculeBatch(
            x=x,
            a=a,
            c=c_1,  # always take the predicted charge
            e=e,
            edge_index=edge_index,
            batch=batch_id,
        )

        # Validate: check for cross-batch edges after edge symmetrization
        is_cb = validate_no_cross_batch_edges(
            edge_index, batch_id, "integration_adjusted: after edge symmetrization"
        )
        if not is_cb:
            exit()

        if self.n_atoms_strategy != "fixed":
            # Build index mapping: original_idx -> post_deletion_idx (or -1 if deleted)
            N_original = mol.num_nodes
            keep_mask = ~do_del

            # Fail-safe: prevent deletion of entire sample (keep at least 2 nodes per batch_id)
            num_graphs_safe = batch_id.max().item() + 1
            for g in range(num_graphs_safe):
                graph_mask = batch_id == g
                n_kept = (keep_mask & graph_mask).sum().item()
                if n_kept < 2:
                    n_to_restore = 2 - n_kept
                    deleted_in_graph_idx = torch.where(graph_mask & do_del)[0]
                    n_restore = min(n_to_restore, deleted_in_graph_idx.shape[0])
                    if n_restore > 0:
                        restore_indices = deleted_in_graph_idx[:n_restore]
                        keep_mask[restore_indices] = True

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
                # Validate: check for cross-batch edges after filtering
                is_cb = validate_no_cross_batch_edges(
                    mol.edge_index,
                    mol.batch,
                    "integration_adjusted: after filter_nodes",
                )
                if not is_cb:
                    exit()

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

                new_atoms = self.sample_insertions(
                    ins_gmm_dict,
                    len(self.vocab.atom_tokens),
                    len(self.vocab.charge_tokens),
                )

                new_atoms.batch = batch_id[do_ins_valid]

                # Determine fallback edge distribution
                if (
                    self.cat_strategy == "uniform-sample"
                    and self.distributions.edge_type_distribution is not None
                ):
                    edge_dist = self.distributions.edge_type_distribution.to(
                        self.device
                    )
                else:
                    edge_dist = torch.zeros(len(self.vocab.edge_tokens))
                    edge_dist[self.edge_mask_index] = 1.0

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

                if ins_edge_head is not None and h_latent is not None:
                    # Predict using original indices (h_latent uses original indexing)
                    orig_spawn_idx, orig_target_idx, ins_edge_logits = (
                        ins_edge_head.predict_edges_for_insertion(
                            h=h_latent,
                            x=x_1,
                            gmm_dict=ins_gmm_preds,
                            batch=batch_id,
                            insertion_mask=do_ins_valid,
                        )
                    )

                    if ins_edge_logits is not None and ins_edge_logits.numel() > 0:
                        # Map spawn indices to new atom indices (0, 1, 2, ...)
                        # This works for both deleted and non-deleted spawn nodes
                        ins_edge_new_atom_idx = orig_to_new_atom[orig_spawn_idx]

                        # Map target indices to post-deletion space
                        # Targets must exist in the post-deletion molecule
                        ins_edge_target_idx = original_to_postdel[orig_target_idx]

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
                        else:
                            ins_edge_logits = None

                # Check if we have predicted edge logits for insertions
                use_predicted_edges = (
                    ins_edge_logits is not None
                    and ins_edge_new_atom_idx is not None
                    and ins_edge_target_idx is not None
                    and ins_edge_logits.numel() > 0
                )

                if use_predicted_edges:
                    # Use predicted edges from InsertionEdgeHead
                    # ins_edge_new_atom_idx: identifies which new atom (0, 1, 2, ...)
                    # ins_edge_target_idx: in post-deletion space (valid indices in mol)
                    mol = join_molecules_with_predicted_edges(
                        mol=mol,
                        new_atoms=new_atoms,
                        ins_edge_logits=ins_edge_logits,
                        spawn_node_idx=ins_edge_new_atom_idx,
                        target_node_idx=ins_edge_target_idx,
                        fallback_edge_dist=edge_dist,
                    )
                else:
                    # Fall back to random edge sampling
                    mol = join_molecules_with_atoms(mol, new_atoms, edge_dist)

                # Validate: check for cross-batch edges after joining
                validate_no_cross_batch_edges(
                    mol.edge_index, mol.batch, "integration_adjusted: after joining"
                )

        return mol
