import torch

from chemflow.flow_matching.gmm import (
    sample_from_typed_gmm,
    sample_from_gmm,
)
from chemflow.utils import (
    token_to_index,
    EdgeAligner,
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


class RateIntegrator:
    def __init__(
        self,
        vocab: Vocab,
        distributions: Distributions,
        gmm_params,
        cat_strategy="uniform-sample",
        n_atoms_strategy="flexible",
        num_integration_steps=100,
        time_strategy="log",
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
        sub_rate_a: torch.Tensor,
        sub_rate_c: torch.Tensor,
        sub_rate_e: torch.Tensor,
        del_rate: torch.Tensor,
        ins_rate: torch.Tensor,
        ins_gmm_preds: torch.Tensor,
        t: torch.Tensor,
        dt: float,
        eps: float = 1e-6,
        h_latent: torch.Tensor = None,
        ins_edge_head=None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Integrate one step of the stochastic process for GNN models.

        Works with variable-sized graphs using batch_ids.
        All tensors are flat (no padding).

        Args:
            mol_t: MoleculeData - current molecule
            mol_1_pred: MoleculeData - predicted molecule
            sub_rate_a: Shape (N,) - node-level atom type substitution rates
            sub_rate_c: Shape (N,) - node-level charge substitution rates
            sub_rate_e: Shape (E,) - edge-level edge type substitution rates
            del_rate: Shape (N,) - node-level deletion rates
            ins_rate: Shape (N,) - node-level insertion rates
            ins_gmm_preds:       - GMM parameters for each node
            t: Shape (num_graphs,) - current time for each graph
            dt: Time step
            eps: Small epsilon value for numerical stability
            h_latent: Shape (N, hidden_dim) - latent node features for edge prediction
            ins_edge_head: InsertionEdgeHead instance for predicting edges (or None)

        Returns:
            mol_t_final: MoleculeData - updated molecule with the new atoms
        """

        x, a, c, e, edge_index, batch_id = mol_t.unpack()
        x_1, a_1, c_1, e_1, edge_index_1, _ = mol_1_pred.unpack()

        # 1. Update positions (Euler-Maruyama scheme)
        velocity = (x_1 - x) / (1 - t[batch_id].unsqueeze(-1)).clamp(
            min=eps, max=1.0 - eps
        )
        x = x + velocity * dt

        # 1. Independent Insertion (as per text)
        # Sample whether to insert with probability h * lambda_ins
        p_ins = ins_rate * dt
        p_ins = p_ins.squeeze(-1)
        do_ins = torch.rand_like(p_ins) < p_ins

        # 2. Deletion OR Substitution (Hierarchical Step)
        # Sample whether to delete or substitute with probability h(lambda_del + lambda_sub)
        total_mod_rate = del_rate + sub_rate_a + sub_rate_c

        p_mod = total_mod_rate * dt
        p_mod = p_mod.squeeze(-1)
        do_mod = torch.rand_like(p_mod) < p_mod

        # Initialize masks
        do_del = torch.zeros_like(do_mod)
        do_sub = torch.zeros_like(do_mod)

        # Only process the positions where a modification actually occurred
        # This saves computation and strictly follows the "exclusive" logic
        mod_indices = torch.nonzero(do_mod)

        if len(mod_indices) > 0:
            # Select del with p = lambda_del / (lambda_del + lambda_sub)
            # Extract rates only for the active indices to save memory/compute
            curr_del_rate = del_rate[do_mod].squeeze(-1)
            curr_total_rate = total_mod_rate[do_mod].squeeze(-1)

            # Calculate conditional split w/ epsilon for stability
            p_cond_del = curr_del_rate / (curr_total_rate + 1e-8)

            # Sample the decision
            is_deletion = torch.rand_like(p_cond_del) < p_cond_del

            # Assign back to the main masks
            do_del[do_mod] = is_deletion
            do_sub[do_mod] = ~is_deletion

        # Do substitution
        a[do_sub] = a_1[do_sub]
        c[do_sub] = c_1[do_sub]

        # 2.5. Update edge types
        # Get current edge types (already indices)

        # we will deal with only the triu edge types and symmetrize later
        edge_infos = self.edge_aligner.align_edges(
            source_group=(edge_index, [e, sub_rate_e]),
            target_group=(edge_index_1, [e_1]),
        )
        e_triu, sub_rate_e_triu, e_1_triu = edge_infos["edge_attr"]
        edge_index_triu, _ = edge_infos["edge_index"]

        p_sub_e = sub_rate_e_triu * dt
        p_sub_e = p_sub_e.squeeze(-1)
        do_sub_e = torch.rand_like(p_sub_e) < p_sub_e
        e_triu[do_sub_e] = e_1_triu[do_sub_e]

        # Finally, symmetrize the edge types
        # edge_index, e = symmetrize_upper_triangle(edge_index_triu, e_triu)
        edge_index, e_attrs = self.edge_aligner.symmetrize_edges(edge_index_triu, [e_triu])
        e = e_attrs[0]

        mol = MoleculeBatch(
            x=x,
            a=a,
            c=c,
            e=e,
            edge_index=edge_index,
            batch=batch_id,
        )

        if self.n_atoms_strategy != "fixed":
            # Build index mapping: original_idx -> post_deletion_idx (or -1 if deleted)
            N_original = mol.num_nodes
            keep_mask = ~do_del
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
            # Only insert from nodes that were NOT deleted
            do_ins_valid = do_ins & keep_mask
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

                # Predict edges for insertions if head is available
                ins_edge_logits = None
                ins_edge_spawn_idx = None
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
                        # Remap indices from original to post-deletion space
                        ins_edge_spawn_idx = original_to_postdel[orig_spawn_idx]
                        ins_edge_target_idx = original_to_postdel[orig_target_idx]

                        # Filter out edges where either endpoint was deleted
                        valid_edges = (ins_edge_spawn_idx >= 0) & (
                            ins_edge_target_idx >= 0
                        )
                        if valid_edges.any():
                            ins_edge_spawn_idx = ins_edge_spawn_idx[valid_edges]
                            ins_edge_target_idx = ins_edge_target_idx[valid_edges]
                            ins_edge_logits = ins_edge_logits[valid_edges]
                        else:
                            ins_edge_logits = None

                # Check if we have predicted edge logits for insertions
                use_predicted_edges = (
                    ins_edge_logits is not None
                    and ins_edge_spawn_idx is not None
                    and ins_edge_target_idx is not None
                    and ins_edge_logits.numel() > 0
                )

                if use_predicted_edges:
                    # Use predicted edges from InsertionEdgeHead
                    # Now indices are in post-deletion space
                    mol = join_molecules_with_predicted_edges(
                        mol=mol,
                        new_atoms=new_atoms,
                        ins_edge_logits=ins_edge_logits,
                        spawn_node_idx=ins_edge_spawn_idx,
                        target_node_idx=ins_edge_target_idx,
                        fallback_edge_dist=edge_dist,
                    )
                else:
                    # Fall back to random edge sampling
                    mol = join_molecules_with_atoms(mol, new_atoms, edge_dist)

        return mol
