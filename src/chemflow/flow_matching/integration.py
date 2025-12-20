import torch
import torch.nn.functional as F
from torch_geometric.utils import subgraph

from chemflow.flow_matching.gmm import sample_from_typed_gmm, sample_from_gmm
from chemflow.utils import (
    token_to_index,
    build_fully_connected_edge_index,
    get_canonical_upper_triangle_with_index,
    symmetrize_upper_triangle,
)


from chemflow.dataset.molecule_data import MoleculeData, PointCloud, MoleculeBatch
from chemflow.dataset.molecule_data import join_molecule_with_atoms


class Integrator:
    def __init__(
        self,
        atom_tokens,
        edge_tokens,
        edge_type_distribution,
        K,
        D,
        cat_strategy="uniform-sample",
        n_atoms_strategy="flexible",
        device="cpu",
    ):
        self.atom_tokens = atom_tokens
        self.K = K
        self.D = D
        self.cat_strategy = cat_strategy
        self.n_atoms_strategy = n_atoms_strategy
        self.device = device
        self.edge_type_distribution = edge_type_distribution

        if self.n_atoms_strategy != "fixed":
            self.death_token_index = token_to_index(atom_tokens, "<DEATH>")

        if self.cat_strategy == "mask":
            self.edge_mask_index = token_to_index(edge_tokens, "<MASK>")
            self.atom_mask_index = token_to_index(atom_tokens, "<MASK>")

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

    def sample_death_process_gnn(
        self,
        global_death_rate: torch.Tensor,
        class_preds: torch.Tensor,
        xt_mask: torch.Tensor,
        batch_id: torch.Tensor,
        death_token_index: int,
        dt: float,
    ) -> torch.Tensor:
        """
        Sample particle deaths using the death process.

        Works with flat tensors and batch_id for variable-sized graphs.
        Selects nodes with highest probability mass on the death token.

        Args:
            global_death_rate: Shape (num_graphs,) - graph-level death rates
            class_preds: Shape (N_total, num_classes) - class predictions (softmaxed logits)
            xt_mask: Shape (N_total,) - mask indicating which nodes are alive
            batch_id: Shape (N_total,) - batch assignment for each node
            death_token_index: Index of the death token in the class vocabulary
            dt: Time step
            device: Device to use

        Returns:
            death_mask: Shape (N_total,) - boolean mask indicating which nodes die
        """
        num_graphs = global_death_rate.shape[0]
        death_mask = torch.zeros_like(xt_mask, dtype=torch.bool, device=self.device)

        # Extract probability for death token for each node
        death_token_probs = class_preds[:, death_token_index]  # Shape: (N_total,)

        # Sample number of deaths using Poisson distribution
        # global_death_rate is already the expected count over remaining time
        death_intensity = global_death_rate * dt
        num_deaths = torch.poisson(death_intensity)

        for graph_id in range(num_graphs):
            # Get nodes in this graph
            graph_mask = (batch_id == graph_id) & xt_mask
            if not graph_mask.any():
                continue

            # Get valid particles for this graph
            valid_indices = torch.where(graph_mask)[0]
            num_valid = len(valid_indices)
            num_deaths_g = num_deaths[graph_id].to(torch.int32).item()

            if num_valid > 0 and num_deaths_g > 0:
                # Get death token probabilities for this graph's nodes
                graph_death_probs = death_token_probs[valid_indices]

                # Select the top num_deaths_g particles with highest death token probability
                num_to_kill = min(num_deaths_g, num_valid)

                # Get indices of particles with highest death token probability
                _, death_indices = torch.topk(
                    graph_death_probs, num_to_kill, largest=True
                )

                # Get the actual indices in the flat tensor
                actual_death_indices = valid_indices[death_indices]

                # Apply death mask
                death_mask[actual_death_indices] = True

        return death_mask

    def sample_birth_process_gnn(
        self,
        birth_rate: torch.Tensor,
        birth_gmm_dict: dict,
        batch_id: torch.Tensor,
        dt: float,
        N_types: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample birth process for GNN (works with batch_ids).

        Args:
            birth_rate: Shape (num_graphs,) - graph-level birth rates
            # GMM parameters per graph
            birth_gmm_dict: dict - GMM parameters per graph
            batch_id: Shape (N_total,) - current batch assignments (for determining num_graphs)
            dt: Time step
            N_types: Number of atom types

        Returns:
            new_particles: Shape (N_new, D) - new particle positions
            new_types: Shape (N_new, N_types) - new particle types (one-hot)
            new_batch_ids: Shape (N_new,) - batch assignments for new particles
        """
        num_graphs = birth_rate.shape[0]
        new_atoms = []
        for graph_id in range(num_graphs):
            # Sample number of births for this graph
            # Expected number of births = birth_rate * dt
            expected_births = birth_rate[graph_id] * dt

            # Use Poisson sampling (approximate with normal for large values)
            if expected_births > 0:
                num_births = torch.poisson(expected_births.unsqueeze(0)).item()
                num_births = int(num_births)
            else:
                num_births = 0

            if num_births == 0:
                continue

            # Sample from GMM for this graph
            # Keep batch dimension
            gmm_params = {
                "mu": birth_gmm_dict["mu"][graph_id].unsqueeze(0),
                "sigma": birth_gmm_dict["sigma"][graph_id].unsqueeze(0),
                "pi": birth_gmm_dict["pi"][graph_id].unsqueeze(0),
            }
            if self.cat_strategy == "uniform-sample":
                sampled_x, sampled_a, sampled_c = sample_from_typed_gmm(
                    gmm_params, num_births, self.K, self.D, N_types
                )
            else:
                sampled_x = sample_from_gmm(gmm_params, num_births, self.K, self.D)
                sampled_a = self.atom_mask_index * torch.ones(
                    (num_births,),
                    dtype=torch.long,
                    device=self.device,
                )
                sampled_c = self.charge_mask_index * torch.ones(
                    (num_births,),
                    dtype=torch.long,
                    device=self.device,
                )

            # Convert to one-hot for types
            # sampled_types_onehot = F.one_hot(sampled_types, num_classes=N_types).float()

            # Squeeze batch dimension if needed
            if sampled_x.dim() == 3:
                sampled_x = sampled_x.squeeze(0)
            # if sampled_types_onehot.dim() == 3:
            #    sampled_types_onehot = sampled_types_onehot.squeeze(0)

            new_atoms_i = PointCloud(
                x=sampled_x,
                a=sampled_a,
                c=sampled_c,
            )
            new_atoms.append(new_atoms_i)

        if len(new_atoms) == 0:
            # No births, return empty tensors
            new_atoms = PointCloud(
                x=torch.empty((0, self.D), device=self.device),
                a=torch.empty((0, N_types), device=self.device),
                batch=torch.empty((0,), device=self.device, dtype=batch_id.dtype),
            )

            return new_atoms

        new_atoms = MoleculeBatch.from_data_list(new_atoms)

        return new_atoms

    def integrate_step_gnn(
        self,
        mol_t: MoleculeData,
        mol_1_pred: MoleculeData,
        global_death_rate: torch.Tensor,
        birth_rate: torch.Tensor,
        birth_gmm_dict: dict,
        t: torch.Tensor,
        dt: float,
        cat_noise_level: float = 0.0,
        eps: float = 1e-6,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Integrate one step of the stochastic process for GNN models.

        Works with variable-sized graphs using batch_ids.
        All tensors are flat (no padding).

        Args:
            mol_t: MoleculeData - current molecule
            mol_1_pred: MoleculeData - predicted molecule
            global_death_rate: Shape (num_graphs,) - graph-level death rates
            birth_rate: Shape (num_graphs,) - graph-level birth rates
            birth_gmm_dict: dict - GMM parameters per graph
            t: Shape (num_graphs,) - current time for each graph
            dt: Time step
            cat_noise_level: Categorical noise level for type updates
            eps: Small epsilon value for numerical stability

        Returns:
            mol_t_final: MoleculeData - updated molecule
        """
        # setup the input molecules
        mol_t_out = mol_t.clone()
        mol_1_pred = mol_1_pred.clone()

        x, a, c, e, edge_index, batch_id = mol_t_out.unpack()
        x_1, a_1, c_1, e_1, edge_index_1, _ = mol_1_pred.unpack()

        # keeps track of which nodes are alive
        xt_mask = torch.ones_like(batch_id, dtype=torch.bool, device=self.device)

        # 1. Update positions (Euler-Maruyama scheme)
        velocity = (x_1 - x) / (1 - t[batch_id].unsqueeze(-1)).clamp(
            min=eps, max=1.0 - eps
        )
        x = x + velocity * dt

        # 2. Update node types
        if self.cat_strategy == "uniform-sample":
            a = self.discrete_integration_step_gnn(
                class_curr=a,
                class_probs_1=a_1,
                t=t,
                batch_idx=batch_id,
                dt=dt,
                noise_scale=cat_noise_level,
                mask=None,
            )
            a = a.view(-1)

        else:
            # Get time for each node (expand t to match nodes)
            num_graphs = t.shape[0]
            node_times = torch.zeros_like(batch_id, dtype=t.dtype, device=self.device)
            for graph_id in range(num_graphs):
                graph_mask = batch_id == graph_id
                node_times[graph_mask] = t[graph_id]

            # Choose elements to unmask
            limit = dt * (1 + (cat_noise_level * node_times)) / (1 - node_times + 1e-8)
            unmask = torch.rand_like(a_1.float()) < limit
            unmask = unmask & (a == self.atom_mask_index) & xt_mask

            # Choose elements to mask
            mask_prob = dt * cat_noise_level
            mask = torch.rand_like(a_1.float()) < mask_prob
            mask = mask & (a != self.atom_mask_index) & xt_mask
            # Do not mask at the end
            end_mask = (node_times + dt) >= 1.0
            mask = mask & (~end_mask)

            # Apply unmasking and re-masking
            a[unmask] = a_1[unmask]
            a[mask] = self.atom_mask_index

        # 2.5. Update edge types
        # Get current edge types (already indices)
        e_curr = e.clone()
        e_pred = e_1.clone()

        # we will deal with only the upper triangle of the edge types and symmetrize them later
        edge_index_triu_curr, e_curr_triu = get_canonical_upper_triangle_with_index(
            edge_index, e_curr
        )
        edge_index_triu_pred, e_pred_triu = get_canonical_upper_triangle_with_index(
            edge_index_1, e_pred
        )

        assert torch.all(edge_index_triu_curr == edge_index_triu_pred), (
            "The edge indices must be the same."
        )

        # Only update edges that exist in edge_index
        if self.cat_strategy == "uniform-sample":
            batch_idx_edge = batch_id[edge_index_triu_curr[0]]

            et_new_triu = self.discrete_integration_step_gnn(
                class_curr=e_curr_triu,
                class_probs_1=e_pred_triu,
                t=t,
                batch_idx=batch_idx_edge,
                dt=dt,
                noise_scale=cat_noise_level,
                mask=None,
            )
            et_new_triu = et_new_triu.view(-1)

        else:
            # Get edge times for valid edges
            num_graphs = t.shape[0]
            node_times = torch.zeros_like(batch_id, dtype=t.dtype, device=self.device)
            for graph_id in range(num_graphs):
                node_times[batch_id == graph_id] = t[graph_id]
            edge_times = (node_times[edge_index[0]] + node_times[edge_index[1]]) / 2.0

            # Sample predictions for valid edges
            e_pred_triu = torch.distributions.Categorical(e_pred_triu).sample()

            # Unmask and mask logic for valid edges
            limit = dt * (1 + (cat_noise_level * edge_times)) / (1 - edge_times + 1e-8)
            node_alive_mask = xt_mask[edge_index[0]] & xt_mask[edge_index[1]]
            unmask = (
                (torch.rand_like(e_pred_triu.float()) < limit)
                & (e_curr_triu == self.edge_mask_index)
                & node_alive_mask
            )
            mask = (
                (torch.rand_like(e_curr_triu.float()) < dt * cat_noise_level)
                & (e_curr_triu != self.edge_mask_index)
                & node_alive_mask
                & ((edge_times + dt) < 1.0)
            )

            et_new_valid = e_curr_triu.clone()
            et_new_valid[unmask] = e_pred_triu[unmask]
            et_new_valid[mask] = self.edge_mask_index

            et_new_triu[edge_index[0], edge_index[1]] = et_new_valid
            et_new_triu[edge_index[1], edge_index[0]] = et_new_valid

        # finally, symmetrize the edge types
        edge_index, e = symmetrize_upper_triangle(edge_index_triu_pred, et_new_triu)

        if self.n_atoms_strategy != "fixed":
            # 3.1. Sample death process
            death_mask = self.sample_death_process_gnn(
                global_death_rate,
                a_1,
                xt_mask,
                batch_id,
                self.death_token_index,
                dt,
            )

            alive_mask = xt_mask & (~death_mask)

            # 3.2. Remove dead nodes and combine with new particles
            # Keep only alive nodes from existing particles
            x_alive = x[alive_mask]
            a_alive = a[alive_mask]
            c_alive = c[alive_mask]
            batch_id_alive = batch_id[alive_mask]

            edge_index_alive, e_alive = subgraph(
                subset=alive_mask,
                edge_index=edge_index,
                edge_attr=e,
                relabel_nodes=True,
                num_nodes=x.shape[0],
            )

            mol_t_alive = MoleculeBatch(
                x=x_alive,
                a=a_alive,
                c=c_alive,
                e=e_alive,
                edge_index=edge_index_alive,
                batch=batch_id_alive,
            )

            # 4. Sample birth process
            new_atoms = self.sample_birth_process_gnn(
                birth_rate,
                birth_gmm_dict,
                batch_id,
                dt,
                N_types=a_1.shape[-1],
            )

            # Combine alive nodes with new particles
            if new_atoms.num_nodes > 0:
                # adjust which edge distribution to sample from
                if (
                    self.cat_strategy == "uniform-sample"
                    and self.edge_type_distribution is not None
                ):
                    # Sample from edge_type_distribution
                    edge_dist = torch.distributions.Categorical(
                        self.edge_type_distribution.to(self.device)
                    )
                else:
                    edge_dist = torch.zeros(len(self.edge_tokens))
                    edge_dist[self.edge_mask_index] = 1.0

                # add the new atoms to the existing molecules
                mol_t_final = join_molecule_with_atoms(
                    mol_t_alive, new_atoms, edge_dist
                )

            else:
                mol_t_final = mol_t_alive

        else:
            # No death or birth process, just movement
            mol_t_final = MoleculeBatch(
                x=x,
                a=a,
                c=c,
                e=e,
                edge_index=edge_index,
                batch=batch_id,
            )

        return mol_t_final
