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
        K,
        D,
        cat_strategy="uniform-sample",
        device="cpu",
        edge_type_distribution=None,
        edge_tokens=None,
    ):
        self.atom_tokens = atom_tokens
        self.atom_mask_index = token_to_index(atom_tokens, "<MASK>")
        self.death_token_index = token_to_index(atom_tokens, "<DEATH>")
        self.K = K
        self.D = D
        self.cat_strategy = cat_strategy
        self.device = device
        self.edge_type_distribution = edge_type_distribution
        if edge_tokens is not None:
            self.edge_mask_index = token_to_index(edge_tokens, "<MASK>")
        else:
            self.edge_mask_index = None

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
        # a_pred = torch.distributions.Categorical(mol_1_pred.a).sample()

        if self.cat_strategy == "uniform-sample":
            # probability to stay in the current type
            p_a_stay_curr = torch.gather(a_1, -1, a.unsqueeze(-1))

            # Setup batched time tensor and noise tensor
            ones = [1] * (len(a_1.shape) - 1)
            times = t[batch_id].view(-1, *ones).clamp(min=1e-3, max=1.0 - 1e-3)
            noise = torch.zeros_like(times)
            noise[times + dt < 1.0] = cat_noise_level

            # Off-diagonal step probs
            mult = (1 + ((2 * noise) * (a_1.shape[-1] - 1) * times)) / (1 - times)
            first_term = dt * mult * a_1
            second_term = dt * noise * p_a_stay_curr
            a_step_probs = (first_term + second_term).clamp(max=1.0)

            # On-diagonal step probs
            a_step_probs.scatter_(-1, a.unsqueeze(-1), 0.0)
            diags = (1.0 - a_step_probs.sum(dim=-1, keepdim=True)).clamp(min=0.0)
            a_step_probs.scatter_(-1, a.unsqueeze(-1), diags)

            # set mask and death probability to 0 for the nodes that are not valid
            a_step_probs[:, self.atom_mask_index] = 0.0
            a_step_probs[:, self.death_token_index] = 0.0

            a_pred = torch.distributions.Categorical(a_step_probs).sample()
            a = a_pred

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
        num_edge_classes = e_1.shape[-1]

        # Get current edge types (already indices)
        e_curr = e.long().clone()
        e_pred = e_1.long().clone()

        # we will deal with only the upper triangle of the edge types and symmetrize them later
        _, e_curr_triu = get_canonical_upper_triangle_with_index(edge_index, e_curr)
        edge_index_triu, e_pred_triu = get_canonical_upper_triangle_with_index(
            edge_index_1, e_pred
        )

        # Only update edges that exist in edge_index
        if self.cat_strategy == "uniform-sample":
            # Get probability of current edge type for valid edges
            # et_curr_valid = et_new[edge_index[0], edge_index[1]]
            # print(et_curr_valid)

            # p_e_stay_curr = mol_1_pred.e.gather(1, et_new.unsqueeze(-1)).squeeze(-1)
            p_e_stay_curr = torch.gather(
                e_pred_triu, -1, e_curr_triu.unsqueeze(-1)
            ).squeeze(-1)

            # Compute edge times (average of connected nodes)
            node_times = t[batch_id]
            edge_times = (
                node_times[edge_index_triu[0]] + node_times[edge_index_triu[1]]
            ) / 2.0
            edge_times = edge_times.clamp(min=1e-3, max=1.0 - 1e-3)
            noise = torch.zeros_like(edge_times)
            noise[edge_times + dt < 1.0] = cat_noise_level

            # Compute step probabilities for valid edges
            mult = (1 + ((2 * noise) * (num_edge_classes - 1) * edge_times)) / (
                1 - edge_times
            )
            first_term = dt * mult.unsqueeze(-1) * e_pred_triu
            second_term = dt * noise.unsqueeze(-1) * p_e_stay_curr.unsqueeze(-1)
            step_probs = (first_term + second_term).clamp(max=1.0)

            # Set diagonal probabilities
            step_probs.scatter_(1, e_curr_triu.unsqueeze(-1), 0.0)
            diags = (1.0 - step_probs.sum(dim=-1, keepdim=True)).clamp(min=0.0)
            step_probs.scatter_(1, e_curr_triu.unsqueeze(-1), diags)

            # set mask probability to 0 for the edges that are not valid
            step_probs[:, self.edge_mask_index] = 0.0

            # Sample new edge types for valid edges
            et_new_triu = torch.distributions.Categorical(step_probs).sample()

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
        edge_index, e_pred = symmetrize_upper_triangle(edge_index_triu, et_new_triu)
        e = e_pred

        # 3.1 Sample death process
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
            mol_t_final = join_molecule_with_atoms(mol_t_alive, new_atoms, edge_dist)

        else:
            mol_t_final = mol_t_alive

        return mol_t_final
