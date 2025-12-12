import torch
import torch.nn.functional as F

from chemflow.flow_matching.gmm import sample_from_typed_gmm, sample_from_gmm
from chemflow.utils import token_to_index


class Integrator:
    def __init__(self, tokens, K, D, typed_gmm=True, cat_strategy="uniform-sample", device="cpu"):
        self.tokens = tokens
        self.mask_index = token_to_index(tokens, "<MASK>")
        self.death_token_index = token_to_index(tokens, "<DEATH>")
        self.K = K
        self.D = D
        self.typed_gmm = typed_gmm
        self.cat_strategy = cat_strategy
        self.device = device
        self.eps = 1e-6

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
        new_particles_list = []
        new_types_list = []
        new_batch_ids_list = []
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
            if self.typed_gmm:
                sampled_locations, sampled_types = sample_from_typed_gmm(
                    gmm_params, num_births, self.K, self.D, N_types
                )
            else:
                sampled_locations = sample_from_gmm(
                    gmm_params, num_births, self.K, self.D
                )
                sampled_types = self.mask_index * torch.ones(
                    (num_births,),
                    dtype=torch.long,
                    device=self.device,
                )

            # Convert to one-hot for types
            sampled_types_onehot = F.one_hot(sampled_types, num_classes=N_types).float()

            # Squeeze batch dimension if needed
            if sampled_locations.dim() == 3:
                sampled_locations = sampled_locations.squeeze(0)
            if sampled_types_onehot.dim() == 3:
                sampled_types_onehot = sampled_types_onehot.squeeze(0)

            new_particles_list.append(sampled_locations)
            new_types_list.append(sampled_types_onehot)
            new_batch_ids_list.append(
                torch.full(
                    (num_births,), graph_id, device=self.device, dtype=batch_id.dtype
                )
            )

        if len(new_particles_list) == 0:
            # No births, return empty tensors
            empty_particles = torch.empty((0, self.D), device=self.device)
            empty_types = torch.empty((0, N_types), device=self.device)
            empty_batch_ids = torch.empty(
                (0,), device=self.device, dtype=batch_id.dtype
            )
            return empty_particles, empty_types, empty_batch_ids

        new_particles = torch.cat(new_particles_list, dim=0)
        new_types = torch.cat(new_types_list, dim=0)
        new_batch_ids = torch.cat(new_batch_ids_list, dim=0)

        return new_particles, new_types, new_batch_ids

    def integrate_step_gnn(
        self,
        x_pred: torch.Tensor,
        type_pred: torch.Tensor,
        global_death_rate: torch.Tensor,
        birth_rate: torch.Tensor,
        birth_gmm_dict: dict,
        xt: torch.Tensor,
        ct: torch.Tensor,
        batch_id: torch.Tensor,
        t: torch.Tensor,
        dt: float,
        mask_index: int,
        death_token_index: int,
        cat_noise_level: float = 0.0,
        eps: float = 1e-6
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Integrate one step of the stochastic process for GNN models.

        Works with variable-sized graphs using batch_ids.
        All tensors are flat (no padding).

        Args:
            velocity: Shape (N_total, D) - velocity predictions for each node
            type_pred: Shape (N_total, num_classes) - type predictions for each node
            global_death_rate: Shape (num_graphs,) - graph-level death rates
            birth_rate: Shape (num_graphs,) - graph-level birth rates
            # GMM parameters per graph
            birth_gmm_dict: dict - GMM parameters per graph
            xt: Shape (N_total, D) - current positions
            ct: Shape (N_total, num_classes) - current types (one-hot)
            batch_id: Shape (N_total,) - batch assignment for each node
            t: Shape (num_graphs,) - current time for each graph
            dt: Time step
            mask_index: Index of the mask token
            death_token_index: Index of the death token in the class vocabulary
            cat_noise_level: Categorical noise level for type updates

        Returns:
            xt_new: Shape (N_final, D) - updated positions
            ct_new: Shape (N_final, num_classes) - updated types
            xt_mask_new: Shape (N_final,) - updated masks
            batch_id_new: Shape (N_final,) - updated batch assignments
        """

        # keeps track of which nodes are alive
        xt_mask = torch.ones_like(batch_id, dtype=torch.bool, device=self.device)

        # 1. Update positions (Euler-Maruyama scheme)
        velocity = (x_pred - xt) / (1 - t[batch_id].unsqueeze(-1)).clamp(min=eps, max=1.0 - eps)
        xt_new = xt + velocity * dt

        # 2. Update types
        curr = torch.argmax(ct, dim=-1)
        pred = torch.distributions.Categorical(type_pred).sample()

        if self.cat_strategy == "uniform-sample":
            # probability to stay in the current type
            pred_probs_curr = torch.gather(type_pred, -1, curr.unsqueeze(-1))

            # Setup batched time tensor and noise tensor
            ones = [1] * (len(type_pred.shape) - 1)
            times = t[batch_id].view(-1, *ones).clamp(min=eps, max=1.0 - eps)
            noise = torch.zeros_like(times)
            noise[times + dt < 1.0] = cat_noise_level

            # Off-diagonal step probs
            mult = (1 + ((2 * noise) * (ct.shape[-1] - 1) * times)) / (1 - times)
            first_term = dt * mult * type_pred
            second_term = dt * noise * pred_probs_curr
            step_probs = (first_term + second_term).clamp(max=1.0)

            # On-diagonal step probs
            step_probs.scatter_(-1, curr.unsqueeze(-1), 0.0)
            diags = (1.0 - step_probs.sum(dim=-1, keepdim=True)).clamp(min=eps)
            step_probs.scatter_(-1, curr.unsqueeze(-1), diags)
            pred = torch.distributions.Categorical(step_probs).sample()
            ct_new = F.one_hot(pred, num_classes=ct.shape[-1]).float()

        if self.cat_strategy == "mask":
            # Get time for each node (expand t to match nodes)
            num_graphs = t.shape[0]
            node_times = torch.zeros_like(batch_id, dtype=t.dtype, device=self.device)
            for graph_id in range(num_graphs):
                graph_mask = batch_id == graph_id
                node_times[graph_mask] = t[graph_id]

            # Choose elements to unmask
            limit = dt * (1 + (cat_noise_level * node_times)) / (1 - node_times).clamp(min=eps, max=1-eps)
            unmask = torch.rand_like(pred.float()) < limit
            unmask = unmask & (curr == mask_index) & xt_mask

            # Choose elements to mask
            mask_prob = dt * cat_noise_level
            mask = torch.rand_like(pred.float()) < mask_prob
            mask = mask & (curr != mask_index) & xt_mask
            # Do not mask at the end
            end_mask = (node_times + dt) >= 1.0
            mask = mask & (~end_mask)

            # Apply unmasking and re-masking
            curr[unmask] = pred[unmask]
            curr[mask] = mask_index
            ct_new = F.one_hot(curr, num_classes=ct.shape[-1]).float()

        # 3. Sample death process
        # TODO ct should be logits?
        death_mask = self.sample_death_process_gnn(
            global_death_rate, type_pred, xt_mask, batch_id, death_token_index, dt
        )
        xt_mask_new = xt_mask & (~death_mask)

        # 4. Sample birth process
        new_particles, new_types, new_batch_ids = self.sample_birth_process_gnn(
            birth_rate,
            birth_gmm_dict,
            batch_id,
            dt,
            ct.shape[-1],
        )

        # 5. Remove dead nodes and combine with new particles
        # Keep only alive nodes from existing particles
        alive_mask = xt_mask_new
        xt_alive = xt_new[alive_mask]
        ct_alive = ct_new[alive_mask]
        batch_id_alive = batch_id[alive_mask]

        # Combine alive nodes with new particles
        if new_particles.shape[0] > 0:
            xt_final = torch.cat([xt_alive, new_particles], dim=0)
            ct_final = torch.cat([ct_alive, new_types], dim=0)
            batch_id_final = torch.cat([batch_id_alive, new_batch_ids], dim=0)

        else:
            xt_final = xt_alive
            ct_final = ct_alive
            batch_id_final = batch_id_alive

        return xt_final, ct_final, batch_id_final
