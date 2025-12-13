import numpy as np
import torch
import torch.nn.functional as F

from chemflow.flow_matching.gmm import interpolate_gmm, interpolate_typed_gmm
from chemflow.flow_matching.sampling import sample_births, sample_deaths
from chemflow.flow_matching.assignment import assign_targets_batched
from chemflow.utils import token_to_index
from external_code.egnn import unsorted_segment_mean


class Interpolator:
    def __init__(
        self,
        tokens,
        atom_type_distribution,
        edge_type_distribution,
        typed_gmm=True,
        N_samples=20,
    ):
        self.tokens = tokens
        self.typed_gmm = typed_gmm
        self.N_samples = N_samples

        self.mask_token = token_to_index(self.tokens, "<MASK>")
        self.death_token = token_to_index(self.tokens, "<DEATH>")

        self.atom_type_distribution = atom_type_distribution
        self.edge_type_distribution = edge_type_distribution

        # TODO I'm lazy and this will always be 3
        self.D = 3
        self.M = len(atom_type_distribution)
        self.E = len(edge_type_distribution)

    def interpolate_x(self, x0, x1, t):
        """
        Continuous interpolation for continuous variables.

        Args:
            x0: (N, D) tensor at time 0
            x1: (N, D) tensor at time 1
            t: float, interpolation time in [0, 1]
        Returns:
            x_t: (N, D) interpolated tensor at time t
        """
        return x0 * (1 - t) + x1 * t

    def interpolate_classes(self, c0, c1, t):
        """
        Discrete interpolation for discrete variables / one-hot classes c.

        Args:
            c0: (N, M) one-hot tensor at time 0
            c1: (N, M) one-hot tensor at time 1
            t: float, interpolation time in [0, 1]
        Returns:
            y_t: (N, M) interpolated one-hot tensor at time t
        """

        N, M = c0.shape

        # Convert one-hot to integer indices
        c0_idx = torch.argmax(c0, dim=-1)
        c1_idx = torch.argmax(c1, dim=-1)

        # Sample Bernoulli mask for which positions to keep from y0
        mask = torch.rand(N, device=t.device) > t  # True = keep from y0
        mask = mask.view(N)

        # Start from y1 and overwrite with y0 where mask=True
        c1_idx[mask] = c0_idx[mask]

        # Convert back to one-hot
        ct = F.one_hot(c1_idx, M)

        return ct

    def interpolate_e(self, e0, e1, t):
        """
        Discrete interpolation for discrete variables / one-hot classes e.

        Args:
            e0: (N, N) one-hot tensor at time 0
            e1: (N, N) one-hot tensor at time 1
            t: float, interpolation time in [0, 1]
        Returns:
            et: (N, N) interpolated one-hot tensor at time t
        """
        et = e1.clone()

        # create upper tri random mask
        mask = torch.triu(
            torch.rand((e0.shape[0], e0.shape[0]), device=e0.device), diagonal=1
        )

        mask = mask > t

        # interpolate e0 and e1
        et[mask] = e0[mask]
        et[mask.T] = e0[mask.T]

        return et

    def interpolate_different_size(
        self,
        x0,
        a0,
        c0,
        edge_types0,
        x0_batch_id,
        x1,
        a1,
        c1,
        edge_types1,
        x1_batch_id,
        t,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        """
        Interpolates between x0 and x1, and c0 and c1 for graphs with flexible sizes using batch_id.
        Handles matched nodes, death processes (unmatched x0), and birth processes.

        Args:
            x0: Shape (N_total, D) - concatenated nodes from all graphs
            a0: Shape (N_total, M+1) - concatenated atom types from all graphs
            c0: Shape (N_total, M+1) - concatenated charge types from all graphs
            edge_types0: Shape (N_total, N_total) - concatenated edge types from all graphs
            x0_batch_id: Shape (N_total,) - batch assignment for each x0 node
            x1: Shape (M_total, D) - concatenated nodes from all graphs
            a1: Shape (M_total, M+1) - concatenated types from all graphs
            c1: Shape (M_total, M+1) - concatenated charge types from all graphs
            edge_types1: Shape (M_total, M_total) - concatenated edge types from all graphs
            x1_batch_id: Shape (M_total,) - batch assignment for each x1 node
            t: Shape (num_graphs,) or scalar - time parameter for each graph

        Returns:
            tuple: A tuple of:
            - xt: Shape (K_total, D) - interpolated positions, concatenated
            - at: Shape (K_total, M+1) - interpolated atom types, concatenated
            - ct: Shape (K_total, M+1) - interpolated charge types, concatenated
            - xt_batch_id: Shape (K_total,) - batch assignment for each interpolated node
            - targets: Dictionary containing the following keys:
                - target_vf: Shape (K_total, D) - velocity field targets
                - target_cvf: Shape (K_total, M+1) - velocity field targets for types
                - birth_rate_target: Shape (num_graphs, 1) - birth rate targets
                - death_rate_target: Shape (num_graphs, 1) - death rate targets
                - birth_locations: Shape (num_graphs, D) - GMM samples
                - birth_types: Shape (num_graphs, M) - GMM samples for types
                - birth_locations_batch_ids: Shape (num_graphs, N_samples) - batch ids for GMM samples
        """
        # 1. Assign targets
        assigned_targets = assign_targets_batched(
            x0, a0, c0, edge_types0, x0_batch_id, x1, a1, c1, edge_types1, x1_batch_id
        )

        matched_x0, matched_x1 = assigned_targets["matched"]["x"]
        matched_a0, matched_a1 = assigned_targets["matched"]["a"]
        _, matched_c1 = assigned_targets["matched"]["c"]
        matched_e0, matched_e1 = assigned_targets["matched"]["edge_types"]

        unmatched_x0, unmatched_x1 = assigned_targets["unmatched"]["x"]
        unmatched_a0, unmatched_a1 = assigned_targets["unmatched"]["a"]
        _, unmatched_c1 = assigned_targets["unmatched"]["c"]
        unmatched_e0, unmatched_e1 = assigned_targets["unmatched"]["edge_types"]

        # 2.1 Interpolate the matched targets
        matched_xt = [
            self.interpolate_x(matched_x0_i, matched_x1_i, t_i)
            for matched_x0_i, matched_x1_i, t_i in zip(matched_x0, matched_x1, t)
        ]
        matched_at = [
            self.interpolate_classes(matched_a0_i, matched_a1_i, t_i)
            for matched_a0_i, matched_a1_i, t_i in zip(matched_a0, matched_a1, t)
        ]

        matched_avf = [
            F.one_hot(torch.argmax(a1_i, dim=-1), num_classes=self.M)
            for a1_i in matched_a1
        ]

        """matched_ct = [
            self.interpolate_c(matched_c0_i, matched_c1_i, t_i)
            for matched_c0_i, matched_c1_i, t_i in zip(matched_c0, matched_c1, t)
        ]"""

        matched_cvf = [
            F.one_hot(torch.argmax(c1_i, dim=-1), num_classes=self.M)
            for c1_i in matched_c1
        ]

        matched_et = [
            self.interpolate_e(matched_e0_i, matched_e1_i, t_i)
            for matched_e0_i, matched_e1_i, t_i in zip(matched_e0, matched_e1, t)
        ]

        # 2.2 Handle unmatched samples / targets
        death_xt = []
        death_x1 = []
        death_vf = []
        death_at = []
        death_avf = []
        death_rate_target = []

        birth_xt = []
        birth_x1 = []
        birth_vf = []
        birth_at = []
        birth_avf = []
        # birth_ct = []
        # birth_cvf = []
        birth_rate_target = []
        birth_locations = []
        birth_types = []

        et_list = []
        target_e_list = []

        empty_x = torch.empty((0, self.D), device=x0.device, dtype=x0.dtype)
        empty_a = torch.empty((0, self.M), device=x0.device, dtype=x0.dtype)
        # empty_c = torch.empty((0, self.C), device=x0.device, dtype=x0.dtype)
        empty_e = torch.empty((0, self.E), device=x0.device, dtype=x0.dtype)

        # create a sink state that all unmatched x0 will move towards
        x_sink = torch.zeros((1, self.D), device=x0.device)

        for index, (
            unmatched_x0_i,
            unmatched_a0_i,
            unmatched_e0_i,
            unmatched_x1_i,
            unmatched_a1_i,
            unmatched_e1_i,
            t_i,
        ) in enumerate(
            zip(
                unmatched_x0,
                unmatched_a0,
                unmatched_e0,
                unmatched_x1,
                unmatched_a1,
                unmatched_e1,
                t,
            )
        ):
            instantaneous_rate = 1 / torch.clamp(1 - t_i, min=1e-8)

            # 2.3 Death process
            if unmatched_x0_i.shape[0] > 0:
                # sample death times
                death_times, is_dead, x0_alive_at_xt, a0_alive_at_xt = sample_deaths(
                    unmatched_x0_i, unmatched_a0_i, t_i
                )

                # interpolate the unmatched x0 to the sink state
                death_xt_i = self.interpolate_x(x0_alive_at_xt, x_sink, t_i)

                death_a1_i = self.death_token * torch.ones(
                    (death_xt_i.shape[0],), dtype=torch.long, device=death_xt_i.device
                )
                death_a1_i = F.one_hot(death_a1_i, num_classes=len(self.tokens))
                death_at_i = self.interpolate_classes(a0_alive_at_xt, death_a1_i, t_i)

                # Global death rate is now the actual number of nodes to remove in this batch item
                N_necessary_deaths = x0_alive_at_xt.shape[0]
                death_rate_target_i = N_necessary_deaths * instantaneous_rate

                # create edge type targets as null
                e_targets_i = matched_et[index] + torch.zeros(
                    (matched_et[index].shape[0] + N_necessary_deaths, self.E),
                    device=x0.device,
                )

                death_xt.append(death_xt_i)
                death_x1.append(x_sink.repeat(death_xt_i.shape[0], 1))
                death_vf.append(x_sink.repeat(death_xt_i.shape[0], 1) - x0_alive_at_xt)

                death_at.append(death_at_i)
                death_avf.append(death_a1_i)
                death_rate_target.append(death_rate_target_i.view(1, 1))

                birth_xt.append(empty_x)
                birth_x1.append(empty_x)
                birth_vf.append(empty_x)

                birth_at.append(empty_a)
                birth_avf.append(empty_a)

                birth_rate_target.append(torch.zeros((1, 1), device=x0.device))
                birth_locations.append(empty_x)
                birth_types.append(torch.zeros((0), device=x0.device))

            # 2.4 Birth process
            elif unmatched_x1_i.shape[0] > 0:
                # sample birth times and locations
                (
                    birth_times,
                    birth_xt_i,
                    birth_x1_i,
                    birth_a1_i,
                    birth_e1_i,
                    unborn_x1_i,
                    unborn_a1_i,
                ) = sample_births(
                    unmatched_x1_i, unmatched_a1_i, unmatched_e1_i, t_i, sigma=0.5
                )

                # check if any births are happening
                if birth_xt_i.shape[0] > 0:
                    # interpolate the birth locations to the birth times
                    t_birth_interpolation = (
                        t_i.repeat(birth_times.shape[0]) - birth_times
                    ).unsqueeze(-1)

                    birth_xt_i = self.interpolate_x(
                        birth_xt_i,
                        birth_x1_i,
                        t_birth_interpolation,
                    )

                    if self.typed_gmm:
                        # use atom distribution (e.g. empirical distribution) for birth types
                        birth_c0_i = self.atom_type_distribution.unsqueeze(0).to(
                            x0.device
                        )
                        birth_c0_i = birth_c0_i.repeat(birth_xt_i.shape[0], 1)
                        birth_at_i = self.interpolate_classes(
                            birth_c0_i,
                            birth_a1_i,
                            t_birth_interpolation.view(-1),
                        )
                    else:
                        # use mask token for birth types
                        birth_a0_i = self.mask_token * torch.ones(
                            (birth_xt_i.shape[0],),
                            dtype=torch.long,
                            device=birth_xt_i.device,
                        )
                        birth_a0_i = F.one_hot(birth_a0_i, num_classes=self.M)

                        birth_at_i = self.interpolate_classes(
                            birth_a0_i,
                            birth_a1_i,
                            t_birth_interpolation.view(-1),
                        )
                    # sample birth types from atom type distribution
                    birth_at_distr = torch.distributions.Categorical(
                        probs=self.atom_type_distribution
                    )
                    birth_at_i = birth_at_distr.sample((birth_xt_i.shape[0],))
                    birth_at_i = F.one_hot(birth_at_i, num_classes=self.M)
                    birth_at_i = birth_at_i.to(x0.device)

                    matched_e_targets_i = matched_et[index]
                    adj_size = matched_e_targets_i.shape[0] + birth_xt_i.shape[0]

                    target_e_i = torch.zeros(
                        (adj_size, adj_size),
                        device=x0.device,
                    )
                    target_e_i[
                        : matched_e_targets_i.shape[0], : matched_e_targets_i.shape[0]
                    ] = matched_e_targets_i
                    # TODO this will not work
                    target_e_i[
                        matched_e_targets_i.shape[0] :, matched_e_targets_i.shape[0] :
                    ] = birth_e1_i

                    birth_xt.append(birth_xt_i)
                    birth_x1.append(birth_x1_i)
                    birth_vf.append((birth_x1_i - birth_xt_i) * instantaneous_rate)

                    birth_at.append(birth_at_i)
                    birth_avf.append(birth_a1_i)

                else:
                    # no birth was sampled
                    birth_xt.append(empty_x)
                    birth_x1.append(empty_x)
                    birth_vf.append(empty_x)

                    birth_at.append(empty_a)
                    birth_avf.append(empty_a)

                death_xt.append(empty_x)
                death_x1.append(empty_x)
                death_vf.append(empty_x)

                death_at.append(empty_a)
                death_avf.append(empty_a)
                death_rate_target.append(torch.zeros((1, 1), device=x0.device))

                N_necessary_births = unmatched_x1_i.shape[0] - birth_xt_i.shape[0]
                birth_rate_target_i = N_necessary_births * instantaneous_rate
                birth_rate_target.append(birth_rate_target_i.view(1, 1))

                if unborn_x1_i.shape[0] > 0:
                    # Store the GMM samples for NLL calculation during training
                    if self.typed_gmm:
                        p_a_0 = self.atom_type_distribution.unsqueeze(0).to(x0.device)
                        p_a_1 = unborn_a1_i
                        # TODO make sigma hyperparameter
                        sampled_locations, sampled_types = interpolate_typed_gmm(
                            p_x_1=unborn_x1_i,
                            p_c_0=p_a_0,
                            p_c_1=p_a_1,
                            t=t_i,
                            num_samples=self.N_samples,
                            sigma=0.5,
                        )
                    else:
                        sampled_locations = interpolate_gmm(
                            unborn_x1_i, t_i, num_samples=self.N_samples
                        )
                        sampled_types = self.mask_token * torch.ones(
                            (sampled_locations.shape[0],),
                            dtype=torch.long,
                            device=birth_xt_i.device,
                        )
                    birth_locations.append(sampled_locations)
                    birth_types.append(sampled_types)
                else:
                    # if no unborn x1, you're done, so pad with very negative numbers
                    birth_locations.append(empty_x)
                    birth_types.append(torch.zeros((0), device=birth_xt_i.device))

            # 2.5 No birth or death process, just movement
            else:
                death_xt.append(empty_x)
                death_x1.append(empty_x)
                death_vf.append(empty_x)

                # death_ct.append(empty_c)
                # death_cvf.append(empty_c)
                death_at.append(empty_a)
                death_avf.append(empty_a)
                death_rate_target.append(torch.zeros((1, 1), device=x0.device))

                birth_xt.append(empty_x)
                birth_x1.append(empty_x)
                birth_vf.append(empty_x)

                birth_at.append(empty_a)
                birth_avf.append(empty_a)
                # birth_ct.append(empty_c)
                # birth_cvf.append(empty_c)

                birth_rate_target.append(torch.zeros((1, 1), device=x0.device))
                birth_locations.append(empty_x)
                birth_types.append(torch.zeros((0), device=x0.device))

                et_list.append(matched_et[index])
                target_e_list.append(matched_e1[index])

        # 3. Concatenate the matched, death, and birth processes
        xt_list = [
            torch.cat([matched_xt_i, death_xt_i, birth_xt_i], dim=0)
            for matched_xt_i, death_xt_i, birth_xt_i in zip(
                matched_xt, death_xt, birth_xt
            )
        ]
        at_list = [
            torch.cat([matched_at_i, death_at_i, birth_at_i], dim=0)
            for matched_at_i, death_at_i, birth_at_i in zip(
                matched_at, death_at, birth_at
            )
        ]

        """ct_list = [
            torch.cat([matched_ct_i, death_ct_i, birth_ct_i], dim=0)
            for matched_ct_i, death_ct_i, birth_ct_i in zip(
                matched_ct, death_ct, birth_ct
            )
        ]"""

        # 4. Concatenate the matched, death, and birth targets
        target_x_list = [
            torch.cat([matched_x1_i, death_x1_i, birth_x1_i], dim=0)
            for matched_x1_i, death_x1_i, birth_x1_i in zip(
                matched_x1, death_x1, birth_x1
            )
        ]

        target_avf_list = [
            torch.cat([matched_avf_i, death_avf_i, birth_avf_i], dim=0)
            for matched_avf_i, death_avf_i, birth_avf_i in zip(
                matched_avf, death_avf, birth_avf
            )
        ]

        # TODO placeholder for same number of atoms experiment
        target_cvf = torch.cat(matched_cvf, dim=0)

        """target_cvf_list = [
            torch.cat([matched_cvf_i, death_cvf_i, birth_cvf_i], dim=0)
            for matched_cvf_i, death_cvf_i, birth_cvf_i in zip(
                matched_cvf, death_cvf, birth_cvf
            )
        ]"""

        N_t = torch.tensor([xt.shape[0] for xt in xt_list])
        xt_batch_id = torch.repeat_interleave(torch.arange(len(xt_list)), N_t).to(
            x0.device
        )
        # concatenate and remove mean of xt and target_x
        xt = torch.cat(xt_list, dim=0)
        xt_mean = unsorted_segment_mean(xt, xt_batch_id, len(xt_list))
        xt = xt - xt_mean[xt_batch_id]

        target_x = torch.cat(target_x_list, dim=0)
        target_x = target_x - xt_mean[xt_batch_id]

        at = torch.cat(at_list, dim=0)
        target_avf = torch.cat(target_avf_list, dim=0)

        # ct = torch.cat(ct_list, dim=0)
        # target_cvf = torch.cat(target_cvf_list, dim=0)

        et = torch.block_diag(*et_list)
        target_evf = torch.block_diag(*target_e_list)
        target_evf = F.one_hot(target_evf, num_classes=self.E)

        birth_rate_target = torch.cat(birth_rate_target, dim=0)
        death_rate_target = torch.cat(death_rate_target, dim=0)

        # deal with samples for typed gmm
        N_birth_samples = torch.tensor(
            [birth_locations_i.shape[0] for birth_locations_i in birth_locations]
        )
        birth_locations_batch_ids = torch.repeat_interleave(
            torch.arange(len(birth_locations)), N_birth_samples, dim=0
        ).to(x0.device)

        # remove xt_mean from birth locations
        birth_locations = torch.cat(birth_locations, dim=0)
        birth_locations = birth_locations - xt_mean[birth_locations_batch_ids]

        birth_types = torch.cat(birth_types, dim=0)

        targets = {
            "target_x": target_x,
            "target_a": target_avf,
            "target_c": target_cvf,
            "target_e": target_evf,
            "birth_rate_target": birth_rate_target,
            "death_rate_target": death_rate_target,
            # targets for gmm
            "birth_locations": birth_locations,
            "birth_types": birth_types,
            "birth_batch_ids": birth_locations_batch_ids,
        }

        return (xt, at, et, xt_batch_id, targets)
