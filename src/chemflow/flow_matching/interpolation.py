import numpy as np
import torch
import torch.nn.functional as F

from torch_geometric.utils import to_dense_adj
from torch_geometric.data import Batch

from chemflow.flow_matching.gmm import interpolate_gmm, interpolate_typed_gmm
from chemflow.flow_matching.sampling import sample_births, sample_deaths
from chemflow.flow_matching.assignment import assign_targets_batched
from chemflow.utils import token_to_index, build_fully_connected_edge_index
from external_code.egnn import unsorted_segment_mean

from chemflow.dataset.molecule_data import (
    PointCloud,
    MoleculeData,
    MoleculeBatch,
    join_molecules,
)
from chemflow.flow_matching.sampling import sample_prior_graph


class Interpolator:
    def __init__(
        self,
        atom_tokens,
        edge_tokens,
        charge_tokens,
        atom_type_distribution,
        edge_type_distribution,
        cat_strategy="uniform-sample",
        n_atoms_strategy="flexible",
        N_samples=20,
    ):
        self.atom_tokens = atom_tokens
        self.edge_tokens = edge_tokens
        self.charge_tokens = charge_tokens
        self.cat_strategy = cat_strategy
        self.n_atoms_strategy = n_atoms_strategy
        self.N_samples = N_samples

        if self.n_atoms_strategy != "fixed":
            self.atom_death_token = token_to_index(self.atom_tokens, "<DEATH>")
        if self.cat_strategy == "mask":
            self.atom_mask_token = token_to_index(self.atom_tokens, "<MASK>")
            self.edge_mask_token = token_to_index(self.edge_tokens, "<MASK>")

        self.atom_type_distribution = atom_type_distribution
        self.edge_type_distribution = edge_type_distribution

        # TODO I'm lazy and this will always be 3
        self.D = 3
        self.M = len(atom_type_distribution)
        self.E = len(edge_type_distribution)
        self.C = len(charge_tokens)

    def interpolate_mols(self, m_0: MoleculeData, m_1: MoleculeData, t):
        """
        Interpolates between two molecules.

        Args:
            m0: MoleculeData at time 0
            m1: MoleculeData at time 1
            t: float, interpolation time in [0, 1]
        Returns:
            m_t: MoleculeData at time t
        """
        x_0, a_0, c_0, e_0, edge_index_0 = m_0.unpack()
        x_1, a_1, c_1, e_1, edge_index_1 = m_1.unpack()

        x_t = self.interpolate_continuous(x_0, x_1, t)
        a_t = self.interpolate_discrete(a_0, a_1, t)
        c_t = self.interpolate_discrete(c_0, c_1, t)

        # e_0 and e_1 can be different sizes. Need to convert to dense adjacency matrices.
        # Then interpolate the dense adjacency matrices.
        # Then convert back to sparse adjacency matrices.
        e_0_dense = to_dense_adj(edge_index_0, edge_attr=e_0)
        e_1_dense = to_dense_adj(edge_index_1, edge_attr=e_1)

        _, N, N = e_0_dense.shape
        e_t_dense = self.interpolate_discrete(
            e_0_dense.flatten(), e_1_dense.flatten(), t
        )
        e_t_dense = e_t_dense.view(N, N)

        # NOTE assuming fully connected graph for now
        edge_index_t = build_fully_connected_edge_index(N, device=x_t.device)

        e_t = e_t_dense[edge_index_t[0], edge_index_t[1]]
        return MoleculeData(x=x_t, a=a_t, e=e_t, edge_index=edge_index_t, c=c_t)

    def interpolate_continuous(self, x0, x1, t):
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

    def interpolate_discrete(self, c0_idx, c1_idx, t):
        """
        Discrete interpolation for discrete variables / one-hot classes c.

        Args:
            c0_idx: (N,) class indices tensor at time 0
            c1_idx: (N,) class indices tensor at time 1
            t: float, interpolation time in [0, 1]
        Returns:
            y_t: (N, ) interpolated class indices tensor at time t
        """

        N = c0_idx.shape[0]

        # Sample Bernoulli mask for which positions to keep from y0
        mask = torch.rand(N, device=t.device) > t  # True = keep from y0
        mask = mask.view(N)

        # Start from y1 and overwrite with y0 where mask=True
        ct_idx = c1_idx.clone()
        ct_idx[mask] = c0_idx[mask]

        return ct_idx

    def interpolate_different_size(
        self, samples_batched, targets_batched, t, optimal_transport="equivariant"
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        """
        Interpolates between x0 and x1, and c0 and c1 for graphs with flexible sizes using batch_id.
        Handles matched nodes, death processes (unmatched x0), and birth processes.

        Args:
            samples_batched (Batch): Batch of samples
            targets_batched (Batch): Batch of targets
            t: Shape (num_graphs,) or scalar - time parameter for each graph
            optimal_transport (str): Optimal transport strategy

        Returns:
            tuple: A tuple of:
            - mols_t: MoleculeBatch of interpolated molecules
            - targets: Dictionary containing the following keys:
                - mols_1: MoleculeBatch of target molecules
                - birth_rate_target: Shape (num_graphs, 1) - birth rate targets
                - death_rate_target: Shape (num_graphs, 1) - death rate targets
                - atoms_to_birth: PointCloud of atoms to birth
        """
        device = samples_batched.x.device

        # 1. Assign targets
        matched_samples, matched_targets, unmatched_samples, unmatched_targets = (
            assign_targets_batched(samples_batched, targets_batched, optimal_transport)
        )

        # 2.1 Interpolate the matched targets
        matched_m_t = [
            self.interpolate_mols(m_0, m_1, t_i)
            for m_0, m_1, t_i in zip(matched_samples, matched_targets, t)
        ]

        matched_m_1 = []
        # for the targets, adjust the edges to be fully connected w/o self-loops
        for matched_target in matched_targets:
            matched_m_1_i = matched_target.clone()

            edge_index_1 = matched_m_1_i.edge_index
            e_1 = matched_m_1_i.e
            e_1_dense = to_dense_adj(edge_index_1, edge_attr=e_1)[0]

            # NOTE assuming fully connected graph for now
            edge_index_1 = build_fully_connected_edge_index(
                matched_m_1_i.num_nodes, device=device
            )

            e_1 = e_1_dense[edge_index_1[0], edge_index_1[1]]
            matched_m_1_i.e = e_1
            matched_m_1_i.edge_index = edge_index_1
            matched_m_1.append(matched_m_1_i)

        # 2.2 Handle unmatched samples / targets
        death_atoms_t = []
        death_atoms_1 = []
        death_rate_target = []

        born_atoms_t = []
        born_atoms_1 = []
        birth_rate_target = []
        atoms_to_birth = []

        empty_x = torch.empty((0, self.D), device=device, dtype=torch.float32)
        empty_a = torch.empty((0), device=device, dtype=torch.long)
        empty_c = torch.empty((0), device=device, dtype=torch.long)
        empty_e = torch.empty((0), device=device, dtype=torch.long)
        empty_edge_index = torch.empty((2, 0), device=device, dtype=torch.long)

        empty_mol = MoleculeData(
            x=empty_x, a=empty_a, e=empty_e, edge_index=empty_edge_index, c=empty_c
        )
        empty_point_cloud = PointCloud(x=empty_x, a=empty_a, c=empty_c)

        # create a sink state that all unmatched x0 will move towards
        x_sink = torch.zeros((1, self.D), device=device)

        for index, (
            unmatched_sample_i,
            unmatched_target_i,
            t_i,
        ) in enumerate(
            zip(
                unmatched_samples,
                unmatched_targets,
                t,
            )
        ):
            instantaneous_rate = 1 / torch.clamp(1 - t_i, min=1e-8)

            # 2.3 Death process
            if unmatched_sample_i.num_nodes > 0:
                # sample death times
                death_times, is_dead, x0_alive_at_xt, a0_alive_at_xt = sample_deaths(
                    unmatched_sample_i.x, unmatched_sample_i.a, t_i
                )

                # interpolate the unmatched x0 to the sink state
                death_xt_i = self.interpolate_x(x0_alive_at_xt, x_sink, t_i)

                death_a1_i = self.death_token * torch.ones(
                    (death_xt_i.shape[0],), dtype=torch.long, device=death_xt_i.device
                )
                death_a1_i = F.one_hot(death_a1_i, num_classes=len(self.atom_tokens))
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

                born_atoms_t.append(empty_mol.clone())
                born_atoms_1.append(empty_mol.clone())

                birth_rate_target.append(torch.zeros((1, 1), device=device))
                atoms_to_birth.append(empty_point_cloud.clone())

            # 2.4 Birth process
            elif unmatched_target_i.num_nodes > 0:
                # sample birth times and locations
                (
                    birth_times,
                    born_atoms_1_i,
                    unborn_atoms_1_i,
                ) = sample_births(
                    unmatched_target_i,
                    t_i,
                )

                # check if any births are happening
                if born_atoms_1_i.num_nodes > 0:
                    raise ValueError("Not implemented")
                    # sample prior graph for born nodes randomly
                    born_atoms_0_i = sample_prior_graph(
                        self.atom_type_distribution,
                        self.edge_type_distribution,
                        self.charge_type_distribution,
                        self.n_atoms_distribution,
                        n_atoms=born_atoms_1_i.num_nodes,
                    )

                    # do another matching between 0 and 1 for born nodes
                    (
                        born_atoms_0_i,
                        born_atoms_1_i,
                        _,
                        _,
                    ) = assign_targets_batched(
                        born_atoms_0_i, born_atoms_1_i, optimal_transport
                    )

                    born_atoms_t_i = [
                        self.interpolate_mols(m_0, m_1, t_i)
                        for m_0, m_1, t_i in zip(born_atoms_0_i, born_atoms_1_i, t_i)
                    ]

                    born_atoms_t.append(born_atoms_t_i)
                    born_atoms_1.append(born_atoms_1_i)

                else:
                    # no birth was sampled
                    born_atoms_t.append(empty_mol.clone())
                    born_atoms_1.append(empty_mol.clone())

                death_atoms_t.append(empty_mol.clone())
                death_atoms_1.append(empty_mol.clone())
                death_rate_target.append(torch.zeros((1, 1), device=device))

                N_necessary_births = unborn_atoms_1_i.num_nodes
                birth_rate_target_i = N_necessary_births * instantaneous_rate
                birth_rate_target.append(birth_rate_target_i.view(1, 1))

                # check if there are any yet unborn atoms
                if unborn_atoms_1_i.num_nodes > 0:
                    # Store the GMM samples for NLL calculation during training
                    if self.cat_strategy == "uniform-sample":
                        p_a_0 = self.atom_type_distribution.unsqueeze(0).to(device)
                        p_a_1 = unborn_a1_i
                        # TODO make sigma hyperparameter
                        # TODO add charge type
                        sampled_x, sampled_a, sampled_c = interpolate_typed_gmm(
                            p_x_1=unborn_x1_i,
                            p_a_0=p_a_0,
                            p_a_1=p_a_1,
                            t=t_i,
                            num_samples=self.N_samples,
                            sigma=0.5,
                        )
                    else:
                        sampled_x = interpolate_gmm(
                            unborn_x1_i, t_i, num_samples=self.N_samples
                        )
                        sampled_a = self.atom_mask_token * torch.ones(
                            (sampled_locations.shape[0],),
                            dtype=torch.long,
                            device=birth_xt_i.device,
                        )
                        sampled_c = self.charge_mask_token * torch.ones(
                            (sampled_x.shape[0],),
                            dtype=torch.long,
                            device=birth_xt_i.device,
                        )
                    atoms_to_birth.append(
                        PointCloud(x=sampled_x, a=sampled_a, c=sampled_c)
                    )

                else:
                    # if no unborn x1, you're done
                    atoms_to_birth.append(empty_point_cloud.clone())

            # 2.5 No birth or death process, just movement
            else:
                death_atoms_t.append(empty_mol.clone())
                death_atoms_1.append(empty_mol.clone())
                death_rate_target.append(torch.zeros((1, 1), device=device))

                born_atoms_t.append(empty_mol.clone())
                born_atoms_1.append(empty_mol.clone())
                birth_rate_target.append(torch.zeros((1, 1), device=device))

                atoms_to_birth.append(empty_point_cloud.clone())

        # 3. Concatenate the matched, death, and birth processes
        mols_t = [
            join_molecules([m_t_i, b_t_i, d_t_i])
            for m_t_i, b_t_i, d_t_i in zip(matched_m_t, born_atoms_t, death_atoms_t)
        ]
        mols_t = MoleculeBatch.from_data_list(mols_t)

        # 4. Concatenate the matched, death, and birth targets
        mols_1 = [
            join_molecules([m_1_i, b_1_i, d_1_i])
            for m_1_i, b_1_i, d_1_i in zip(matched_m_1, born_atoms_1, death_atoms_1)
        ]
        mols_1 = MoleculeBatch.from_data_list(mols_1)

        # TODO must add edges etc between unconnected components!!!

        # remove mean of xt and target_x
        # TODO do we need to do this? unsure, because we add / remove nodes
        xt_mean = unsorted_segment_mean(mols_t.x, mols_t.batch, mols_t.num_graphs)
        _ = mols_t.remove_com(xt_mean)
        _ = mols_1.remove_com(xt_mean)

        # deal with samples for typed gmm
        atoms_to_birth = Batch.from_data_list(atoms_to_birth)
        atoms_to_birth.x = atoms_to_birth.x - xt_mean[atoms_to_birth.batch]

        birth_rate_target = torch.cat(birth_rate_target, dim=0)
        death_rate_target = torch.cat(death_rate_target, dim=0)

        targets = {
            # positions, atom types, charge types, edge types
            "mols_1": mols_1,
            # rate targets
            "birth_rate_target": birth_rate_target,
            "death_rate_target": death_rate_target,
            # targets for gmm
            "atoms_to_birth": atoms_to_birth,
        }

        return mols_t, targets
