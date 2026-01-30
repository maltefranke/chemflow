import numpy as np
import torch
import torch.nn.functional as F

from torch_geometric.utils import to_dense_adj

from chemflow.flow_matching.assignment import partial_optimal_transport

from chemflow.utils import (
    token_to_index,
    EdgeAligner,
    validate_no_cross_batch_edges,
)
from external_code.egnn import unsorted_segment_mean

from chemflow.dataset.molecule_data import (
    AugmentedMoleculeData,
    MoleculeData,
    MoleculeBatch,
    filter_nodes,
)

from chemflow.dataset.vocab import Vocab, Distributions

from chemflow.flow_matching.schedules import FastPowerSchedule, KappaSchedule


class Interpolator:
    def __init__(
        self,
        vocab: Vocab,
        distributions: Distributions,
        cat_strategy="uniform-sample",
        n_atoms_strategy="flexible",
        optimal_transport="equivariant",
        ins_noise_scale=1.0,
        ins_schedule: KappaSchedule | None = None,
        del_schedule: KappaSchedule | None = None,
        c_move=1.0,
        c_sub=0.0,
        c_ins=1e8,
    ):
        self.vocab = vocab
        self.distributions = distributions
        self.cat_strategy = cat_strategy
        self.n_atoms_strategy = n_atoms_strategy

        self.optimal_transport = optimal_transport
        self.ins_noise_scale = ins_noise_scale

        if del_schedule is None:
            self.del_schedule = FastPowerSchedule(beta=2.5)
        else:
            self.del_schedule = del_schedule

        if ins_schedule is None:
            self.ins_schedule = FastPowerSchedule(beta=2.5)
        else:
            self.ins_schedule = ins_schedule

        self.c_move = c_move
        self.c_sub = c_sub
        self.c_ins = c_ins

        if self.cat_strategy == "mask":
            self.atom_mask_token = token_to_index(self.atom_tokens, "<MASK>")
            self.edge_mask_token = token_to_index(self.edge_tokens, "<MASK>")

        # TODO I'm lazy and this will always be 3
        self.D = 3
        self.M = len(self.distributions.atom_type_distribution)
        self.E = len(self.distributions.edge_type_distribution)
        self.C = len(self.distributions.charge_type_distribution)

        self.edge_aligner = EdgeAligner()

    def interpolate_mols(self, m_0: MoleculeData, m_1: MoleculeData, t):
        """
        Interpolates between two molecules.

        Args:
            m0: AugmentedMoleculeData at time 0
            m1: AugmentedMoleculeData at time 1
            t: float, interpolation time in [0, 1]
        Returns:
            m_t: AugmentedMoleculeData at time t
        """

        x_0, a_0, c_0, e_0, edge_index_0 = m_0.unpack()
        x_1, a_1, c_1, e_1, edge_index_1 = m_1.unpack()
        is_auxiliary_0 = m_0.is_auxiliary

        assert e_0.shape == e_1.shape, "Edge types must have the same shape"
        assert torch.all(edge_index_0 == edge_index_1), "Edge indices must be the same"

        x_t = self.interpolate_continuous(x_0, x_1, t)
        a_t = self.interpolate_discrete(a_0, a_1, t)
        c_t = self.interpolate_discrete(c_0, c_1, t)
        e_t = self.interpolate_discrete(e_0, e_1, t)

        # NOTE edge_index is the same for both m_0 and m_1, just take the first one
        # we make sure they are the same with the assert above
        return AugmentedMoleculeData(
            x=x_t,
            a=a_t,
            e=e_t,
            edge_index=edge_index_0,
            c=c_t,
            is_auxiliary=is_auxiliary_0,  # Will not be needed for interpolation
        )

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
        self,
        samples_batched,
        targets_batched,
        t: float,
    ):
        """Interpolation handling insertion, deletion, and substitution.

        Args:
            samples_batched: MoleculeBatch - batch of sample molecules
            targets_batched: MoleculeBatch - batch of target molecules
            t: float - interpolation time in [0, 1]

        Returns:
            mol_t: MoleculeBatch - batch of interpolated molecules
            mol_1: MoleculeBatch - batch of target molecules
            ins_targets: MoleculeBatch - batch of insertion targets
        """
        # 1. Transport & Alignment (Uses your existing helpers)
        # Returns list of (AugmentedMoleculeData, AugmentedMoleculeData)
        aligned_pairs = partial_optimal_transport(
            samples_batched,
            targets_batched,
            c_move=self.c_move,
            c_sub=self.c_sub,
            c_ins=self.c_ins,  # High birth cost forces matches where possible
            optimal_transport=self.optimal_transport,
        )

        interpolated_graphs = []
        target_graphs = []
        ins_targets_list = []

        total_num_nodes_mol_t = 0
        for (sample, target), t_i in zip(aligned_pairs, t):
            device = sample.x.device
            N = sample.x.shape[0]

            # --- 2. Classification ---
            # Create boolean masks for the three processes based on auxiliary status
            is_sub = (~sample.is_auxiliary) & (~target.is_auxiliary)  # Real -> Real
            is_del = (~sample.is_auxiliary) & (target.is_auxiliary)  # Real -> Dummy
            is_ins = (sample.is_auxiliary) & (~target.is_auxiliary)  # Dummy -> Real

            # squeeze the masks
            is_sub = is_sub.squeeze()
            is_del = is_del.squeeze()
            is_ins = is_ins.squeeze()

            # Sanity check: Dummies mapping to Dummies should have been filtered by OT already,
            # but we can ignore them safely via these masks.

            # --- 3. Scheduling (Vectorized) ---
            # Assign a random event time tau_i ~ U[0,t_end] to every node.

            # We will do substitution during the whole interpolation.
            inst_rate_sub = 1 / torch.clamp(1 - t_i, min=1e-8)

            tau_del = torch.rand(N, 1, device=device)
            t_del = self.del_schedule.kappa_t(t_i)
            inst_rate_del = self.del_schedule.rate(t_i)

            tau_ins = torch.rand(N, 1, device=device)
            t_ins = self.ins_schedule.kappa_t(t_i)
            inst_rate_ins = self.ins_schedule.rate(t_i)

            # Substitution nodes (always active)
            mask_keep_sub = is_sub

            # Deletion nodes that have not been deleted yet
            mask_keep_del = is_del & (t_del < tau_del).squeeze()

            # Insertion nodes that have already been inserted
            mask_keep_ins = is_ins & (t_ins > tau_ins).squeeze()

            # Future insertion nodes waiting to be born
            mask_future_ins = is_ins & (t_ins <= tau_ins).squeeze()

            # The mask of all nodes that exist in the INTERPOLATED state
            mask_exists = mask_keep_sub | mask_keep_del | mask_keep_ins

            # --- 4. Interpolation Logic ---

            # A. Handle Deletions (Frozen at Source until disappearance)
            if mask_keep_del.any():
                # Nodes that have to die will not move and stay at sample
                target.x[mask_keep_del] = sample.x[mask_keep_del]
                target.a[mask_keep_del] = sample.a[mask_keep_del]
                target.c[mask_keep_del] = sample.c[mask_keep_del]

            # B. Handle Births (Jump to Target)
            if mask_keep_ins.any():
                n_ins = mask_keep_ins.sum()

                # Once born, they appear around the target position
                # NOTE this will likely break the optimal transport!
                sample.x[mask_keep_ins] = (
                    target.x[mask_keep_ins]
                    + torch.randn(n_ins, self.D, device=device) * self.ins_noise_scale
                )

                # sample marginal atom types and charges from the empirical distribution
                a_ins = torch.distributions.Categorical(
                    probs=self.distributions.atom_type_distribution
                ).sample((n_ins,))

                c_ins = torch.distributions.Categorical(
                    probs=self.distributions.charge_type_distribution
                ).sample((n_ins,))

                sample.a[mask_keep_ins] = a_ins
                sample.c[mask_keep_ins] = c_ins

            # C. Handle edges
            # C.1 Generate Indices for Upper Triangle (excluding diagonal)
            # This is our canonical list of edges to interpolate
            triu_rows, triu_cols = torch.triu_indices(N, N, offset=1, device=device)
            triu_edge_index = torch.stack([triu_rows, triu_cols], dim=0)

            # C.2 Helper: Extract Upper Triangular Features from Sparse Object
            def _extract_triu_feats(data_obj):
                # C.2.1 To Dense (N, N, F)
                # Handle case where edge_attr is missing or 1D
                attr = data_obj.e
                if attr is None or attr.numel() == 0:
                    # Default to No-Bond (0)
                    # Use a default feature dim, e.g., 1 or whatever your model expects
                    feat_dim = 1
                    return torch.zeros(
                        (triu_rows.shape[0], feat_dim), device=device, dtype=torch.long
                    )

                if attr.dim() == 1:
                    attr = attr.unsqueeze(-1)

                # to_dense_adj fills missing edges with 0
                dense_adj = to_dense_adj(
                    data_obj.edge_index, edge_attr=attr, max_num_nodes=N
                )[0]

                # C.2.2 Extract strictly upper triangular part + diagonal
                return dense_adj[triu_rows, triu_cols].squeeze()  # Shape (N_triu, )

            e0_triu = _extract_triu_feats(sample)
            e1_triu = _extract_triu_feats(target)

            # C.3 Create Edge Masks (for the Upper Triangle)
            # An edge (u, v) is affected if u OR v is in the mask
            # We map node masks to the upper triangular edges
            node_mask_del = mask_keep_del.squeeze()
            edge_mask_del = node_mask_del[triu_rows] | node_mask_del[triu_cols]

            node_mask_ins = mask_keep_ins.squeeze()
            edge_mask_ins = node_mask_ins[triu_rows] | node_mask_ins[triu_cols]

            # D. Apply Logic (Deletion: Freeze / Insertion: Jump)

            # Deletion: Freeze Target edges to Sample values
            if edge_mask_del.any():
                e1_triu[edge_mask_del] = e0_triu[edge_mask_del]

            # Birth: Initialize Sample edges with Random Noise
            if edge_mask_ins.any():
                n_edges_ins = edge_mask_ins.sum()
                # Draw from edge distribution
                rand_edges = (
                    torch.distributions.Categorical(
                        probs=self.distributions.edge_type_distribution
                    )
                    .sample((n_edges_ins,))
                    .to(device)
                )

                e0_triu[edge_mask_ins] = rand_edges.to(dtype=e0_triu.dtype)

            # E. Assign Upper Triangle Edges back to objects
            # This prepares them for interpolate_mols
            sample.edge_index = triu_edge_index
            sample.e = e0_triu

            target.edge_index = triu_edge_index
            target.e = e1_triu

            # D. Do the actual interpolation
            interp_state = self.interpolate_mols(sample, target, t_i)

            # --- 5. Symmetrization ---
            # Reconstruct fully connected graph from interpolated upper triangle
            full_idx_interp, full_attr_interp = self.edge_aligner.symmetrize_edges(
                interp_state.edge_index, [interp_state.e]
            )
            interp_state.edge_index = full_idx_interp
            interp_state.e = full_attr_interp[0]

            full_idx_target, full_attr_target = self.edge_aligner.symmetrize_edges(
                target.edge_index, [target.e]
            )
            target.edge_index = full_idx_target
            target.e = full_attr_target[0]

            # --- 6. Construct Result Objects (Filtering ONCE) ---

            # The Intermediary State (Graph at time t)
            # We filter the 'exists' mask. Indices are handled automatically.
            interp_state = filter_nodes(interp_state, mask_exists.squeeze())
            target_state = filter_nodes(target, mask_exists.squeeze())

            # The Future Insertion Targets (Special object for GMM target)
            # We want the *Target* properties for these nodes
            future_ins_nodes = filter_nodes(target, mask_future_ins.squeeze())

            # --- 7. Spawn Node Logic ---
            # Link each future insertion to the closest existing node in interp_state.
            # This "spawn node" is the one that predicts/triggers the insertion via GMM.

            num_interp_nodes = interp_state.x.shape[0]
            num_future_ins_nodes = future_ins_nodes.x.shape[0]

            # This tensor holds the count of insertions spawned by each node
            ins_counts = torch.zeros(num_interp_nodes, device=device)

            if num_future_ins_nodes > 0 and num_interp_nodes > 0:
                # Find closest node in interp_state for each future insertion
                dists = torch.cdist(future_ins_nodes.x, interp_state.x)
                spawn_idx_local = torch.argmin(dists, dim=1).to(device)

                # Accumulate counts (handle multiple insertions from same spawn node)
                ones = torch.ones(spawn_idx_local.shape[0], device=device)
                ins_counts.index_add_(0, spawn_idx_local, ones)

                # Convert to global indices (offset by nodes already processed)
                spawn_idx_global = spawn_idx_local + total_num_nodes_mol_t
                future_ins_nodes.spawn_node_idx = spawn_idx_global

                # --- 7b. Compute target edges for insertions ---
                # For each insertion node, find its edges to existing nodes in target
                # These are the training targets for the InsertionEdgeHead

                # Create index mappings from original to filtered indices
                exists_indices_original = torch.where(mask_exists.squeeze())[0]
                ins_indices_original = torch.where(mask_future_ins.squeeze())[0]

                # Map: original_idx -> local_idx in interp_state (-1 if not exists)
                orig_to_interp = torch.full((N,), -1, dtype=torch.long, device=device)
                orig_to_interp[exists_indices_original] = torch.arange(
                    num_interp_nodes, device=device
                )

                # Map: original_idx -> local_idx in future_ins_nodes (-1 if not ins)
                orig_to_ins = torch.full((N,), -1, dtype=torch.long, device=device)
                orig_to_ins[ins_indices_original] = torch.arange(
                    num_future_ins_nodes, device=device
                )

                # Extract edges where one endpoint is insertion, other is existing
                src, dst = target.edge_index

                src_is_ins = orig_to_ins[src] >= 0
                dst_is_exists = orig_to_interp[dst] >= 0
                ins_to_exists_mask = src_is_ins & dst_is_exists

                if ins_to_exists_mask.any():
                    # Get the edges and their types
                    ins_edge_src_orig = src[ins_to_exists_mask]
                    ins_edge_dst_orig = dst[ins_to_exists_mask]
                    ins_edge_types = target.e[ins_to_exists_mask]

                    # Convert to local indices
                    ins_edge_src_local = orig_to_ins[ins_edge_src_orig]
                    ins_edge_dst_local = orig_to_interp[ins_edge_dst_orig]

                    # Offset dst indices to global mol_t indices
                    ins_edge_target_global = ins_edge_dst_local + total_num_nodes_mol_t

                    # For each edge from insertion node, get the spawn node
                    # (the node in mol_t that predicts/triggers this insertion)
                    ins_edge_spawn_global = spawn_idx_global[ins_edge_src_local]

                    future_ins_nodes.ins_edge_spawn_idx = ins_edge_spawn_global
                    future_ins_nodes.ins_edge_target_idx = ins_edge_target_global
                    future_ins_nodes.ins_edge_types = ins_edge_types
                else:
                    # No insertion edges
                    future_ins_nodes.ins_edge_spawn_idx = torch.empty(
                        (0,), device=device, dtype=torch.long
                    )
                    future_ins_nodes.ins_edge_target_idx = torch.empty(
                        (0,), device=device, dtype=torch.long
                    )
                    future_ins_nodes.ins_edge_types = torch.empty(
                        (0,), device=device, dtype=torch.long
                    )

            else:
                # Empty tensors for proper batching
                future_ins_nodes.spawn_node_idx = torch.empty(
                    (0,), device=device, dtype=torch.long
                )
                future_ins_nodes.ins_edge_spawn_idx = torch.empty(
                    (0,), device=device, dtype=torch.long
                )
                future_ins_nodes.ins_edge_target_idx = torch.empty(
                    (0,), device=device, dtype=torch.long
                )
                future_ins_nodes.ins_edge_types = torch.empty(
                    (0,), device=device, dtype=torch.long
                )

            # 3. Calculate Rate and Attach to Object
            # This ensures the target travels with the batch object
            # instantaneous_rate should be the scalar lambda(t)

            # INSERTION
            interp_state.lambda_ins = ins_counts * inst_rate_ins
            interp_state.n_ins = ins_counts

            # DELETION
            node_del_rate_target = torch.zeros((N,), device=device)
            node_del_rate_target[mask_keep_del] = inst_rate_del
            interp_state.lambda_del = node_del_rate_target[mask_exists.squeeze()]

            need_a_sub = (interp_state.a != target_state.a).int()
            need_c_sub = (interp_state.c != target_state.c).int()
            need_e_sub = (interp_state.e != target_state.e).int()

            interp_state.lambda_a_sub = need_a_sub * inst_rate_sub
            interp_state.lambda_c_sub = need_c_sub * inst_rate_sub
            interp_state.lambda_e_sub = need_e_sub * inst_rate_sub

            # Graph-level counts of insertions and deletions still missing
            # These are used by the global budget heads for graph-level predictions
            interp_state.n_ins_missing = mask_future_ins.sum().float()
            interp_state.n_del_missing = mask_keep_del.sum().float()

            # Collect results
            interpolated_graphs.append(interp_state)
            target_graphs.append(target_state)
            ins_targets_list.append(future_ins_nodes)

            total_num_nodes_mol_t += interp_state.num_nodes

        # Re-batch the list of data objects into a single Batch object
        mol_t = MoleculeBatch.from_data_list(interpolated_graphs)
        mol_1 = MoleculeBatch.from_data_list(target_graphs)
        ins_targets = MoleculeBatch.from_data_list(ins_targets_list)

        # Remove the centers of mass
        x_mean = unsorted_segment_mean(mol_t.x, mol_t.batch, mol_t.num_graphs)
        _ = mol_t.remove_com(x_mean)
        _ = mol_1.remove_com(x_mean)
        _ = ins_targets.remove_com(x_mean)

        # Validate: check for cross-batch edges after interpolation
        validate_no_cross_batch_edges(
            mol_t.edge_index, mol_t.batch, "interpolate_different_size mol_t"
        )
        validate_no_cross_batch_edges(
            mol_1.edge_index, mol_1.batch, "interpolate_different_size mol_1"
        )

        return mol_t, mol_1, ins_targets
