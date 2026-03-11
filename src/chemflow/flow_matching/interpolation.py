import torch

from torch_geometric.utils import to_dense_adj

from chemflow.flow_matching.assignment import partial_optimal_transport_single

from chemflow.utils.utils import EdgeAligner

from chemflow.dataset.molecule_data import (
    AugmentedMoleculeData,
    MoleculeData,
    MoleculeBatch,
    filter_nodes,
)

from chemflow.dataset.vocab import Vocab, Distributions

from chemflow.flow_matching.schedules import (
    FastPowerSchedule,
    KappaSchedule,
    LinearSchedule,
)


class Interpolator:
    def __init__(
        self,
        vocab: Vocab,
        distributions: Distributions,
        n_atoms_strategy="flexible",
        optimal_transport="equivariant",
        ins_noise_scale=1.0,
        ins_schedule: KappaSchedule | None = None,
        del_schedule: KappaSchedule | None = None,
        sub_schedule: KappaSchedule | None = None,
        sub_e_schedule: KappaSchedule | None = None,
        c_move=1.0,
        c_sub=0.0,
        c_ins=1e8,
        c_del=0.0,
    ):
        self.vocab = vocab
        self.distributions = distributions
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

        if sub_schedule is None:
            self.sub_schedule = LinearSchedule()
        else:
            self.sub_schedule = sub_schedule

        # Separate schedule for edge type substitutions; falls back to sub_schedule
        if sub_e_schedule is None:
            self.sub_e_schedule = self.sub_schedule
        else:
            self.sub_e_schedule = sub_e_schedule

        self.c_move = c_move
        self.c_sub = c_sub
        self.c_ins = c_ins
        self.c_del = c_del

        self.edge_aligner = EdgeAligner()

        self._cat_atom = torch.distributions.Categorical(
            probs=distributions.atom_type_distribution
        )
        self._cat_charge = torch.distributions.Categorical(
            probs=distributions.charge_type_distribution
        )
        self._cat_edge = torch.distributions.Categorical(
            probs=distributions.edge_type_distribution
        )

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
        # A node is auxiliary in the interpolated paired representation if it is
        # auxiliary on either side of the OT-aligned pair.
        # This is not used downstream, just for consistency.
        is_auxiliary_t = m_0.is_auxiliary | m_1.is_auxiliary

        assert e_0.shape == e_1.shape, "Edge types must have the same shape"
        assert torch.all(edge_index_0 == edge_index_1), "Edge indices must be the same"

        x_t = self.interpolate_continuous(x_0, x_1, t)

        # take kappa_t for discrete variables
        t_kappa = self.sub_schedule.kappa_t(t)
        t_kappa_e = self.sub_e_schedule.kappa_t(t)

        a_t = self.interpolate_discrete(a_0, a_1, t_kappa)
        c_t = self.interpolate_discrete(c_0, c_1, t_kappa)
        e_t = self.interpolate_discrete(e_0, e_1, t_kappa_e)

        # NOTE edge_index is the same for both m_0 and m_1, just take the first one
        # we make sure they are the same with the assert above
        return AugmentedMoleculeData(
            x=x_t,
            a=a_t,
            e=e_t,
            edge_index=edge_index_0,
            c=c_t,
            is_auxiliary=is_auxiliary_t,
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

    def interpolate_discrete(self, y0_idx, y1_idx, t):
        """
        Discrete interpolation for discrete variables / one-hot classes c.

        Args:
            y0_idx: (N,) class indices tensor at time 0
            y1_idx: (N,) class indices tensor at time 1
            t: float, interpolation time in [0, 1]
        Returns:
            y_t: (N, ) interpolated class indices tensor at time t
        """

        N = y0_idx.shape[0]

        # Sample Bernoulli mask for which positions to keep from y0
        mask = torch.rand(N, device=t.device) > t  # True = keep from y0
        mask = mask.view(N)

        # Start from y1 and overwrite with y0 where mask=True
        yt_idx = y1_idx.clone()
        yt_idx[mask] = y0_idx[mask]

        return yt_idx

    def interpolate_batch(
        self,
        samples_batched,
        targets_batched,
        t: torch.Tensor,
    ):
        """Batch interpolation that delegates to interpolate_single per pair.

        Args:
            samples_batched: MoleculeBatch of prior samples.
            targets_batched: MoleculeBatch of target molecules.
            t: (batch_size,) tensor of interpolation times in [0, 1).

        Returns:
            mol_t: MoleculeBatch of interpolated molecules at time t.
            mol_1: MoleculeBatch of target molecules (filtered to existing nodes).
            ins_targets: MoleculeBatch of insertion targets.
        """
        mol_t_list = []
        mol_1_list = []
        ins_targets_list = []

        offset = 0
        for b in range(len(t)):
            mol_t, mol_1, ins_targets = self.interpolate_single(
                samples_batched[b], targets_batched[b], t[b].item()
            )

            # Offset local spawn / edge indices to global mol_t indices
            if (
                hasattr(ins_targets, "spawn_node_idx")
                and ins_targets.spawn_node_idx.numel() > 0
            ):
                ins_targets.spawn_node_idx.add_(offset)
            if (
                hasattr(ins_targets, "ins_edge_spawn_idx")
                and ins_targets.ins_edge_spawn_idx.numel() > 0
            ):
                ins_targets.ins_edge_spawn_idx.add_(offset)
            if (
                hasattr(ins_targets, "ins_edge_target_idx")
                and ins_targets.ins_edge_target_idx.numel() > 0
            ):
                ins_targets.ins_edge_target_idx.add_(offset)

            mol_t_list.append(mol_t)
            mol_1_list.append(mol_1)
            ins_targets_list.append(ins_targets)
            offset += mol_t.num_nodes

        mol_t = MoleculeBatch.from_data_list(mol_t_list)
        mol_1 = MoleculeBatch.from_data_list(mol_1_list)
        ins_targets = MoleculeBatch.from_data_list(ins_targets_list)

        return mol_t, mol_1, ins_targets

    def interpolate_single(
        self,
        sample_mol: MoleculeData,
        target_mol: MoleculeData,
        t_scalar: float,
    ):
        """Per-sample interpolation (OT alignment + interpolation + rates).

        Performs OT alignment and interpolation for a single (sample, target)
        pair. Returns objects with LOCAL node indices suitable for later
        batching with global offset correction in ``interpolate_batch`` or
        ``train_collate_fn``.

        Args:
            sample_mol: Prior sample graph.
            target_mol: Target graph from the dataset.
            t_scalar: Interpolation time in [0, 1).

        Returns:
            interp_state: Interpolated molecule at time t (with rate attributes).
            target_state: Target molecule filtered to existing nodes.
            future_ins_nodes: Insertion targets (with local spawn/edge indices).
        """
        device = sample_mol.x.device
        t_i = torch.tensor(t_scalar, device=device)

        # ===== 1. OT Alignment =====
        sample, target = partial_optimal_transport_single(
            sample_mol,
            target_mol,
            c_move=self.c_move,
            c_sub=self.c_sub,
            c_ins=self.c_ins,
            c_del=self.c_del,
            optimal_transport=self.optimal_transport,
        )

        # ===== 2. Per-pair interpolation =====
        N = sample.x.shape[0]

        is_sub = (~sample.is_auxiliary) & (~target.is_auxiliary)  # Real -> Real
        is_del = (~sample.is_auxiliary) & (target.is_auxiliary)  # Real -> Dummy
        is_ins = (sample.is_auxiliary) & (~target.is_auxiliary)  # Dummy -> Real

        is_sub = is_sub.squeeze()
        is_del = is_del.squeeze()
        is_ins = is_ins.squeeze()

        # Scheduling
        # Assign a random event time tau_i ~ U[0,1] to every node.
        # We then determine if the event happens with kappa_t.
        inst_rate_sub = self.sub_schedule.rate(t_i)
        inst_rate_sub_e = self.sub_e_schedule.rate(t_i)

        tau_del = torch.rand(N, 1, device=device)
        t_del = self.del_schedule.kappa_t(t_i)
        inst_rate_del = self.del_schedule.rate(t_i)

        tau_ins = torch.rand(N, 1, device=device)
        t_ins = self.ins_schedule.kappa_t(t_i)
        inst_rate_ins = self.ins_schedule.rate(t_i)

        # Masks to determine present nodes in the interpolated state.
        mask_keep_sub = is_sub
        mask_keep_del = is_del & (t_del < tau_del).squeeze()
        mask_keep_ins = is_ins & (t_ins > tau_ins).squeeze()
        mask_future_ins = is_ins & (t_ins <= tau_ins).squeeze()
        mask_exists = mask_keep_sub | mask_keep_del | mask_keep_ins

        # A. Deletions: freeze at source
        if mask_keep_del.any():
            target.x[mask_keep_del] = sample.x[mask_keep_del]
            target.a[mask_keep_del] = sample.a[mask_keep_del]
            target.c[mask_keep_del] = sample.c[mask_keep_del]

        # B. Insertions: jump to target with noise
        if mask_keep_ins.any():
            n_ins = mask_keep_ins.sum()
            sample.x[mask_keep_ins] = (
                target.x[mask_keep_ins]
                + torch.randn(n_ins, sample.x.shape[-1], device=device)
                * self.ins_noise_scale
            )
            a_ins = self._cat_atom.sample((n_ins,))
            c_ins = self._cat_charge.sample((n_ins,))
            sample.a[mask_keep_ins] = a_ins
            sample.c[mask_keep_ins] = c_ins

        # C. Edge handling
        triu_rows, triu_cols = torch.triu_indices(N, N, offset=1, device=device)
        triu_edge_index = torch.stack([triu_rows, triu_cols], dim=0)

        def _extract_triu_feats(data_obj):
            attr = data_obj.e
            if attr is None or attr.numel() == 0:
                feat_dim = 1
                return torch.zeros(
                    (triu_rows.shape[0], feat_dim), device=device, dtype=torch.long
                )
            if attr.dim() == 1:
                attr = attr.unsqueeze(-1)
            dense_adj = to_dense_adj(
                data_obj.edge_index, edge_attr=attr, max_num_nodes=N
            )[0]
            return dense_adj[triu_rows, triu_cols].squeeze()

        e0_triu = _extract_triu_feats(sample)
        e1_triu = _extract_triu_feats(target)

        node_mask_del = mask_keep_del.squeeze()
        node_mask_ins = mask_keep_ins.squeeze()

        # Use mutually exclusive edge classes to avoid double-editing del-ins edges.
        edge_has_del = node_mask_del[triu_rows] | node_mask_del[triu_cols]
        edge_has_ins = node_mask_ins[triu_rows] | node_mask_ins[triu_cols]
        edge_mask_del = edge_has_del & ~edge_has_ins
        edge_mask_ins = edge_has_ins & ~edge_has_del

        # del edges are frozen at source
        if edge_mask_del.any():
            e1_triu[edge_mask_del] = e0_triu[edge_mask_del]

        if edge_mask_ins.any():
            n_edges_ins = edge_mask_ins.sum()
            rand_edges = self._cat_edge.sample((n_edges_ins,)).to(device)
            e0_triu[edge_mask_ins] = rand_edges.to(dtype=e0_triu.dtype)

        sample.edge_index = triu_edge_index
        sample.e = e0_triu
        target.edge_index = triu_edge_index
        target.e = e1_triu

        # D. Interpolate
        interp_state = self.interpolate_mols(sample, target, t_i)

        # E. Symmetrize edges
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

        # F. Filter nodes
        interp_state = filter_nodes(interp_state, mask_exists.squeeze())
        target_state = filter_nodes(target, mask_exists.squeeze())
        future_ins_nodes = filter_nodes(target, mask_future_ins.squeeze())

        # G. Spawn node logic (LOCAL indices)
        num_interp_nodes = interp_state.x.shape[0]
        num_future_ins_nodes = future_ins_nodes.x.shape[0]
        ins_counts = torch.zeros(num_interp_nodes, device=device)

        if num_future_ins_nodes > 0 and num_interp_nodes > 0:
            dists = torch.cdist(future_ins_nodes.x, interp_state.x)
            spawn_idx_local = torch.argmin(dists, dim=1).to(device)

            ones = torch.ones(spawn_idx_local.shape[0], device=device)
            ins_counts.index_add_(0, spawn_idx_local, ones)

            future_ins_nodes.spawn_node_idx = spawn_idx_local

            # Edge targets for insertions
            exists_indices_original = torch.where(mask_exists.squeeze())[0]
            ins_indices_original = torch.where(mask_future_ins.squeeze())[0]

            orig_to_interp = torch.full((N,), -1, dtype=torch.long, device=device)
            orig_to_interp[exists_indices_original] = torch.arange(
                num_interp_nodes, device=device
            )

            orig_to_ins = torch.full((N,), -1, dtype=torch.long, device=device)
            orig_to_ins[ins_indices_original] = torch.arange(
                num_future_ins_nodes, device=device
            )

            src, dst = target.edge_index
            src_is_ins = orig_to_ins[src] >= 0
            dst_is_exists = orig_to_interp[dst] >= 0
            ins_to_exists_mask = src_is_ins & dst_is_exists

            if ins_to_exists_mask.any():
                ins_edge_src_orig = src[ins_to_exists_mask]
                ins_edge_dst_orig = dst[ins_to_exists_mask]
                ins_edge_types = target.e[ins_to_exists_mask]

                ins_edge_src_local = orig_to_ins[ins_edge_src_orig]
                ins_edge_dst_local = orig_to_interp[ins_edge_dst_orig]

                future_ins_nodes.ins_edge_spawn_idx = spawn_idx_local[
                    ins_edge_src_local
                ]
                future_ins_nodes.ins_edge_target_idx = ins_edge_dst_local
                future_ins_nodes.ins_edge_types = ins_edge_types
            else:
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

        # H. Attach rate targets
        interp_state.lambda_ins = ins_counts * inst_rate_ins
        interp_state.n_ins = ins_counts

        node_del_rate_target = torch.zeros((N,), device=device)
        node_del_rate_target[mask_keep_del] = inst_rate_del
        interp_state.lambda_del = node_del_rate_target[mask_exists.squeeze()]

        need_a_sub = (interp_state.a != target_state.a).int()
        need_c_sub = (interp_state.c != target_state.c).int()
        need_e_sub = (interp_state.e != target_state.e).int()

        interp_state.lambda_a_sub = need_a_sub * inst_rate_sub
        interp_state.lambda_c_sub = need_c_sub * inst_rate_sub
        interp_state.lambda_e_sub = need_e_sub * inst_rate_sub_e

        interp_state.n_ins_missing = mask_future_ins.sum().float()
        interp_state.n_del_missing = mask_keep_del.sum().float()

        # I. Remove center of mass (mol_t defines the reference frame)
        x_mean = interp_state.x.mean(dim=0)
        interp_state.x = interp_state.x - x_mean
        target_state.x = target_state.x - x_mean
        if future_ins_nodes.x.shape[0] > 0:
            future_ins_nodes.x = future_ins_nodes.x - x_mean

        return interp_state, target_state, future_ins_nodes
