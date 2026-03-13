from torch_geometric.data import Data, Batch, HeteroData
import torch
from torch_geometric.utils import to_dense_adj
from rdkit import Chem
import numpy as np

from chemflow.utils.utils import index_to_token, EdgeAligner
from external_code.egnn import unsorted_segment_mean

from torch_geometric.utils import sort_edge_index
from torch.distributions import Categorical

from chemflow.utils.repr import tensors_to_rdkit_mol


# RDKit bond type to index mapping
IDX_BOND_MAP = {
    "1": Chem.BondType.SINGLE,
    "2": Chem.BondType.DOUBLE,
    "3": Chem.BondType.TRIPLE,
    "4": Chem.BondType.AROMATIC,
}


class PointCloud(Data):
    """
    A generic point cloud data object. Contains only x, a, and c - no edges.
    """

    def __init__(
        self,
        x: torch.Tensor | None = None,
        a: torch.Tensor | None = None,
        c: torch.Tensor | None = None,
        **kwargs,
    ):
        super().__init__(x=x, a=a, c=c, **kwargs)


class MoleculeData(PointCloud):
    """
    A generic molecule data object. Contains x, a, c, and the typed edges between them.

    Attributes:
        x: The node coordinates.
        a: The atom types.
        e: The edge types.
        edge_index: The edge index.
        c: The charge.
    """

    def __init__(
        self,
        x: torch.Tensor | None = None,
        a: torch.Tensor | None = None,
        e: torch.Tensor | None = None,
        edge_index: torch.Tensor | None = None,
        c: torch.Tensor | None = None,
        **kwargs,
    ):
        # Allow PyTorch Geometric's separate() to create empty instances
        # It will populate attributes via stores_as() mechanism
        super().__init__(x=x, a=a, c=c, e=e, edge_index=edge_index, **kwargs)

    @property
    def get_dims(self):
        """
        Get the dimensions of the data.
        Returns:
            D: The dimension of the coordinates.
            M: The number of atom types.
            C: The number of charge types.
        """
        D = self.x.shape[-1]
        M = self.a.shape[-1] - 1
        C = self.c.shape[-1] - 1
        return D, M, C

    def unpack(self):
        """
        Unpack the data into a tuple of tensors.
        Returns:
            x: The coordinates.
            a: The atom types.
            e: The edge types.
            edge_index: The edge index.
        """
        return (
            self.x,
            self.a,
            self.c,
            self.e,
            self.edge_index,
        )

    def get_e_triu(self):
        """
        Get the upper triangle of the edge types.
        Returns:
            e_triu: The upper triangle of the edge types.
        """

        dense = to_dense_adj(
            self.edge_index, edge_attr=self.e, max_num_nodes=self.num_nodes
        )

        dense = dense.squeeze()
        edge_index = torch.triu_indices(self.num_nodes, self.num_nodes, offset=1)
        e_triu = dense[edge_index[0], edge_index[1]]
        return e_triu, edge_index

    def get_permuted_subgraph(self, subset):
        """
        Creates a subgraph where nodes are ordered exactly as in 'subset'.
        Topological connections are remapped to match this new order.
        """
        device = self.x.device
        num_nodes = self.num_nodes

        # 1. Create a mapping from old_index -> new_index
        # We initialize with -1 to detect edges connecting to non-subset nodes
        mapping = torch.full((num_nodes,), -1, dtype=torch.long, device=device)

        # The key difference: We map subset[i] to i
        # This ensures new node 0 is subset[0], new node 1 is subset[1], etc.
        mapping[subset] = torch.arange(subset.size(0), device=device)

        # 2. Filter and Remap Edges
        # Find edges where both start and end nodes are in the subset
        mask = (mapping[self.edge_index[0]] >= 0) & (mapping[self.edge_index[1]] >= 0)

        # Remap the edge indices to the new 0...K IDs
        new_edge_index = mapping[self.edge_index[:, mask]]

        # 3. Filter Attributes
        # Node attributes (Selecting by index respects the permutation)
        new_x = self.x[subset] if self.x is not None else None
        new_a = self.a[subset] if self.a is not None else None
        new_c = self.c[subset] if self.c is not None else None

        # Edge attributes (Use the mask computed above)
        new_e = self.e[mask] if self.e is not None else None

        # 4. Return new MoleculeData object
        return MoleculeData(
            x=new_x, a=new_a, e=new_e, edge_index=new_edge_index, c=new_c
        )

    def to_rdkit_mol(
        self,
        atom_tokens: list[str],
        edge_tokens: list[str],
        charge_tokens: list[str],
        sanitize: bool = True,
    ):
        a = self.a.clone().detach().cpu().numpy()
        x = self.x.clone().detach().cpu().numpy()
        c = self.c.clone().detach().cpu().numpy()

        atom_tokens = [index_to_token(atom_tokens, int(index)) for index in a]
        charge_tokens = [int(index_to_token(charge_tokens, int(index))) for index in c]

        # handle edge case where there is only one atom (no edges)
        if self.num_nodes == 0:
            return None
        elif self.num_nodes == 1:
            edge_types = []
            edge_index_list = []
        else:
            if torch.max(self.e.view(-1)) > len(edge_tokens):
                print("e max was outside of edge tokens already")
                print(self.e.max(), len(edge_tokens), self.e.shape)
                exit()

            e_triu, edge_index_triu = self.get_e_triu()
            if torch.max(e_triu.view(-1)) > len(edge_tokens):
                print("e_triu max was outside of edge tokens")
                print(
                    self.edge_index,
                    self.e,
                    e_triu,
                    torch.max(self.e),
                    torch.max(e_triu),
                    len(edge_tokens),
                )
                exit()

            e = e_triu.clone().detach().cpu().numpy()
            edge_tokens = [index_to_token(edge_tokens, int(index)) for index in e]

            # make edge_index (2, N) -> (N, 2)
            edge_index = edge_index_triu.clone().detach().cpu().numpy()
            edge_index = edge_index.T.tolist()

            edge_types = []
            edge_index_list = []
            for edge, edge_type in zip(edge_index, edge_tokens):
                if edge_type == "<NO_BOND>":
                    continue
                edge_types.append(IDX_BOND_MAP[edge_type])
                edge_index_list.append(edge)

        mol = tensors_to_rdkit_mol(
            atom_tokens, x, charge_tokens, edge_types, edge_index_list
        )

        return mol


class MoleculeBatch(Batch):
    """
    A batch of molecule data objects.
    """

    _data_cls = MoleculeData  # Tell PyG which class to use for separation

    def unpack(self):
        """
        Unpack the data into a tuple of tensors.
        Returns:
            x: The coordinates.
            a: The atom types.
            c: The charge types.
            e: The edge types.
            edge_index: The edge index.
            batch: The batch index.
        """
        return self.x, self.a, self.c, self.e, self.edge_index, self.batch

    def remove_com(self, x_mean=None):
        """Remove the center of mass from the coordinates."""
        if x_mean is None:
            x_mean = unsorted_segment_mean(self.x, self.batch, self.num_graphs)
        else:
            assert x_mean.shape == (self.num_graphs, self.x.shape[-1]), (
                "x_mean must have the same shape as the coordinates"
            )
        self.x = self.x - x_mean[self.batch]
        return x_mean

    def to_data_list(self):
        """
        Manually reconstructs a list of MoleculeData objects.
        Required because manual batch instantiation breaks internal slice metadata.
        """
        data_list = []

        # Iterate over each graph ID present in the batch
        # We use unique() to skip graphs that might have been fully filtered out
        for i in self.batch.unique():
            # 1. Create Masks
            node_mask = self.batch == i
            # Check BOTH endpoints are in this batch (not just source)
            edge_mask = node_mask[self.edge_index[0]] & node_mask[self.edge_index[1]]

            # 2. Re-center Edge Indices (Global -> Local 0-N)
            # Find the global index of the first node in this graph
            min_node_idx = node_mask.nonzero(as_tuple=False).min()
            local_edge_index = self.edge_index[:, edge_mask] - min_node_idx

            # 3. Construct the Single Data Object
            data = self._data_cls(
                x=self.x[node_mask],
                a=self.a[node_mask],
                c=self.c[node_mask],
                e=self.e[edge_mask],
                edge_index=local_edge_index,
            )
            data_list.append(data)

        return data_list


def join_molecules(mol_list: list[MoleculeData]) -> MoleculeData:
    # 1. Use Batch to merge everything correctly (shifts edge_index, cats features)
    batch = MoleculeBatch.from_data_list(mol_list)

    # 2. Extract the merged attributes to create your single Data object
    #    (This strips away the 'batch', 'ptr', and 'slice_dict' attributes)
    merged_mol = MoleculeData(
        x=batch.x,
        a=batch.a,
        e=batch.e,
        edge_index=batch.edge_index,
        c=batch.c,
        # Add any other custom attributes here
    )

    return merged_mol


class AugmentedMoleculeData(Data):
    """
    An Augmented molecule object that contains auxiliary nodes. Used for aligning samples and targets.
    Simplifies interpolation of birth and death nodes.
    """

    def __init__(
        self,
        x: torch.Tensor | None = None,
        a: torch.Tensor | None = None,
        is_auxiliary: torch.Tensor | None = None,
        edge_index: torch.Tensor | None = None,
        e: torch.Tensor | None = None,
        c: torch.Tensor | None = None,
        spawn_node_idx: torch.Tensor | None = None,
        **kwargs,
    ):
        # Initialize PyG Data base
        super().__init__(x=x, edge_index=edge_index, **kwargs)

        # Explicitly assign attributes to ensure they are registered
        if a is not None:
            self.a = a
        if c is not None:
            self.c = c
        if e is not None:
            self.e = e
        if is_auxiliary is not None:
            self.is_auxiliary = is_auxiliary
        if spawn_node_idx is not None:
            self.spawn_node_idx = spawn_node_idx

    def unpack(self):
        """
        Unpack the data into a tuple of tensors.
        Returns:
            x: The coordinates.
            a: The atom types.
            c: The charge types.
        """
        return self.x, self.a, self.c, self.e, self.edge_index  # , self.is_auxiliary

    @classmethod
    def from_molecule(cls, mol) -> "AugmentedMoleculeData":
        """Creates an AugmentedMoleculeData from your existing MoleculeData/PointCloud."""
        N = mol.x.shape[0]
        device = mol.x.device

        return cls(
            x=mol.x.clone().to(device),
            a=mol.a.clone().to(device),
            c=mol.c.clone().to(device),
            e=mol.e.clone().to(device),
            edge_index=mol.edge_index.clone().to(device),
            is_auxiliary=torch.zeros((N, 1), dtype=torch.bool, device=device),
        )

    def pad(self, num_auxiliary: int) -> "AugmentedMoleculeData":
        """Appends `num_auxiliary` zero-initialized nodes (with is_auxiliary=True)."""
        if num_auxiliary <= 0:
            return self

        device = self.x.device

        # Helper to concatenate padding
        def _pad(tensor, width):
            if tensor is None:
                return None
            pad_block = torch.zeros(
                (num_auxiliary, width), dtype=tensor.dtype, device=device
            )
            return torch.cat([tensor, pad_block], dim=0)

        # Pad Node Attributes
        self.x = _pad(self.x, self.x.shape[1])

        # a and c are (N, ) tensors, so we need to unsqueeze them and pad them to (N, 1)
        self.a = self.a.unsqueeze(1)
        self.a = _pad(self.a, 1).squeeze(-1)

        self.c = self.c.unsqueeze(1)
        self.c = _pad(self.c, 1).squeeze(-1)

        # Pad Mask
        dummy_mask = torch.ones((num_auxiliary, 1), dtype=torch.bool, device=device)
        self.is_auxiliary = torch.cat([self.is_auxiliary, dummy_mask], dim=0)

        return self

    def permute_nodes(
        self, indices: np.ndarray | torch.Tensor
    ) -> "AugmentedMoleculeData":
        """Permutes nodes and updates edge connectivity to match new indices."""
        device = self.x.device
        N_total = self.x.shape[0]

        # 1. Update Topology
        # We map the VALUES in edge_index to the new node positions.
        # The edges themselves (and their features `e`) do not change order.
        if self.edge_index is not None and self.edge_index.numel() > 0:
            # Create a map: old_index -> new_index
            mapping = torch.full((N_total,), -1, dtype=torch.long, device=device)
            perm_tensor = torch.as_tensor(indices, device=device, dtype=torch.long)
            mapping[perm_tensor] = torch.arange(len(indices), device=device)

            # Apply map to edge_index
            self.edge_index = mapping[self.edge_index]

        # 2. Permute Node Tensors
        self.x = self.x[indices]
        self.a = self.a[indices]
        self.is_auxiliary = self.is_auxiliary[indices]

        if hasattr(self, "c") and self.c is not None:
            self.c = self.c[indices]

        return self


def filter_nodes(
    mol: Data | Batch | AugmentedMoleculeData, mask: torch.Tensor
) -> AugmentedMoleculeData:
    """Keeps only nodes where mask is True. Removes edges connected to dropped nodes."""

    device = mol.x.device

    # Fail-safe: Ensure at least one atom remains
    mask = mask.clone()  # Work on a copy to avoid modifying the original

    # 1. Update Topology
    # Check for both edge_index AND e to prevent crashes
    has_edges = mol.edge_index is not None and mol.edge_index.numel() > 0

    if has_edges:
        # Map: current_index -> new_compact_index (or -1 if dropped)
        indices_kept = torch.nonzero(mask.squeeze(), as_tuple=True)[0]

        # Safe initialization
        mapping = torch.full((mol.x.shape[0],), -1, dtype=torch.long, device=device)
        mapping[indices_kept] = torch.arange(len(indices_kept), device=device)

        # Remap edges
        new_edge_index = mapping[mol.edge_index]

        # Identify valid edges (both endpoints must exist in the new set)
        valid_edge_mask = (new_edge_index[0] >= 0) & (new_edge_index[1] >= 0)

        # Slice edge_index
        edge_index = new_edge_index[:, valid_edge_mask]

        # Slice edge attributes safely
        e = None
        if hasattr(mol, "e") and mol.e is not None:
            e = mol.e[valid_edge_mask]

    else:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        e = (
            torch.empty((0,), dtype=torch.long, device=device)
            if hasattr(mol, "e")
            else None
        )

    # 2. Slice Node Tensors
    x = mol.x[mask]
    a = mol.a[mask]
    c = mol.c[mask]

    # CRITICAL: Slice the batch vector if it exists
    batch = None
    if hasattr(mol, "batch") and mol.batch is not None:
        batch = mol.batch[mask]

    if isinstance(mol, MoleculeBatch):
        return MoleculeBatch(
            x=x,
            a=a,
            c=c,
            e=e,
            edge_index=edge_index,
            batch=batch,
        )

    elif isinstance(mol, AugmentedMoleculeData):
        return AugmentedMoleculeData(
            x=x,
            a=a,
            c=c,
            e=e,
            edge_index=edge_index,
            is_auxiliary=mol.is_auxiliary[mask],
        )
    else:
        return MoleculeData(
            x=x,
            a=a,
            c=c,
            e=e,
            edge_index=edge_index,
        )


def sort_nodes_by_batch(data):
    """
    Sorts nodes in a Data/Batch object so that all nodes belonging to
    batch 0 come first, then batch 1, etc. Remaps edge indices accordingly.
    """
    if not hasattr(data, "batch") or data.batch is None:
        return data

    # 1. Get the sorting permutation
    # stable=True keeps original order of nodes within the same batch
    perm = data.batch.argsort(stable=True)

    # 2. Reorder node features
    num_nodes = perm.size(0)

    if hasattr(data, "x") and data.x is not None:
        data.x = data.x[perm]
    if hasattr(data, "a") and data.a is not None:
        data.a = data.a[perm]
    if hasattr(data, "c") and data.c is not None:
        data.c = data.c[perm]

    # Apply to batch last so we still have the original for reference if needed
    data.batch = data.batch[perm]

    # 3. Remap edge_index
    if hasattr(data, "edge_index") and data.edge_index is not None:
        # Create the mapping: map[old_position] = new_position
        perm_map = torch.empty(num_nodes, dtype=torch.long, device=data.x.device)

        # Ensure arange matches perm size exactly
        perm_map[perm] = torch.arange(num_nodes, device=data.x.device)

        # Apply mapping
        data.edge_index = perm_map[data.edge_index]

        # 4. Sort edges (Optional but recommended for PyG)
        # Safely handle edge attributes
        edge_attr = data.e if hasattr(data, "e") else None

        # sort_edge_index returns (edge_index, edge_attr)
        data.edge_index, sorted_edge_attr = sort_edge_index(
            data.edge_index, edge_attr, num_nodes=num_nodes
        )

        # Only assign back if we had attributes to start with
        if edge_attr is not None:
            data.e = sorted_edge_attr

    return data


def join_molecules_with_atoms(
    mol: MoleculeData | MoleculeBatch | AugmentedMoleculeData,
    atoms: PointCloud,
    edge_dist: torch.Tensor,
) -> MoleculeData | MoleculeBatch | AugmentedMoleculeData:
    """
    Joins a molecule with atoms. Adds fully connected edges between the molecule and the atoms.
    The new edges are sampled from the edge distribution.
    """
    assert hasattr(mol, "batch") and mol.batch is not None, "Molecule must have a batch"
    assert hasattr(atoms, "batch") and atoms.batch is not None, (
        "Atoms must have a batch"
    )

    # get the batch ids
    batch_ids = mol.batch
    batch_ids_atoms = atoms.batch

    # get the number of nodes in the molecule
    num_nodes = mol.x.shape[0]

    device = mol.x.device
    edge_aligner = EdgeAligner()

    # 1. Concatenate node features
    new_x = torch.cat([mol.x, atoms.x], dim=0)
    new_a = torch.cat([mol.a, atoms.a], dim=0)
    new_c = torch.cat([mol.c, atoms.c], dim=0)
    new_batch = torch.cat([batch_ids, batch_ids_atoms], dim=0)

    # 2. Get existing edges from molecule (keep as is)
    if mol.edge_index is not None:
        existing_edge_index = mol.edge_index.clone()
    else:
        existing_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
    if mol.e is not None:
        existing_e = mol.e.clone()
    else:
        existing_e = torch.empty((0,), dtype=torch.long, device=device)

    # 3. Create new edges: fully connected between mol nodes and atom nodes,
    # and between atom nodes. We create only upper triangle edges, then symmetrize.
    # We need to do this per batch.
    new_edges_triu_list = []
    new_e_triu_list = []

    # Sample edge types from distribution
    edge_dist_categorical = Categorical(probs=edge_dist)

    # IMPORTANT: Iterate over ALL unique batch IDs from both mol and atoms
    # This handles cases where all nodes of a molecule were deleted but new atoms inserted
    all_batch_ids = torch.cat([batch_ids, batch_ids_atoms]).unique()

    for batch_id in all_batch_ids:
        # Get indices for this batch
        mol_mask = batch_ids == batch_id
        atom_mask = batch_ids_atoms == batch_id

        mol_indices = torch.where(mol_mask)[0]
        # Offset by num_nodes to account for mol nodes
        atom_indices = torch.where(atom_mask)[0] + num_nodes

        # Create edges between mol nodes and atom nodes (upper triangle only)
        # Since mol_indices < atom_indices (mol nodes come first), all mol->atom edges
        # are in upper triangle, but we filter to be explicit
        if len(mol_indices) > 0 and len(atom_indices) > 0:
            mol_atom_edges = torch.cartesian_prod(mol_indices, atom_indices)
            mol_atom_edges = mol_atom_edges.T  # Shape: (2, N)
            # Keep only upper triangle (row < col)
            triu_mask = mol_atom_edges[0] < mol_atom_edges[1]
            mol_atom_edges_triu = mol_atom_edges[:, triu_mask]

            if mol_atom_edges_triu.shape[1] > 0:
                new_edges_triu_list.append(mol_atom_edges_triu)
                # Sample edge types for mol-atom edges
                n_mol_atom_edges = mol_atom_edges_triu.shape[1]
                mol_atom_e = edge_dist_categorical.sample((n_mol_atom_edges,)).to(
                    device
                )
                new_e_triu_list.append(mol_atom_e)

        # Create edges between atom nodes (upper triangle only, no self-loops)
        if len(atom_indices) > 1:
            atom_atom_edges = torch.cartesian_prod(atom_indices, atom_indices)
            atom_atom_edges = atom_atom_edges.T  # Shape: (2, N)
            # Keep only upper triangle (row < col, which also removes self-loops)
            triu_mask = atom_atom_edges[0] < atom_atom_edges[1]
            atom_atom_edges_triu = atom_atom_edges[:, triu_mask]

            if atom_atom_edges_triu.shape[1] > 0:
                new_edges_triu_list.append(atom_atom_edges_triu)
                # Sample edge types for atom-atom edges
                n_atom_atom_edges = atom_atom_edges_triu.shape[1]
                atom_atom_e = edge_dist_categorical.sample((n_atom_atom_edges,)).to(
                    device
                )
                new_e_triu_list.append(atom_atom_e)

    # 4. Combine all triu edges and symmetrize
    if len(new_edges_triu_list) > 0:
        new_edge_index_triu = torch.cat(new_edges_triu_list, dim=1)
        new_e_triu = torch.cat(new_e_triu_list, dim=0)

        # Symmetrize the upper triangle edges to get full symmetric representation
        new_edge_index, new_e_attrs = edge_aligner.symmetrize_edges(
            new_edge_index_triu, [new_e_triu]
        )
        new_e = new_e_attrs[0]

        # Combine with existing edges
        combined_edge_index = torch.cat([existing_edge_index, new_edge_index], dim=1)
        combined_e = torch.cat([existing_e, new_e], dim=0)
    else:
        combined_edge_index = existing_edge_index
        combined_e = existing_e

    # 5. Create the result object
    # Determine return type based on input type
    result = MoleculeBatch(
        x=new_x,
        a=new_a,
        c=new_c,
        e=combined_e,
        edge_index=combined_edge_index,
        batch=new_batch,
    )

    result = sort_nodes_by_batch(result)

    return result


def join_molecules_with_predicted_edges(
    mol: MoleculeBatch,
    new_atoms: PointCloud,
    e_ins: torch.Tensor,
    spawn_node_idx: torch.Tensor,
    target_node_idx: torch.Tensor,
    fallback_edge_dist: torch.Tensor,
    e_ins_to_ins: torch.Tensor | None = None,
    ins_to_ins_src_idx: torch.Tensor | None = None,
    ins_to_ins_dst_idx: torch.Tensor | None = None,
) -> MoleculeBatch:
    """Join existing molecule with newly inserted atoms.

    Args:
        mol: Existing molecule batch.
        new_atoms: New atoms to insert (with .batch attribute).
        e_ins: Predicted edge types for (new_atom -> existing) pairs.
        spawn_node_idx: Local new-atom indices (0..num_new-1) for e_ins source.
        target_node_idx: Existing-node indices (in mol) for e_ins destination.
        fallback_edge_dist: Categorical prior for ins->ins edges when
            e_ins_to_ins is not provided.
        e_ins_to_ins: Predicted edge types for (new_atom_i -> new_atom_j)
            upper-triangular pairs.  If None, falls back to random sampling.
        ins_to_ins_src_idx: Local new-atom index for source of each ii-edge.
        ins_to_ins_dst_idx: Local new-atom index for destination of each ii-edge.
    """

    device = mol.x.device
    batch_ids = mol.batch
    batch_ids_atoms = new_atoms.batch
    num_existing_nodes = mol.x.shape[0]
    edge_aligner = EdgeAligner()

    # 1. Concatenate node features
    new_x = torch.cat([mol.x, new_atoms.x], dim=0)
    new_a = torch.cat([mol.a, new_atoms.a], dim=0)
    new_c = torch.cat([mol.c, new_atoms.c], dim=0)
    new_batch = torch.cat([batch_ids, batch_ids_atoms], dim=0)

    # 2. Keep existing edges
    if mol.edge_index is not None and mol.edge_index.numel() > 0:
        existing_edge_index = mol.edge_index.clone()
        existing_e = mol.e.clone()
    else:
        existing_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        existing_e = torch.empty((0,), dtype=torch.long, device=device)

    # 3. spawn_node_idx is expected to index into new_atoms (local new-atom ids).
    # We map to global ids by offsetting with num_existing_nodes.

    # 4. Create NEW predicted edges (Upper Triangular): new_atom -> existing
    new_edges_triu_list = []
    new_e_triu_list = []

    if e_ins is not None and e_ins.numel() > 0:
        edge_types = e_ins.to(dtype=torch.long, device=device)

        # Map the local spawn indices to the global graph space
        global_new_atom_idx = spawn_node_idx + num_existing_nodes

        # Enforce upper triangular convention (target_idx < global_new_atom_idx is always True)
        row = target_node_idx
        col = global_new_atom_idx

        # Stack into a (2, E) tensor
        predicted_edges_triu = torch.stack([row, col], dim=0)

        new_edges_triu_list.append(predicted_edges_triu)
        new_e_triu_list.append(edge_types)

    # 5. Create edges between new atoms
    use_predicted_ii = (
        e_ins_to_ins is not None
        and e_ins_to_ins.numel() > 0
        and ins_to_ins_src_idx is not None
        and ins_to_ins_dst_idx is not None
    )

    if use_predicted_ii:
        # Use model-predicted ins->ins edge types (upper triangular, src < dst by construction)
        global_src = ins_to_ins_src_idx + num_existing_nodes
        global_dst = ins_to_ins_dst_idx + num_existing_nodes
        # Ensure triu ordering (src < dst in global space; src_local < dst_local was
        # already enforced during interpolation, so this holds after the offset too)
        ii_edge_index = torch.stack([global_src, global_dst], dim=0)
        new_edges_triu_list.append(ii_edge_index)
        new_e_triu_list.append(e_ins_to_ins.to(dtype=torch.long, device=device))
    else:
        # Fallback: sample ins->ins edge types from the prior distribution
        edge_dist_cat = Categorical(probs=fallback_edge_dist)
        for batch_id in batch_ids_atoms.unique():
            atom_mask_local = batch_ids_atoms == batch_id
            atom_indices_global = torch.where(atom_mask_local)[0] + num_existing_nodes

            if len(atom_indices_global) > 1:
                # Transpose combinations to yield shape (2, N_pairs) to match Step 4
                pairs = torch.combinations(atom_indices_global, r=2).T
                new_edges_triu_list.append(pairs)

                n_pairs = pairs.shape[1]
                sampled_types = edge_dist_cat.sample((n_pairs,)).to(device)
                new_e_triu_list.append(sampled_types)

    # 6. Symmetrize and Combine
    if len(new_edges_triu_list) > 0:
        # All items are guaranteed to be tensors now. Direct concatenation.
        new_edge_index_triu = torch.cat(
            new_edges_triu_list, dim=1
        )  # Shape: (2, E_total)
        new_e_triu = torch.cat(new_e_triu_list, dim=0)  # Shape: (E_total,)

        # Symmetrize
        new_edge_index, new_e_attrs = edge_aligner.symmetrize_edges(
            new_edge_index_triu, [new_e_triu]
        )
        new_e = new_e_attrs[0]

        combined_edge_index = torch.cat([existing_edge_index, new_edge_index], dim=1)
        combined_e = torch.cat([existing_e, new_e], dim=0)

        # Sort
        combined_edge_index, combined_e = sort_edge_index(
            combined_edge_index, combined_e
        )
    else:
        combined_edge_index = existing_edge_index
        combined_e = existing_e

    # 7. Result
    result = MoleculeBatch(
        x=new_x,
        a=new_a,
        c=new_c,
        e=combined_e,
        edge_index=combined_edge_index,
        batch=new_batch,
    )

    # Ensure this function correctly updates edge_index permutations!
    result = sort_nodes_by_batch(result)

    return result
