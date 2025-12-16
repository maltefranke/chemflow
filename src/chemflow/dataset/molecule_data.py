from torch_geometric.data import Data, Batch, HeteroData
import torch
from torch_geometric.utils import to_dense_adj
from external_code.egnn import unsorted_segment_mean


from torch_geometric.utils import to_dense_batch, dense_to_sparse
from torch.distributions import Categorical


class PointCloud(Data):
    """
    A generic point cloud data object.
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
    A generic molecule data object.

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
        e_triu = dense[torch.triu_indices(self.num_nodes, self.num_nodes, offset=1)]
        return e_triu

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
            edge_mask = node_mask[self.edge_index[0]]

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


def join_molecule_with_atoms(
    mol: MoleculeBatch, atoms: PointCloud, edge_distribution: torch.Tensor
) -> MoleculeBatch:
    """
    Merges a batch of molecules with a batch of atoms and samples new edges
    for the atoms based on the provided distribution.

    Args:
        mol: Batch of existing molecules.
        atoms: Batch of new atoms to add.
        edge_distribution: 1D Tensor of probs [P(No_Edge), P(Type_1), P(Type_2)...].
                           Index 0 is assumed to be 'No Edge'.
    """
    raise NotImplementedError("This function must be debugged.")
    assert mol.batch is not None
    assert atoms.batch is not None
    device = mol.x.device

    # --- 1. Convert to Dense Representations ---
    # Convert node features to dense [Batch, Max_Nodes, Features]
    # mask_mol is [Batch, Max_Mol_Nodes] (True = Real Node, False = Padding)
    x_mol, mask_mol = to_dense_batch(mol.x, mol.batch)
    a_mol, _ = to_dense_batch(mol.a, mol.batch)
    c_mol, _ = to_dense_batch(mol.c, mol.batch)

    # Do the same for the new atoms
    x_atom, mask_atom = to_dense_batch(atoms.x, atoms.batch)
    a_atom, _ = to_dense_batch(atoms.a, atoms.batch)
    c_atom, _ = to_dense_batch(atoms.c, atoms.batch)

    # Convert edge attributes to dense [Batch, Max_Mol_Nodes, Max_Mol_Nodes]
    # Note: If mol.e is None, to_dense_adj creates a binary adjacency.
    # We assume mol.e contains integer edge types (or 1s).
    adj_mol = to_dense_adj(mol.edge_index, mol.batch, edge_attr=mol.e)

    # Handle case where adj might have an extra last dimension if edge_attr was [E, 1]
    if adj_mol.dim() > 3:
        adj_mol = adj_mol.squeeze(-1)

    # --- 2. Combine Dense Tensors ---
    # Concatenate along the node dimension (dim 1)
    # The new layout for each graph is: [Old_Nodes, New_Atoms, Padding]
    x_combined = torch.cat([x_mol, x_atom], dim=1)
    a_combined = torch.cat([a_mol, a_atom], dim=1)
    c_combined = torch.cat([c_mol, c_atom], dim=1)
    mask_combined = torch.cat([mask_mol, mask_atom], dim=1)

    batch_size, n_mol_max = x_mol.shape[:2]
    _, n_atom_max = x_atom.shape[:2]
    n_total = n_mol_max + n_atom_max

    # --- 3. Construct and Sample New Adjacency ---
    # Initialize full adjacency with zeros (No Edge)
    adj_new = torch.zeros(
        (batch_size, n_total, n_total), device=device, dtype=adj_mol.dtype
    )

    # A. Copy existing edges (Top-Left block)
    adj_new[:, :n_mol_max, :n_mol_max] = adj_mol

    # B. Sample New Edges
    # We need to sample for the interaction blocks:
    # 1. Mol <-> Atom (Top-Right)
    # 2. Atom <-> Atom (Bottom-Right)
    # 3. Atom <-> Mol (Bottom-Left) -- Symmetric to Top-Right

    # Create a sampler based on the provided distribution
    # We sample indices [0, 1, 2...] where 0 is 'No Edge'
    sampler = Categorical(probs=edge_distribution)

    # Sample a full matrix of random edge types
    # Shape: [Batch, Total, Total]
    random_edges = sampler.sample((batch_size, n_total, n_total)).to(adj_mol.dtype)

    # Create a mask for where we are ALLOWED to add new edges
    # We only want to touch the blocks involving the new atoms
    sampling_mask = torch.zeros(
        (batch_size, n_total, n_total), device=device, dtype=torch.bool
    )

    # Enable sampling for Mol-Atom (Top-Right)
    sampling_mask[:, :n_mol_max, n_mol_max:] = True
    # Enable sampling for Atom-Atom (Bottom-Right)
    sampling_mask[:, n_mol_max:, n_mol_max:] = True

    # Enforce symmetry? (Molecules are undirected)
    # If undirected, we only sample Upper Triangle and copy to Lower
    upper_tri_mask = torch.triu(
        torch.ones((n_total, n_total), device=device, dtype=torch.bool), diagonal=1
    )
    final_sampling_mask = sampling_mask & upper_tri_mask.unsqueeze(0)

    # Apply sampled values to the allowed regions
    # Where mask is True, use random_edges. Where False, keep existing adj_new (which has old edges + zeros)
    adj_new = torch.where(final_sampling_mask, random_edges, adj_new)

    # Symmetrize: Copy Upper Triangle to Lower Triangle
    # adj_new = adj_new + adj_new.transpose(1, 2)
    # WARNING: Simple addition doubles the diagonal and fails for categorical types.
    # Correct way for categorical adjacency symmetrization:
    adj_new_t = adj_new.transpose(1, 2)
    adj_new = torch.max(
        adj_new, adj_new_t
    )  # Max works if 0 is No Edge and >0 are types

    # --- 4. Convert Back to Sparse (MoleculeData) ---
    # dense_to_sparse filters out elements that are 0 (No Edge)
    # It returns edge_index and the values (edge_attr)
    edge_index, edge_attr = dense_to_sparse(adj_new, mask=mask_combined)

    # Extract the flattened node features using the mask
    # The mask ensures we drop the padding nodes
    new_x = x_combined[mask_combined]
    new_a = a_combined[mask_combined]
    new_c = c_combined[mask_combined]

    # Reconstruct the batch vector
    # We count how many real nodes are in each graph to recreate the batch indices
    nodes_per_graph = mask_combined.sum(dim=1)
    new_batch = torch.repeat_interleave(
        torch.arange(batch_size, device=device), nodes_per_graph
    )

    return MoleculeData(
        x=new_x, a=new_a, e=edge_attr, edge_index=edge_index, c=new_c, batch=new_batch
    )


class ConditioningData(Data):
    """
    A generic conditioning data object.

    Attributes:
        x: The conditioning embeddings.
    """

    def __init__(self, x: torch.Tensor | None = None, **kwargs):
        # Allow PyTorch Geometric's separate() to create empty instances
        # It will populate attributes via stores_as() mechanism
        super().__init__(x=x, **kwargs)


class EditData(HeteroData):
    """
    A tuple of input molecule M0, edit instructions E, and output molecule M1.
    M0 and M1 are MoleculeData objects and E is a TextData object.

    Must have m1 (the target molecule) for unconditional or conditional generation.
    Can have m0 (input molecule) and e (conditioning data) for conditional generation.
    """

    def __init__(
        self,
        m1: MoleculeData,
        m0: MoleculeData | None = None,
        e: ConditioningData | None = None,
    ):
        super().__init__()
        self.m0 = m0
        self.e = e
        self.m1 = m1
