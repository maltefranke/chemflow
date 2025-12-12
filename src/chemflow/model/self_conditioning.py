"""Adapted from FlowMol3"""

import torch
import torch.nn as nn


def _norm_no_nan(x, axis=-1, keepdims=False, eps=1e-8, sqrt=True):
    """
    L2 norm of tensor clamped above a minimum value `eps`.

    :param sqrt: if `False`, returns the square of the L2 norm
    """
    out = torch.clamp(torch.sum(torch.square(x), axis, keepdims), min=eps)
    return torch.sqrt(out) if sqrt else out


def _rbf(D, D_min=0.0, D_max=20.0, D_count=16):
    """
    From https://github.com/jingraham/neurips19-graph-protein-design

    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have
    shape [...dims, D_count].
    """
    device = D.device
    D_mu = torch.linspace(D_min, D_max, D_count, device=device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    RBF = torch.exp(-(((D_expand - D_mu) / D_sigma) ** 2))
    return RBF


def rbf_twoscale(
    D, D_min=0.0, D_max=10, D_count=32, dividing_point: float = 3.5, high_res_frac=0.6
):
    device = D.device

    n_highres_points = int(D_count * high_res_frac)
    n_lowres_points = D_count - n_highres_points

    D_sigma_highres = (dividing_point - D_min) / n_highres_points
    D_sigma_lowres = (D_max - dividing_point) / n_lowres_points
    sigmas = [D_sigma_highres, D_sigma_lowres]

    sections = [
        torch.linspace(D_min, dividing_point, n_highres_points, device=device),
        torch.linspace(dividing_point, D_max, n_lowres_points, device=device)[1:],
    ]
    rbf_embeddings = []
    for D_mu, D_sigma in zip(sections, sigmas):
        D_mu = D_mu.view([1, -1])
        D_expand = torch.unsqueeze(D, -1)
        RBF_i = torch.exp(-(((D_expand - D_mu) / D_sigma) ** 2))
        rbf_embeddings.append(RBF_i)

    return torch.cat(rbf_embeddings, dim=-1)


class SelfConditioningResidualLayer(nn.Module):
    """
    Self-conditioning residual layer for flow matching models.

    This layer takes the current state and a predicted endpoint (from previous
    forward pass) and computes residual updates to node and edge features.
    """

    def __init__(
        self,
        n_atom_types: int,
        n_bond_types: int,
        node_embedding_dim: int,
        edge_embedding_dim: int,
        rbf_dim: int = 16,
        rbf_dmax: float = 20,
    ):
        """
        Args:
            n_atom_types: Number of atom type classes
            n_charges: Number of charge classes (set to 1 if not used)
            n_bond_types: Number of bond type classes
            node_embedding_dim: Dimension of node embeddings
            edge_embedding_dim: Dimension of edge embeddings
            rbf_dim: Dimension of RBF embeddings for distances
            rbf_dmax: Maximum distance for RBF
        """
        super().__init__()

        self.rbf_dim = rbf_dim
        self.rbf_dmax = rbf_dmax

        # Node residual MLP
        # Input: node_embedding + atom_types_onehot + charges_onehot + rbf_distance
        self.node_residual_mlp = nn.Sequential(
            nn.Linear(
                node_embedding_dim + n_atom_types + rbf_dim,
                node_embedding_dim,
            ),
            nn.SiLU(),
            nn.Linear(node_embedding_dim, node_embedding_dim),
            nn.SiLU(),
        )

        # Edge residual MLP
        # Input: edge_embedding + bond_types_onehot + rbf_distance_change
        self.edge_residual_mlp = nn.Sequential(
            nn.Linear(edge_embedding_dim + n_bond_types + rbf_dim, edge_embedding_dim),
            nn.SiLU(),
            nn.Linear(edge_embedding_dim, edge_embedding_dim),
            nn.SiLU(),
        )

    def compute_edge_distances(
        self,
        edge_index: torch.Tensor,
        node_positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute pairwise distances for edges in PyTorch Geometric format.

        Args:
            edge_index: Edge connectivity of shape (2, E) where E is number of edges
            node_positions: Node positions of shape (N, D) where N is number of nodes

        Returns:
            RBF embeddings of edge distances of shape (E, rbf_dim)
        """
        row, col = edge_index[0], edge_index[1]

        # Compute edge vectors
        edge_vec = node_positions[row] - node_positions[col]

        # Compute distances
        edge_dist = _norm_no_nan(edge_vec, keepdims=True) + 1e-8

        # Convert to RBF embeddings
        rbf_embeddings = _rbf(
            edge_dist.squeeze(1), D_max=self.rbf_dmax, D_count=self.rbf_dim
        )

        return rbf_embeddings

    def forward(
        self,
        h: torch.Tensor,
        coord: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        prev_outs: dict,
        atom_types: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through self-conditioning residual layer.

        Args:
            h: Node embeddings of shape (N, node_embedding_dim)
            coord: Node coordinates of shape (N, 3)
            edge_index: Edge connectivity of shape (2, E)
            edge_attr: Edge attributes of shape (E, edge_embedding_dim)
            prev_outs: Dictionary from previous forward pass containing:
                - "pos_head": predicted final positions (N, 3)
                - "class_head": predicted atom types (N, n_atom_types) or indices
                - "edge_type_head": predicted edge types (E, n_bond_types) or indices
            atom_types: Current atom type indices of shape (N,)
            charges: Current charge indices of shape (N,) - optional, defaults to zeros

        Returns:
            Tuple of (h_out, coord_out, edge_attr_out)
        """
        # Extract predicted endpoint from previous outputs
        x_pred = prev_outs.get("pos_head", coord)  # Predicted final positions

        # Get predicted atom types (convert from logits if needed)
        a_pred = prev_outs.get("class_head")
        if a_pred is not None:
            if a_pred.dim() > 1:
                # If it's logits, convert to one-hot
                a_pred_onehot = torch.softmax(a_pred, dim=-1)
            else:
                # If it's indices, convert to one-hot
                n_atom_types = a_pred.max().item() + 1
                a_pred_onehot = torch.nn.functional.one_hot(
                    a_pred, num_classes=n_atom_types
                )
        else:
            # Fallback: use current atom types if available
            if atom_types is not None:
                n_atom_types = atom_types.max().item() + 1
                a_pred_onehot = torch.nn.functional.one_hot(
                    atom_types, num_classes=n_atom_types
                )
            else:
                # No atom type info available, use zeros
                n_atom_types = (
                    self.node_residual_mlp[0].in_features - h.shape[-1] - self.rbf_dim
                )
                if n_atom_types > 0:
                    a_pred_onehot = torch.zeros(
                        h.shape[0], n_atom_types, device=h.device
                    )
                else:
                    a_pred_onehot = torch.zeros(h.shape[0], 0, device=h.device)

        # Compute node-to-node distances (current position vs predicted final position)
        d_node = _norm_no_nan(coord - x_pred, keepdims=True)  # (N, 1)
        d_node_rbf = _rbf(
            d_node.squeeze(1), D_max=self.rbf_dmax, D_count=self.rbf_dim
        )  # (N, rbf_dim)

        # Concatenate node residual inputs
        node_residual_inputs = [
            h,  # Current node embeddings
            a_pred_onehot,  # Predicted atom types
            d_node_rbf,  # Distance RBF
        ]
        node_residual = self.node_residual_mlp(torch.cat(node_residual_inputs, dim=-1))

        # Compute edge distances at current time and predicted final time
        d_edge_t = self.compute_edge_distances(edge_index, coord)  # (E, rbf_dim)
        d_edge_pred = self.compute_edge_distances(edge_index, x_pred)  # (E, rbf_dim)

        # Compute change in edge length (distance at t=1 - distance at t)
        # This represents how much each edge will stretch/compress
        d_edge_change = d_edge_pred - d_edge_t  # (E, rbf_dim)

        # Get predicted edge types
        e_pred = prev_outs.get("edge_type_head")
        if e_pred is not None:
            if e_pred.dim() > 1:
                # If it's logits, convert to one-hot
                e_pred_onehot = torch.softmax(e_pred, dim=-1)
            else:
                # If it's indices, convert to one-hot
                n_bond_types = e_pred.max().item() + 1
                e_pred_onehot = torch.nn.functional.one_hot(
                    e_pred, num_classes=n_bond_types
                )
        else:
            # No edge type prediction, use zeros
            n_bond_types = (
                self.edge_residual_mlp[0].in_features
                - edge_attr.shape[-1]
                - self.rbf_dim
            )
            if n_bond_types > 0:
                e_pred_onehot = torch.zeros(
                    edge_attr.shape[0], n_bond_types, device=edge_attr.device
                )
            else:
                e_pred_onehot = torch.zeros(
                    edge_attr.shape[0], 0, device=edge_attr.device
                )

        # Concatenate edge residual inputs
        edge_residual_inputs = [
            edge_attr,  # Current edge embeddings
            e_pred_onehot,  # Predicted edge types
            d_edge_change,  # Change in edge length
        ]
        edge_residual = self.edge_residual_mlp(torch.cat(edge_residual_inputs, dim=-1))

        # Apply residuals
        h_out = h + node_residual
        coord_out = coord  # Positions are not modified by this layer
        edge_attr_out = edge_attr + edge_residual

        return h_out, coord_out, edge_attr_out
