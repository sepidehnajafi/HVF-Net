"""Hypergraph construction over the electrode montage.

Implements Section 3.7.2 (Equations 5-8): a K-nearest-neighbor hyperedge is
defined for every electrode based on scalp Euclidean distance, and hyperedge
weights are set to the inverse of the number of unique pairwise combinations
within the hyperedge, so that denser hyperedges are down-weighted relative to
sparser ones.
"""

from __future__ import annotations

from math import comb

import numpy as np


def build_incidence_matrix(
    electrode_coords: np.ndarray,
    k_neighbors: int = 6,
) -> np.ndarray:
    """K-NN hyperedge incidence matrix H.

    Every electrode v defines one hyperedge e_v containing v and its
    ``k_neighbors`` spatially nearest electrodes (Euclidean distance in 3-D
    scalp coordinates).

    Args:
        electrode_coords: array of shape (num_channels, 3).
        k_neighbors: K in the K-NN hyperedge definition.

    Returns:
        Binary incidence matrix H of shape (num_channels, num_channels),
        H[v, e] = 1 if node v belongs to hyperedge e.
    """
    num_channels = electrode_coords.shape[0]
    dist = np.linalg.norm(
        electrode_coords[:, None, :] - electrode_coords[None, :, :], axis=-1
    )
    H = np.zeros((num_channels, num_channels), dtype=np.float32)
    for e in range(num_channels):
        # k_neighbors nearest to electrode e, excluding itself, plus itself.
        nearest = np.argsort(dist[e])[1:k_neighbors + 1]
        H[e, e] = 1.0
        H[nearest, e] = 1.0
    return H


def hyperedge_weights(H: np.ndarray) -> np.ndarray:
    """Hyperedge weight w(e) = 1 / C(|e| + 1, 2), Equation (7).

    Larger hyperedges (more nodes) receive proportionally smaller weight so
    that their contribution to the hypergraph Laplacian is normalized by the
    number of unique pairwise interactions they represent.

    Args:
        H: incidence matrix of shape (num_channels, num_channels).

    Returns:
        Diagonal weight vector of shape (num_channels,).
    """
    edge_sizes = H.sum(axis=0).astype(int)  # |e| for each hyperedge
    weights = np.array([1.0 / comb(int(size) + 1, 2) for size in edge_sizes])
    return weights.astype(np.float32)


def hypergraph_laplacian(H: np.ndarray, w: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normalized hypergraph Laplacian, Equation (8):

        L_H = I - D_v^{-1/2} H W D_e^{-1} H^T D_v^{-1/2}

    Args:
        H: incidence matrix, shape (num_channels, num_channels).
        w: hyperedge weight vector, shape (num_channels,).
        eps: numerical floor to avoid division by zero for isolated nodes.

    Returns:
        L_H, shape (num_channels, num_channels).
    """
    W = np.diag(w)
    d_v = H @ w  # node degree, shape (num_channels,)
    d_e = H.sum(axis=0)  # hyperedge degree, shape (num_channels,)

    D_v_inv_sqrt = np.diag(1.0 / np.sqrt(d_v + eps))
    D_e_inv = np.diag(1.0 / (d_e + eps))

    propagation = D_v_inv_sqrt @ H @ W @ D_e_inv @ H.T @ D_v_inv_sqrt
    return np.eye(H.shape[0], dtype=np.float32) - propagation.astype(np.float32)
