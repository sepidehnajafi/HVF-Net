"""Learnable hypergraph convolution layers (Equation 9).

    X^{(l+1)} = sigma( D_v^{-1/2} H W D_e^{-1} H^T D_v^{-1/2} X^{(l)} Theta^{(l)} )

The propagation matrix ``D_v^{-1/2} H W D_e^{-1} H^T D_v^{-1/2}`` is fixed by
the electrode montage (see ``src/data/hypergraph.py``) and precomputed once;
only Theta^{(l)} is learned.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HypergraphConvLayer(nn.Module):
    """A single hypergraph convolution layer operating on node features.

    Args:
        in_dim: input node feature dimension.
        out_dim: output node feature dimension.
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.theta = nn.Linear(in_dim, out_dim, bias=True)

    def forward(self, x: torch.Tensor, propagation: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: node features, shape (batch, num_nodes, in_dim).
            propagation: precomputed
                ``D_v^{-1/2} H W D_e^{-1} H^T D_v^{-1/2}`` matrix, shape
                (num_nodes, num_nodes). Shared across the batch since it
                depends only on the fixed electrode montage.

        Returns:
            Updated node features, shape (batch, num_nodes, out_dim).
        """
        # (batch, num_nodes, in_dim) -> aggregate over the hypergraph
        aggregated = torch.einsum("nm,bmf->bnf", propagation, x)
        return F.relu(self.theta(aggregated))


class HypergraphEncoder(nn.Module):
    """Stack of :class:`HypergraphConvLayer` (Table 2: 2 layers by default).

    Maps raw per-electrode connectivity summaries to a ``node_embedding_dim``
    representation, which is subsequently volumetrized and passed to the 3D
    CNN backbone (Section 3.7.1-3.7.2).
    """

    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int = 2):
        super().__init__()
        dims = [in_dim] + [hidden_dim] * num_layers
        self.layers = nn.ModuleList(
            [HypergraphConvLayer(dims[i], dims[i + 1]) for i in range(num_layers)]
        )

    def forward(self, x: torch.Tensor, propagation: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, propagation)
        return x
