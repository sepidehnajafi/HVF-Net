"""Bidirectional Cross-Modal Interaction (BCMI) module (Equations 23a-23b).

Two gated cross-attention pathways let the connectivity embedding f_c and
spectral embedding f_s refine each other:

  C -> S:  f_s' = f_s + g_{c->s} * CrossAttn_{c->s}(f_c, f_s)
  S -> C:  f_c' = f_c + g_{s->c} * CrossAttn_{s->c}(f_s, f_c)

with a learnable sigmoid gate g in each direction, computed from the
concatenation of both embeddings. This is the most novel component of the
architecture: it lets each modality selectively amplify or suppress the
other's contribution per-sample, rather than using a fixed fusion weight.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class CrossModalPathway(nn.Module):
    """One directional gated cross-attention pathway (query modality attends
    to key/value modality).

    Args:
        query_dim: embedding dimension of the querying modality.
        kv_dim: embedding dimension of the key/value (attended-to) modality.
        d_k: attention key/query projection dimension.
    """

    def __init__(self, query_dim: int, kv_dim: int, d_k: int = 16):
        super().__init__()
        self.d_k = d_k
        self.w_q = nn.Linear(query_dim, d_k, bias=False)
        self.w_k = nn.Linear(kv_dim, d_k, bias=False)
        self.w_v = nn.Linear(kv_dim, kv_dim, bias=False)
        self.gate = nn.Linear(query_dim + kv_dim, kv_dim)

    def forward(self, query_emb: torch.Tensor, kv_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query_emb: (batch, query_dim) embedding providing the query
                (e.g. f_c when computing the C->S pathway).
            kv_emb: (batch, kv_dim) embedding being attended to and updated
                (e.g. f_s when computing the C->S pathway).

        Returns:
            Updated ``kv_emb``, shape (batch, kv_dim).
        """
        q = self.w_q(query_emb)   # (batch, d_k)
        k = self.w_k(kv_emb)      # (batch, d_k)
        v = self.w_v(kv_emb)      # (batch, kv_dim)

        # Single-vector (pooled) attention score per sample: scaled dot
        # product reduces to a scalar gate on the value projection.
        score = (q * k).sum(dim=-1, keepdim=True) / math.sqrt(self.d_k)
        attn = torch.sigmoid(score)  # (batch, 1); degenerate softmax over 1 key
        cross_attn_out = attn * v    # (batch, kv_dim)

        gate = torch.sigmoid(self.gate(torch.cat([query_emb, kv_emb], dim=-1)))
        return kv_emb + gate * cross_attn_out


class BCMIModule(nn.Module):
    """Full bidirectional module producing refined (f_c', f_s') embeddings."""

    def __init__(self, connectivity_dim: int = 128, spectral_dim: int = 64, d_k: int = 16):
        super().__init__()
        self.c_to_s = CrossModalPathway(query_dim=connectivity_dim, kv_dim=spectral_dim, d_k=d_k)
        self.s_to_c = CrossModalPathway(query_dim=spectral_dim, kv_dim=connectivity_dim, d_k=d_k)

    def forward(self, f_c: torch.Tensor, f_s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            f_c: (batch, connectivity_dim) connectivity embedding.
            f_s: (batch, spectral_dim) spectral embedding.

        Returns:
            (f_c', f_s'), each with the same shape as the corresponding
            input.
        """
        f_s_updated = self.c_to_s(query_emb=f_c, kv_emb=f_s)
        f_c_updated = self.s_to_c(query_emb=f_s, kv_emb=f_c)
        return f_c_updated, f_s_updated
