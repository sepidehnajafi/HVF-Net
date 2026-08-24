"""Spectral stream: per-electrode band-power tokens processed by a
Transformer encoder (Section 3.7.3).

Each electrode is treated as one token, with a 5-dimensional feature vector
(log power in delta/theta/alpha/beta/gamma). Tokens are linearly projected to
``projection_dim``, combined with a learnable positional encoding, and passed
through a standard pre-norm Transformer encoder before global average
pooling to a 64-dimensional spectral embedding.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SpectralTokenizer(nn.Module):
    def __init__(self, token_dim: int, projection_dim: int, num_electrodes: int):
        super().__init__()
        self.proj = nn.Linear(token_dim, projection_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_electrodes, projection_dim) * 0.02)

    def forward(self, spectral: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spectral: (batch, num_electrodes, token_dim) log band-power matrix.

        Returns:
            (batch, num_electrodes, projection_dim) token embeddings.
        """
        return self.proj(spectral) + self.pos_embedding


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 4,
        num_layers: int = 4,
        ffn_dim: int = 1024,
        dropout: float = 0.2,
    ):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.encoder(tokens)


class SpectralStream(nn.Module):
    """Full spectral branch: tokenize -> Transformer -> global average pool
    -> project to 64-D spectral embedding f_s.
    """

    def __init__(
        self,
        num_electrodes: int,
        token_dim: int = 5,
        projection_dim: int = 64,
        d_model: int = 256,
        num_heads: int = 4,
        num_layers: int = 4,
        ffn_dim: int = 1024,
        dropout: float = 0.2,
        spectral_embedding_dim: int = 64,
    ):
        super().__init__()
        self.tokenizer = SpectralTokenizer(token_dim, projection_dim, num_electrodes)
        self.in_proj = nn.Linear(projection_dim, d_model) if projection_dim != d_model else nn.Identity()
        self.transformer = TransformerEncoder(d_model, num_heads, num_layers, ffn_dim, dropout)
        self.out_proj = nn.Linear(d_model, spectral_embedding_dim)

    def forward(self, spectral: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spectral: (batch, num_electrodes, token_dim).

        Returns:
            (batch, spectral_embedding_dim) spectral embedding f_s.
        """
        tokens = self.tokenizer(spectral)
        tokens = self.in_proj(tokens)
        encoded = self.transformer(tokens)      # (batch, num_electrodes, d_model)
        pooled = encoded.mean(dim=1)             # global average pool over electrodes
        return self.out_proj(pooled)
