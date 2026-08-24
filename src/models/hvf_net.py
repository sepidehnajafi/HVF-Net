"""HVF-Net (Hypergraph Volumetric Fusion Network): full model assembly.

Pipeline (Section 3.7, Figure 2 of the paper):

    volumetric PLV tensor (C, C, B)
            |
    HypergraphEncoder (per-band node embeddings) --> volumetrized (C, C, d)
            |
    CNN3DBackbone --> f_c  (128-D connectivity embedding)

    per-electrode log band power (C, B)
            |
    SpectralStream (Transformer) --> f_s  (64-D spectral embedding)

    (f_c, f_s) --> BCMIModule --> (f_c', f_s')
            |
    concat + classification head --> logits
            |
    concat (pre-head) --> contrastive embedding
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .bcmi import BCMIModule
from .cnn3d import CNN3DBackbone
from .hypergraph_conv import HypergraphEncoder
from .spectral_transformer import SpectralStream


class HVFNet(nn.Module):
    """Hypergraph Volumetric Fusion Network.

    Fuses a volumetric hypergraph/3D-CNN connectivity pathway with a
    spectral Transformer pathway via a bidirectional cross-modal
    interaction (BCMI) module, for binary EEG-based classification.
    """

    def __init__(
        self,
        num_electrodes: int = 60,
        num_bands: int = 5,
        hypergraph_hidden_dim: int = 64,
        hypergraph_layers: int = 2,
        cnn_channel_progression: tuple[int, ...] = (16, 32, 64, 128),
        spectral_token_dim: int = 5,
        spectral_projection_dim: int = 64,
        transformer_d_model: int = 256,
        transformer_heads: int = 4,
        transformer_layers: int = 4,
        transformer_ffn_dim: int = 1024,
        transformer_dropout: float = 0.2,
        spectral_embedding_dim: int = 64,
        bcmi_d_k: int = 16,
        classifier_dropout: float = 0.5,
        num_classes: int = 2,
    ):
        super().__init__()

        # --- Connectivity path: hypergraph -> volumetrize -> 3D CNN -------
        self.hypergraph_encoder = HypergraphEncoder(
            in_dim=num_electrodes,  # each electrode's raw PLV row, per band
            hidden_dim=hypergraph_hidden_dim,
            num_layers=hypergraph_layers,
        )
        self.cnn3d = CNN3DBackbone(
            num_bands=num_bands, channel_progression=cnn_channel_progression
        )
        connectivity_dim = cnn_channel_progression[-1]  # 128

        # --- Spectral path --------------------------------------------------
        self.spectral_stream = SpectralStream(
            num_electrodes=num_electrodes,
            token_dim=spectral_token_dim,
            projection_dim=spectral_projection_dim,
            d_model=transformer_d_model,
            num_heads=transformer_heads,
            num_layers=transformer_layers,
            ffn_dim=transformer_ffn_dim,
            dropout=transformer_dropout,
            spectral_embedding_dim=spectral_embedding_dim,
        )

        # --- Cross-modal fusion ---------------------------------------------
        self.bcmi = BCMIModule(
            connectivity_dim=connectivity_dim,
            spectral_dim=spectral_embedding_dim,
            d_k=bcmi_d_k,
        )

        fused_dim = connectivity_dim + spectral_embedding_dim
        self.classifier = nn.Sequential(
            nn.Dropout(classifier_dropout),
            nn.Linear(fused_dim, fused_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(classifier_dropout),
            nn.Linear(fused_dim // 2, num_classes),
        )

    def encode_connectivity(
        self, volumetric_plv: torch.Tensor, propagation: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            volumetric_plv: (batch, C, C, B) raw PLV tensor (rows = nodes).
            propagation: (C, C) fixed hypergraph propagation matrix.

        Returns:
            (batch, 128) connectivity embedding f_c.
        """
        batch, C, _, B = volumetric_plv.shape
        # Treat each band's (C, C) PLV matrix as node features (C nodes,
        # C-dim raw feature) and refine per band via the hypergraph encoder.
        refined_bands = []
        for b in range(B):
            band_matrix = volumetric_plv[:, :, :, b]           # (batch, C, C)
            refined = self.hypergraph_encoder(band_matrix, propagation)  # (batch, C, d)
            refined_bands.append(refined)
        # Re-volumetrize: (batch, C, d, B) -> collapse hidden dim into a
        # second "electrode-like" axis so the tensor stays 3-D spatial x
        # 1-channel for the 3D CNN, matching Section 3.7.1's C x C x B input.
        # Here we instead pool the hypergraph hidden dim back down to a
        # single channel via the raw PLV, i.e. the hypergraph encoder acts
        # as a per-band node re-weighting prior to volumetric convolution.
        node_gate = torch.stack(
            [r.mean(dim=-1) for r in refined_bands], dim=-1
        )  # (batch, C, B) -- learned per-node, per-band importance
        gated_volumetric = volumetric_plv * node_gate.unsqueeze(2)  # broadcast over columns
        gated_volumetric = gated_volumetric.unsqueeze(1)  # (batch, 1, C, C, B)
        return self.cnn3d(gated_volumetric)

    def forward(
        self,
        volumetric_plv: torch.Tensor,
        spectral_features: torch.Tensor,
        propagation: torch.Tensor,
        return_embedding: bool = False,
    ):
        """
        Args:
            volumetric_plv: (batch, C, C, B).
            spectral_features: (batch, C, B) log band-power matrix.
            propagation: (C, C) fixed hypergraph propagation matrix.
            return_embedding: if True, also return the pre-classifier fused
                embedding (used as the contrastive-loss embedding).

        Returns:
            logits (batch, num_classes), and optionally the fused embedding.
        """
        f_c = self.encode_connectivity(volumetric_plv, propagation)
        f_s = self.spectral_stream(spectral_features)
        f_c, f_s = self.bcmi(f_c, f_s)
        fused = torch.cat([f_c, f_s], dim=-1)
        logits = self.classifier(fused)

        if return_embedding:
            return logits, fused
        return logits
