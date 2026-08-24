"""3D convolutional backbone over the volumetric connectivity tensor.

Consumes a (batch, 1, C, C, B) volumetric PLV tensor -- electrode x electrode
x frequency band -- and produces a 128-dimensional connectivity embedding via
four Conv3D-BN-ReLU-Pool blocks followed by global average pooling
(Section 3.7.1, Equations 16-17). Spatial dimensions are halved at each stage
via ceil-mode pooling; the frequency axis is preserved throughout.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv3d(
            in_ch, out_ch, kernel_size=(3, 3, 3), stride=1, padding="same"
        )
        self.bn = nn.BatchNorm3d(out_ch)
        self.act = nn.ReLU(inplace=True)
        # Spatial-only pooling: kernel/stride 2 on the two electrode axes,
        # kernel/stride 1 on the frequency-band axis. ceil_mode=True so that
        # odd spatial sizes (e.g. 15 -> 8) round up rather than truncate.
        self.pool = nn.MaxPool3d(kernel_size=(2, 2, 1), stride=(2, 2, 1), ceil_mode=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.act(self.bn(self.conv(x))))


class CNN3DBackbone(nn.Module):
    """Four-block 3D CNN with channel progression 1->16->32->64->128.

    Args:
        num_channels_electrodes: number of scalp electrodes C (spatial extent
            of both the row and column axes of the volumetric tensor).
        num_bands: number of frequency bands B (preserved, unpooled axis).
        channel_progression: output channels of each of the 4 blocks.
    """

    def __init__(
        self,
        num_bands: int = 5,
        channel_progression: tuple[int, ...] = (16, 32, 64, 128),
    ):
        super().__init__()
        in_channels = [1] + list(channel_progression[:-1])
        self.blocks = nn.ModuleList(
            [
                ConvBlock3D(in_channels[i], channel_progression[i])
                for i in range(len(channel_progression))
            ]
        )
        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)
        self.out_dim = channel_progression[-1]

    def forward(self, volumetric: torch.Tensor) -> torch.Tensor:
        """
        Args:
            volumetric: (batch, 1, C, C, B) connectivity tensor.

        Returns:
            (batch, out_dim) connectivity embedding, out_dim=128 by default.
        """
        x = volumetric
        for block in self.blocks:
            x = block(x)
        x = self.global_avg_pool(x)  # (batch, out_dim, 1, 1, 1)
        return x.flatten(1)
