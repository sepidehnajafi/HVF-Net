"""Neuro-Inspired Spectral Perturbation (NISP) augmentation.

Implements the training-time-only augmentation described in Section 3.7.3:
additive, band- and class-specific Gaussian noise on the log band-power
features, calibrated per fold on training data only (Equation set around
sigma_delta ... sigma_gamma).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class NISPConfig:
    """Per-band noise standard deviations. ``beta`` is class-conditional:
    a larger perturbation is applied to PD trials than HC trials, reflecting
    the well-documented sensitivity of beta-band power to dopaminergic state.
    """

    sigma_delta: float = 0.15
    sigma_theta: float = 0.18
    sigma_alpha: float = 0.25
    sigma_beta_hc: float = 0.22
    sigma_beta_pd: float = 0.28
    sigma_gamma: float = 0.10

    # Canonical band order matching `spectral_features.spectral_feature_matrix`
    band_order: tuple = field(default=("delta", "theta", "alpha", "beta", "gamma"))

    def sigma_vector(self, labels: torch.Tensor) -> torch.Tensor:
        """Per-sample, per-band noise std, shape (batch, num_bands).

        Args:
            labels: binary label tensor, shape (batch,); 1 = PD, 0 = HC.
        """
        batch_size = labels.shape[0]
        base = torch.tensor(
            [self.sigma_delta, self.sigma_theta, self.sigma_alpha, 0.0, self.sigma_gamma]
        ).unsqueeze(0).repeat(batch_size, 1)
        beta_sigma = torch.where(
            labels.bool(),
            torch.full_like(labels, self.sigma_beta_pd, dtype=torch.float32),
            torch.full_like(labels, self.sigma_beta_hc, dtype=torch.float32),
        )
        base[:, 3] = beta_sigma
        return base


class NISPAugmentation(torch.nn.Module):
    """Applies NISP noise to a (batch, channels, num_bands) spectral tensor.

    Active during training only; a no-op in eval mode.
    """

    def __init__(self, cfg: NISPConfig | None = None):
        super().__init__()
        self.cfg = cfg or NISPConfig()

    def forward(self, spectral_features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spectral_features: (batch, channels, num_bands) log band power.
            labels: (batch,) binary PD/HC labels.

        Returns:
            Perturbed spectral features, same shape as input.
        """
        if not self.training:
            return spectral_features

        sigma = self.cfg.sigma_vector(labels).to(spectral_features.device)
        # broadcast over channels: (batch, 1, num_bands)
        sigma = sigma.unsqueeze(1)
        noise = torch.randn_like(spectral_features) * sigma
        return spectral_features + noise
