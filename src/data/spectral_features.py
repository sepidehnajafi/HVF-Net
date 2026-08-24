"""Per-electrode log band-power features for the spectral stream.

Produces the (channels, num_bands) matrix consumed by the spectral
Transformer branch (Section 3.7.3): one row per electrode, one column per
canonical frequency band.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import welch


def band_power(
    signal: np.ndarray,
    sampling_rate_hz: int,
    low_hz: float,
    high_hz: float,
) -> np.ndarray:
    """Welch power spectral density integrated over a frequency band.

    Args:
        signal: array of shape (channels, samples).
        sampling_rate_hz: sampling rate.
        low_hz, high_hz: band edges.

    Returns:
        Array of shape (channels,) with band power per channel.
    """
    freqs, psd = welch(signal, fs=sampling_rate_hz, axis=-1, nperseg=min(256, signal.shape[-1]))
    band_mask = (freqs >= low_hz) & (freqs <= high_hz)
    trapezoid = getattr(np, "trapezoid", None) or np.trapz  # numpy >=2.0 renamed trapz
    return trapezoid(psd[..., band_mask], freqs[band_mask], axis=-1)


def spectral_feature_matrix(
    epoch: np.ndarray,
    sampling_rate_hz: int,
    bands: dict[str, tuple[float, float]],
    log_transform: bool = True,
    eps: float = 1e-12,
) -> np.ndarray:
    """Build the (channels, num_bands) log band-power matrix for one epoch.

    Args:
        epoch: array of shape (channels, samples).
        sampling_rate_hz: sampling rate.
        bands: mapping of band name -> (low_hz, high_hz).
        log_transform: apply log10 for numerical stability / near-Gaussian
            distribution, consistent with Section 3.7.3.
        eps: numerical floor before taking the log.

    Returns:
        Array of shape (channels, num_bands).
    """
    columns = []
    for _, (low_hz, high_hz) in bands.items():
        power = band_power(epoch, sampling_rate_hz, low_hz, high_hz)
        columns.append(power)
    matrix = np.stack(columns, axis=-1)  # (channels, num_bands)
    if log_transform:
        matrix = np.log10(matrix + eps)
    return matrix.astype(np.float32)
