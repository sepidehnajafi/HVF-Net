"""Volumetric dynamic functional connectivity via band-limited Phase-Locking
Value (PLV).

Implements Section 3.7.1 of the paper: for each frequency band b and epoch,
the instantaneous phase of every channel is extracted via the analytic
(Hilbert) signal, and pairwise PLV is computed to form a
``(channels, channels, bands)`` volumetric tensor per epoch.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import hilbert

from .preprocessing import bandpass_filter


def instantaneous_phase(band_signal: np.ndarray) -> np.ndarray:
    """Instantaneous phase via the analytic signal.

    Args:
        band_signal: array of shape (channels, time), already band-limited.

    Returns:
        Phase array of shape (channels, time), values in (-pi, pi].
    """
    analytic = hilbert(band_signal, axis=-1)
    return np.angle(analytic)


def plv_matrix(phase: np.ndarray) -> np.ndarray:
    """Pairwise Phase-Locking Value.

    PLV_ik = | (1/T) * sum_t exp(j * (phi_i(t) - phi_k(t))) |

    Args:
        phase: array of shape (channels, time).

    Returns:
        Symmetric PLV matrix of shape (channels, channels), diagonal = 1.
    """
    # complex phase vectors, shape (channels, time)
    z = np.exp(1j * phase)
    num_timepoints = phase.shape[-1]
    # (channels, channels) complex cross-spectrum averaged over time
    cross = z @ z.conj().T / num_timepoints
    return np.abs(cross)


def volumetric_connectivity_tensor(
    epoch: np.ndarray,
    sampling_rate_hz: int,
    bands: dict[str, tuple[float, float]],
) -> np.ndarray:
    """Build the (C, C, B) volumetric dynamic functional connectivity tensor
    for a single epoch.

    Args:
        epoch: array of shape (channels, samples).
        sampling_rate_hz: sampling rate of ``epoch``.
        bands: mapping of band name -> (low_hz, high_hz), evaluated in a
            fixed canonical order (delta, theta, alpha, beta, gamma).

    Returns:
        Array of shape (channels, channels, num_bands).
    """
    num_channels = epoch.shape[0]
    num_bands = len(bands)
    tensor = np.zeros((num_channels, num_channels, num_bands), dtype=np.float32)

    for b_idx, (_, (low_hz, high_hz)) in enumerate(bands.items()):
        band_signal = bandpass_filter(epoch, low_hz, high_hz, sampling_rate_hz)
        phase = instantaneous_phase(band_signal)
        tensor[:, :, b_idx] = plv_matrix(phase)

    return tensor


def batch_volumetric_connectivity(
    epochs: np.ndarray,
    sampling_rate_hz: int,
    bands: dict[str, tuple[float, float]],
) -> np.ndarray:
    """Vectorized wrapper of :func:`volumetric_connectivity_tensor` over a
    batch of epochs.

    Args:
        epochs: array of shape (num_epochs, channels, samples).
        sampling_rate_hz: sampling rate.
        bands: see :func:`volumetric_connectivity_tensor`.

    Returns:
        Array of shape (num_epochs, channels, channels, num_bands).
    """
    return np.stack(
        [volumetric_connectivity_tensor(e, sampling_rate_hz, bands) for e in epochs],
        axis=0,
    )
