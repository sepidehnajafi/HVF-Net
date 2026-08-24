"""EEG preprocessing utilities: filtering and epoching.

This module operates on raw continuous EEG (channels x time) and produces
fixed-length, overlapping epochs time-locked to stimulus markers. It assumes
artifact rejection (e.g. ICA-based EOG/EMG removal) has already been applied
upstream; see Section 3.1 of the paper for the full acquisition pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EpochingConfig:
    sampling_rate_hz: int = 500
    epoch_duration_s: float = 2.0
    overlap: float = 0.5

    @property
    def epoch_len_samples(self) -> int:
        return int(round(self.epoch_duration_s * self.sampling_rate_hz))

    @property
    def step_samples(self) -> int:
        return int(round(self.epoch_len_samples * (1.0 - self.overlap)))


def bandpass_filter(
    signal: np.ndarray,
    low_hz: float,
    high_hz: float,
    sampling_rate_hz: int,
    order: int = 4,
) -> np.ndarray:
    """Zero-phase Butterworth bandpass filter.

    Args:
        signal: array of shape (channels, time).
        low_hz, high_hz: passband edges.
        sampling_rate_hz: sampling rate of ``signal``.
        order: filter order (applied twice via ``filtfilt`` -> effective
            order is doubled).

    Returns:
        Filtered signal, same shape as input.
    """
    from scipy.signal import butter, filtfilt

    nyquist = sampling_rate_hz / 2.0
    b, a = butter(order, [low_hz / nyquist, high_hz / nyquist], btype="band")
    return filtfilt(b, a, signal, axis=-1)


def make_overlapping_epochs(
    continuous: np.ndarray,
    marker_samples: np.ndarray,
    cfg: EpochingConfig,
    pre_marker_s: float = 0.2,
) -> np.ndarray:
    """Segment continuous EEG into overlapping epochs locked to markers.

    Args:
        continuous: array of shape (channels, time_samples).
        marker_samples: 1-D array of stimulus-onset sample indices.
        cfg: :class:`EpochingConfig`.
        pre_marker_s: seconds of pre-stimulus baseline to include.

    Returns:
        Array of shape (num_epochs, channels, epoch_len_samples).
    """
    epoch_len = cfg.epoch_len_samples
    step = cfg.step_samples
    pre_samples = int(round(pre_marker_s * cfg.sampling_rate_hz))

    epochs = []
    for onset in marker_samples:
        start = onset - pre_samples
        # Slide overlapping windows across the response window for this trial.
        window_end = start + epoch_len
        while window_end <= continuous.shape[-1]:
            if start >= 0:
                epochs.append(continuous[:, start:start + epoch_len])
            start += step
            window_end = start + epoch_len
            # Only take windows within a single trial-length span; callers
            # that want a fixed number of windows per trial should slice
            # the returned array downstream.
            if start - (onset - pre_samples) >= epoch_len:
                break

    if not epochs:
        return np.empty((0, continuous.shape[0], epoch_len))
    return np.stack(epochs, axis=0)


def reject_high_amplitude_epochs(
    epochs: np.ndarray,
    peak_to_peak_uv: float = 100.0,
) -> np.ndarray:
    """Boolean mask of epochs to keep, rejecting artifact-contaminated ones.

    Args:
        epochs: array of shape (num_epochs, channels, samples), in microvolts.
        peak_to_peak_uv: rejection threshold.

    Returns:
        Boolean array of shape (num_epochs,), True = keep.
    """
    ptp = epochs.max(axis=-1) - epochs.min(axis=-1)  # (num_epochs, channels)
    return (ptp.max(axis=-1) <= peak_to_peak_uv)
