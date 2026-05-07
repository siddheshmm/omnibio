"""Signal preprocessor — per-window signal transforms.

Applies signal-level preprocessing to raw windows:
  - DC offset removal (subtract mean)
  - Bandpass filter (Butterworth)
  - Normalization (z-score)
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.signal import butter, sosfilt

from autolabeler.config import MLConfig

logger = logging.getLogger(__name__)


def preprocess(
    X_raw: np.ndarray,
    config: Optional[MLConfig] = None,
) -> np.ndarray:
    """Apply preprocessing to raw windows.

    Args:
        X_raw: shape (n_trials, n_samples, n_channels)
        config: ML configuration. If None, uses defaults.

    Returns:
        X_processed: same shape, preprocessed signal.
    """
    if config is None:
        config = MLConfig()

    X = X_raw.copy()

    if config.dc_offset_removal:
        X = remove_dc_offset(X)

    if config.bandpass_filter:
        X = apply_bandpass(
            X,
            low=config.bandpass_low,
            high=config.bandpass_high,
            sample_rate=1000,  # Inferred from data; can be overridden
        )

    if config.normalize:
        X = normalize_zscore(X)

    return X


def remove_dc_offset(X: np.ndarray) -> np.ndarray:
    """Remove DC offset (subtract mean per window per channel).

    Args:
        X: shape (n_trials, n_samples, n_channels)

    Returns:
        DC-removed signal, same shape.
    """
    # Mean along the samples axis (axis=1), keep dims for broadcasting
    return X - X.mean(axis=1, keepdims=True)


def apply_bandpass(
    X: np.ndarray,
    low: float = 0.1,
    high: float = 500.0,
    sample_rate: int = 1000,
    order: int = 4,
) -> np.ndarray:
    """Apply Butterworth bandpass filter.

    Args:
        X: shape (n_trials, n_samples, n_channels)
        low: Low cutoff frequency (Hz).
        high: High cutoff frequency (Hz).
        sample_rate: Sample rate (Hz).
        order: Filter order.

    Returns:
        Filtered signal, same shape.
    """
    nyquist = sample_rate / 2.0
    low_norm = max(low / nyquist, 0.001)
    high_norm = min(high / nyquist, 0.999)

    if low_norm >= high_norm:
        logger.warning(
            f"Invalid bandpass range [{low}–{high}] Hz for {sample_rate} Hz. "
            f"Skipping filter."
        )
        return X

    sos = butter(order, [low_norm, high_norm], btype="band", output="sos")

    X_filtered = np.empty_like(X)
    for trial in range(X.shape[0]):
        for ch in range(X.shape[2]):
            X_filtered[trial, :, ch] = sosfilt(sos, X[trial, :, ch])

    return X_filtered


def normalize_zscore(X: np.ndarray) -> np.ndarray:
    """Z-score normalize per window per channel.

    Args:
        X: shape (n_trials, n_samples, n_channels)

    Returns:
        Normalized signal, same shape.
    """
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True)
    std = np.where(std == 0, 1.0, std)  # Avoid division by zero
    return (X - mean) / std
