"""Feature extraction — compute statistical features from signal windows.

Extracts simple time-domain features from each preprocessed window.
Each feature is computed per-channel, producing a flat feature vector.
"""

import logging
from typing import Optional

import numpy as np

from autolabeler.config import MLConfig

logger = logging.getLogger(__name__)

# Registry of available feature functions
FEATURE_REGISTRY: dict[str, callable] = {}


def _register(name: str):
    """Decorator to register a feature function."""
    def decorator(func):
        FEATURE_REGISTRY[name] = func
        return func
    return decorator


@_register("rms")
def _rms(signal: np.ndarray) -> float:
    """Root Mean Square."""
    return float(np.sqrt(np.mean(signal ** 2)))


@_register("peak_to_peak")
def _peak_to_peak(signal: np.ndarray) -> float:
    """Peak-to-peak amplitude."""
    return float(np.max(signal) - np.min(signal))


@_register("std")
def _std(signal: np.ndarray) -> float:
    """Standard deviation."""
    return float(np.std(signal))


@_register("p90")
def _p90(signal: np.ndarray) -> float:
    """90th percentile of absolute value."""
    return float(np.percentile(np.abs(signal), 90))


@_register("mean_abs")
def _mean_abs(signal: np.ndarray) -> float:
    """Mean absolute value."""
    return float(np.mean(np.abs(signal)))


def extract_features(
    X: np.ndarray,
    config: Optional[MLConfig] = None,
) -> tuple[np.ndarray, list[str]]:
    """Extract features from preprocessed windows.

    Args:
        X: shape (n_trials, n_samples, n_channels) — preprocessed signal.
        config: ML configuration with feature list. If None, uses defaults.

    Returns:
        Tuple of:
            X_features: shape (n_trials, n_features) — feature matrix
            feature_names: list of feature name strings
    """
    if config is None:
        config = MLConfig()

    n_trials, n_samples, n_channels = X.shape
    feature_names = []
    feature_list = []

    for feat_name in config.features:
        if feat_name not in FEATURE_REGISTRY:
            logger.warning(f"Unknown feature '{feat_name}', skipping.")
            continue
        feat_func = FEATURE_REGISTRY[feat_name]

        for ch in range(n_channels):
            col_name = f"{feat_name}_ch{ch+1}" if n_channels > 1 else feat_name
            feature_names.append(col_name)

            values = np.array([
                feat_func(X[trial, :, ch]) for trial in range(n_trials)
            ])
            feature_list.append(values)

    X_features = np.column_stack(feature_list)
    logger.info(
        f"Extracted {len(feature_names)} features from "
        f"{n_trials} windows ({n_channels} ch)"
    )
    return X_features, feature_names


def list_available_features() -> list[str]:
    """Return list of registered feature names."""
    return list(FEATURE_REGISTRY.keys())
