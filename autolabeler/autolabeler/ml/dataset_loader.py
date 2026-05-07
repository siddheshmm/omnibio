"""Dataset loader — reads trial window CSVs into uniform NumPy arrays.

Loads all trial windows from a session directory, pads/truncates
to a uniform sample count, and returns (X_raw, y, label_names).
"""

import logging
import re
from pathlib import Path
from typing import Optional, Iterable, Sequence

import numpy as np

logger = logging.getLogger(__name__)


def load_dataset(
    session_dir: Path,
    expected_samples: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load trial windows from a session directory.

    Reads all `windows/trial_XXX_label.csv` files, pads or truncates
    to uniform length, and returns arrays ready for preprocessing.

    Args:
        session_dir: Path to the session directory containing `windows/`.
        expected_samples: Expected number of samples per window. If None,
            uses the median window length from the dataset.

    Returns:
        Tuple of:
            X_raw: shape (n_trials, n_samples, n_channels) — raw signal data
            y: shape (n_trials,) — integer class labels
            label_names: list of class name strings, indexed by y values
    """
    windows_dir = Path(session_dir) / "windows"
    if not windows_dir.exists():
        raise FileNotFoundError(f"Windows directory not found: {windows_dir}")

    # Find all trial CSV files
    trial_files = sorted(windows_dir.glob("trial_*.csv"))
    if not trial_files:
        raise ValueError(f"No trial window files found in {windows_dir}")

    # Parse filenames to get class labels
    pattern = re.compile(r"trial_(\d+)_(.+)\.csv")
    raw_windows = []
    labels = []

    for f in trial_files:
        match = pattern.match(f.name)
        if not match:
            logger.warning(f"Skipping non-matching file: {f.name}")
            continue

        class_label = match.group(2)

        # Load signal data (skip timestamp column)
        try:
            data = np.loadtxt(f, delimiter=",", skiprows=1)
        except Exception as e:
            logger.warning(f"Failed to load {f.name}: {e}")
            continue

        if data.ndim == 1:
            # Single row or single column — skip if too small
            if data.shape[0] <= 1:
                logger.warning(f"Skipping empty/tiny file: {f.name}")
                continue
            # If it's a flat array, it has timestamp + channels
            # Reshape: assume timestamp is col 0
            data = data.reshape(1, -1)

        # Drop timestamp column (column 0)
        signal = data[:, 1:]
        if signal.ndim == 1:
            signal = signal.reshape(-1, 1)

        raw_windows.append(signal)
        labels.append(class_label)

    if not raw_windows:
        raise ValueError("No valid trial windows loaded.")

    # Determine target length
    lengths = [w.shape[0] for w in raw_windows]
    if expected_samples is None:
        expected_samples = int(np.median(lengths))

    n_channels = raw_windows[0].shape[1]
    n_trials = len(raw_windows)

    logger.info(
        f"Loaded {n_trials} windows, {n_channels} channels, "
        f"target length {expected_samples} samples "
        f"(range: {min(lengths)}–{max(lengths)})"
    )

    # Pad/truncate to uniform length
    X_raw = np.zeros((n_trials, expected_samples, n_channels), dtype=np.float64)
    for i, window in enumerate(raw_windows):
        n = min(window.shape[0], expected_samples)
        X_raw[i, :n, :] = window[:n, :]

    # Encode labels
    unique_labels = sorted(set(labels))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    y = np.array([label_to_idx[l] for l in labels], dtype=np.int64)

    return X_raw, y, unique_labels


def load_multiple_sessions(
    session_dirs: Iterable[Path],
    expected_samples: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load and combine trial windows from multiple session directories.

    This is primarily used to train a single model on several sessions
    from the same subject. It enforces a consistent label set across all
    sessions and concatenates their windows along the trial axis.

    Args:
        session_dirs: Iterable of session directory paths, each containing
            a ``windows/`` subdirectory with trial CSVs.
        expected_samples: Optional target length in samples. If None,
            each session is first loaded with its own median length and
            then all sessions are truncated to the shortest window length
            across sessions so they can be stacked.

    Returns:
        Tuple of:
            X_raw: shape (sum_trials, n_samples, n_channels)
            y: shape (sum_trials,)
            label_names: list of class name strings, shared across sessions

    Raises:
        ValueError: If sessions have inconsistent label sets.
    """
    session_dirs = list(session_dirs)
    if not session_dirs:
        raise ValueError("No session directories provided.")

    X_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    label_ref: Optional[Sequence[str]] = None

    # First load each session independently.
    for sess in session_dirs:
        X_sess, y_sess, labels = load_dataset(Path(sess))

        if label_ref is None:
            label_ref = list(labels)
        elif list(labels) != list(label_ref):
            raise ValueError(
                "Inconsistent class labels across sessions. "
                f"First session labels: {list(label_ref)}, "
                f"session {sess} labels: {list(labels)}"
            )

        X_list.append(X_sess)
        y_list.append(y_sess)

    # Determine common target length.
    if expected_samples is None:
        # Truncate all sessions to the shortest window length.
        min_len = min(X.shape[1] for X in X_list)
        target_len = min_len
    else:
        target_len = expected_samples

    X_trimmed: list[np.ndarray] = []
    for X in X_list:
        if X.shape[1] >= target_len:
            X_trimmed.append(X[:, :target_len, :])
        else:
            # Pad with zeros at the end if a session has shorter windows.
            n_trials, _, n_channels = X.shape
            padded = np.zeros((n_trials, target_len, n_channels), dtype=X.dtype)
            padded[:, : X.shape[1], :] = X
            X_trimmed.append(padded)

    X_all = np.concatenate(X_trimmed, axis=0)
    y_all = np.concatenate(y_list, axis=0)

    return X_all, y_all, list(label_ref or [])
