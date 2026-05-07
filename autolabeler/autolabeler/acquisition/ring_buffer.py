"""Thread-safe ring buffer for storing timestamped multi-channel signal data."""

import threading
import time
import numpy as np
from typing import Optional


class RingBuffer:
    """Thread-safe circular buffer for real-time signal acquisition.

    Stores timestamped multi-channel samples in a pre-allocated NumPy array.
    When full, oldest samples are overwritten. Designed for high-throughput
    data from UDP listener at rates up to tens of kHz.

    Args:
        max_samples: Maximum number of samples to store.
        channels: Number of signal channels.
        dtype: NumPy dtype for sample data.
    """

    def __init__(
        self,
        max_samples: int = 600_000,
        channels: int = 1,
        dtype: np.dtype = np.float64,
    ):
        self._max_samples = max_samples
        self._channels = channels
        self._dtype = dtype

        # Pre-allocated storage
        self._data = np.zeros((max_samples, channels), dtype=dtype)
        self._timestamps = np.zeros(max_samples, dtype=np.float64)

        # Write position and sample counter
        self._write_pos = 0
        self._total_written = 0

        self._lock = threading.Lock()

    @classmethod
    def from_duration(
        cls,
        duration_seconds: float,
        sample_rate: int,
        channels: int = 1,
        dtype: np.dtype = np.float64,
    ) -> "RingBuffer":
        """Create a ring buffer sized for a given duration at a given sample rate."""
        max_samples = int(duration_seconds * sample_rate)
        return cls(max_samples=max_samples, channels=channels, dtype=dtype)

    @property
    def max_samples(self) -> int:
        return self._max_samples

    @property
    def channels(self) -> int:
        return self._channels

    @property
    def total_written(self) -> int:
        with self._lock:
            return self._total_written

    @property
    def available_samples(self) -> int:
        """Number of valid samples currently in the buffer."""
        with self._lock:
            return min(self._total_written, self._max_samples)

    def write(self, samples: np.ndarray, timestamps: Optional[np.ndarray] = None) -> None:
        """Write samples into the buffer.

        Args:
            samples: Array of shape (N,) for single-channel or (N, channels) for multi-channel.
            timestamps: Array of shape (N,) with timestamps. If None, uses time.time().
        """
        samples = np.asarray(samples, dtype=self._dtype)
        if samples.ndim == 1:
            if self._channels == 1:
                samples = samples.reshape(-1, 1)
            else:
                raise ValueError(
                    f"Expected {self._channels} channels, got 1D array. "
                    f"Provide shape (N, {self._channels})."
                )

        n = samples.shape[0]
        if samples.shape[1] != self._channels:
            raise ValueError(
                f"Channel mismatch: buffer has {self._channels}, got {samples.shape[1]}"
            )

        if timestamps is None:
            now = time.time()
            timestamps = np.linspace(now - n / 1000, now, n, endpoint=False)
        timestamps = np.asarray(timestamps, dtype=np.float64)

        with self._lock:
            if n >= self._max_samples:
                # More data than buffer can hold — keep last max_samples
                samples = samples[-self._max_samples:]
                timestamps = timestamps[-self._max_samples:]
                n = self._max_samples
                self._data[:] = samples
                self._timestamps[:] = timestamps
                self._write_pos = 0
                self._total_written += n
            else:
                end = self._write_pos + n
                if end <= self._max_samples:
                    self._data[self._write_pos:end] = samples
                    self._timestamps[self._write_pos:end] = timestamps
                else:
                    # Wrap around
                    first_part = self._max_samples - self._write_pos
                    self._data[self._write_pos:] = samples[:first_part]
                    self._timestamps[self._write_pos:] = timestamps[:first_part]
                    remainder = n - first_part
                    self._data[:remainder] = samples[first_part:]
                    self._timestamps[:remainder] = timestamps[first_part:]
                self._write_pos = end % self._max_samples
                self._total_written += n

    def read_latest(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """Read the N most recent samples.

        Returns:
            Tuple of (data, timestamps) where data has shape (M, channels)
            and timestamps has shape (M,). M = min(n, available_samples).
        """
        with self._lock:
            available = min(self._total_written, self._max_samples)
            n = min(n, available)
            if n == 0:
                return (
                    np.empty((0, self._channels), dtype=self._dtype),
                    np.empty(0, dtype=np.float64),
                )

            end = self._write_pos
            start = (end - n) % self._max_samples

            if start < end:
                data = self._data[start:end].copy()
                ts = self._timestamps[start:end].copy()
            else:
                data = np.concatenate([self._data[start:], self._data[:end]])
                ts = np.concatenate([self._timestamps[start:], self._timestamps[:end]])

            return data, ts

    def read_time_range(
        self, start_time: float, end_time: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Read samples within a time range.

        Returns:
            Tuple of (data, timestamps) for samples where start_time <= t < end_time.
        """
        with self._lock:
            available = min(self._total_written, self._max_samples)
            if available == 0:
                return (
                    np.empty((0, self._channels), dtype=self._dtype),
                    np.empty(0, dtype=np.float64),
                )

            # Get all valid data in chronological order
            end = self._write_pos
            if available < self._max_samples:
                all_data = self._data[:available].copy()
                all_ts = self._timestamps[:available].copy()
            elif end == 0:
                all_data = self._data.copy()
                all_ts = self._timestamps.copy()
            else:
                all_data = np.concatenate([self._data[end:], self._data[:end]])
                all_ts = np.concatenate([self._timestamps[end:], self._timestamps[:end]])

            mask = (all_ts >= start_time) & (all_ts < end_time)
            return all_data[mask], all_ts[mask]

    def read_all(self) -> tuple[np.ndarray, np.ndarray]:
        """Read all valid samples in chronological order."""
        with self._lock:
            available = min(self._total_written, self._max_samples)
            if available == 0:
                return (
                    np.empty((0, self._channels), dtype=self._dtype),
                    np.empty(0, dtype=np.float64),
                )

            end = self._write_pos
            if available < self._max_samples:
                return self._data[:available].copy(), self._timestamps[:available].copy()
            elif end == 0:
                return self._data.copy(), self._timestamps.copy()
            else:
                data = np.concatenate([self._data[end:], self._data[:end]])
                ts = np.concatenate([self._timestamps[end:], self._timestamps[:end]])
                return data, ts

    def clear(self) -> None:
        """Reset the buffer."""
        with self._lock:
            self._data[:] = 0
            self._timestamps[:] = 0
            self._write_pos = 0
            self._total_written = 0
