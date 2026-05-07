"""Signal segmenter — extract labeled windows from the ring buffer.

Uses event timestamps (from events.csv) to slice the ring buffer into
individual trial windows, each tagged with its class label.
"""

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from autolabeler.acquisition.ring_buffer import RingBuffer

logger = logging.getLogger(__name__)


@dataclass
class TrialWindow:
    """A single extracted trial window.

    Attributes:
        trial_number: 1-indexed trial number.
        class_label: Class name (e.g. "Touch", "No Touch").
        block_number: Block this trial belonged to.
        start_time: Unix timestamp of window start.
        end_time: Unix timestamp of window end.
        data: Signal data, shape (N, channels).
        timestamps: Sample timestamps, shape (N,).
    """
    trial_number: int
    class_label: str
    block_number: int
    start_time: float
    end_time: float
    data: np.ndarray
    timestamps: np.ndarray

    @property
    def num_samples(self) -> int:
        return self.data.shape[0]

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


def extract_windows(
    ring_buffer: RingBuffer,
    events_csv_path: Path,
) -> list[TrialWindow]:
    """Extract trial windows from the ring buffer using event timestamps.

    Reads each trial from events.csv and extracts the corresponding
    signal segment from the ring buffer using its time range.

    Args:
        ring_buffer: The ring buffer containing signal data.
        events_csv_path: Path to the events.csv file from the experiment.

    Returns:
        List of TrialWindow objects, one per trial in events.csv.
    """
    events_csv_path = Path(events_csv_path)
    if not events_csv_path.exists():
        raise FileNotFoundError(f"Events file not found: {events_csv_path}")

    windows: list[TrialWindow] = []

    with open(events_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            trial_num = int(row["trial_number"])
            class_label = row["class_label"]
            block_num = int(row["block_number"])
            start_time = float(row["start_time"])
            end_time = float(row["end_time"])

            # Extract signal window from ring buffer
            data, timestamps = ring_buffer.read_time_range(start_time, end_time)

            if data.shape[0] == 0:
                logger.warning(
                    f"Trial {trial_num} ({class_label}): no samples found in "
                    f"buffer for time range [{start_time:.3f}, {end_time:.3f}). "
                    f"Data may have been overwritten."
                )

            windows.append(TrialWindow(
                trial_number=trial_num,
                class_label=class_label,
                block_number=block_num,
                start_time=start_time,
                end_time=end_time,
                data=data,
                timestamps=timestamps,
            ))

    logger.info(
        f"Extracted {len(windows)} trial windows from {events_csv_path.name}"
    )
    return windows


def dump_raw_recording(ring_buffer: RingBuffer) -> tuple[np.ndarray, np.ndarray]:
    """Dump the full raw recording from the ring buffer.

    Returns:
        Tuple of (data, timestamps) — the complete signal in chronological order.
    """
    data, timestamps = ring_buffer.read_all()
    logger.info(f"Raw recording: {data.shape[0]} samples, {data.shape[1]} channels")
    return data, timestamps
