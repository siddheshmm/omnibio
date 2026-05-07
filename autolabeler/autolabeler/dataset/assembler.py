"""Dataset assembler — save raw recording, trial windows, and metadata.

Orchestrates the full dataset save after an experiment completes:
  1. Dump full raw recording from ring buffer → raw_recording.csv
  2. Extract and save individual trial windows → windows/trial_XXX_label.csv
  3. Update metadata.json with dataset statistics
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

from autolabeler.acquisition.ring_buffer import RingBuffer
from autolabeler.dataset.segmenter import extract_windows, dump_raw_recording, TrialWindow

logger = logging.getLogger(__name__)


def save_array_csv(
    path: Path, data: np.ndarray, timestamps: np.ndarray, channels: int
) -> None:
    """Save timestamped signal data as CSV.

    Format: timestamp, ch1, ch2, ...
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    header_cols = ["timestamp"] + [f"ch{i+1}" for i in range(channels)]
    header = ",".join(header_cols)

    combined = np.column_stack([timestamps, data])
    np.savetxt(path, combined, delimiter=",", header=header, comments="", fmt="%.6f")


def save_dataset(
    ring_buffer: RingBuffer,
    output_dir: Path,
    events_csv_path: Optional[Path] = None,
) -> dict:
    """Save the complete dataset to disk.

    Args:
        ring_buffer: The ring buffer containing signal data.
        output_dir: Directory to save all dataset files.
        events_csv_path: Path to events.csv. Defaults to output_dir/events.csv.

    Returns:
        Summary dict with dataset statistics.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if events_csv_path is None:
        events_csv_path = output_dir / "events.csv"

    channels = ring_buffer.channels
    summary = {
        "raw_samples": 0,
        "windows_saved": 0,
        "windows_empty": 0,
        "total_window_samples": 0,
    }

    # 1. Save raw recording
    raw_path = output_dir / "raw_recording.csv"
    raw_data, raw_ts = dump_raw_recording(ring_buffer)
    if raw_data.shape[0] > 0:
        save_array_csv(raw_path, raw_data, raw_ts, channels)
        summary["raw_samples"] = raw_data.shape[0]
        logger.info(f"Saved raw recording: {raw_path} ({raw_data.shape[0]} samples)")
    else:
        logger.warning("Ring buffer empty — no raw recording to save.")

    # 2. Extract and save trial windows
    if events_csv_path.exists():
        windows = extract_windows(ring_buffer, events_csv_path)
        windows_dir = output_dir / "windows"
        windows_dir.mkdir(parents=True, exist_ok=True)

        # Clear stale window files from previous experiments
        for old_file in windows_dir.glob("trial_*.csv"):
            old_file.unlink()

        for w in windows:
            filename = f"trial_{w.trial_number:03d}_{w.class_label}.csv"
            window_path = windows_dir / filename

            if w.num_samples > 0:
                save_array_csv(window_path, w.data, w.timestamps, channels)
                summary["windows_saved"] += 1
                summary["total_window_samples"] += w.num_samples
            else:
                summary["windows_empty"] += 1

        logger.info(
            f"Saved {summary['windows_saved']} trial windows to {windows_dir} "
            f"({summary['windows_empty']} empty)"
        )
    else:
        logger.warning(f"Events CSV not found at {events_csv_path} — skipping windows.")

    # 3. Update metadata.json with dataset stats
    meta_path = output_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    else:
        meta = {}

    meta["dataset"] = summary
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Dataset saved: {summary}")
    return summary
