"""Tests for the dataset segmenter and assembler."""

import csv
import json
import time
import numpy as np
from pathlib import Path

from autolabeler.acquisition.ring_buffer import RingBuffer
from autolabeler.dataset.segmenter import extract_windows, dump_raw_recording
from autolabeler.dataset.assembler import save_dataset


def _write_events_csv(path: Path, events: list[dict]) -> None:
    """Helper: write a minimal events.csv."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["trial_number", "class_label", "block_number",
                         "start_time", "end_time", "duration"],
        )
        writer.writeheader()
        writer.writerows(events)


def _create_buffer_with_data(
    n_samples: int = 5000,
    channels: int = 1,
    sample_rate: int = 1000,
) -> tuple[RingBuffer, float]:
    """Create a ring buffer filled with known data, return (buffer, start_time)."""
    buf = RingBuffer(max_samples=n_samples * 2, channels=channels)
    start = time.time()
    dt = 1.0 / sample_rate
    timestamps = np.array([start + i * dt for i in range(n_samples)])
    data = np.arange(n_samples, dtype=np.float64).reshape(-1, 1)
    if channels > 1:
        data = np.column_stack([data * (ch + 1) for ch in range(channels)])
    buf.write(data, timestamps)
    return buf, start


class TestSegmenter:
    """Test signal window extraction."""

    def test_extract_single_window(self, tmp_path):
        buf, start = _create_buffer_with_data(5000, 1, 1000)

        # Write events for a 1-second trial
        trial_start = start + 1.0
        trial_end = start + 2.0
        _write_events_csv(tmp_path / "events.csv", [{
            "trial_number": 1,
            "class_label": "Touch",
            "block_number": 1,
            "start_time": f"{trial_start:.6f}",
            "end_time": f"{trial_end:.6f}",
            "duration": "1.000",
        }])

        windows = extract_windows(buf, tmp_path / "events.csv")

        assert len(windows) == 1
        w = windows[0]
        assert w.trial_number == 1
        assert w.class_label == "Touch"
        assert w.block_number == 1
        assert w.num_samples > 0
        assert w.num_samples <= 1000  # ~1 second at 1000 Hz
        assert w.data.shape[1] == 1

    def test_extract_multiple_windows(self, tmp_path):
        buf, start = _create_buffer_with_data(10000, 1, 1000)

        events = []
        for i in range(5):
            t0 = start + i * 2.0
            t1 = t0 + 1.0
            events.append({
                "trial_number": i + 1,
                "class_label": "A" if i % 2 == 0 else "B",
                "block_number": 1,
                "start_time": f"{t0:.6f}",
                "end_time": f"{t1:.6f}",
                "duration": "1.000",
            })

        _write_events_csv(tmp_path / "events.csv", events)
        windows = extract_windows(buf, tmp_path / "events.csv")

        assert len(windows) == 5
        assert windows[0].class_label == "A"
        assert windows[1].class_label == "B"

    def test_extract_multichannel(self, tmp_path):
        buf, start = _create_buffer_with_data(5000, 3, 1000)

        _write_events_csv(tmp_path / "events.csv", [{
            "trial_number": 1,
            "class_label": "X",
            "block_number": 1,
            "start_time": f"{start + 0.5:.6f}",
            "end_time": f"{start + 1.5:.6f}",
            "duration": "1.000",
        }])

        windows = extract_windows(buf, tmp_path / "events.csv")
        assert windows[0].data.shape[1] == 3

    def test_out_of_range_returns_empty(self, tmp_path):
        buf, start = _create_buffer_with_data(1000, 1, 1000)

        # Event far in the future
        _write_events_csv(tmp_path / "events.csv", [{
            "trial_number": 1,
            "class_label": "X",
            "block_number": 1,
            "start_time": f"{start + 100:.6f}",
            "end_time": f"{start + 101:.6f}",
            "duration": "1.000",
        }])

        windows = extract_windows(buf, tmp_path / "events.csv")
        assert len(windows) == 1
        assert windows[0].num_samples == 0

    def test_dump_raw_recording(self):
        buf, _ = _create_buffer_with_data(500, 2, 1000)
        data, ts = dump_raw_recording(buf)

        assert data.shape == (500, 2)
        assert ts.shape == (500,)

    def test_missing_events_file_raises(self, tmp_path):
        buf, _ = _create_buffer_with_data(100)
        try:
            extract_windows(buf, tmp_path / "nope.csv")
            assert False, "Expected FileNotFoundError"
        except FileNotFoundError:
            pass


class TestAssembler:
    """Test full dataset save pipeline."""

    def test_save_creates_all_files(self, tmp_path):
        buf, start = _create_buffer_with_data(3000, 1, 1000)

        # Write events
        events = [
            {"trial_number": 1, "class_label": "A", "block_number": 1,
             "start_time": f"{start + 0.5:.6f}", "end_time": f"{start + 1.5:.6f}",
             "duration": "1.000"},
            {"trial_number": 2, "class_label": "B", "block_number": 1,
             "start_time": f"{start + 1.5:.6f}", "end_time": f"{start + 2.5:.6f}",
             "duration": "1.000"},
        ]
        _write_events_csv(tmp_path / "events.csv", events)

        summary = save_dataset(buf, tmp_path)

        # Check files exist
        assert (tmp_path / "raw_recording.csv").exists()
        assert (tmp_path / "windows" / "trial_001_A.csv").exists()
        assert (tmp_path / "windows" / "trial_002_B.csv").exists()
        assert (tmp_path / "metadata.json").exists()

        # Check summary
        assert summary["raw_samples"] == 3000
        assert summary["windows_saved"] == 2

    def test_raw_recording_shape(self, tmp_path):
        buf, start = _create_buffer_with_data(2000, 2, 1000)

        # Empty events
        _write_events_csv(tmp_path / "events.csv", [])
        save_dataset(buf, tmp_path)

        raw = np.loadtxt(tmp_path / "raw_recording.csv", delimiter=",", skiprows=1)
        assert raw.shape == (2000, 3)  # timestamp + 2 channels

    def test_window_file_contents(self, tmp_path):
        buf, start = _create_buffer_with_data(2000, 1, 1000)

        _write_events_csv(tmp_path / "events.csv", [{
            "trial_number": 1, "class_label": "test", "block_number": 1,
            "start_time": f"{start + 0.1:.6f}", "end_time": f"{start + 0.6:.6f}",
            "duration": "0.500",
        }])

        save_dataset(buf, tmp_path)

        window_data = np.loadtxt(
            tmp_path / "windows" / "trial_001_test.csv",
            delimiter=",", skiprows=1,
        )
        # Should have ~500 samples (0.5s at 1000 Hz), timestamp + 1 channel
        assert window_data.shape[0] > 0
        assert window_data.shape[1] == 2  # timestamp + ch1

    def test_metadata_updated(self, tmp_path):
        buf, start = _create_buffer_with_data(1000, 1, 1000)

        # Pre-existing metadata (like from event logger)
        with open(tmp_path / "metadata.json", "w") as f:
            json.dump({"experiment_config": {"classes": ["A"]}}, f)

        _write_events_csv(tmp_path / "events.csv", [{
            "trial_number": 1, "class_label": "A", "block_number": 1,
            "start_time": f"{start:.6f}", "end_time": f"{start + 1:.6f}",
            "duration": "1.000",
        }])

        save_dataset(buf, tmp_path)

        with open(tmp_path / "metadata.json", "r") as f:
            meta = json.load(f)

        assert "dataset" in meta
        assert meta["dataset"]["windows_saved"] == 1
        assert meta["dataset"]["raw_samples"] == 1000
        # Original metadata preserved
        assert meta["experiment_config"]["classes"] == ["A"]

    def test_no_events_file_saves_raw_only(self, tmp_path):
        buf, _ = _create_buffer_with_data(500, 1, 1000)
        # Don't create events.csv

        summary = save_dataset(buf, tmp_path, tmp_path / "events.csv")

        assert (tmp_path / "raw_recording.csv").exists()
        assert summary["raw_samples"] == 500
        assert summary["windows_saved"] == 0
