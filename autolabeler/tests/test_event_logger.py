"""Tests for the experiment event logger module."""

import csv
import json
import tempfile
from pathlib import Path

from autolabeler.config import ExperimentConfig, HardwareConfig
from autolabeler.experiment.event_logger import EventLogger
from autolabeler.experiment.scheduler import Trial


class TestEventLogger:
    """Test crash-safe CSV event logging."""

    def test_creates_csv_with_header(self, tmp_path):
        config = ExperimentConfig(classes=["A", "B"], trials_per_class=5)
        logger = EventLogger(tmp_path, config)
        logger.start()
        logger.stop()

        csv_path = tmp_path / "events.csv"
        assert csv_path.exists()

        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            header = next(reader)
        assert header == EventLogger.CSV_HEADER

    def test_logs_trial_events(self, tmp_path):
        config = ExperimentConfig(classes=["Touch", "No Touch"], trials_per_class=3)
        logger = EventLogger(tmp_path, config)
        logger.start()

        trial = Trial(trial_number=1, class_label="Touch", block_number=1, index_in_block=0)
        logger.log_trial(trial, start_time=1000.0, end_time=1005.0)

        trial2 = Trial(trial_number=2, class_label="No Touch", block_number=1, index_in_block=1)
        logger.log_trial(trial2, start_time=1008.0, end_time=1013.0)

        logger.stop()

        # Read back
        with open(tmp_path / "events.csv", "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 2
        assert rows[0]["trial_number"] == "1"
        assert rows[0]["class_label"] == "Touch"
        assert rows[0]["block_number"] == "1"
        assert float(rows[0]["duration"]) == 5.0

        assert rows[1]["trial_number"] == "2"
        assert rows[1]["class_label"] == "No Touch"

    def test_events_written_count(self, tmp_path):
        config = ExperimentConfig(classes=["A"], trials_per_class=1)
        logger = EventLogger(tmp_path, config)
        logger.start()

        assert logger.events_written == 0

        trial = Trial(trial_number=1, class_label="A", block_number=1, index_in_block=0)
        logger.log_trial(trial, start_time=0, end_time=5)

        assert logger.events_written == 1

        count = logger.stop()
        assert count == 1

    def test_crash_safety_data_flushed(self, tmp_path):
        """Verify data is on disk even before stop() is called."""
        config = ExperimentConfig(classes=["A", "B"], trials_per_class=5)
        logger = EventLogger(tmp_path, config)
        logger.start()

        trial = Trial(trial_number=1, class_label="A", block_number=1, index_in_block=0)
        logger.log_trial(trial, start_time=100.0, end_time=105.0)

        # Read WITHOUT calling stop() — simulates crash
        with open(tmp_path / "events.csv", "r") as f:
            content = f.read()

        assert "A" in content
        assert "105.0" in content

        logger.stop()

    def test_metadata_json_created(self, tmp_path):
        config = ExperimentConfig(
            classes=["Touch", "No Touch"],
            trials_per_class=10,
            trial_duration=5.0,
        )
        hw_config = HardwareConfig(name="Test Device", sample_rate=1000)
        logger = EventLogger(tmp_path, config, hw_config)
        logger.start()
        logger.stop()

        meta_path = tmp_path / "metadata.json"
        assert meta_path.exists()

        with open(meta_path, "r") as f:
            meta = json.load(f)

        assert "experiment_config" in meta
        assert meta["experiment_config"]["classes"] == ["Touch", "No Touch"]
        assert meta["experiment_config"]["trial_duration"] == 5.0
        assert "hardware_config" in meta
        assert meta["hardware_config"]["name"] == "Test Device"
        assert "experiment_start" in meta

    def test_log_without_start_raises(self, tmp_path):
        config = ExperimentConfig(classes=["A"], trials_per_class=1)
        logger = EventLogger(tmp_path, config)

        trial = Trial(trial_number=1, class_label="A", block_number=1, index_in_block=0)
        try:
            logger.log_trial(trial, start_time=0, end_time=5)
            assert False, "Expected RuntimeError"
        except RuntimeError as e:
            assert "not started" in str(e).lower()

    def test_csv_path_property(self, tmp_path):
        config = ExperimentConfig(classes=["A"], trials_per_class=1)
        logger = EventLogger(tmp_path, config)
        assert logger.csv_path == tmp_path / "events.csv"

    def test_output_dir_created_automatically(self, tmp_path):
        nested = tmp_path / "sub" / "dir"
        config = ExperimentConfig(classes=["A"], trials_per_class=1)
        logger = EventLogger(nested, config)
        logger.start()
        logger.stop()

        assert (nested / "events.csv").exists()
