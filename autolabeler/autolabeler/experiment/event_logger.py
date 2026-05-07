"""Crash-safe CSV event logger for experiment trials.

Writes trial events to a CSV file in real-time, flushing after every
write to ensure no data loss on crash. Also stores experiment metadata
as a JSON sidecar file.
"""

import csv
import json
import time
import logging
from pathlib import Path
from typing import Optional, TextIO

from autolabeler.config import ExperimentConfig, HardwareConfig
from autolabeler.experiment.scheduler import Trial

logger = logging.getLogger(__name__)


class EventLogger:
    """Records experiment trial events to CSV.

    Each trial is written as a row immediately upon completion, with
    flush after every write for crash safety.

    Args:
        output_dir: Directory to write events.csv and metadata.json.
        experiment_config: Experiment configuration for metadata.
        hardware_config: Hardware configuration for metadata.
    """

    CSV_HEADER = [
        "trial_number",
        "class_label",
        "block_number",
        "start_time",
        "end_time",
        "duration",
    ]

    def __init__(
        self,
        output_dir: Path,
        experiment_config: ExperimentConfig,
        hardware_config: Optional[HardwareConfig] = None,
    ):
        self._output_dir = Path(output_dir)
        self._exp_config = experiment_config
        self._hw_config = hardware_config
        self._csv_path = self._output_dir / "events.csv"
        self._meta_path = self._output_dir / "metadata.json"
        self._file: Optional[TextIO] = None
        self._writer: Optional[csv.writer] = None
        self._events_written = 0
        self._start_timestamp: Optional[float] = None

    def start(self) -> Path:
        """Open the CSV file and write the header.

        Returns:
            Path to the events.csv file.
        """
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._file = open(self._csv_path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow(self.CSV_HEADER)
        self._file.flush()
        self._start_timestamp = time.time()
        self._events_written = 0

        # Write metadata sidecar
        self._write_metadata()

        logger.info(f"Event logger started: {self._csv_path}")
        return self._csv_path

    def log_trial(self, trial: Trial, start_time: float, end_time: float) -> None:
        """Log a completed trial event.

        Args:
            trial: The Trial object that was just completed.
            start_time: Unix timestamp when the trial started.
            end_time: Unix timestamp when the trial ended.
        """
        if self._writer is None:
            raise RuntimeError("EventLogger not started. Call start() first.")

        duration = end_time - start_time
        self._writer.writerow([
            trial.trial_number,
            trial.class_label,
            trial.block_number,
            f"{start_time:.6f}",
            f"{end_time:.6f}",
            f"{duration:.3f}",
        ])
        self._file.flush()
        self._events_written += 1

        logger.debug(
            f"Logged trial {trial.trial_number}: {trial.class_label} "
            f"({duration:.1f}s)"
        )

    def stop(self) -> int:
        """Close the CSV file.

        Returns:
            Number of events written.
        """
        if self._file:
            self._file.close()
            self._file = None
            self._writer = None
        logger.info(
            f"Event logger stopped. {self._events_written} events written "
            f"to {self._csv_path}"
        )
        return self._events_written

    @property
    def events_written(self) -> int:
        return self._events_written

    @property
    def csv_path(self) -> Path:
        return self._csv_path

    def _write_metadata(self) -> None:
        """Write experiment metadata as JSON sidecar."""
        from dataclasses import asdict

        meta = {
            "experiment_start": self._start_timestamp,
            "experiment_config": asdict(self._exp_config),
        }
        if self._hw_config:
            meta["hardware_config"] = asdict(self._hw_config)

        with open(self._meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
