"""Experiment prompter — QTimer-based state machine for driving experiments.

Manages the experiment lifecycle: countdown → stimulus → rest → ...  → done.
Emits Qt signals for GUI updates. Runs entirely on the GUI thread via QTimer
to avoid threading issues with Qt widgets.
"""

import time
import logging
from enum import Enum, auto
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QObject, QTimer, pyqtSignal

from autolabeler.config import ExperimentConfig, HardwareConfig
from autolabeler.experiment.scheduler import Trial, generate_schedule, schedule_summary
from autolabeler.experiment.event_logger import EventLogger
from autolabeler.dataset.assembler import save_dataset

logger = logging.getLogger(__name__)


class ExperimentState(Enum):
    """States of the experiment state machine."""
    IDLE = auto()
    COUNTDOWN = auto()
    STIMULUS = auto()
    REST = auto()
    DONE = auto()


class Prompter(QObject):
    """Drives the experiment through its states, emitting signals for the GUI.

    Signals:
        state_changed(state_name): Emitted when the state machine transitions.
        countdown_tick(seconds_remaining): Emitted each second during countdown.
        trial_started(trial_number, total_trials, class_label, block_number):
            Emitted when a stimulus trial begins.
        trial_ended(trial_number, class_label): Emitted when a trial ends.
        rest_started(seconds): Emitted when a rest period begins.
        rest_tick(seconds_remaining): Emitted each second during rest.
        progress_updated(trial_number, total_trials): Emitted for progress bar.
        experiment_finished(summary_dict): Emitted when all trials are complete.
    """

    # Signals
    state_changed = pyqtSignal(str)
    countdown_tick = pyqtSignal(int)
    trial_started = pyqtSignal(int, int, str, int)  # trial#, total, label, block
    trial_ended = pyqtSignal(int, str)  # trial#, label
    rest_started = pyqtSignal(float)  # rest duration
    rest_tick = pyqtSignal(float)  # remaining seconds
    progress_updated = pyqtSignal(int, int)  # current, total
    experiment_finished = pyqtSignal(dict)

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._state = ExperimentState.IDLE
        self._schedule: list[Trial] = []
        self._exp_config: Optional[ExperimentConfig] = None
        self._hw_config: Optional[HardwareConfig] = None
        self._event_logger: Optional[EventLogger] = None
        self._ring_buffer = None
        self._output_dir: Optional[Path] = None

        self._current_trial_idx = 0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._tick_interval_ms = 100  # 10 Hz tick rate for smooth countdown

        self._phase_start_time = 0.0
        self._phase_duration = 0.0
        self._countdown_remaining = 0
        self._experiment_start_time = 0.0

    @property
    def state(self) -> ExperimentState:
        return self._state

    @property
    def is_running(self) -> bool:
        return self._state not in (ExperimentState.IDLE, ExperimentState.DONE)

    def start_experiment(
        self,
        exp_config: ExperimentConfig,
        output_dir: Path,
        hw_config: Optional[HardwareConfig] = None,
        ring_buffer=None,
        seed: Optional[int] = None,
    ) -> None:
        """Start a new experiment.

        Args:
            exp_config: Experiment configuration.
            output_dir: Directory to write events.csv.
            hw_config: Optional hardware config for metadata.
            seed: Optional random seed for schedule reproducibility.
        """
        if self.is_running:
            logger.warning("Experiment already running, ignoring start request.")
            return

        self._exp_config = exp_config
        self._hw_config = hw_config
        self._ring_buffer = ring_buffer
        self._output_dir = output_dir

        # Generate schedule
        self._schedule = generate_schedule(exp_config, seed=seed)
        self._current_trial_idx = 0

        # Start event logger
        self._event_logger = EventLogger(output_dir, exp_config, hw_config)
        self._event_logger.start()

        # Begin countdown
        self._experiment_start_time = time.time()
        self._countdown_remaining = int(exp_config.countdown_duration)
        self._enter_state(ExperimentState.COUNTDOWN)

        logger.info(
            f"Experiment started: {len(self._schedule)} trials, "
            f"{len(exp_config.classes)} classes"
        )

    def abort(self) -> None:
        """Abort the running experiment."""
        if not self.is_running:
            return
        self._timer.stop()
        if self._event_logger:
            self._event_logger.stop()
        # Save whatever we have so far
        self._save_dataset()
        self._enter_state(ExperimentState.IDLE)
        logger.info("Experiment aborted.")

    # --- State machine ---

    def _enter_state(self, new_state: ExperimentState) -> None:
        """Transition to a new state."""
        self._state = new_state
        self.state_changed.emit(new_state.name)

        if new_state == ExperimentState.COUNTDOWN:
            self._phase_start_time = time.time()
            self._phase_duration = self._exp_config.countdown_duration
            self.countdown_tick.emit(self._countdown_remaining)
            self._timer.start(self._tick_interval_ms)

        elif new_state == ExperimentState.STIMULUS:
            trial = self._schedule[self._current_trial_idx]
            self._phase_start_time = time.time()
            self._phase_duration = self._exp_config.trial_duration
            self.trial_started.emit(
                trial.trial_number,
                len(self._schedule),
                trial.class_label,
                trial.block_number,
            )
            self.progress_updated.emit(trial.trial_number, len(self._schedule))
            self._timer.start(self._tick_interval_ms)

        elif new_state == ExperimentState.REST:
            self._phase_start_time = time.time()
            self._phase_duration = self._exp_config.rest_duration
            self.rest_started.emit(self._exp_config.rest_duration)
            self._timer.start(self._tick_interval_ms)

        elif new_state == ExperimentState.DONE:
            self._timer.stop()
            if self._event_logger:
                self._event_logger.stop()
            # Save dataset
            dataset_summary = self._save_dataset()
            summary = schedule_summary(self._schedule, self._exp_config)
            summary["actual_duration"] = time.time() - self._experiment_start_time
            summary["dataset"] = dataset_summary
            summary["output_dir"] = str(self._output_dir)
            self.experiment_finished.emit(summary)
            logger.info("Experiment completed.")

    def _save_dataset(self) -> dict:
        """Save dataset from ring buffer + events."""
        if self._ring_buffer is None or self._output_dir is None:
            return {}
        try:
            return save_dataset(
                self._ring_buffer,
                self._output_dir,
                self._output_dir / "events.csv",
            )
        except Exception as e:
            logger.error(f"Failed to save dataset: {e}", exc_info=True)
            return {"error": str(e)}

    def _tick(self) -> None:
        """Called by QTimer at 10 Hz. Manages phase transitions."""
        elapsed = time.time() - self._phase_start_time

        if self._state == ExperimentState.COUNTDOWN:
            remaining = int(self._exp_config.countdown_duration - elapsed)
            if remaining != self._countdown_remaining:
                self._countdown_remaining = max(0, remaining)
                self.countdown_tick.emit(self._countdown_remaining)
            if elapsed >= self._exp_config.countdown_duration:
                self._timer.stop()
                self._enter_state(ExperimentState.STIMULUS)

        elif self._state == ExperimentState.STIMULUS:
            remaining = self._phase_duration - elapsed
            self.rest_tick.emit(max(0.0, remaining))  # reuse for trial countdown
            if elapsed >= self._phase_duration:
                self._timer.stop()
                # Log the completed trial
                trial = self._schedule[self._current_trial_idx]
                end_time = time.time()
                self._event_logger.log_trial(
                    trial, self._phase_start_time, end_time
                )
                self.trial_ended.emit(trial.trial_number, trial.class_label)
                self._current_trial_idx += 1

                # Next: rest or done?
                if self._current_trial_idx >= len(self._schedule):
                    self._enter_state(ExperimentState.DONE)
                else:
                    self._enter_state(ExperimentState.REST)

        elif self._state == ExperimentState.REST:
            remaining = self._phase_duration - elapsed
            self.rest_tick.emit(max(0.0, remaining))
            if elapsed >= self._phase_duration:
                self._timer.stop()
                self._enter_state(ExperimentState.STIMULUS)
