# -*- coding: utf-8 -*-
"""
Temporal classification dosing automation.

Encodes short pseudorandom glucose/salt stimulus motifs as a sequence of
dosing cycles, then leaves response and washout windows for classification.

Default pilot:
  - 6-minute cycles
  - 4 stimulus cycles
  - 2 response cycles
  - 2 washout cycles
  - 6 total trials: A,C,B,A,B,C

Class motifs:
  A: glucose, glucose, salt, salt
  B: glucose, salt, glucose, salt
  C: salt, salt, glucose, glucose

Use glucose media in the media pump and salt media in the alt-media pump.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from pioreactor.actions.pump import add_media, add_alt_media, remove_waste
from pioreactor.automations import events
from pioreactor.automations.dosing.base import DosingAutomationJobContrib
from pioreactor.config import config
from pioreactor.utils import sqlite_worker
from pioreactor.utils.timing import current_utc_timestamp


__plugin_summary__ = "Short temporal glucose/salt classification dosing"
__plugin_version__ = "0.1.0"
__plugin_name__ = "Temporal Classification Dosing"
__plugin_author__ = "Siddhesh"
__plugin_homepage__ = "https://pioreactor.com"


@dataclass(frozen=True)
class CycleAction:
    pump: str  # "media", "alt_media", or "none"
    volume_ml: float


'''CLASS_LIBRARY = {
    "A": [
        CycleAction("media", 0.08),
        CycleAction("media", 0.08),
        CycleAction("alt_media", 0.08),
        CycleAction("alt_media", 0.08),
    ],
    "B": [
        CycleAction("media", 0.08),
        CycleAction("alt_media", 0.08),
        CycleAction("media", 0.08),
        CycleAction("alt_media", 0.08),
    ],
    "C": [
        CycleAction("alt_media", 0.08),
        CycleAction("alt_media", 0.08),
        CycleAction("media", 0.08),
        CycleAction("media", 0.08),
    ],
}'''


CLASS_LIBRARY = {
    # Rising ramp — slow start, accelerating glucose feed
    "A": [
        CycleAction("media", 0.05),
        CycleAction("media", 0.08),
        CycleAction("media", 0.12),
        CycleAction("media", 0.16),
        CycleAction("media", 0.21),
        CycleAction("media", 0.25),
    ],
    # Falling ramp — heavy front-load, tapering off
    "B": [
        CycleAction("media", 0.25),
        CycleAction("media", 0.21),
        CycleAction("media", 0.16),
        CycleAction("media", 0.12),
        CycleAction("media", 0.08),
        CycleAction("media", 0.05),
    ],
    # Pulse pair — two sharp spikes with low baseline
    "C": [
        CycleAction("media", 0.05),
        CycleAction("media", 0.22),
        CycleAction("media", 0.05),
        CycleAction("media", 0.05),
        CycleAction("media", 0.22),
        CycleAction("media", 0.05),
    ],
    # Mackey-Glass motif — flat then nonlinear surge
    "D": [
        CycleAction("media", 0.06),
        CycleAction("media", 0.05),
        CycleAction("media", 0.08),
        CycleAction("media", 0.14),
        CycleAction("media", 0.22),
        CycleAction("media", 0.25),
    ],
}


class TemporalClassificationDosing(DosingAutomationJobContrib):
    """
    Short temporal classification dosing.

    cycle_minutes:
      Scheduled interval between execute() calls.

    trial_sequence:
      Comma-separated class IDs, for example: "A,C,B,A,B,C"

    response_cycles / washout_cycles:
      Number of no-dose cycles after the 4-cycle stimulus motif.

    pulse_volume_ml:
      Overrides the default class motif pulse volume.
    """

    automation_name = "temporal_classification_dosing"

    published_settings = {
        "trial_sequence": {"datatype": "string", "settable": False, "unit": None},
        "stimulus_cycles": {"datatype": "integer", "settable": False, "unit": "cycles"},
        "response_cycles": {"datatype": "integer", "settable": False, "unit": "cycles"},
        "washout_cycles": {"datatype": "integer", "settable": False, "unit": "cycles"},
        "pulse_volume_ml": {"datatype": "float", "settable": False, "unit": "mL"},
        "current_trial": {"datatype": "string", "settable": False, "unit": None},
        "current_phase": {"datatype": "string", "settable": False, "unit": None},
    }

    def __init__(
        self,
        trial_sequence: str = "A,C,B,A,B,C",
        pulse_volume_ml: float = 0.08,
        stimulus_cycles: int = 4,
        response_cycles: int = 2,
        washout_cycles: int = 2,
        pause_between_pumps_s: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.trial_sequence = str(trial_sequence)
        self._trial_sequence = [
            token.strip().upper()
            for token in self.trial_sequence.split(",")
            if token.strip()
        ]
        if not self._trial_sequence:
            raise ValueError("trial_sequence must contain at least one class ID")

        self.pulse_volume_ml = float(pulse_volume_ml)
        self.stimulus_cycles = int(stimulus_cycles)
        self.response_cycles = int(response_cycles)
        self.washout_cycles = int(washout_cycles)
        self.pause_between_pumps_s = float(pause_between_pumps_s)

        # if self.stimulus_cycles != 4:
        #     raise ValueError("This pilot job expects stimulus_cycles=4")
        if self.pulse_volume_ml <= 0:
            raise ValueError("pulse_volume_ml must be > 0")
        if self.response_cycles < 0 or self.washout_cycles < 0:
            raise ValueError("response_cycles and washout_cycles must be >= 0")

        self._trial_count = 0
        self._cycle_in_trial = 0
        self._phase = "idle"
        self.current_trial = ""
        self.current_phase = "idle"

        self._trial_db = None
        self._cycle_db = None

        self._trial_len = self.stimulus_cycles + self.response_cycles + self.washout_cycles
        self._setup_db()

        self.logger.info(
            "Temporal Classification Dosing ready | duration=%s min | trials=%d | "
            "stimulus=%d cycles | response=%d cycles | washout=%d cycles | pulse=%.3f mL"
            % (
                self.duration,
                len(self._trial_sequence),
                self.stimulus_cycles,
                self.response_cycles,
                self.washout_cycles,
                self.pulse_volume_ml,
            )
        )

    def _setup_db(self):
        db_path = config.get("storage", "database")
        self._db = sqlite_worker.Sqlite3Worker(db_path)
        self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS temporal_classification_trials (
                experiment      TEXT NOT NULL,
                pioreactor_unit TEXT NOT NULL,
                timestamp       TEXT NOT NULL,
                trial_index     INTEGER NOT NULL,
                class_id        TEXT NOT NULL,
                status          TEXT NOT NULL
            )
            """
        )
        self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS temporal_classification_cycles (
                experiment      TEXT NOT NULL,
                pioreactor_unit TEXT NOT NULL,
                timestamp       TEXT NOT NULL,
                trial_index     INTEGER NOT NULL,
                class_id        TEXT NOT NULL,
                cycle_in_trial  INTEGER NOT NULL,
                phase           TEXT NOT NULL,
                pump            TEXT NOT NULL,
                volume_ml       REAL NOT NULL
            )
            """
        )

    def _pump_kwargs(self):
        return dict(
            unit=self.unit,
            experiment=self.experiment,
            source_of_event=f"{self.job_name}:{self.automation_name}",
            mqtt_client=self.pub_client,
            logger=self.logger,
        )

    def _class_action(self, class_id: str, cycle_in_trial: int) -> CycleAction:
        try:
            return CLASS_LIBRARY[class_id][cycle_in_trial]
        except KeyError as e:
            raise ValueError(f"Unknown class_id '{class_id}'") from e
        except IndexError as e:
            raise ValueError(f"class_id '{class_id}' does not have cycle {cycle_in_trial}") from e

    def _log_trial(self, trial_index: int, class_id: str, status: str):
        self._db.execute(
            """
            INSERT INTO temporal_classification_trials
            (experiment, pioreactor_unit, timestamp, trial_index, class_id, status)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (self.experiment, self.unit, current_utc_timestamp(), trial_index, class_id, status),
        )

    def _log_cycle(self, trial_index: int, class_id: str, cycle_in_trial: int, phase: str, pump: str, volume_ml: float):
        self._db.execute(
            """
            INSERT INTO temporal_classification_cycles
            (experiment, pioreactor_unit, timestamp, trial_index, class_id, cycle_in_trial, phase, pump, volume_ml)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                self.experiment,
                self.unit,
                current_utc_timestamp(),
                trial_index,
                class_id,
                cycle_in_trial,
                phase,
                pump,
                volume_ml,
            ),
        )

    def _fire_pump(self, pump: str, volume_ml: float):
        if volume_ml <= 0:
            return
        if pump == "media":
            add_media(ml=volume_ml, **self._pump_kwargs())
        elif pump == "alt_media":
            add_alt_media(ml=volume_ml, **self._pump_kwargs())
        elif pump == "waste":
            remove_waste(ml=volume_ml, **self._pump_kwargs())
        else:
            raise ValueError(f"Unknown pump '{pump}'")

    def execute(self):
        if self._trial_count >= len(self._trial_sequence):
            self.logger.info("All %d trials complete. Stopping automation.", len(self._trial_sequence))
            self.set_state(self.DISCONNECTED)
            return None

        class_id = self._trial_sequence[self._trial_count]
        self.current_trial = class_id

        if self._cycle_in_trial == 0:
            self.current_phase = "stimulus"
            self._phase = "stimulus"
            self._log_trial(self._trial_count, class_id, "started")
            self.logger.info(
                "Trial %d/%d | class=%s | stimulus starting"
                % (self._trial_count + 1, len(self._trial_sequence), class_id)
            )

        if self._cycle_in_trial < self.stimulus_cycles:
            action = self._class_action(class_id, self._cycle_in_trial)
            self.current_phase = "stimulus"
            self._phase = "stimulus"
            self.logger.info(
                "Trial %d/%d | class=%s | cycle %d/%d | %s %.4f mL"
                % (
                    self._trial_count + 1,
                    len(self._trial_sequence),
                    class_id,
                    self._cycle_in_trial + 1,
                    self.stimulus_cycles,
                    action.pump,
                    action.volume_ml,
                )
            )
            remove_waste(ml=action.volume_ml, **self._pump_kwargs())
            self._fire_pump(action.pump, action.volume_ml)
            self._log_cycle(
                self._trial_count,
                class_id,
                self._cycle_in_trial,
                "stimulus",
                action.pump,
                action.volume_ml,
            )
            event = None
        else:
            phase_index = self._cycle_in_trial - self.stimulus_cycles
            if phase_index < self.response_cycles:
                self.current_phase = "response"
                self._phase = "response"
                self._log_cycle(
                    self._trial_count,
                    class_id,
                    self._cycle_in_trial,
                    "response",
                    "none",
                    0.0,
                )
                event = None
            else:
                self.current_phase = "washout"
                self._phase = "washout"
                self._log_cycle(
                    self._trial_count,
                    class_id,
                    self._cycle_in_trial,
                    "washout",
                    "none",
                    0.0,
                )
                event = None

        self._cycle_in_trial += 1

        if self._cycle_in_trial >= self._trial_len:
            self._log_trial(self._trial_count, class_id, "completed")
            self.logger.info(
                "Trial %d/%d | class=%s complete"
                    % (self._trial_count + 1, len(self._trial_sequence), class_id)
            )
            self._trial_count += 1
            self._cycle_in_trial = 0
            self.current_trial = ""
            self.current_phase = "idle"
            return None

        return None

    def on_disconnected(self):
        try:
            self._db.close()
        except Exception:
            pass
