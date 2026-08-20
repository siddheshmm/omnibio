# -*- coding: utf-8 -*-
"""
manual_dosing_control.py  — v1.0.0 (Pioreactor 26.5.0+ Compatible)
───────────────────────────────────────────────────────────────────
Independent scheduled dosing automation for Pioreactor 26.5.0+.

Features:
  - Fully self-scheduled: uses an internal daemon Timer for rock-solid timing
    on Pioreactor 26.5.0+ (where base class shared duration timers were removed).
  - Supports custom volume sequences (e.g. "0.05, 0.10, 0.20, 0.25, 0.20, 0.10")
    or continuous mathematical sine wave modulation.
  - Immediate execution of Cycle 0 on startup at t=0s.
  - Dedicated SQLite table logging (`sine_media_volumes`).
  - Safe lifecycle cleanup on pause, sleep, and disconnect.
"""
from __future__ import annotations
import time
import math
from threading import Timer
from typing import Any

from pioreactor.automations.dosing.base import DosingAutomationJobContrib
from pioreactor.automations import events
from pioreactor.actions.pump import add_media, add_alt_media, remove_waste
from pioreactor.utils import sqlite_worker
from pioreactor.utils.timing import current_utc_timestamp
from pioreactor.config import config


__plugin_summary__ = "Self-scheduled dosing automation with sine wave and sequence support"
__plugin_version__ = "1.0.0"
__plugin_name__    = "Manual Dosing Control"
__plugin_author__  = "Siddhesh"
__plugin_homepage__ = "https://pioreactor.com"


# ─── Custom events ────────────────────────────────────────────────────────────

class ScheduledDoseEvent(events.AutomationEvent):
    pass

class NoDoseEvent(events.AutomationEvent):
    pass

class MaxCyclesReachedEvent(events.AutomationEvent):
    pass


# ─── Automation ───────────────────────────────────────────────────────────────

class ManualDosingControl(DosingAutomationJobContrib):
    """
    Scheduled dosing with explicit volume sequence or sine wave media volume modulation.

    Mode A (Explicit Sequence - Recommended):
      Provide volume_sequence="0.05, 0.10, 0.20, 0.25, 0.20, 0.10".
      Fires Cycle 0 immediately at t=0, then every `duration` minutes.

    Mode B (Mathematical Sine Wave):
      media_volume(cycle) = media_ml_mean
                          + media_ml_amplitude * sin(2π * cycle / sine_period_cycles)

    Set sine_period_cycles = 0 to use fixed media_ml_mean (no sine wave).
    """

    automation_name = "manual_dosing_control"

    published_settings = {
        "duration": {
            "datatype": "float",
            "settable": True,
            "unit": "min",
        },
        "volume_sequence": {
            "datatype": "string",
            "settable": True,
            "unit": "mL",
        },
        "media_ml_mean": {
            "datatype": "float",
            "settable": True,
            "unit": "mL",
        },
        "media_ml_amplitude": {
            "datatype": "float",
            "settable": True,
            "unit": "mL",
        },
        "sine_period_cycles": {
            "datatype": "float",
            "settable": True,
            "unit": "cycles",
        },
        "alt_media_ml": {
            "datatype": "float",
            "settable": True,
            "unit": "mL",
        },
        "waste_ml": {
            "datatype": "float",
            "settable": True,
            "unit": "mL",
        },
        "pump_sequence": {
            "datatype": "string",
            "settable": True,
            "unit": None,
        },
        "pause_between_pumps_s": {
            "datatype": "float",
            "settable": True,
            "unit": "s",
        },
        "max_cycles": {
            "datatype": "integer",
            "settable": True,
            "unit": "cycles",
        },
        "current_media_ml": {
            "datatype": "float",
            "settable": False,
            "unit": "mL",
        },
        "current_cycle": {
            "datatype": "integer",
            "settable": False,
            "unit": None,
        },
    }

    def __init__(
        self,
        duration: float | str = 40.0,
        volume_sequence: str = "0.05, 0.10, 0.20, 0.25, 0.20, 0.10",
        media_ml_mean: float = 0.15,
        media_ml_amplitude: float = 0.10,
        sine_period_cycles: float = 6.0,
        alt_media_ml: float = 0.0,
        waste_ml: float = 0.15,
        pump_sequence: str = "waste_first",
        pause_between_pumps_s: float = 0.0,
        max_cycles: int = 18,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        self.duration              = float(duration)
        self.volume_sequence       = str(volume_sequence).strip()
        self.media_ml_mean         = float(media_ml_mean)
        self.media_ml_amplitude    = float(media_ml_amplitude)
        self.sine_period_cycles    = float(sine_period_cycles)
        self.alt_media_ml          = float(alt_media_ml)
        self.waste_ml              = float(waste_ml)
        self.pump_sequence         = str(pump_sequence).strip().lower()
        self.pause_between_pumps_s = float(pause_between_pumps_s)
        self.max_cycles            = int(max_cycles)

        if self.duration <= 0:
            raise ValueError("duration must be greater than 0")

        self.current_cycle = 0
        self._timer: Timer | None = None

        # Parse explicit sequence if provided
        self.sequence = [
            float(x.strip())
            for x in self.volume_sequence.split(",")
            if x.strip()
        ]

        self.current_media_ml = self._calculate_media_volume()

        # Set up dedicated SQLite writer
        db_path = config.get("storage", "database")
        self._db = sqlite_worker.Sqlite3Worker(db_path)
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS sine_media_volumes (
                experiment       TEXT NOT NULL,
                pioreactor_unit  TEXT NOT NULL,
                timestamp        TEXT NOT NULL,
                volume_ml        REAL NOT NULL
            )
        """)

        if self.sequence:
            mode_desc = f"explicit sequence={self.sequence}"
        elif self.sine_period_cycles > 0:
            mode_desc = f"sine (mean={self.media_ml_mean}mL, amp={self.media_ml_amplitude}mL, period={self.sine_period_cycles}c)"
        else:
            mode_desc = f"fixed (media={self.media_ml_mean}mL)"

        self.logger.info(
            f"Manual Dosing Control v1.0.0 ready | "
            f"Schedule every {self.duration} min | "
            f"mode: {mode_desc} | "
            f"alt_media={self.alt_media_ml} mL, "
            f"waste={self.waste_ml} mL | "
            + (f"max_cycles={self.max_cycles}" if self.max_cycles > 0 else "max_cycles=unlimited")
        )

    # ── SQLite writer ─────────────────────────────────────────────────────────

    def _save_media_volume(self, volume_ml: float) -> None:
        """Write the current sine wave media volume to the dedicated table."""
        try:
            self._db.execute(
                """INSERT INTO sine_media_volumes
                   (experiment, pioreactor_unit, timestamp, volume_ml)
                   VALUES (?, ?, ?, ?)""",
                (self.experiment, self.unit, current_utc_timestamp(), volume_ml),
            )
        except Exception as e:
            self.logger.debug(f"Could not save sine media volume to DB: {e}")

    # ── State Machine & Timers ────────────────────────────────────────────────

    def execute(self):
        """Required by DosingAutomationJobContrib base class."""
        return None

    def on_ready(self) -> None:
        super().on_ready()
        self.logger.info("Manual Dosing Control started. Triggering Step 0...")
        self._run_step()

    def _run_step(self) -> None:
        if self.state != self.READY:
            return

        # Check if max_cycles reached
        if self.max_cycles > 0 and self.current_cycle >= self.max_cycles:
            self.logger.info(
                f"Completed all {self.max_cycles} configured cycle(s). Stopping automation."
            )
            self._stop_everything()
            self.set_state(self.DISCONNECTED)
            return

        # Calculate volume for this cycle
        media_volume = self._calculate_media_volume()
        self.current_media_ml = media_volume
        self._save_media_volume(media_volume)

        self.logger.info(
            f"Cycle {self.current_cycle}"
            + (f"/{self.max_cycles}" if self.max_cycles > 0 else "")
            + f" | media = {media_volume} mL"
        )

        # Determine pump order
        if self.pump_sequence == "waste_first":
            order = ["waste", "media", "alt_media"]
        else:
            order = ["media", "alt_media", "waste"]

        volumes = {
            "media":     media_volume,
            "alt_media": self.alt_media_ml,
            "waste":     self.waste_ml,
        }

        pumps_to_run = [p for p in order if volumes[p] > 0]

        for i, pump in enumerate(pumps_to_run):
            self._fire_pump(pump, volumes[pump])
            if i < len(pumps_to_run) - 1 and self.pause_between_pumps_s > 0:
                time.sleep(self.pause_between_pumps_s)

        self.current_cycle += 1

        # Check if another step should be scheduled
        if self.max_cycles > 0 and self.current_cycle >= self.max_cycles:
            self.logger.info(
                f"Completed all {self.max_cycles} cycle(s). Automation complete."
            )
            self._stop_everything()
            self.set_state(self.DISCONNECTED)
            return

        duration_s = self.duration * 60.0
        self.logger.info(
            f"Next dosing cycle in {self.duration} min ({duration_s:.0f}s)"
        )
        self._timer = Timer(duration_s, self._run_step)
        self._timer.daemon = True
        self._timer.start()

    def _stop_everything(self) -> None:
        """Cancel active timer immediately."""
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    # ── Volume calculator ─────────────────────────────────────────────────────

    def _calculate_media_volume(self) -> float:
        if self.sequence:
            return round(self.sequence[self.current_cycle % len(self.sequence)], 4)

        if self.sine_period_cycles <= 0:
            return self.media_ml_mean

        volume = (
            self.media_ml_mean
            + self.media_ml_amplitude
            * math.sin(2 * math.pi * self.current_cycle / self.sine_period_cycles)
        )
        return max(0.0, round(volume, 4))

    # ── Pump sequence setter ──────────────────────────────────────────────────

    def set_pump_sequence(self, value) -> None:
        value = str(value).strip().lower()
        if value not in ("waste_first", "media_first"):
            self.logger.warning(
                f"Invalid pump_sequence '{value}'. "
                f"Must be 'waste_first' or 'media_first'. "
                f"Keeping: {self.pump_sequence}"
            )
            return
        self.pump_sequence = value
        self.logger.info(f"Pump sequence changed to: {self.pump_sequence}")

    # ── Shared pump kwargs ────────────────────────────────────────────────────

    @property
    def _pump_kwargs(self) -> dict:
        return dict(
            unit=self.unit,
            experiment=self.experiment,
            source_of_event=f"{self.job_name}:{self.automation_name}",
            mqtt_client=self.pub_client,
            logger=self.logger,
        )

    # ── Individual pump firing ────────────────────────────────────────────────

    def _fire_pump(self, pump: str, volume_ml: float) -> bool:
        if volume_ml <= 0:
            self.logger.debug(f"[scheduled] {pump} skipped (volume = {volume_ml} mL)")
            return False
        self.logger.info(f"[scheduled] {pump} pump → {volume_ml} mL")
        if pump == "media":
            add_media(ml=volume_ml, **self._pump_kwargs)
        elif pump == "alt_media":
            add_alt_media(ml=volume_ml, **self._pump_kwargs)
        elif pump == "waste":
            remove_waste(ml=volume_ml, **self._pump_kwargs)
        return True

    # ── Lifecycle Handlers ────────────────────────────────────────────────────

    def on_sleeping(self) -> None:
        super().on_sleeping()
        self.logger.info("Manual Dosing Control sleeping. Pausing timer.")
        self._stop_everything()

    def on_disconnected(self) -> None:
        self.logger.info("Manual Dosing Control disconnected.")
        self._stop_everything()
        try:
            self._db.close()
        except Exception:
            pass
        super().on_disconnected()