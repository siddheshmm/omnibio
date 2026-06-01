# -*- coding: utf-8 -*-
"""
manual_dosing_control.py  — v0.6.0
────────────────────────────────────
Independent scheduled control of media, alt-media, and waste pumps,
with sine wave modulation of media volume.

Changes in v0.6.0:
  - Added max_cycles parameter: automation stops automatically after
    max_cycles cycles. Set to 0 (default) to run indefinitely.
"""
from __future__ import annotations
import time
import math

from pioreactor.automations.dosing.base import DosingAutomationJobContrib
from pioreactor.automations import events
from pioreactor.actions.pump import add_media, add_alt_media, remove_waste
from pioreactor.utils import sqlite_worker
from pioreactor.utils.timing import current_utc_timestamp
from pioreactor.config import config


__plugin_summary__ = "Scheduled dosing with sine wave modulation and optional auto-stop"
__plugin_version__ = "0.6.0"
__plugin_name__    = "Manual Dosing Control"
__plugin_author__  = "Your Name"
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
    Scheduled dosing with sine wave media volume modulation.

    media_volume(cycle) = media_ml_mean
                        + media_ml_amplitude * sin(2π * cycle / sine_period_cycles)

    Set sine_period_cycles = 0 to use fixed media_ml_mean (no sine wave).

    pump_sequence:
      "waste_first"  → waste → media → alt-media
      "media_first"  → media → alt-media → waste

    pause_between_pumps_s:
      Seconds to wait between each pump firing within a cycle.

    max_cycles:
      Stop automatically after this many cycles.
      Set to 0 to run indefinitely (default).
    """

    automation_name = "manual_dosing_control"

    published_settings = {
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
    }

    def __init__(
        self,
        media_ml_mean: float = 1.0,
        media_ml_amplitude: float = 0.0,
        sine_period_cycles: float = 0.0,
        alt_media_ml: float = 0.0,
        waste_ml: float = 0.0,
        pump_sequence: str = "waste_first",
        pause_between_pumps_s: float = 0.0,
        max_cycles: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.media_ml_mean         = float(media_ml_mean)
        self.media_ml_amplitude    = float(media_ml_amplitude)
        self.sine_period_cycles    = float(sine_period_cycles)
        self.alt_media_ml          = float(alt_media_ml)
        self.waste_ml              = float(waste_ml)
        self.pump_sequence         = str(pump_sequence).strip().lower()
        self.pause_between_pumps_s = float(pause_between_pumps_s)
        self.max_cycles            = int(max_cycles)
        self._cycle_count          = 0
        self.current_media_ml      = self.media_ml_mean

        # ── Set up dedicated SQLite writer ────────────────────────────────
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

        self.logger.info(
            f"Manual Dosing Control v0.6.0 ready | "
            f"Schedule every {self.duration} min | "
            f"sequence={self.pump_sequence} | "
            f"media_mean={self.media_ml_mean} mL, "
            f"amplitude={self.media_ml_amplitude} mL, "
            f"sine_period={self.sine_period_cycles} cycles | "
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

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def on_disconnected(self) -> None:
        try:
            self._db.close()
        except Exception:
            pass

    # ── Sine wave calculator ──────────────────────────────────────────────────

    def _calculate_media_volume(self) -> float:
        if self.sine_period_cycles <= 0:
            return self.media_ml_mean
        volume = (
            self.media_ml_mean
            + self.media_ml_amplitude
            * math.sin(2 * math.pi * self._cycle_count / self.sine_period_cycles)
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

    # ── Scheduled execute ─────────────────────────────────────────────────────

    def execute(self) -> events.AutomationEvent:

        # ── Check max_cycles before doing anything ────────────────────────
        if self.max_cycles > 0 and self._cycle_count >= self.max_cycles:
            self.logger.info(
                f"Reached max_cycles={self.max_cycles}. "
                f"Stopping dosing automation."
            )
            self.set_state(self.DISCONNECTED)
            return MaxCyclesReachedEvent(
                f"Stopped after {self.max_cycles} cycles as configured.",
                {"max_cycles": self.max_cycles, "total_cycles_run": self._cycle_count},
            )

        # ── Calculate sine wave volume for this cycle ─────────────────────
        media_volume = self._calculate_media_volume()

        # Publish to MQTT (live chart)
        self.current_media_ml = media_volume

        # Save to dedicated SQLite table (historical chart)
        self._save_media_volume(media_volume)

        self.logger.info(
            f"Cycle {self._cycle_count}"
            + (f"/{self.max_cycles}" if self.max_cycles > 0 else "")
            + f" | media = {media_volume} mL"
            + (" (sine)" if self.sine_period_cycles > 0 else " (fixed)")
        )

        # ── Determine pump order ──────────────────────────────────────────
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

        if not pumps_to_run:
            self._cycle_count += 1
            return NoDoseEvent("All volumes are 0 — nothing dosed this cycle.")

        fired = []
        for i, pump in enumerate(pumps_to_run):
            self._fire_pump(pump, volumes[pump])
            fired.append(f"{pump}={volumes[pump]} mL")
            if i < len(pumps_to_run) - 1 and self.pause_between_pumps_s > 0:
                self.logger.info(
                    f"Waiting {self.pause_between_pumps_s}s before next pump..."
                )
                time.sleep(self.pause_between_pumps_s)

        self._cycle_count += 1
        summary = ", ".join(fired)
        return ScheduledDoseEvent(
            f"Cycle {self._cycle_count - 1}"
            + (f"/{self.max_cycles}" if self.max_cycles > 0 else "")
            + f" ({self.pump_sequence}): {summary}",
            {
                "media_ml":              media_volume,
                "alt_media_ml":          self.alt_media_ml,
                "waste_ml":              self.waste_ml,
                "pump_sequence":         self.pump_sequence,
                "sine_period_cycles":    self.sine_period_cycles,
                "cycle_count":           self._cycle_count - 1,
                "max_cycles":            self.max_cycles,
            },
        )