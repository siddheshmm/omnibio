# -*- coding: utf-8 -*-
"""
sine_led_automation.py  — v1.1.0
────────────────────────────────────
Scheduled sinusoidal and stepped intensity LED automation for Pioreactor.

Features:
  - Full Web UI & CLI compatibility:
      • Channel mapping supports integers (1=A, 2=B, 3=C, 4=D) and letters ('A', 'B', 'C', 'D').
      • Default fallback sequence: auto-loads "0, 10, 25, 35, 25, 10, 0" (3-hour cycle)
        even if UI numeric input sends 0.
  - Dedicated SQLite table logging (`sine_led_intensities`)
  - Live MQTT publishing for UI monitoring
  - Automatic turn-off (0.0%) on disconnect or stop
"""
from __future__ import annotations

from threading import Timer
from typing import Any

from pioreactor.automations.led.base import LEDAutomationJob
from pioreactor.actions.led_intensity import led_intensity
from pioreactor.automations import events
from pioreactor.config import config
from pioreactor.utils import sqlite_worker
from pioreactor.utils.timing import current_utc_timestamp


__plugin_name__    = "Sine LED Automation"
__plugin_summary__ = "Scheduled sine wave and stepped intensity LED automation"
__plugin_version__ = "1.1.0"
__plugin_author__  = "Siddhesh"
__plugin_homepage__ = "https://pioreactor.com"

# Standard 3-hour sine wave default: 0% -> 10% -> 25% -> 35% -> 25% -> 10% -> 0%
DEFAULT_INTENSITY_SEQUENCE = [0.0, 10.0, 25.0, 35.0, 25.0, 10.0, 0.0]

CHANNEL_MAP = {
    "1": "A", "2": "B", "3": "C", "4": "D",
    "A": "A", "B": "B", "C": "C", "D": "D",
    1: "A", 2: "B", 3: "C", 4: "D",
}


class SineLedAutomation(LEDAutomationJob):
    """
    Scheduled sine wave / stepped intensity LED automation.

    intensity_sequence:
      Comma-separated list of intensity percentages (e.g. "0, 10, 25, 35, 25, 10, 0").
      If 0 or blank is passed from UI, the default 3-hour sine sequence is automatically used.

    minutes_per_step:
      Duration in minutes to hold each intensity level before advancing to the next.

    led_channel:
      LED channel. Accepts 'A', 'B', 'C', 'D' or integers 1=A, 2=B (default UV), 3=C, 4=D.

    repeat_cycles:
      Number of times to repeat the full sequence. Set to 0 to run indefinitely.
    """

    automation_name = "sine_led_automation"

    published_settings = {
        "led_channel": {
            "datatype": "string",
            "settable": True,
            "unit": None,
        },
        "intensity_sequence": {
            "datatype": "string",
            "settable": True,
            "unit": None,
        },
        "minutes_per_step": {
            "datatype": "float",
            "settable": True,
            "unit": "min",
        },
        "current_intensity": {
            "datatype": "float",
            "settable": False,
            "unit": "%",
        },
        "current_step": {
            "datatype": "integer",
            "settable": False,
            "unit": None,
        },
        "repeat_cycles": {
            "datatype": "integer",
            "settable": True,
            "unit": "cycles",
        },
    }

    def __init__(
        self,
        led_channel: str | int = 2,
        intensity_sequence: str | float | int = "0, 10, 25, 35, 25, 10, 0",
        minutes_per_step: float | str = 30.0,
        repeat_cycles: int | str = 0,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        self.led_channel = self._parse_channel(led_channel)
        self.minutes_per_step = float(minutes_per_step)
        self.repeat_cycles = int(repeat_cycles)

        self.sequence = self._parse_sequence(intensity_sequence)
        self.intensity_sequence = ", ".join(f"{x:g}" for x in self.sequence)

        if self.minutes_per_step <= 0:
            raise ValueError("minutes_per_step must be > 0")

        self.current_step = 0
        self._total_steps_executed = 0
        self.current_intensity = 0.0
        self._timer = None

        # Set up dedicated SQLite writer table
        db_path = config.get("storage", "database")
        self._db = sqlite_worker.Sqlite3Worker(db_path)
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS sine_led_intensities (
                experiment       TEXT NOT NULL,
                pioreactor_unit  TEXT NOT NULL,
                timestamp        TEXT NOT NULL,
                led_channel      TEXT NOT NULL,
                intensity_pct    REAL NOT NULL
            )
        """)

        self.logger.info(
            f"Sine LED Automation ready on Channel {self.led_channel} | "
            f"sequence={self.sequence} | "
            f"step_duration={self.minutes_per_step} min | "
            + (f"repeat_cycles={self.repeat_cycles}" if self.repeat_cycles > 0 else "repeat_cycles=unlimited")
        )

    # ── Input Parsing Helpers ─────────────────────────────────────────────────

    def _parse_channel(self, raw: Any) -> str:
        raw_key = str(raw).strip().upper()
        if raw_key in CHANNEL_MAP:
            return CHANNEL_MAP[raw_key]
        if raw in CHANNEL_MAP:
            return CHANNEL_MAP[raw]
        self.logger.warning(f"Unknown channel '{raw}'. Defaulting to Channel 'B' (UV LED).")
        return "B"

    def _parse_sequence(self, raw: Any) -> list[float]:
        if raw is None:
            return list(DEFAULT_INTENSITY_SEQUENCE)
        if isinstance(raw, (int, float)):
            # If UI sends a single 0, load default 3-hour sine wave
            if float(raw) == 0:
                return list(DEFAULT_INTENSITY_SEQUENCE)
            return [float(raw)]

        raw_str = str(raw).strip()
        if not raw_str or raw_str in ("0", "default", "sine"):
            return list(DEFAULT_INTENSITY_SEQUENCE)

        try:
            parsed = [
                float(x.strip())
                for x in raw_str.replace(";", ",").split(",")
                if x.strip()
            ]
            if not parsed or (len(parsed) == 1 and parsed[0] == 0.0):
                return list(DEFAULT_INTENSITY_SEQUENCE)
            return parsed
        except Exception:
            self.logger.warning(
                f"Could not parse intensity_sequence '{raw_str}'. Using default: {DEFAULT_INTENSITY_SEQUENCE}"
            )
            return list(DEFAULT_INTENSITY_SEQUENCE)

    # ── Database Logging ──────────────────────────────────────────────────────

    def _save_led_intensity(self, intensity: float) -> None:
        """Record the active LED intensity to the dedicated SQLite table."""
        try:
            self._db.execute(
                """INSERT INTO sine_led_intensities
                   (experiment, pioreactor_unit, timestamp, led_channel, intensity_pct)
                   VALUES (?, ?, ?, ?, ?)""",
                (self.experiment, self.unit, current_utc_timestamp(), self.led_channel, intensity),
            )
        except Exception as e:
            self.logger.debug(f"Could not save LED intensity to DB: {e}")

    # ── LED Control ───────────────────────────────────────────────────────────

    def _set_led(self, intensity: float) -> None:
        """Set LED intensity and log state."""
        self.current_intensity = intensity
        self._save_led_intensity(intensity)
        self.logger.info(
            f"LED Channel {self.led_channel} set to {intensity:.1f}% "
            f"(Step {self.current_step + 1}/{len(self.sequence)})"
        )
        led_intensity(
            {self.led_channel: intensity},
            unit=self.unit,
            experiment=self.experiment,
        )

    # ── State Machine & Timers ────────────────────────────────────────────────

    def execute(self):
        """Required by LEDAutomationJob base class."""
        return None

    def on_ready(self) -> None:
        super().on_ready()
        self.logger.info("Sine LED Automation started.")
        self._run_step()

    def _run_step(self) -> None:
        if self.state != self.READY:
            return

        total_sequence_len = len(self.sequence)
        completed_cycles = self._total_steps_executed // total_sequence_len

        # Check if max repeat cycles reached
        if self.repeat_cycles > 0 and completed_cycles >= self.repeat_cycles:
            self.logger.info(
                f"Completed all {self.repeat_cycles} full sine cycle(s). Stopping automation."
            )
            self._stop_everything()
            self.set_state(self.DISCONNECTED)
            return

        target_intensity = self.sequence[self.current_step % total_sequence_len]
        self._set_led(target_intensity)

        self.current_step = (self.current_step + 1) % total_sequence_len
        self._total_steps_executed += 1

        # Schedule next transition
        duration_s = self.minutes_per_step * 60.0
        self.logger.info(
            f"Holding {target_intensity:.1f}% for {self.minutes_per_step} min "
            f"(next update in {duration_s:.0f}s)"
        )
        self._timer = Timer(duration_s, self._run_step)
        self._timer.daemon = True
        self._timer.start()

    def _stop_everything(self) -> None:
        """Cancel active timer and turn off LED immediately."""
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        self._set_led(0.0)

    # ── Lifecycle Handlers ────────────────────────────────────────────────────

    def on_sleeping(self) -> None:
        super().on_sleeping()
        self.logger.info("Sine LED Automation sleeping. Turning off LED.")
        self._stop_everything()

    def on_disconnected(self) -> None:
        self.logger.info("Sine LED Automation disconnected. Turning off LED.")
        self._stop_everything()
        try:
            self._db.close()
        except Exception:
            pass
        super().on_disconnected()
