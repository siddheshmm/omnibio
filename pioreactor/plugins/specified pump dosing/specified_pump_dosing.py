# -*- coding: utf-8 -*-
"""
specified_pump_dosing.py — v1.1.0
──────────────────────────────────
Flexible single-pump sequence dosing plugin for the Pioreactor.

Allows running a sequence of doses using any specified pump:
  - 'media'     (Nutrient / Media pump)
  - 'alt_media' (Alternative Media / Salt / Inducer pump)
  - 'waste'     (Effluent / Waste pump)

Key Features:
  1. Complete Pump Isolation: Only actuates the specified pump. The other two pumps
     (e.g., continuous media & waste pumps) are NEVER touched or stopped.
  2. Standalone Background Job Architecture: Inherits from BackgroundJobContrib
     rather than DosingAutomationJobContrib. This ensures that when the sequence
     finishes or disconnects, Pioreactor's Dosing Automation manager does NOT
     trigger an emergency global pump shutdown.
  3. Clean Shutdown: Cancels only its own cycle timer and closes its own resources.
     Does NOT touch the hardware PWM controller directly, keeping all other
     continuous pump channels running smoothly.
"""
from __future__ import annotations

import re
import time
from threading import Timer
from typing import Any

import click

from pioreactor.actions.pump import add_alt_media
from pioreactor.actions.pump import add_media
from pioreactor.actions.pump import remove_waste
from pioreactor.background_jobs.base import BackgroundJobContrib
from pioreactor.cli.run import run
from pioreactor.config import config
from pioreactor.utils import sqlite_worker
from pioreactor.utils.timing import current_utc_timestamp
from pioreactor.whoami import get_assigned_experiment_name
from pioreactor.whoami import get_latest_experiment_name
from pioreactor.whoami import get_unit_name


__plugin_name__ = "Specified Pump Dosing"
__plugin_summary__ = "Execute a sequence of doses using any specified pump without touching other pumps"
__plugin_version__ = "1.1.0"
__plugin_author__ = "Siddhesh"
__plugin_homepage__ = "https://pioreactor.com"


class SpecifiedPumpDosing(BackgroundJobContrib):
    job_name = "specified_pump_dosing"

    published_settings = {
        "pump": {
            "datatype": "string",
            "settable": True,
            "unit": None,
        },
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
        "default_volume_ml": {
            "datatype": "float",
            "settable": True,
            "unit": "mL",
        },
        "max_cycles": {
            "datatype": "integer",
            "settable": True,
            "unit": "cycles",
        },
        "current_dose_ml": {
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
        unit: str,
        experiment: str,
        pump: str = "media",
        duration: float | str = 20.0,
        volume_sequence: str = "",
        default_volume_ml: float | str = 0.10,
        max_cycles: int | str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            unit=unit,
            experiment=experiment,
            plugin_name="specified_pump_dosing",
            **kwargs,
        )

        self.pump = str(pump).strip().lower()
        if self.pump not in ("media", "alt_media", "waste"):
            self.logger.warning(
                f"Unknown pump '{self.pump}', falling back to 'media'. "
                "Valid options: 'media', 'alt_media', 'waste'."
            )
            self.pump = "media"

        try:
            self.duration = float(duration)
        except (ValueError, TypeError):
            self.duration = 20.0
        if self.duration <= 0:
            raise ValueError("duration must be greater than 0")

        self.volume_sequence = str(volume_sequence).strip() if volume_sequence is not None else ""
        try:
            self.default_volume_ml = max(0.0, float(default_volume_ml))
        except (ValueError, TypeError):
            self.default_volume_ml = 0.10
        self.sequence = self._parse_sequence_string(self.volume_sequence)

        # Safely parse max_cycles: if provided (>0), use it; if None/empty, default to sequence length
        try:
            if max_cycles is not None and str(max_cycles).strip() != "":
                self.max_cycles = int(float(str(max_cycles).strip()))
            elif self.sequence:
                self.max_cycles = len(self.sequence)
            else:
                self.max_cycles = 0
        except (ValueError, TypeError):
            self.max_cycles = len(self.sequence) if self.sequence else 0

        self.current_cycle = 0
        self.current_dose_ml = self._calculate_dose_volume(0)
        self._timer: Timer | None = None

        # Dedicated SQLite table for logging doses
        db_path = config.get("storage", "database")
        self._db = sqlite_worker.Sqlite3Worker(db_path)
        self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS specified_pump_dosing_records (
                experiment       TEXT NOT NULL,
                pioreactor_unit  TEXT NOT NULL,
                timestamp        TEXT NOT NULL,
                cycle            INTEGER NOT NULL,
                pump             TEXT NOT NULL,
                volume_ml        REAL NOT NULL
            )
            """
        )

        seq_info = f"{len(self.sequence)} steps" if self.sequence else f"fixed {self.default_volume_ml} mL"
        cycle_info = f"max_cycles={self.max_cycles}" if self.max_cycles > 0 else "max_cycles=unlimited"
        self.logger.info(
            f"Specified Pump Dosing initialized | Target Pump='{self.pump}' | "
            f"Sequence: {seq_info} | Interval: {self.duration} min | {cycle_info}"
        )

    def _parse_sequence_string(self, raw_str: str) -> list[float]:
        """Extract floating point numbers from formatted strings, line-breaks, or commas."""
        if not raw_str:
            return []
        try:
            cleaned = str(raw_str).replace("\n", " ").replace("\r", " ").replace(";", ",")
            tokens = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", cleaned)
            return [max(0.0, float(t)) for t in tokens]
        except Exception as e:
            self.logger.warning(f"Could not parse sequence string: {e}")
            return []

    def _calculate_dose_volume(self, cycle_idx: int | None = None) -> float:
        """Get the dose volume for the current or specified cycle index."""
        c = self.current_cycle if cycle_idx is None else cycle_idx
        if self.sequence:
            return round(self.sequence[c % len(self.sequence)], 4)
        return round(self.default_volume_ml, 4)

    def _save_record(self, volume_ml: float) -> None:
        """Record dosing event into local SQLite database."""
        try:
            self._db.execute(
                """
                INSERT INTO specified_pump_dosing_records
                (experiment, pioreactor_unit, timestamp, cycle, pump, volume_ml)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    self.experiment,
                    self.unit,
                    current_utc_timestamp(),
                    self.current_cycle,
                    self.pump,
                    volume_ml,
                ),
            )
        except Exception as e:
            self.logger.debug(f"Could not save dosing record to DB: {e}")

    def on_ready(self) -> None:
        super().on_ready()
        self.logger.info(f"Specified Pump Dosing ready. Initiating cycle 0 on pump '{self.pump}'...")
        self._run_step()

    def _run_step(self) -> None:
        if self.state != self.READY:
            return

        # Check if max_cycles has already been reached
        if self.max_cycles > 0 and self.current_cycle >= self.max_cycles:
            self.logger.info(
                f"Completed all {self.max_cycles} cycle(s) on pump '{self.pump}'. "
                "Stopping job without touching other pumps."
            )
            self._stop_timer()
            self.set_state(self.DISCONNECTED)
            return

        dose_vol = self._calculate_dose_volume()
        self.current_dose_ml = dose_vol
        self._save_record(dose_vol)

        cycle_str = f"{self.current_cycle}" + (f"/{self.max_cycles}" if self.max_cycles > 0 else "")
        self.logger.info(f"Cycle {cycle_str} | Dosing {dose_vol:.4f} mL on '{self.pump}' pump only.")

        # Actuate ONLY the specified pump
        if dose_vol > 0:
            self._fire_specified_pump(dose_vol)

        self.current_cycle += 1

        # Check if max_cycles reached after completing current dose
        if self.max_cycles > 0 and self.current_cycle >= self.max_cycles:
            self.logger.info(
                f"Finished all {self.max_cycles} cycle(s) on pump '{self.pump}'. "
                "Shutting down specified pump job."
            )
            self._stop_timer()
            self.set_state(self.DISCONNECTED)
            return

        # Schedule next dosing cycle
        duration_s = self.duration * 60.0
        self.logger.info(f"Next dose on '{self.pump}' in {self.duration} min ({duration_s:.0f}s).")
        self._timer = Timer(duration_s, self._run_step)
        self._timer.daemon = True
        self._timer.start()

    def _fire_specified_pump(self, volume_ml: float) -> bool:
        """Actuates ONLY the specified pump. Leaves all other pumps completely untouched."""
        kwargs = dict(
            unit=self.unit,
            experiment=self.experiment,
            source_of_event=f"{self.job_name}:{self.pump}",
            mqtt_client=self.pub_client,
            logger=self.logger,
        )

        try:
            if self.pump == "media":
                add_media(ml=volume_ml, **kwargs)
            elif self.pump == "alt_media":
                add_alt_media(ml=volume_ml, **kwargs)
            elif self.pump == "waste":
                remove_waste(ml=volume_ml, **kwargs)
            return True
        except Exception as e:
            self.logger.error(f"Failed to dose {volume_ml} mL on pump '{self.pump}': {e}")
            return False

    def _stop_timer(self) -> None:
        """Cancel active cycle timer."""
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def on_sleeping(self) -> None:
        self.logger.info(f"Specified Pump Dosing paused/sleeping for pump '{self.pump}'.")
        self._stop_timer()
        super().on_sleeping()

    def on_disconnected(self) -> None:
        self.logger.info(f"Specified Pump Dosing disconnected. Only pump '{self.pump}' was managed.")
        self._stop_timer()
        try:
            self._db.close()
        except Exception:
            pass
        super().on_disconnected()


# ── Standalone CLI Command ───────────────────────────────────────────────────

@run.command(name="specified_pump_dosing", help=__plugin_summary__)
@click.option(
    "--pump",
    default="media",
    type=click.Choice(["media", "alt_media", "waste"], case_sensitive=False),
    help="Target pump to run ('media', 'alt_media', or 'waste')",
)
@click.option(
    "--duration",
    default=20.0,
    type=float,
    help="Minutes between dosing steps (schedule interval)",
)
@click.option(
    "--volume-sequence",
    default="",
    type=str,
    help="Comma-separated volume sequence in mL (e.g. '0.05, 0.10, 0.20')",
)
@click.option(
    "--default-volume-ml",
    default=0.10,
    type=float,
    help="Fixed volume in mL if volume_sequence is empty",
)
@click.option(
    "--max-cycles",
    default=None,
    type=int,
    help="Number of cycles to run (defaults to sequence length if provided, or 0 for unlimited)",
)
@click.option("--unit", default=None, type=str, help="Target Pioreactor unit name")
@click.option("--experiment", default=None, type=str, help="Experiment name")
def click_specified_pump_dosing(
    pump: str,
    duration: float,
    volume_sequence: str,
    default_volume_ml: float,
    max_cycles: int | None,
    unit: str | None,
    experiment: str | None,
) -> None:
    """Run specified pump dosing as an independent background job from CLI."""
    unit = unit or get_unit_name()
    experiment = experiment or get_assigned_experiment_name(unit) or get_latest_experiment_name()

    with SpecifiedPumpDosing(
        unit=unit,
        experiment=experiment,
        pump=pump,
        duration=duration,
        volume_sequence=volume_sequence,
        default_volume_ml=default_volume_ml,
        max_cycles=max_cycles,
    ) as job:
        job.block_until_disconnected()
