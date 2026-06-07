# -*- coding: utf-8 -*-
"""
Record ESP32/SCD4x CO2 readings published to Pioreactor MQTT.

ESP32 publishes scalar values to:
    pioreactor/<unit>/<experiment>/co2_reading/co2
    pioreactor/<unit>/<experiment>/co2_reading/temperature
    pioreactor/<unit>/<experiment>/co2_reading/relative_humidity

This job stores readings in SQLite table `co2_readings`, which the chart YAML
files use for historical data. It uses the existing Pioreactor-side CO2 column
name `co2_reading_ppm`.

Run on the Pioreactor:
    pio run co2_recorder
"""
from __future__ import annotations

import time
import sqlite3
from dataclasses import dataclass

import click

from pioreactor.background_jobs.base import BackgroundJob
from pioreactor.config import config
from pioreactor.pubsub import create_client
from pioreactor.utils import sqlite_worker
from pioreactor.utils.timing import current_utc_timestamp
from pioreactor.whoami import get_latest_experiment_name
from pioreactor.whoami import get_unit_name


__plugin_name__ = "CO2 Recorder"
__plugin_summary__ = "Record ESP32 SCD4x CO2 readings from MQTT"
__plugin_version__ = "0.1.0"
__plugin_author__ = "Siddhesh"


@dataclass
class PendingReading:
    co2_reading_ppm: float | None = None
    temperature_c: float | None = None
    relative_humidity: float | None = None


class CO2Recorder(BackgroundJob):
    job_name = "co2_recorder"

    published_settings = {
        "latest_co2_ppm": {"datatype": "float", "settable": False, "unit": "ppm"},
        "latest_temperature_c": {"datatype": "float", "settable": False, "unit": "C"},
        "latest_relative_humidity": {"datatype": "float", "settable": False, "unit": "%"},
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.latest_co2_ppm = 0.0
        self.latest_temperature_c = 0.0
        self.latest_relative_humidity = 0.0

        self._pending = PendingReading()
        self._db = sqlite_worker.Sqlite3Worker(config.get("storage", "database"))
        self._setup_db()

        self._client = create_client(
            hostname=config.get("mqtt", "broker_address"),
            client_id=f"{self.unit}-{self.job_name}",
        )
        self._client.on_message = self._on_message

        topic = f"pioreactor/{self.unit}/{self.experiment}/co2_reading/+"
        self._client.subscribe(topic)
        self._client.loop_start()

        self.logger.info("CO2 recorder subscribed to %s", topic)

    def _setup_db(self):
        db_path = config.get("storage", "database")
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS co2_readings (
                    experiment         TEXT NOT NULL,
                    pioreactor_unit    TEXT NOT NULL,
                    timestamp          TEXT NOT NULL,
                    co2_reading_ppm    REAL
                )
                """
            )

            existing = {
                row[1]
                for row in conn.execute("PRAGMA table_info(co2_readings)").fetchall()
            }
            if "temperature_c" not in existing:
                conn.execute("ALTER TABLE co2_readings ADD COLUMN temperature_c REAL")
            if "relative_humidity" not in existing:
                conn.execute("ALTER TABLE co2_readings ADD COLUMN relative_humidity REAL")
            conn.commit()

    def _ensure_column(self, column_name: str, column_type: str):
        try:
            self._db.execute(
                f"ALTER TABLE co2_readings ADD COLUMN {column_name} {column_type}"
            )
        except Exception as exc:
            if "duplicate column" not in str(exc).lower():
                raise

    def _on_message(self, client, userdata, message):
        try:
            field = message.topic.split("/")[-1]
            value = float(message.payload.decode("utf-8"))
        except Exception as exc:
            self.logger.debug("Could not parse CO2 MQTT message: %s", exc)
            return

        if field == "co2":
            self._pending.co2_reading_ppm = value
            self.latest_co2_ppm = value
        elif field == "temperature":
            self._pending.temperature_c = value
            self.latest_temperature_c = value
        elif field == "relative_humidity":
            self._pending.relative_humidity = value
            self.latest_relative_humidity = value
        else:
            return

        if (
            self._pending.co2_reading_ppm is not None
            and self._pending.temperature_c is not None
            and self._pending.relative_humidity is not None
        ):
            self._save_pending()
            self._pending = PendingReading()

    def _save_pending(self):
        self._db.execute(
            """
            INSERT INTO co2_readings
            (experiment, pioreactor_unit, timestamp, co2_reading_ppm, temperature_c, relative_humidity)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                self.experiment,
                self.unit,
                current_utc_timestamp(),
                self._pending.co2_reading_ppm,
                self._pending.temperature_c,
                self._pending.relative_humidity,
            ),
        )

    def block_until_disconnected(self):
        try:
            while self.state != self.DISCONNECTED:
                time.sleep(1)
        except KeyboardInterrupt:
            self.set_state(self.DISCONNECTED)

    def on_disconnected(self):
        try:
            self._client.loop_stop()
            self._client.disconnect()
        except Exception:
            pass

        try:
            self._db.close()
        except Exception:
            pass

        return super().on_disconnected()


@click.command(name="co2_recorder")
@click.option("--unit", default=None, help="Pioreactor unit name. Defaults to this unit.")
@click.option("--experiment", default=None, help="Experiment name. Defaults to latest experiment.")
def click_co2_recorder(unit, experiment):
    unit = unit or get_unit_name()
    experiment = experiment or get_latest_experiment_name()

    with CO2Recorder(unit=unit, experiment=experiment) as job:
        job.block_until_disconnected()
