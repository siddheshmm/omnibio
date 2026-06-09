# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import sqlite3
import time

import smbus2

from pioreactor.background_jobs.base import BackgroundJobContrib
from pioreactor.background_jobs.leader.mqtt_to_db_streaming import produce_metadata
from pioreactor.background_jobs.leader.mqtt_to_db_streaming import register_source_to_sink
from pioreactor.background_jobs.leader.mqtt_to_db_streaming import TopicToParserToTable
from pioreactor.cli.run import run
from pioreactor.config import config
from pioreactor.utils import timing
from pioreactor.whoami import get_assigned_experiment_name
from pioreactor.whoami import get_unit_name

__plugin_name__ = "pH Reader"
__plugin_summary__ = "Record pH readings from the pH sensor board via ADS1115 (analog-to-digital converter)"
__plugin_version__ = "0.1.0"
__plugin_author__ = "Siddhesh"

def parser(topic, payload) -> dict:
    metadata = produce_metadata(topic)
    data = json.loads(payload)
    return {
        "experiment": metadata.experiment,
        "pioreactor_unit": metadata.pioreactor_unit,
        "timestamp": timing.current_utc_timestamp(),
        "ph_reading": float(data["ph"]),
        "voltage": float(data["voltage"]),
    }


register_source_to_sink(
    TopicToParserToTable(
        "pioreactor/+/+/ph_reading/reading",
        parser,
        "ph_readings",
    )
)


class PHReading(BackgroundJobContrib):

    job_name = "ph_reading"

    published_settings = {
        "interval": {"datatype": "float", "unit": "s", "settable": True},
        "ph":       {"datatype": "float", "unit": "pH", "settable": False},
        "voltage":  {"datatype": "float", "unit": "V",  "settable": False},
    }

    def __init__(self, unit: str, experiment: str) -> None:
        super().__init__(unit=unit, experiment=experiment, plugin_name="ph_reading")

        self.interval = config.getfloat("ph_reading.config", "interval", fallback=30.0)
        self.voltage_at_ph7 = config.getfloat("ph_reading.config", "voltage_at_ph7", fallback=2.5)
        self.slope = config.getfloat("ph_reading.config", "slope", fallback=0.1816)
        self.i2c_bus = config.getint("ph_reading.config", "i2c_bus", fallback=3)
        self.i2c_address = 0x48

        self._setup_db()
        self._setup_adc()

        self.read_timer = timing.RepeatedTimer(
            self.interval,
            self.read_and_publish,
            run_immediately=True,
        )
        self.read_timer.start()

    def _setup_db(self) -> None:
        db_path = config.get("storage", "database")
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ph_readings (
                    experiment      TEXT NOT NULL,
                    pioreactor_unit TEXT NOT NULL,
                    timestamp       TEXT NOT NULL,
                    ph_reading      REAL NOT NULL,
                    voltage         REAL NOT NULL,
                    FOREIGN KEY (experiment) REFERENCES experiments (experiment) ON DELETE CASCADE
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS ph_readings_ix
                ON ph_readings (experiment, pioreactor_unit, timestamp)
                """
            )
            conn.commit()

    def _setup_adc(self) -> None:
        self.bus = smbus2.SMBus(self.i2c_bus)

    def _read_voltage(self) -> float:
        self.bus.write_i2c_block_data(self.i2c_address, 1, [0xC1, 0x83])
        time.sleep(0.2)
        data = self.bus.read_i2c_block_data(self.i2c_address, 0, 2)
        raw = (data[0] << 8) | data[1]
        if raw > 32767:
            raw -= 65536
        return raw * 4.096 / 32767

    def _voltage_to_ph(self, voltage: float) -> float:
        return 7.0 + (self.voltage_at_ph7 - voltage) / self.slope

    def read_and_publish(self) -> None:
        try:
            voltage = self._read_voltage()
            ph = self._voltage_to_ph(voltage)
            self.voltage = round(voltage, 4)
            self.ph = round(ph, 3)
            self.publish(
                f"pioreactor/{self.unit}/{self.experiment}/ph_reading/reading",
                json.dumps({"ph": self.ph, "voltage": self.voltage}),
            )
            self.logger.debug(f"pH={self.ph}, V={self.voltage}")
        except Exception as e:
            self.logger.error(f"pH read error: {e}")

    def set_interval(self, new_interval: float) -> None:
        self.read_timer.interval = new_interval
        self.interval = new_interval

    def on_sleeping(self) -> None:
        self.read_timer.pause()

    def on_sleeping_to_ready(self) -> None:
        self.read_timer.unpause()

    def on_disconnected(self) -> None:
        self.read_timer.cancel()
        self.bus.close()


@run.command(name="ph_reading")
def start_ph_reading() -> None:
    """
    Read pH from ADS1115 on software I2C bus and publish to MQTT.
    """
    unit = get_unit_name()
    job = PHReading(
        unit=unit,
        experiment=get_assigned_experiment_name(unit),
    )
    job.block_until_disconnected()