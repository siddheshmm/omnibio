# -*- coding: utf-8 -*-
"""
circulation_pump.py

Toggles the 3-pump closed-loop circulation between pio01 vial, pio02
vial, and the external beaker. The pumps themselves are driven by an
Arduino; this job just publishes the on/off command over MQTT.

Place in /home/pioreactor/.pioreactor/plugins/circulation_pump.py
"""
import click
from pioreactor.whoami import get_unit_name, get_assigned_experiment_name
from pioreactor.background_jobs.base import BackgroundJob
from pioreactor.cli.run import run

__plugin_summary__ = "Toggles the 3-pump closed-loop circulation (pio01 <-> pio02 <-> external beaker) via an Arduino over MQTT"
__plugin_version__ = "0.1.0"
__plugin_name__ = "Circulation Pump"
__plugin_author__ = "Siddhesh"
__plugin_homepage__ = "https://docs.pioreactor.com"

# flat topic, since this loop spans two Pioreactor units + one Arduino,
# not tied to a single unit/experiment namespace
ARDUINO_TOPIC = "pioreactor/circulation_pump/run"


class CirculationPump(BackgroundJob):

    job_name = "circulation_pump"

    published_settings = {
        "circulating": {"datatype": "boolean", "settable": True, "default": False},
    }

    def __init__(self, unit, experiment, **kwargs):
        super().__init__(unit=unit, experiment=experiment)
        self.circulating = False

    def set_circulating(self, value):
        # MQTT payloads arrive as strings, e.g. "1"/"0"/"true"/"false"
        value = str(value).strip().lower() in ("1", "true", "yes", "on")
        self.circulating = value
        self.publish(ARDUINO_TOPIC, "1" if value else "0", retain=True)
        self.logger.notice(f"Circulation {'started' if value else 'stopped'}")

    def on_disconnected(self):
        # safety: always stop the pumps if this job exits/crashes
        self.publish(ARDUINO_TOPIC, "0", retain=True)


@run.command(name="circulation_pump", help=__plugin_summary__)
def click_circulation_pump():
    unit = get_unit_name()
    experiment = get_assigned_experiment_name(unit)
    job = CirculationPump(unit=unit, experiment=experiment)
    job.block_until_disconnected()
