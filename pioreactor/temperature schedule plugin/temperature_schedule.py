# -*- coding: utf-8 -*-

from __future__ import annotations

from pioreactor.automations.temperature.thermostat import Thermostat
from pioreactor.utils.timing import RepeatedTimer


__plugin_name__ = "Temperature Schedule"
__plugin_summary__ = "Scheduled temperature cycling automation"
__plugin_version__ = "0.2.0"
__plugin_author__ = "Siddhesh"


class TemperatureSchedule(Thermostat):

    automation_name = "temperature_schedule"

    published_settings = {
        "temperature_sequence": {
            "datatype": "string",
            "settable": False,
            "unit": None,
        },
        "hours_per_step": {
            "datatype": "float",
            "settable": False,
            "unit": "hours",
        },
        "current_step": {
            "datatype": "integer",
            "settable": False,
            "unit": None,
        },
        "target_temperature": {
            "datatype": "float",
            "settable": True,
            "unit": "C",
        },
    }

    def __init__(
        self,
        temperature_sequence: str = "21,28,34,37,34,28",
        hours_per_step: float = 8.0,
        target_temperature: float = 30.0,
        **kwargs,
    ):

        super().__init__(
            target_temperature=target_temperature,
            **kwargs,
        )

        self.temperature_sequence = temperature_sequence
        self.hours_per_step = float(hours_per_step)

        self.sequence = [
            float(x.strip())
            for x in temperature_sequence.split(",")
            if x.strip()
        ]

        if not self.sequence:
            raise ValueError(
                "temperature_sequence cannot be empty"
            )

        if self.hours_per_step <= 0:
            raise ValueError(
                "hours_per_step must be > 0"
            )

        self.current_step = 0

        self.logger.info(
            "Temperature schedule initialized | "
            f"sequence={self.sequence} | "
            f"step_interval={self.hours_per_step}h"
        )

        # execute immediately once
        self.execute()

        # safe managed timer
        self.timer = RepeatedTimer(
            interval=60 * 60 * self.hours_per_step,
            function=self.execute,
            job_name=self.job_name,
            logger=self.logger,
        )

        self.timer.start()

    def execute(self):

        try:

            target_temp = self.sequence[
                self.current_step % len(self.sequence)
            ]

            self.logger.info(
                f"Setting target temperature -> "
                f"{target_temp}C"
            )

            # Proper thermostat update
            self.target_temperature = target_temp

            self.current_step += 1

            return None

        except Exception as e:

            self.logger.error(
                f"Failed updating temperature: {e}"
            )

            return None

    def on_disconnected(self):

        try:
            self.timer.cancel()
        except Exception:
            pass

        self.logger.info(
            "Temperature schedule stopped cleanly."
        )

        return super().on_disconnected()
