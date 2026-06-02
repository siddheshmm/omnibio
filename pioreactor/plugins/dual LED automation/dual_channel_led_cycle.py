# -*- coding: utf-8 -*-
# dual_channel_led_cycle.py
#
# LED automation for the light/colour experiment.
# Pin C
# Pin D
#
# Each channel has its own ON duration, OFF duration, and intensity.
# The two channels run completely independently — their timers do not
# share state, so you can have red ON while blue is OFF and vice-versa.
#
# Typical use for the phased experiment:
#   Phase A (dark baseline)  → both intensities set to 0
#   Phase B (red only)       → channel_C_intensity = <value>, channel_D_intensity = 0
#   B→C buffer (dark)        → both intensities set to 0
#   Phase C (blue only)      → channel_C_intensity = 0, channel_D_intensity = <value>
#   C→D buffer + recovery    → both intensities set to 0
#
# Settings can be changed live from the UI (they are all marked settable=True).
#
#

# -*- coding: utf-8 -*-

from threading import Timer
from typing import Any

from pioreactor.automations.led.base import LEDAutomationJob
from pioreactor.actions.led_intensity import led_intensity


class DualChannelLedCycle(LEDAutomationJob):

    automation_name = "dual_channel_led_cycle"

    published_settings = {
        "channel_C_intensity": {
            "datatype": "float",
            "settable": True,
            "unit": "%",
        },
        "channel_C_duration_min": {
            "datatype": "float",
            "settable": True,
            "unit": "min",
        },

        "channel_D_intensity": {
            "datatype": "float",
            "settable": True,
            "unit": "%",
        },
        "channel_D_duration_min": {
            "datatype": "float",
            "settable": True,
            "unit": "min",
        },

        "gap_duration_min": {
            "datatype": "float",
            "settable": True,
            "unit": "min",
        },
        "repeat_cycle": {
            "datatype": "integer",
            "settable": True,
        },
    }

    def __init__(
        self,
        channel_C_intensity: float | str = 50,
        channel_C_duration_min: float | str = 60,

        channel_D_intensity: float | str = 50,
        channel_D_duration_min: float | str = 60,

        gap_duration_min: float | str = 60,
        
        repeat_cycle: int | str = 1,

        **kwargs: Any,
    ):

        super().__init__(**kwargs)

        self.channel_C_intensity = float(channel_C_intensity)
        self.channel_C_duration_min = float(channel_C_duration_min)

        self.channel_D_intensity = float(channel_D_intensity)
        self.channel_D_duration_min = float(channel_D_duration_min)

        self.gap_duration_min = float(gap_duration_min)
        
        self.repeat_cycle = int(repeat_cycle)
        if self.repeat_cycle < 1:
            raise ValueError("repeat_cycle must be >= 1")

        self._timer = None

        # STATES:
        # C_ON
        # GAP_AFTER_C
        # D_ON
        # GAP_AFTER_D

        self._state = "C_ON"
        self._cycles_completed = 0

    # ==========================================================
    # REQUIRED
    # ==========================================================

    def execute(self):
        return None

    # ==========================================================
    # LED HELPER
    # ==========================================================

    def _set_leds(self, c_intensity=0, d_intensity=0):

        self.logger.info(
            f"LED COMMAND -> C={c_intensity}, D={d_intensity}"
        )

        led_intensity(
            {
                "C": c_intensity,
                "D": d_intensity,
            },
            unit=self.unit,
            experiment=self.experiment,
        )

    # ==========================================================
    # LIFECYCLE
    # ==========================================================

    def on_ready(self):

        super().on_ready()

        self.logger.info("Automation READY")

        self._start_cycle()

    def on_sleeping(self):

        super().on_sleeping()

        self.logger.info("Automation SLEEPING")

        self._stop_everything()

    def on_disconnected(self):

        self.logger.info("Automation DISCONNECTED")

        self._stop_everything()

        super().on_disconnected()

    # ==========================================================
    # STOP EVERYTHING
    # ==========================================================

    def _stop_everything(self):

        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

        # FORCE LEDs OFF
        self._set_leds(0, 0)

    def _schedule_next_state(self, duration_s: float):

        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

        self.logger.info(
            f"Scheduling next LED transition in {duration_s:.2f} s"
        )

        self._timer = Timer(duration_s, self._run_next_state)
        self._timer.daemon = True
        self._timer.start()

    # ==========================================================
    # MAIN STATE MACHINE
    # ==========================================================

    def _start_cycle(self):

        self.logger.info("Starting LED sequence")

        self._run_next_state()

    def _run_next_state(self):

        if self.state != self.READY:
            return

        # ------------------------------------------------------
        # STATE: C ON
        # ------------------------------------------------------

        if self._state == "C_ON":

            self.logger.info("STATE -> C ON")

            self._set_leds(
                c_intensity=self.channel_C_intensity,
                d_intensity=0,
            )

            duration = self.channel_C_duration_min * 60

            self._state = "GAP_AFTER_C"
            self._schedule_next_state(duration)

        # ------------------------------------------------------
        # STATE: GAP AFTER C
        # ------------------------------------------------------

        elif self._state == "GAP_AFTER_C":

            self.logger.info("STATE -> GAP AFTER C")

            self._set_leds(0, 0)

            duration = self.gap_duration_min * 60

            self._state = "D_ON"
            self._schedule_next_state(duration)

        # ------------------------------------------------------
        # STATE: D ON
        # ------------------------------------------------------

        elif self._state == "D_ON":

            self.logger.info("STATE -> D ON")

            self._set_leds(
                c_intensity=0,
                d_intensity=self.channel_D_intensity,
            )

            duration = self.channel_D_duration_min * 60

            self._state = "GAP_AFTER_D"
            self._schedule_next_state(duration)

        # ------------------------------------------------------
        # STATE: GAP AFTER D
        # ------------------------------------------------------

        elif self._state == "GAP_AFTER_D":

            self.logger.info("STATE -> GAP AFTER D")

            self._set_leds(0, 0)

            self._cycles_completed += 1

            if self._cycles_completed < self.repeat_cycle:

                duration = self.gap_duration_min * 60
                self._state = "C_ON"

                self.logger.info(
                    "Cycle %d/%d complete -> restarting after gap"
                    % (self._cycles_completed, self.repeat_cycle)
                )
                self._schedule_next_state(duration)

            else:

                self.logger.info(
                    "Cycle %d/%d complete -> automation finished"
                    % (self._cycles_completed, self.repeat_cycle)
                )

                self._state = "DONE"
                self._stop_everything()

                try:
                    self.set_state(self.DISCONNECTED)
                except Exception:
                    pass

                return
