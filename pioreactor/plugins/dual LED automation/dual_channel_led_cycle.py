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
        "initial_dark_min": {
            "datatype": "float",
            "settable": True,
            "unit": "min",
        },
        "phase_sequence": {
            "datatype": "string",
            "settable": True,
            "unit": None,
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
        initial_dark_min: float | str = 0,
        phase_sequence: str = "",

        repeat_cycle: int | str = 1,

        **kwargs: Any,
    ):

        super().__init__(**kwargs)

        self.channel_C_intensity = float(channel_C_intensity)
        self.channel_C_duration_min = float(channel_C_duration_min)

        self.channel_D_intensity = float(channel_D_intensity)
        self.channel_D_duration_min = float(channel_D_duration_min)

        self.gap_duration_min = float(gap_duration_min)
        self.initial_dark_min = float(initial_dark_min)
        self.phase_sequence = str(phase_sequence)
        
        self.repeat_cycle = int(repeat_cycle)
        if self.repeat_cycle < 1:
            raise ValueError("repeat_cycle must be >= 1")
        if self.initial_dark_min < 0:
            raise ValueError("initial_dark_min must be >= 0")
        if self.gap_duration_min < 0:
            raise ValueError("gap_duration_min must be >= 0")

        self._phase_sequence = self._parse_phase_sequence(self.phase_sequence)

        self._timer = None

        # STATES:
        # INITIAL_DARK
        # SCHEDULE_PHASE
        # SCHEDULE_GAP
        # C_ON
        # GAP_AFTER_C
        # D_ON
        # GAP_AFTER_D

        self._state = "C_ON"
        self._cycles_completed = 0
        self._schedule_index = 0

    # ==========================================================
    # REQUIRED
    # ==========================================================

    def execute(self):
        return None

    def _parse_phase_sequence(self, raw: str) -> list[str]:

        if not raw.strip():
            return []

        aliases = {
            "C": "C",
            "R": "C",
            "RED": "C",
            "CHANNEL_C": "C",
            "D": "D",
            "B": "D",
            "BLUE": "D",
            "CHANNEL_D": "D",
        }

        sequence = []
        for token in raw.replace(";", ",").split(","):
            key = token.strip().upper()
            if not key:
                continue
            if key not in aliases:
                raise ValueError(
                    "phase_sequence must contain only C/R/red or D/B/blue tokens"
                )
            sequence.append(aliases[key])

        if not sequence:
            raise ValueError("phase_sequence cannot be empty if provided")

        return sequence

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

        if self._phase_sequence:
            self._schedule_index = 0
            self._cycles_completed = 0
            if self.initial_dark_min > 0:
                self._state = "INITIAL_DARK"
            else:
                self._state = "SCHEDULE_PHASE"

        self._run_next_state()

    def _finish_automation(self):

        self._state = "DONE"
        self._stop_everything()

        try:
            self.set_state(self.DISCONNECTED)
        except Exception:
            pass

    def _run_schedule_phase(self):

        channel = self._phase_sequence[self._schedule_index]

        if channel == "C":
            self.logger.info(
                "SCHEDULE -> RED/C phase %d/%d"
                % (self._schedule_index + 1, len(self._phase_sequence))
            )
            self._set_leds(
                c_intensity=self.channel_C_intensity,
                d_intensity=0,
            )
            duration = self.channel_C_duration_min * 60
        else:
            self.logger.info(
                "SCHEDULE -> BLUE/D phase %d/%d"
                % (self._schedule_index + 1, len(self._phase_sequence))
            )
            self._set_leds(
                c_intensity=0,
                d_intensity=self.channel_D_intensity,
            )
            duration = self.channel_D_duration_min * 60

        self._state = "SCHEDULE_GAP"
        self._schedule_next_state(duration)

    def _run_schedule_gap(self):

        self.logger.info("SCHEDULE -> DARK GAP")
        self._set_leds(0, 0)

        self._schedule_index += 1

        if self._schedule_index < len(self._phase_sequence):
            self._state = "SCHEDULE_PHASE"
            self._schedule_next_state(self.gap_duration_min * 60)
            return

        self._cycles_completed += 1

        if self._cycles_completed < self.repeat_cycle:
            self.logger.info(
                "Schedule repeat %d/%d complete -> restarting after gap"
                % (self._cycles_completed, self.repeat_cycle)
            )
            self._schedule_index = 0
            self._state = "SCHEDULE_PHASE"
            self._schedule_next_state(self.gap_duration_min * 60)
            return

        self.logger.info(
            "Schedule repeat %d/%d complete -> automation finished"
            % (self._cycles_completed, self.repeat_cycle)
        )
        self._finish_automation()

    def _run_next_state(self):

        if self.state != self.READY:
            return

        # ------------------------------------------------------
        # PSEUDORANDOM / EXPLICIT SCHEDULE MODE
        # ------------------------------------------------------

        if self._state == "INITIAL_DARK":

            self.logger.info("SCHEDULE -> INITIAL DARK")
            self._set_leds(0, 0)
            self._state = "SCHEDULE_PHASE"
            self._schedule_next_state(self.initial_dark_min * 60)
            return

        if self._state == "SCHEDULE_PHASE":
            self._run_schedule_phase()
            return

        if self._state == "SCHEDULE_GAP":
            self._run_schedule_gap()
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

                self._finish_automation()

                return
