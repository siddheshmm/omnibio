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

from threading import Timer
from time import monotonic
from typing import Any

from pioreactor import structs
from pioreactor.automations import events
from pioreactor.automations.led.base import LEDAutomationJob
from pioreactor.utils.timing import current_utc_datetime


class DualChannelLedCycle(LEDAutomationJob):
    """
    Independent light/dark cycles for two LED channels.

    Channel C
    --------------------------------
    channel_C_intensity         : brightness while ON  (0–100 %)
    channel_C_on_duration_min   : how long C stays ON  (minutes)
    channel_C_off_duration_min  : how long C stays OFF (minutes)

    Channel D
    --------------------------------
    channel_D_intensity         : brightness while ON  (0–100 %)
    channel_D_on_duration_min   : how long D stays ON  (minutes)
    channel_D_off_duration_min  : how long D stays OFF (minutes)

    Setting either intensity to 0 effectively keeps that channel dark
    regardless of its timer, which is the cleanest way to enforce a
    "dark phase" for one channel while the other runs normally.
    """

    automation_name: str = "dual_channel_led_cycle"

    published_settings = {
        # Channel C (Red)
        "channel_C_intensity": {
            "datatype": "float",
            "settable": True,
            "unit": "%",
        },
        "channel_C_on_duration_min": {
            "datatype": "float",
            "settable": True,
            "unit": "min",
        },
        "channel_C_off_duration_min": {
            "datatype": "float",
            "settable": True,
            "unit": "min",
        },
        # Channel D (Blue)
        "channel_D_intensity": {
            "datatype": "float",
            "settable": True,
            "unit": "%",
        },
        "channel_D_on_duration_min": {
            "datatype": "float",
            "settable": True,
            "unit": "min",
        },
        "channel_D_off_duration_min": {
            "datatype": "float",
            "settable": True,
            "unit": "min",
        },
    }

    def __init__(
        self,
        channel_C_intensity: float | str = 50.0,
        channel_C_on_duration_min: float | str = 60.0,
        channel_C_off_duration_min: float | str = 60.0,
        channel_D_intensity: float | str = 50.0,
        channel_D_on_duration_min: float | str = 60.0,
        channel_D_off_duration_min: float | str = 60.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        # ── Channel C state ──────────────────────────────────────────────
        self.channel_C_intensity = float(channel_C_intensity)
        self.channel_C_on_duration_min = float(channel_C_on_duration_min)
        self.channel_C_off_duration_min = float(channel_C_off_duration_min)
        self._C_active: bool = False
        self._C_cycle_started_at = current_utc_datetime()
        self._C_timer: Timer | None = None
        self._C_timer_expected_at: float | None = None

        # ── Channel D state ──────────────────────────────────────────────
        self.channel_D_intensity = float(channel_D_intensity)
        self.channel_D_on_duration_min = float(channel_D_on_duration_min)
        self.channel_D_off_duration_min = float(channel_D_off_duration_min)
        self._D_active: bool = False
        self._D_cycle_started_at = current_utc_datetime()
        self._D_timer: Timer | None = None
        self._D_timer_expected_at: float | None = None

        # _start_both_channels is called from on_ready() below,
        # which is guaranteed to fire in all Pioreactor versions.

    # ────────────────────────────────────────────────────────────────────
    # LEDAutomationJob required hook (called on each polling tick, but we
    # do not rely on polling — we use self-rescheduling Timers instead).
    # ────────────────────────────────────────────────────────────────────
    def execute(self) -> structs.AutomationEvent | None:
        return None

    # ════════════════════════════════════════════════════════════════════
    # PUBLIC SETTERS  (called automatically by the framework when a
    # published_setting is updated from the UI or MQTT)
    # ════════════════════════════════════════════════════════════════════

    def set_channel_C_intensity(self, value: float | str) -> None:
        self.channel_C_intensity = float(value)
        # If channel is currently ON, apply new intensity immediately
        if self._C_active:
            self.set_led_intensity("C", self.channel_C_intensity)
            self.logger.info(f"Channel C intensity updated to {self.channel_C_intensity}% (live).")

    def set_channel_C_on_duration_min(self, value: float | str) -> None:
        self.channel_C_on_duration_min = float(value)
        self._restart_channel_C()

    def set_channel_C_off_duration_min(self, value: float | str) -> None:
        self.channel_C_off_duration_min = float(value)
        self._restart_channel_C()

    def set_channel_D_intensity(self, value: float | str) -> None:
        self.channel_D_intensity = float(value)
        if self._D_active:
            self.set_led_intensity("D", self.channel_D_intensity)
            self.logger.info(f"Channel D intensity updated to {self.channel_D_intensity}% (live).")

    def set_channel_D_on_duration_min(self, value: float | str) -> None:
        self.channel_D_on_duration_min = float(value)
        self._restart_channel_D()

    def set_channel_D_off_duration_min(self, value: float | str) -> None:
        self.channel_D_off_duration_min = float(value)
        self._restart_channel_D()

    # ════════════════════════════════════════════════════════════════════
    # LIFECYCLE HOOKS
    # ════════════════════════════════════════════════════════════════════

    def on_sleeping(self) -> None:
        super().on_sleeping()
        self._cancel_C_timer()
        self._cancel_D_timer()
        # Turn both channels off while sleeping
        self.set_led_intensity("C", 0)
        self.set_led_intensity("D", 0)
        self._C_active = False
        self._D_active = False

    def on_ready(self) -> None:
        super().on_ready()
        # Fires when the job first enters READY state.
        # Guard with timer check so re-entering READY after sleep
        # doesn't double-start the timers.
        if self._C_timer is None and self._D_timer is None:
            self._start_both_channels()

    def on_sleeping_to_ready(self) -> None:
        super().on_sleeping_to_ready()
        self._start_both_channels()

    def on_disconnected(self) -> None:
        self._cancel_C_timer()
        self._cancel_D_timer()
        super().on_disconnected()

    # ════════════════════════════════════════════════════════════════════
    # INTERNAL — CHANNEL C (Red)
    # ════════════════════════════════════════════════════════════════════

    def _start_both_channels(self) -> None:
        self._run_C_phase()
        self._run_D_phase()

    def _restart_channel_C(self) -> None:
        """Cancel current C timer and restart the cycle from scratch."""
        self._cancel_C_timer()
        self._C_cycle_started_at = current_utc_datetime()
        self._C_active = False
        self._run_C_phase()

    def _restart_channel_D(self) -> None:
        self._cancel_D_timer()
        self._D_cycle_started_at = current_utc_datetime()
        self._D_active = False
        self._run_D_phase()

    def _run_C_phase(self) -> None:
        """Toggle channel C and schedule the next toggle."""
        if self.state != self.READY:
            return

        self._C_active = not self._C_active
        intensity = self.channel_C_intensity if self._C_active else 0.0
        self.set_led_intensity("C", intensity)

        action = "ON" if self._C_active else "OFF"
        self.logger.info(f"Channel C (Red) turned {action} at {intensity:.1f}%.")

        # Schedule the next toggle
        duration_min = (
            self.channel_C_on_duration_min
            if self._C_active
            else self.channel_C_off_duration_min
        )
        self._schedule_C_timer(duration_min * 60.0)

    def _run_D_phase(self) -> None:
        """Toggle channel D and schedule the next toggle."""
        if self.state != self.READY:
            return

        self._D_active = not self._D_active
        intensity = self.channel_D_intensity if self._D_active else 0.0
        self.set_led_intensity("D", intensity)

        action = "ON" if self._D_active else "OFF"
        self.logger.info(f"Channel D (Blue) turned {action} at {intensity:.1f}%.")

        duration_min = (
            self.channel_D_on_duration_min
            if self._D_active
            else self.channel_D_off_duration_min
        )
        self._schedule_D_timer(duration_min * 60.0)

    def _schedule_C_timer(self, seconds: float) -> None:
        self._cancel_C_timer()
        self._C_timer_expected_at = monotonic() + seconds
        self._C_timer = Timer(seconds, self._run_C_phase)
        self._C_timer.daemon = True
        self._C_timer.start()
        self.logger.debug(f"Channel C next toggle in {seconds / 60:.2f} min.")

    def _schedule_D_timer(self, seconds: float) -> None:
        self._cancel_D_timer()
        self._D_timer_expected_at = monotonic() + seconds
        self._D_timer = Timer(seconds, self._run_D_phase)
        self._D_timer.daemon = True
        self._D_timer.start()
        self.logger.debug(f"Channel D next toggle in {seconds / 60:.2f} min.")

    def _cancel_C_timer(self) -> None:
        if self._C_timer is not None:
            self._C_timer.cancel()
            self._C_timer = None

    def _cancel_D_timer(self) -> None:
        if self._D_timer is not None:
            self._D_timer.cancel()
            self._D_timer = None