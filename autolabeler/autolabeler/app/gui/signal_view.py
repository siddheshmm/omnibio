"""Low-latency live signal visualization using pyqtgraph.

Provides a real-time scrolling waveform display that reads from the
ring buffer at ~30 FPS. Designed for monitoring incoming sensor data
during experiment setup and recording.
"""

import numpy as np
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox
import pyqtgraph as pg

from autolabeler.acquisition.ring_buffer import RingBuffer


class SignalView(QWidget):
    """Live signal visualization widget.

    Displays a scrolling waveform of the most recent N seconds from the
    ring buffer. Uses pyqtgraph for GPU-accelerated, low-latency rendering.

    Args:
        ring_buffer: Ring buffer to read samples from.
        sample_rate: Expected sample rate (Hz) — used for x-axis scaling.
        window_seconds: Duration of visible window in seconds.
        update_interval_ms: Refresh interval in milliseconds (~30 FPS default).
        parent: Parent widget.
    """

    def __init__(
        self,
        ring_buffer: RingBuffer,
        sample_rate: int = 1000,
        window_seconds: float = 5.0,
        update_interval_ms: int = 33,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self._buffer = ring_buffer
        self._sample_rate = sample_rate
        self._window_seconds = window_seconds
        self._window_samples = int(window_seconds * sample_rate)
        self._channels = ring_buffer.channels
        self._is_running = False

        # Colors for multi-channel display
        self._channel_colors = [
            (46, 204, 113),    # Green
            (52, 152, 219),    # Blue
            (231, 76, 60),     # Red
            (241, 196, 15),    # Yellow
            (155, 89, 182),    # Purple
            (230, 126, 34),    # Orange
            (26, 188, 156),    # Teal
            (236, 240, 241),   # Light gray
        ]

        self._setup_ui()
        self._setup_plot()

        # Timer for periodic updates
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update_plot)
        self._timer.setInterval(update_interval_ms)

    def _setup_ui(self) -> None:
        """Build the widget layout."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Info bar
        info_bar = QHBoxLayout()
        self._status_label = QLabel("⏸ Not receiving data")
        self._status_label.setStyleSheet("color: #e74c3c; font-weight: bold; font-size: 13px;")
        info_bar.addWidget(self._status_label)

        info_bar.addStretch()

        info_bar.addWidget(QLabel("Window:"))
        self._window_combo = QComboBox()
        self._window_combo.addItems(["1s", "2s", "5s", "10s", "30s"])
        self._window_combo.setCurrentText(f"{int(self._window_seconds)}s")
        self._window_combo.currentTextChanged.connect(self._on_window_changed)
        info_bar.addWidget(self._window_combo)

        layout.addLayout(info_bar)

        # pyqtgraph plot widget
        self._plot_widget = pg.PlotWidget()
        self._plot_widget.setBackground("#1e1e2e")
        self._plot_widget.showGrid(x=True, y=True, alpha=0.15)
        self._plot_widget.setLabel("bottom", "Time", units="s")
        self._plot_widget.setLabel("left", "Amplitude")
        self._plot_widget.setMouseEnabled(x=False, y=True)
        self._plot_widget.setMinimumHeight(250)
        layout.addWidget(self._plot_widget)

    def _setup_plot(self) -> None:
        """Initialize plot curves for each channel."""
        self._curves: list[pg.PlotDataItem] = []
        for ch in range(self._channels):
            color = self._channel_colors[ch % len(self._channel_colors)]
            pen = pg.mkPen(color=color, width=1.5)
            curve = self._plot_widget.plot(pen=pen, name=f"Ch {ch + 1}")
            self._curves.append(curve)

    def _on_window_changed(self, text: str) -> None:
        """Handle window duration change."""
        try:
            secs = float(text.replace("s", ""))
            self._window_seconds = secs
            self._window_samples = int(secs * self._sample_rate)
        except ValueError:
            pass

    def _update_plot(self) -> None:
        """Read latest data from ring buffer and update the plot."""
        data, timestamps = self._buffer.read_latest(self._window_samples)

        if data.shape[0] == 0:
            if self._is_running:
                self._status_label.setText("⏸ No data")
                self._status_label.setStyleSheet("color: #e74c3c; font-weight: bold; font-size: 13px;")
            return

        n = data.shape[0]

        # Update status
        if not self._is_running or n > 0:
            self._is_running = True
            rate = self._sample_rate
            self._status_label.setText(
                f"✓ Receiving — {rate} Hz × {self._channels} ch — {n} samples shown"
            )
            self._status_label.setStyleSheet("color: #2ecc71; font-weight: bold; font-size: 13px;")

        # Generate time axis relative to now (negative = past)
        if n > 1:
            time_axis = np.linspace(-self._window_seconds, 0, n)
        else:
            time_axis = np.array([0.0])

        # Update each channel curve
        for ch in range(self._channels):
            if ch < data.shape[1]:
                self._curves[ch].setData(time_axis, data[:, ch])

    def start(self) -> None:
        """Start the live plot update timer."""
        self._timer.start()

    def stop(self) -> None:
        """Stop the live plot update timer."""
        self._timer.stop()
        self._is_running = False
        self._status_label.setText("⏸ Stopped")
        self._status_label.setStyleSheet("color: #f39c12; font-weight: bold; font-size: 13px;")

    def set_sample_rate(self, rate: int) -> None:
        """Update the expected sample rate."""
        self._sample_rate = rate
        self._window_samples = int(self._window_seconds * rate)
