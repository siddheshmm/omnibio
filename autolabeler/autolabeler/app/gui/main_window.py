"""Main application window for AutoLabeler.

Tab-based layout:
  - Setup: Hardware profile selection, UDP listener control
  - Record: Live signal view + experiment prompts (Phase 2+)
  - Dataset: Dataset browser (Phase 3+)
  - Train: ML pipeline configuration (Phase 4+)
"""

import sys
import logging
from pathlib import Path
from functools import partial
from typing import Optional

from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QFont, QIcon, QAction
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QTabWidget,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QComboBox,
    QSpinBox,
    QLineEdit,
    QFormLayout,
    QMessageBox,
    QStatusBar,
    QSplitter,
)

from autolabeler.config import AppConfig, HardwareConfig
from autolabeler.acquisition.ring_buffer import RingBuffer
from autolabeler.acquisition.udp_listener import UDPListener
from autolabeler.acquisition.hardware_profiles import (
    list_profiles,
    get_profile,
    ensure_builtin_profiles,
)
from autolabeler.app.gui.signal_view import SignalView
from autolabeler.app.gui.experiment_panel import ExperimentPanel
from autolabeler.app.gui.dataset_panel import DatasetPanel
from autolabeler.app.gui.train_panel import TrainPanel


logger = logging.getLogger(__name__)


# --- Stylesheet ---
DARK_STYLESHEET = """
QMainWindow, QWidget {
    background-color: #1e1e2e;
    color: #cdd6f4;
    font-family: 'Segoe UI', 'Inter', sans-serif;
    font-size: 13px;
}
QTabWidget::pane {
    border: 1px solid #313244;
    background: #1e1e2e;
    border-radius: 6px;
}
QTabBar::tab {
    background: #313244;
    color: #a6adc8;
    padding: 10px 22px;
    margin-right: 2px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    font-weight: bold;
    font-size: 13px;
}
QTabBar::tab:selected {
    background: #45475a;
    color: #cdd6f4;
}
QTabBar::tab:hover {
    background: #585b70;
}
QGroupBox {
    border: 1px solid #313244;
    border-radius: 8px;
    margin-top: 14px;
    padding: 16px;
    font-weight: bold;
    font-size: 13px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 14px;
    padding: 0 6px;
    color: #89b4fa;
}
QPushButton {
    background: #45475a;
    color: #cdd6f4;
    border: 1px solid #585b70;
    border-radius: 6px;
    padding: 8px 20px;
    font-weight: bold;
    font-size: 13px;
}
QPushButton:hover {
    background: #585b70;
    border-color: #89b4fa;
}
QPushButton:pressed {
    background: #89b4fa;
    color: #1e1e2e;
}
QPushButton:disabled {
    background: #313244;
    color: #585b70;
    border-color: #313244;
}
QPushButton#startBtn {
    background: #2ecc71;
    color: #1e1e2e;
    border-color: #27ae60;
}
QPushButton#startBtn:hover {
    background: #27ae60;
}
QPushButton#stopBtn {
    background: #e74c3c;
    color: #fff;
    border-color: #c0392b;
}
QPushButton#stopBtn:hover {
    background: #c0392b;
}
QComboBox, QSpinBox, QLineEdit {
    background: #313244;
    color: #cdd6f4;
    border: 1px solid #585b70;
    border-radius: 5px;
    padding: 6px 10px;
    font-size: 13px;
}
QComboBox:hover, QSpinBox:hover, QLineEdit:hover {
    border-color: #89b4fa;
}
QComboBox::drop-down {
    border: none;
    padding-right: 8px;
}
QLabel {
    font-size: 13px;
}
QStatusBar {
    background: #181825;
    color: #a6adc8;
    border-top: 1px solid #313244;
    font-size: 12px;
}
"""


class SetupTab(QWidget):
    """Hardware & connection setup tab."""

    def __init__(self, main_window: "MainWindow", parent: QWidget | None = None):
        super().__init__(parent)
        self._main = main_window
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        # --- Hardware Profile Group ---
        hw_group = QGroupBox("Hardware Profile")
        hw_layout = QFormLayout()

        self._profile_combo = QComboBox()
        self._refresh_profiles()
        hw_layout.addRow("Profile:", self._profile_combo)

        self._name_edit = QLineEdit()
        hw_layout.addRow("Device Name:", self._name_edit)

        self._host_edit = QLineEdit("127.0.0.1")
        hw_layout.addRow("UDP Host:", self._host_edit)

        self._port_spin = QSpinBox()
        self._port_spin.setRange(1024, 65535)
        self._port_spin.setValue(5000)
        hw_layout.addRow("UDP Port:", self._port_spin)

        self._rate_spin = QSpinBox()
        self._rate_spin.setRange(1, 200000)
        self._rate_spin.setValue(1000)
        self._rate_spin.setSuffix(" Hz")
        hw_layout.addRow("Sample Rate:", self._rate_spin)

        self._channels_spin = QSpinBox()
        self._channels_spin.setRange(1, 32)
        self._channels_spin.setValue(1)
        hw_layout.addRow("Channels:", self._channels_spin)

        self._format_combo = QComboBox()
        self._format_combo.addItems(["csv", "json", "binary", "raw"])
        hw_layout.addRow("Packet Format:", self._format_combo)

        hw_group.setLayout(hw_layout)
        layout.addWidget(hw_group)

        # Profile combo change handler
        self._profile_combo.currentTextChanged.connect(self._on_profile_changed)
        if self._profile_combo.count() > 0:
            self._on_profile_changed(self._profile_combo.currentText())

        # --- Connection Controls ---
        ctrl_group = QGroupBox("Connection")
        ctrl_layout = QHBoxLayout()

        self._start_btn = QPushButton("▶  Start Listening")
        self._start_btn.setObjectName("startBtn")
        self._start_btn.clicked.connect(self._on_start)
        ctrl_layout.addWidget(self._start_btn)

        self._stop_btn = QPushButton("■  Stop")
        self._stop_btn.setObjectName("stopBtn")
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._on_stop)
        ctrl_layout.addWidget(self._stop_btn)

        ctrl_group.setLayout(ctrl_layout)
        layout.addWidget(ctrl_group)

        layout.addStretch()

    def _refresh_profiles(self) -> None:
        """Reload available profiles into the combo box."""
        self._profile_combo.clear()
        profiles = list_profiles()
        for name in sorted(profiles.keys()):
            self._profile_combo.addItem(name)

    def _on_profile_changed(self, name: str) -> None:
        """Populate form fields from the selected profile."""
        config = get_profile(name)
        if config:
            self._name_edit.setText(config.name)
            self._host_edit.setText(config.udp_host)
            self._port_spin.setValue(config.udp_port)
            self._rate_spin.setValue(config.sample_rate)
            self._channels_spin.setValue(config.channels)
            idx = self._format_combo.findText(config.packet_format)
            if idx >= 0:
                self._format_combo.setCurrentIndex(idx)

    def get_hardware_config(self) -> HardwareConfig:
        """Build a HardwareConfig from the current form values."""
        return HardwareConfig(
            name=self._name_edit.text() or "Custom",
            sample_rate=self._rate_spin.value(),
            channels=self._channels_spin.value(),
            udp_port=self._port_spin.value(),
            udp_host=self._host_edit.text() or "127.0.0.1",
            packet_format=self._format_combo.currentText(),
        )

    def _on_start(self) -> None:
        """Start the UDP listener."""
        config = self.get_hardware_config()
        self._main.start_acquisition(config)
        self._start_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._set_form_enabled(False)

    def _on_stop(self) -> None:
        """Stop the UDP listener."""
        self._main.stop_acquisition()
        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._set_form_enabled(True)

    def _set_form_enabled(self, enabled: bool) -> None:
        """Enable/disable form inputs."""
        for w in (
            self._profile_combo, self._name_edit, self._host_edit,
            self._port_spin, self._rate_spin, self._channels_spin,
            self._format_combo,
        ):
            w.setEnabled(enabled)





class MainWindow(QMainWindow):
    """Main application window for AutoLabeler."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("AutoLabeler")
        self.setMinimumSize(QSize(1000, 700))
        self.resize(1200, 800)

        self._config = AppConfig()
        self._ring_buffer: Optional[RingBuffer] = None
        self._listener: Optional[UDPListener] = None
        self._signal_view: Optional[SignalView] = None

        # Ensure built-in profiles exist on disk
        ensure_builtin_profiles()

        self._setup_ui()

    def _setup_ui(self) -> None:
        """Build the main window layout."""
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)

        # Header
        header = QLabel("AutoLabeler")
        header.setStyleSheet(
            "font-size: 22px; font-weight: bold; color: #89b4fa; padding: 8px 0;"
        )
        main_layout.addWidget(header)

        # Splitter: tabs on left, signal view on right (when active)
        self._splitter = QSplitter(Qt.Orientation.Horizontal)

        # Tab widget
        self._tabs = QTabWidget()
        self._setup_tab = SetupTab(self)
        self._experiment_panel = ExperimentPanel()
        self._dataset_panel = DatasetPanel()
        self._train_panel = TrainPanel()
        self._tabs.addTab(self._setup_tab, "⚙  Setup")
        self._tabs.addTab(self._experiment_panel, "🔴  Record")
        self._tabs.addTab(self._dataset_panel, "📊  Dataset")
        self._tabs.addTab(self._train_panel, "🧠  Train")
        self._splitter.addWidget(self._tabs)

        # Signal view (initially hidden, shown when acquisition starts)
        self._signal_container = QWidget()
        signal_layout = QVBoxLayout(self._signal_container)
        signal_layout.setContentsMargins(0, 0, 0, 0)
        signal_label = QLabel("Live Signal")
        signal_label.setStyleSheet(
            "font-size: 15px; font-weight: bold; color: #89b4fa; padding: 4px 0;"
        )
        signal_layout.addWidget(signal_label)
        # Placeholder — real SignalView is created in start_acquisition
        self._signal_placeholder = QLabel("Start listening to see live signal")
        self._signal_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._signal_placeholder.setStyleSheet("color: #585b70; font-size: 14px; padding: 60px;")
        signal_layout.addWidget(self._signal_placeholder)
        self._splitter.addWidget(self._signal_container)
        self._splitter.setSizes([400, 600])

        main_layout.addWidget(self._splitter)

        # Status bar
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._status_bar.showMessage("Ready — select a hardware profile and start listening")

    def start_acquisition(self, hw_config: HardwareConfig) -> None:
        """Start the UDP listener and live signal view."""
        try:
            # Stop existing if running
            self.stop_acquisition()

            # Create ring buffer
            self._ring_buffer = RingBuffer.from_duration(
                duration_seconds=self._config.buffer_duration,
                sample_rate=hw_config.sample_rate,
                channels=hw_config.channels,
            )

            # Create and start UDP listener
            self._listener = UDPListener(self._ring_buffer, hw_config)
            self._listener.start()

            # Pass hardware config and ring buffer to experiment panel
            self._experiment_panel.set_hardware_config(hw_config)
            self._experiment_panel.set_ring_buffer(self._ring_buffer)

            # Create signal view
            self._signal_view = SignalView(
                ring_buffer=self._ring_buffer,
                sample_rate=hw_config.sample_rate,
                window_seconds=self._config.plot_window_seconds,
                update_interval_ms=self._config.plot_update_interval_ms,
            )

            # Replace placeholder with signal view
            layout = self._signal_container.layout()
            if self._signal_placeholder:
                layout.removeWidget(self._signal_placeholder)
                self._signal_placeholder.deleteLater()
                self._signal_placeholder = None
            layout.addWidget(self._signal_view)
            self._signal_view.start()

            self._status_bar.showMessage(
                f"✓ Listening on {hw_config.udp_host}:{hw_config.udp_port} — "
                f"{hw_config.sample_rate} Hz × {hw_config.channels} ch "
                f"({hw_config.packet_format})"
            )
            logger.info(f"Acquisition started: {hw_config.name}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start acquisition:\n{e}")
            logger.error(f"Acquisition start failed: {e}", exc_info=True)

    def stop_acquisition(self) -> None:
        """Stop the UDP listener and signal view."""
        if self._signal_view:
            self._signal_view.stop()
            layout = self._signal_container.layout()
            layout.removeWidget(self._signal_view)
            self._signal_view.deleteLater()
            self._signal_view = None

            # Restore placeholder
            self._signal_placeholder = QLabel("Start listening to see live signal")
            self._signal_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._signal_placeholder.setStyleSheet(
                "color: #585b70; font-size: 14px; padding: 60px;"
            )
            layout.addWidget(self._signal_placeholder)

        if self._listener:
            self._listener.stop()
            self._listener = None

        self._ring_buffer = None
        self._status_bar.showMessage("Stopped")

    def closeEvent(self, event) -> None:
        """Clean up on window close."""
        self.stop_acquisition()
        event.accept()


def run_app() -> None:
    """Entry point — launch the AutoLabeler GUI."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    app = QApplication(sys.argv)
    app.setApplicationName("AutoLabeler")
    app.setStyleSheet(DARK_STYLESHEET)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())
