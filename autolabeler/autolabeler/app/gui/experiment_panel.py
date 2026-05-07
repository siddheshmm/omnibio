"""Experiment panel widget — config form + live experiment prompts.

Two modes:
  - CONFIG mode: form for defining classes, timing, trials.
  - RUNNING mode: live prompt display, countdown, progress bar.
"""

import logging
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt, QSize
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QSpinBox,
    QDoubleSpinBox,
    QLineEdit,
    QListWidget,
    QFormLayout,
    QStackedWidget,
    QProgressBar,
    QMessageBox,
    QFileDialog,
)

from autolabeler.config import ExperimentConfig, HardwareConfig, DATASETS_DIR
from autolabeler.experiment.prompter import Prompter, ExperimentState

logger = logging.getLogger(__name__)


class ExperimentPanel(QWidget):
    """Experiment configuration and live prompt display.

    Shows a configuration form before the experiment starts, then
    switches to a live prompt view during the experiment.
    """

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._prompter = Prompter(self)
        self._hw_config: Optional[HardwareConfig] = None
        self._ring_buffer = None
        self._setup_ui()
        self._connect_signals()

    def set_hardware_config(self, config: HardwareConfig) -> None:
        """Set the current hardware config (for metadata logging)."""
        self._hw_config = config

    def set_ring_buffer(self, ring_buffer) -> None:
        """Set the ring buffer reference (for dataset saving)."""
        self._ring_buffer = ring_buffer

    # --- UI Setup ---

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        self._stack = QStackedWidget()

        # Page 0: Config form
        self._config_page = self._build_config_page()
        self._stack.addWidget(self._config_page)

        # Page 1: Running experiment view
        self._running_page = self._build_running_page()
        self._stack.addWidget(self._running_page)

        # Page 2: Done / summary page
        self._done_page = self._build_done_page()
        self._stack.addWidget(self._done_page)

        self._stack.setCurrentIndex(0)
        layout.addWidget(self._stack)

    def _build_config_page(self) -> QWidget:
        """Build the experiment configuration form."""
        page = QWidget()
        layout = QVBoxLayout(page)

        # --- Session Identity ---
        id_group = QGroupBox("Session")
        id_layout = QFormLayout()

        self._subject_input = QLineEdit("subject_01")
        self._subject_input.setPlaceholderText("e.g. subject_01")
        id_layout.addRow("Subject ID:", self._subject_input)

        self._session_input = QLineEdit()
        self._session_input.setPlaceholderText("auto-generated if empty")
        self._auto_session_label = QLabel("")
        self._auto_session_label.setStyleSheet("color: #a6adc8; font-size: 11px;")
        id_layout.addRow("Session ID:", self._session_input)
        id_layout.addRow("", self._auto_session_label)
        self._subject_input.textChanged.connect(self._update_auto_session)
        self._update_auto_session()

        id_group.setLayout(id_layout)
        layout.addWidget(id_group)

        # --- Classes Group ---
        cls_group = QGroupBox("Stimulus Classes")
        cls_layout = QVBoxLayout()

        cls_info = QLabel(
            "Define the classes your model should learn to distinguish.\n"
            "Example: 'Touch' and 'No Touch' — where 'No Touch' is the baseline."
        )
        cls_info.setStyleSheet("color: #a6adc8; font-size: 12px; padding-bottom: 4px;")
        cls_info.setWordWrap(True)
        cls_layout.addWidget(cls_info)

        self._class_list = QListWidget()
        self._class_list.setMaximumHeight(120)
        cls_layout.addWidget(self._class_list)

        btn_row = QHBoxLayout()
        self._class_input = QLineEdit()
        self._class_input.setPlaceholderText("Enter class name...")
        self._class_input.returnPressed.connect(self._add_class)
        btn_row.addWidget(self._class_input)

        add_btn = QPushButton("+ Add")
        add_btn.clicked.connect(self._add_class)
        btn_row.addWidget(add_btn)

        remove_btn = QPushButton("− Remove")
        remove_btn.clicked.connect(self._remove_class)
        btn_row.addWidget(remove_btn)

        cls_layout.addLayout(btn_row)
        cls_group.setLayout(cls_layout)
        layout.addWidget(cls_group)

        # --- Timing Group ---
        time_group = QGroupBox("Timing")
        time_layout = QFormLayout()

        self._trial_dur_spin = QDoubleSpinBox()
        self._trial_dur_spin.setRange(0.5, 60.0)
        self._trial_dur_spin.setValue(5.0)
        self._trial_dur_spin.setSuffix(" s")
        self._trial_dur_spin.setSingleStep(0.5)
        self._trial_dur_spin.valueChanged.connect(self._update_estimate)
        time_layout.addRow("Trial Duration:", self._trial_dur_spin)

        self._rest_dur_spin = QDoubleSpinBox()
        self._rest_dur_spin.setRange(0.5, 30.0)
        self._rest_dur_spin.setValue(3.0)
        self._rest_dur_spin.setSuffix(" s")
        self._rest_dur_spin.setSingleStep(0.5)
        self._rest_dur_spin.valueChanged.connect(self._update_estimate)
        time_layout.addRow("Rest Between Trials:", self._rest_dur_spin)

        self._countdown_spin = QDoubleSpinBox()
        self._countdown_spin.setRange(1.0, 30.0)
        self._countdown_spin.setValue(5.0)
        self._countdown_spin.setSuffix(" s")
        time_layout.addRow("Countdown:", self._countdown_spin)

        time_group.setLayout(time_layout)
        layout.addWidget(time_group)

        # --- Trials Group ---
        trial_group = QGroupBox("Trials")
        trial_layout = QFormLayout()

        self._trials_per_class_spin = QSpinBox()
        self._trials_per_class_spin.setRange(1, 500)
        self._trials_per_class_spin.setValue(30)
        self._trials_per_class_spin.valueChanged.connect(self._update_estimate)
        trial_layout.addRow("Trials per Class:", self._trials_per_class_spin)

        self._estimate_label = QLabel("")
        self._estimate_label.setStyleSheet(
            "color: #89b4fa; font-weight: bold; font-size: 13px;"
        )
        trial_layout.addRow("Estimated Duration:", self._estimate_label)

        trial_group.setLayout(trial_layout)
        layout.addWidget(trial_group)

        self._update_estimate()

        # --- Start Button ---
        self._start_btn = QPushButton("▶  Start Experiment")
        self._start_btn.setObjectName("startBtn")
        self._start_btn.setMinimumHeight(44)
        self._start_btn.clicked.connect(self._on_start)
        layout.addWidget(self._start_btn)

        layout.addStretch()
        return page

    def _build_running_page(self) -> QWidget:
        """Build the live experiment prompt view."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # State label
        self._state_label = QLabel("PREPARING...")
        self._state_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._state_label.setStyleSheet(
            "color: #f39c12; font-size: 14px; font-weight: bold;"
        )
        layout.addWidget(self._state_label)

        # Main cue (large text)
        self._cue_label = QLabel("")
        self._cue_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._cue_label.setStyleSheet(
            "font-size: 36px; font-weight: bold; color: #cdd6f4; "
            "padding: 20px; min-height: 80px;"
        )
        self._cue_label.setWordWrap(True)
        layout.addWidget(self._cue_label)

        # Timer display
        self._timer_label = QLabel("")
        self._timer_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._timer_label.setStyleSheet(
            "font-size: 48px; font-weight: bold; color: #89b4fa; "
            "font-family: 'Consolas', 'Courier New', monospace;"
        )
        layout.addWidget(self._timer_label)

        # Progress
        self._progress_label = QLabel("")
        self._progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._progress_label.setStyleSheet(
            "color: #a6adc8; font-size: 14px; padding-top: 12px;"
        )
        layout.addWidget(self._progress_label)

        self._progress_bar = QProgressBar()
        self._progress_bar.setMinimum(0)
        self._progress_bar.setMaximum(100)
        self._progress_bar.setValue(0)
        self._progress_bar.setTextVisible(False)
        self._progress_bar.setMaximumWidth(500)
        self._progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #313244;
                border-radius: 5px;
                background: #313244;
                height: 12px;
            }
            QProgressBar::chunk {
                background: #89b4fa;
                border-radius: 4px;
            }
        """)
        layout.addWidget(self._progress_bar, alignment=Qt.AlignmentFlag.AlignCenter)

        # Abort button
        layout.addSpacing(20)
        self._abort_btn = QPushButton("■  Abort Experiment")
        self._abort_btn.setObjectName("stopBtn")
        self._abort_btn.setMinimumHeight(40)
        self._abort_btn.clicked.connect(self._on_abort)
        layout.addWidget(self._abort_btn, alignment=Qt.AlignmentFlag.AlignCenter)

        layout.addStretch()
        return page

    def _build_done_page(self) -> QWidget:
        """Build the experiment-complete summary view."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        done_icon = QLabel("✅")
        done_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        done_icon.setStyleSheet("font-size: 48px; padding: 10px;")
        layout.addWidget(done_icon)

        self._done_label = QLabel("Experiment Complete!")
        self._done_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._done_label.setStyleSheet(
            "font-size: 22px; font-weight: bold; color: #2ecc71; padding: 8px;"
        )
        layout.addWidget(self._done_label)

        self._summary_label = QLabel("")
        self._summary_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._summary_label.setStyleSheet(
            "color: #cdd6f4; font-size: 14px; padding: 4px;"
        )
        self._summary_label.setWordWrap(True)
        layout.addWidget(self._summary_label)

        self._new_btn = QPushButton("Start New Experiment")
        self._new_btn.setMinimumHeight(40)
        self._new_btn.clicked.connect(self._on_new_experiment)
        layout.addWidget(self._new_btn, alignment=Qt.AlignmentFlag.AlignCenter)

        layout.addStretch()
        return page

    # --- Signal connections ---

    def _connect_signals(self) -> None:
        self._prompter.state_changed.connect(self._on_state_changed)
        self._prompter.countdown_tick.connect(self._on_countdown_tick)
        self._prompter.trial_started.connect(self._on_trial_started)
        self._prompter.trial_ended.connect(self._on_trial_ended)
        self._prompter.rest_started.connect(self._on_rest_started)
        self._prompter.rest_tick.connect(self._on_rest_tick)
        self._prompter.progress_updated.connect(self._on_progress_updated)
        self._prompter.experiment_finished.connect(self._on_experiment_finished)

    # --- Form Actions ---

    def _next_session_id(self) -> str:
        """Find the next available session ID for the current subject."""
        subject = self._subject_input.text().strip() or "subject_01"
        subject_dir = DATASETS_DIR / subject
        if not subject_dir.exists():
            return "session_01"
        existing = sorted(
            [d.name for d in subject_dir.iterdir() if d.is_dir()]
        )
        # Find highest session_XX number and increment
        max_num = 0
        for name in existing:
            if name.startswith("session_"):
                try:
                    num = int(name.split("_")[1])
                    max_num = max(max_num, num)
                except (IndexError, ValueError):
                    pass
        return f"session_{max_num + 1:02d}"

    def _update_auto_session(self) -> None:
        """Update the auto-session hint label."""
        next_id = self._next_session_id()
        self._auto_session_label.setText(
            f"Leave empty to auto-assign: {next_id}"
        )

    def _add_class(self) -> None:
        text = self._class_input.text().strip()
        if text:
            self._class_list.addItem(text)
            self._class_input.clear()
            self._update_estimate()

    def _remove_class(self) -> None:
        row = self._class_list.currentRow()
        if row >= 0:
            self._class_list.takeItem(row)
            self._update_estimate()

    def _update_estimate(self) -> None:
        classes = [self._class_list.item(i).text()
                   for i in range(self._class_list.count())]
        n_classes = max(1, len(classes))
        trials = self._trials_per_class_spin.value()
        trial_dur = self._trial_dur_spin.value()
        rest_dur = self._rest_dur_spin.value()
        total = n_classes * trials
        secs = total * (trial_dur + rest_dur) + self._countdown_spin.value()
        mins, s = divmod(int(secs), 60)
        self._estimate_label.setText(
            f"{total} trials total — ~{mins}m {s}s"
        )

    def _get_experiment_config(self) -> ExperimentConfig:
        """Build an ExperimentConfig from the form."""
        classes = [self._class_list.item(i).text()
                   for i in range(self._class_list.count())]
        return ExperimentConfig(
            subject_id=self._subject_input.text().strip() or "subject_01",
            session_id=self._session_input.text().strip() or self._next_session_id(),
            classes=classes,
            trial_duration=self._trial_dur_spin.value(),
            rest_duration=self._rest_dur_spin.value(),
            trials_per_class=self._trials_per_class_spin.value(),
            countdown_duration=self._countdown_spin.value(),
        )

    def _on_start(self) -> None:
        """Start the experiment."""
        classes = [self._class_list.item(i).text()
                   for i in range(self._class_list.count())]
        if len(classes) < 1:
            QMessageBox.warning(
                self, "No Classes",
                "Add at least one stimulus class before starting."
            )
            return

        config = self._get_experiment_config()

        # Validate that all existing sessions for this subject use the same
        # class set. This prevents mixing incompatible label definitions
        # within a subject.
        subject_dir = DATASETS_DIR / config.subject_id
        if subject_dir.exists():
            from json import load as _json_load

            existing_classes: list[str] | None = None
            for sess_dir in sorted(subject_dir.iterdir()):
                if not sess_dir.is_dir():
                    continue
                meta_path = sess_dir / "metadata.json"
                if not meta_path.exists():
                    continue
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = _json_load(f)
                    exp_cfg = meta.get("experiment_config", {})
                    sess_classes = exp_cfg.get("classes", [])
                except Exception:
                    continue

                if not sess_classes:
                    continue

                if existing_classes is None:
                    existing_classes = list(sess_classes)
                elif list(sess_classes) != list(existing_classes):
                    QMessageBox.critical(
                        self,
                        "Inconsistent Classes",
                        "Existing sessions for this subject use a different set of "
                        "classes.\n\nTo keep datasets consistent, all sessions for "
                        "a subject must share the same class list.\n\n"
                        "Either choose a different subject ID or adjust your "
                        "classes to match previous sessions.",
                    )
                    return

            if existing_classes is not None and list(classes) != list(existing_classes):
                QMessageBox.critical(
                    self,
                    "Classes Mismatch",
                    "The classes defined for this session do not match previous "
                    "sessions for this subject.\n\n"
                    f"Previous classes: {existing_classes}\n"
                    f"Current classes: {classes}\n\n"
                    "Please keep classes consistent within a subject.",
                )
                return

        # Create output directory
        output_dir = DATASETS_DIR / config.subject_id / config.session_id
        output_dir.mkdir(parents=True, exist_ok=True)

        self._stack.setCurrentIndex(1)
        self._prompter.start_experiment(
            exp_config=config,
            output_dir=output_dir,
            hw_config=self._hw_config,
            ring_buffer=self._ring_buffer,
        )

    def _on_abort(self) -> None:
        """Abort the running experiment."""
        reply = QMessageBox.question(
            self, "Abort Experiment",
            "Are you sure you want to abort? Completed trials will be saved.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._prompter.abort()
            self._stack.setCurrentIndex(0)

    def _on_new_experiment(self) -> None:
        """Return to config form for a new experiment."""
        self._stack.setCurrentIndex(0)

    # --- Prompter Signal Handlers ---

    def _on_state_changed(self, state_name: str) -> None:
        colors = {
            "COUNTDOWN": "#f39c12",
            "STIMULUS": "#2ecc71",
            "REST": "#89b4fa",
            "DONE": "#2ecc71",
        }
        color = colors.get(state_name, "#a6adc8")
        self._state_label.setText(state_name)
        self._state_label.setStyleSheet(
            f"color: {color}; font-size: 14px; font-weight: bold;"
        )

    def _on_countdown_tick(self, remaining: int) -> None:
        self._cue_label.setText("Get Ready...")
        self._timer_label.setText(str(remaining))
        self._cue_label.setStyleSheet(
            "font-size: 36px; font-weight: bold; color: #f39c12; "
            "padding: 20px; min-height: 80px;"
        )

    def _on_trial_started(
        self, trial_num: int, total: int, label: str, block: int
    ) -> None:
        self._cue_label.setText(label)
        self._cue_label.setStyleSheet(
            "font-size: 42px; font-weight: bold; color: #2ecc71; "
            "padding: 20px; min-height: 80px;"
        )
        self._progress_label.setText(
            f"Trial {trial_num} / {total}  —  Block {block}"
        )

    def _on_trial_ended(self, trial_num: int, label: str) -> None:
        pass  # rest_started handles the transition

    def _on_rest_started(self, duration: float) -> None:
        self._cue_label.setText("Rest")
        self._cue_label.setStyleSheet(
            "font-size: 36px; font-weight: bold; color: #585b70; "
            "padding: 20px; min-height: 80px;"
        )

    def _on_rest_tick(self, remaining: float) -> None:
        self._timer_label.setText(f"{remaining:.1f}")

    def _on_progress_updated(self, current: int, total: int) -> None:
        pct = int((current / total) * 100) if total else 0
        self._progress_bar.setValue(pct)

    def _on_experiment_finished(self, summary: dict) -> None:
        total = summary["total_trials"]
        classes = ", ".join(summary["classes"])
        actual = summary.get("actual_duration", 0)
        actual_mins, actual_secs = divmod(int(actual), 60)

        # Dataset stats
        ds = summary.get("dataset", {})
        windows_saved = ds.get("windows_saved", 0)
        raw_samples = ds.get("raw_samples", 0)
        output_dir = summary.get("output_dir", "datasets/")

        self._summary_label.setText(
            f"{total} trials completed\n"
            f"Classes: {classes}\n"
            f"Duration: {actual_mins}m {actual_secs}s\n\n"
            f"Dataset saved:\n"
            f"• {windows_saved} trial windows extracted\n"
            f"• {raw_samples:,} raw recording samples\n"
            f"• Location: {output_dir}"
        )
        self._stack.setCurrentIndex(2)
