"""Training panel widget — dataset selection, pipeline config, results display.

Three-page stacked widget:
  - CONFIG: select dataset session, toggle preprocessing/features, start training
  - TRAINING: progress display
  - RESULTS: accuracy table, confusion matrix, export button
"""

import logging
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QComboBox,
    QCheckBox,
    QSpinBox,
    QFormLayout,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QMessageBox,
    QTextEdit,
    QHeaderView,
)

from autolabeler.config import MLConfig, DATASETS_DIR

logger = logging.getLogger(__name__)


class TrainWorker(QThread):
    """Background thread for running the ML pipeline."""

    progress = pyqtSignal(str, str)   # model_name, status
    finished = pyqtSignal(object, object)  # results, model
    error = pyqtSignal(str)

    def __init__(
        self, session_dirs: list[Path], ml_config: MLConfig, sample_rate: int = 1000
    ):
        super().__init__()
        self._session_dirs = session_dirs
        self._ml_config = ml_config
        self._sample_rate = sample_rate

    def run(self):
        try:
            from autolabeler.ml.dataset_loader import load_dataset, load_multiple_sessions
            from autolabeler.ml.preprocessor import preprocess
            from autolabeler.ml.features import extract_features
            from autolabeler.ml.trainer import train, save_model

            self.progress.emit("loader", "Loading dataset...")
            if len(self._session_dirs) == 1:
                X_raw, y, label_names = load_dataset(self._session_dirs[0])
            else:
                X_raw, y, label_names = load_multiple_sessions(self._session_dirs)

            self.progress.emit("preprocessor", "Preprocessing...")
            X_proc = preprocess(X_raw, self._ml_config)

            self.progress.emit("features", "Extracting features...")
            X_feat, feat_names = extract_features(X_proc, self._ml_config)

            def progress_cb(model_name, status):
                self.progress.emit(model_name, status)

            results, best_model = train(
                X_feat, y, label_names, feat_names,
                self._ml_config, progress_callback=progress_cb,
            )

            # Save model (into the first session's directory)
            model_dir = self._session_dirs[0] / "model"
            save_model(best_model, results, model_dir)

            self.finished.emit(results, best_model)

        except Exception as e:
            logger.error(f"Training failed: {e}", exc_info=True)
            self.error.emit(str(e))


class TrainPanel(QWidget):
    """Training configuration and results display."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._worker: Optional[TrainWorker] = None
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        self._stack = QStackedWidget()

        # Page 0: Config
        self._config_page = self._build_config_page()
        self._stack.addWidget(self._config_page)

        # Page 1: Training progress
        self._progress_page = self._build_progress_page()
        self._stack.addWidget(self._progress_page)

        # Page 2: Results
        self._results_page = self._build_results_page()
        self._stack.addWidget(self._results_page)

        self._stack.setCurrentIndex(0)
        layout.addWidget(self._stack)

    def _build_config_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        # --- Dataset Selection ---
        ds_group = QGroupBox("Dataset")
        ds_layout = QFormLayout()

        self._session_combo = QComboBox()
        self._refresh_sessions()
        ds_layout.addRow("Session:", self._session_combo)

        refresh_btn = QPushButton("↻ Refresh")
        refresh_btn.setMaximumWidth(80)
        refresh_btn.clicked.connect(self._refresh_sessions)
        ds_layout.addRow("", refresh_btn)

        ds_group.setLayout(ds_layout)
        layout.addWidget(ds_group)

        # --- Preprocessing ---
        pre_group = QGroupBox("Preprocessing")
        pre_layout = QVBoxLayout()

        self._dc_check = QCheckBox("Remove DC offset (subtract mean)")
        self._dc_check.setChecked(True)
        pre_layout.addWidget(self._dc_check)

        self._bandpass_check = QCheckBox("Bandpass filter")
        self._bandpass_check.setChecked(False)
        pre_layout.addWidget(self._bandpass_check)

        self._normalize_check = QCheckBox("Z-score normalization")
        self._normalize_check.setChecked(False)
        pre_layout.addWidget(self._normalize_check)

        pre_group.setLayout(pre_layout)
        layout.addWidget(pre_group)

        # --- Features ---
        feat_group = QGroupBox("Features")
        feat_layout = QVBoxLayout()

        self._feature_checks = {}
        for feat in ["rms", "peak_to_peak", "std", "p90", "mean_abs"]:
            cb = QCheckBox(feat.replace("_", " ").title())
            cb.setChecked(True)
            self._feature_checks[feat] = cb
            feat_layout.addWidget(cb)

        feat_group.setLayout(feat_layout)
        layout.addWidget(feat_group)

        # --- Models ---
        model_group = QGroupBox("Models & Cross-Validation")
        model_layout = QVBoxLayout()

        self._model_checks = {}
        for model in ["random_forest", "logistic_regression", "gradient_boosting"]:
            cb = QCheckBox(model.replace("_", " ").title())
            cb.setChecked(True)
            self._model_checks[model] = cb
            model_layout.addWidget(cb)

        cv_row = QHBoxLayout()
        cv_row.addWidget(QLabel("CV Folds:"))
        self._cv_spin = QSpinBox()
        self._cv_spin.setRange(2, 20)
        self._cv_spin.setValue(5)
        cv_row.addWidget(self._cv_spin)
        cv_row.addStretch()
        model_layout.addLayout(cv_row)

        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

        # --- Train Button ---
        self._train_btn = QPushButton("🧠  Train Models")
        self._train_btn.setObjectName("startBtn")
        self._train_btn.setMinimumHeight(44)
        self._train_btn.clicked.connect(self._on_train)
        layout.addWidget(self._train_btn)

        layout.addStretch()
        return page

    def _build_progress_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._progress_label = QLabel("Training...")
        self._progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._progress_label.setStyleSheet(
            "font-size: 20px; font-weight: bold; color: #89b4fa; padding: 20px;"
        )
        layout.addWidget(self._progress_label)

        self._progress_detail = QLabel("")
        self._progress_detail.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._progress_detail.setStyleSheet(
            "font-size: 14px; color: #a6adc8; padding: 10px;"
        )
        layout.addWidget(self._progress_detail)

        layout.addStretch()
        return page

    def _build_results_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        # Results header
        self._results_header = QLabel("Training Complete")
        self._results_header.setStyleSheet(
            "font-size: 18px; font-weight: bold; color: #2ecc71; padding: 8px 0;"
        )
        layout.addWidget(self._results_header)

        # Accuracy table
        self._results_table = QTableWidget()
        self._results_table.setColumnCount(4)
        self._results_table.setHorizontalHeaderLabels(
            ["Model", "Accuracy", "± Std", "Time"]
        )
        self._results_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        self._results_table.setMaximumHeight(160)
        self._results_table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers
        )
        self._results_table.setStyleSheet("""
            QTableWidget {
                background: #313244;
                border: 1px solid #45475a;
                border-radius: 4px;
                gridline-color: #45475a;
            }
            QHeaderView::section {
                background: #1e1e2e;
                color: #89b4fa;
                border: 1px solid #313244;
                padding: 6px;
                font-weight: bold;
            }
        """)
        layout.addWidget(self._results_table)

        # Best model label
        self._best_label = QLabel("")
        self._best_label.setStyleSheet(
            "font-size: 14px; color: #f9e2af; padding: 8px 0;"
        )
        layout.addWidget(self._best_label)

        # Classification report
        report_group = QGroupBox("Classification Report (Best Model)")
        report_layout = QVBoxLayout()
        self._report_text = QTextEdit()
        self._report_text.setReadOnly(True)
        self._report_text.setMaximumHeight(180)
        self._report_text.setStyleSheet(
            "background: #1e1e2e; color: #cdd6f4; border: none; "
            "font-family: 'Consolas', monospace; font-size: 12px;"
        )
        report_layout.addWidget(self._report_text)
        report_group.setLayout(report_layout)
        layout.addWidget(report_group)

        # Confusion matrix
        cm_group = QGroupBox("Confusion Matrix (Best Model)")
        cm_layout = QVBoxLayout()
        self._cm_table = QTableWidget()
        self._cm_table.setMaximumHeight(150)
        self._cm_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._cm_table.setStyleSheet("""
            QTableWidget {
                background: #313244;
                border: 1px solid #45475a;
                gridline-color: #45475a;
            }
            QHeaderView::section {
                background: #1e1e2e;
                color: #89b4fa;
                border: 1px solid #313244;
                padding: 4px;
                font-size: 11px;
            }
        """)
        cm_layout.addWidget(self._cm_table)
        cm_group.setLayout(cm_layout)
        layout.addWidget(cm_group)

        # Buttons
        btn_row = QHBoxLayout()
        self._back_btn = QPushButton("← Configure New Training")
        self._back_btn.clicked.connect(lambda: self._stack.setCurrentIndex(0))
        btn_row.addWidget(self._back_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        layout.addStretch()
        return page

    # --- Actions ---

    def _refresh_sessions(self) -> None:
        self._session_combo.clear()
        if not DATASETS_DIR.exists():
            return
        for subj in sorted(DATASETS_DIR.iterdir()):
            if not subj.is_dir():
                continue
            # Collect valid sessions for this subject
            subject_sessions: list[Path] = []
            for sess in sorted(subj.iterdir()):
                if not sess.is_dir():
                    continue
                windows_dir = sess / "windows"
                if windows_dir.exists() and any(windows_dir.glob("trial_*.csv")):
                    subject_sessions.append(sess)

            if not subject_sessions:
                continue

            # Add an "all sessions" option for this subject
            self._session_combo.addItem(
                f"{subj.name} (all sessions)", f"subject:{subj}"
            )

            # Add each individual session
            for sess in subject_sessions:
                self._session_combo.addItem(
                    f"{subj.name}/{sess.name}", str(sess)
                )

    def _get_ml_config(self) -> MLConfig:
        features = [k for k, cb in self._feature_checks.items() if cb.isChecked()]
        models = [k for k, cb in self._model_checks.items() if cb.isChecked()]
        return MLConfig(
            dc_offset_removal=self._dc_check.isChecked(),
            bandpass_filter=self._bandpass_check.isChecked(),
            normalize=self._normalize_check.isChecked(),
            features=features,
            models=models,
            cv_folds=self._cv_spin.value(),
        )

    def _on_train(self) -> None:
        selection = self._session_combo.currentData()
        if not selection:
            QMessageBox.warning(
                self, "No Dataset",
                "No dataset session selected. Record an experiment first."
            )
            return

        ml_config = self._get_ml_config()
        if not ml_config.features:
            QMessageBox.warning(self, "No Features", "Select at least one feature.")
            return
        if not ml_config.models:
            QMessageBox.warning(self, "No Models", "Select at least one model.")
            return

        # Resolve selected sessions: either a single session directory
        # or all sessions under a subject.
        session_dirs: list[Path] = []
        if isinstance(selection, str) and selection.startswith("subject:"):
            subject_dir = Path(selection.split("subject:", 1)[1])
            if not subject_dir.exists():
                QMessageBox.warning(
                    self,
                    "Missing Subject",
                    f"Subject directory not found:\n{subject_dir}",
                )
                return

            # Gather all sessions with windows for this subject
            for sess in sorted(subject_dir.iterdir()):
                if not sess.is_dir():
                    continue
                windows_dir = sess / "windows"
                if windows_dir.exists() and any(windows_dir.glob("trial_*.csv")):
                    session_dirs.append(sess)

            if not session_dirs:
                QMessageBox.warning(
                    self,
                    "No Sessions",
                    f"No sessions with windows found under subject:\n{subject_dir.name}",
                )
                return

            # Validate that all sessions share the same class set
            from json import load as _json_load

            classes_ref: list[str] | None = None
            for sess in session_dirs:
                meta_path = sess / "metadata.json"
                if not meta_path.exists():
                    continue
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = _json_load(f)
                    exp_cfg = meta.get("experiment_config", {})
                    sess_classes = exp_cfg.get("classes", [])
                except Exception:
                    continue

                if classes_ref is None:
                    classes_ref = list(sess_classes)
                elif list(sess_classes) != list(classes_ref):
                    QMessageBox.critical(
                        self,
                        "Inconsistent Classes",
                        "Sessions for this subject define different class sets.\n\n"
                        "Please ensure all sessions under a subject use the same "
                        "classes before training on them together.",
                    )
                    return

        else:
            session_dirs = [Path(selection)]

        self._stack.setCurrentIndex(1)
        self._progress_label.setText("Training...")
        self._progress_detail.setText("Loading dataset...")

        self._worker = TrainWorker(session_dirs, ml_config)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_progress(self, step: str, status: str) -> None:
        self._progress_label.setText(step.replace("_", " ").title())
        self._progress_detail.setText(status)

    def _on_error(self, error_msg: str) -> None:
        QMessageBox.critical(self, "Training Failed", error_msg)
        self._stack.setCurrentIndex(0)

    def _on_finished(self, results, model) -> None:
        # Populate results table
        self._results_table.setRowCount(len(results.model_results))
        for i, r in enumerate(results.model_results):
            name_item = QTableWidgetItem(r.model_name.replace("_", " ").title())
            acc_item = QTableWidgetItem(f"{r.accuracy:.1%}")
            std_item = QTableWidgetItem(f"±{r.std_accuracy:.1%}")
            time_item = QTableWidgetItem(f"{r.train_time:.1f}s")

            # Highlight best row
            if r.model_name == results.best_model_name:
                for item in [name_item, acc_item, std_item, time_item]:
                    item.setBackground(Qt.GlobalColor.darkGreen)

            self._results_table.setItem(i, 0, name_item)
            self._results_table.setItem(i, 1, acc_item)
            self._results_table.setItem(i, 2, std_item)
            self._results_table.setItem(i, 3, time_item)

        # Best model label
        self._best_label.setText(
            f"🏆 Best: {results.best_model_name.replace('_', ' ').title()} "
            f"— {results.best_accuracy:.1%} accuracy"
        )

        # Classification report
        best = results.best_result
        if best:
            self._report_text.setPlainText(best.classification_report)

            # Confusion matrix
            cm = best.confusion_matrix
            n = len(cm)
            self._cm_table.setRowCount(n)
            self._cm_table.setColumnCount(n)
            self._cm_table.setHorizontalHeaderLabels(results.label_names)
            self._cm_table.setVerticalHeaderLabels(results.label_names)
            for r_idx in range(n):
                for c_idx in range(n):
                    self._cm_table.setItem(
                        r_idx, c_idx,
                        QTableWidgetItem(str(cm[r_idx][c_idx]))
                    )

        self._stack.setCurrentIndex(2)
