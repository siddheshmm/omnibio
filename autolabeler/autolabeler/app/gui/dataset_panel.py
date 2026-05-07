"""Dataset browser panel — browse and inspect saved datasets.

Shows a list of saved experiment sessions with basic stats:
  - Subject / Session
  - Number of trials, classes
  - Date, duration
"""

import json
import csv
import logging
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QTextEdit,
)

from autolabeler.config import DATASETS_DIR

logger = logging.getLogger(__name__)


class DatasetPanel(QWidget):
    """Browse and inspect saved experiment datasets."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._setup_ui()
        self.refresh()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Header
        header_row = QHBoxLayout()
        header_label = QLabel("Saved Datasets")
        header_label.setStyleSheet(
            "font-size: 16px; font-weight: bold; color: #89b4fa;"
        )
        header_row.addWidget(header_label)

        refresh_btn = QPushButton("↻ Refresh")
        refresh_btn.clicked.connect(self.refresh)
        refresh_btn.setMaximumWidth(100)
        header_row.addWidget(refresh_btn)
        layout.addLayout(header_row)

        # Tree widget
        self._tree = QTreeWidget()
        self._tree.setHeaderLabels(["Name", "Trials", "Classes", "Details"])
        self._tree.setColumnWidth(0, 200)
        self._tree.setColumnWidth(1, 80)
        self._tree.setColumnWidth(2, 150)
        self._tree.itemClicked.connect(self._on_item_clicked)
        self._tree.setStyleSheet("""
            QTreeWidget {
                background: #313244;
                border: 1px solid #45475a;
                border-radius: 6px;
                font-size: 13px;
            }
            QTreeWidget::item {
                padding: 4px;
            }
            QTreeWidget::item:selected {
                background: #45475a;
            }
            QHeaderView::section {
                background: #1e1e2e;
                color: #89b4fa;
                border: 1px solid #313244;
                padding: 6px;
                font-weight: bold;
            }
        """)
        layout.addWidget(self._tree)

        # Details panel
        details_group = QGroupBox("Session Details")
        details_layout = QVBoxLayout()
        self._details_text = QTextEdit()
        self._details_text.setReadOnly(True)
        self._details_text.setMaximumHeight(200)
        self._details_text.setStyleSheet(
            "background: #1e1e2e; color: #cdd6f4; border: none; "
            "font-family: 'Consolas', monospace; font-size: 12px;"
        )
        self._details_text.setPlaceholderText("Click a session to see details")
        details_layout.addWidget(self._details_text)
        details_group.setLayout(details_layout)
        layout.addWidget(details_group)

    def refresh(self) -> None:
        """Scan the datasets directory and populate the tree."""
        self._tree.clear()

        if not DATASETS_DIR.exists():
            empty = QTreeWidgetItem(["No datasets found"])
            self._tree.addTopLevelItem(empty)
            return

        # Scan: datasets/subject_id/session_id/
        subjects = sorted(
            [d for d in DATASETS_DIR.iterdir() if d.is_dir()],
            key=lambda d: d.name,
        )
        if not subjects:
            empty = QTreeWidgetItem(["No datasets found"])
            self._tree.addTopLevelItem(empty)
            return

        for subject_dir in subjects:
            subject_item = QTreeWidgetItem([subject_dir.name])
            subject_item.setExpanded(True)

            sessions = sorted(
                [d for d in subject_dir.iterdir() if d.is_dir()],
                key=lambda d: d.name,
            )
            for session_dir in sessions:
                meta = self._load_session_meta(session_dir)
                events_path = session_dir / "events.csv"
                trial_count = self._count_events(events_path)
                classes_str = ", ".join(meta.get("classes", []))
                details = meta.get("details", "")

                session_item = QTreeWidgetItem([
                    session_dir.name,
                    str(trial_count) if trial_count > 0 else "—",
                    classes_str or "—",
                    details,
                ])
                session_item.setData(0, Qt.ItemDataRole.UserRole, str(session_dir))
                subject_item.addChild(session_item)

            self._tree.addTopLevelItem(subject_item)

    def _load_session_meta(self, session_dir: Path) -> dict:
        """Load session metadata from metadata.json."""
        meta_path = session_dir / "metadata.json"
        result = {"classes": [], "details": ""}
        if not meta_path.exists():
            return result

        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

            exp_config = meta.get("experiment_config", {})
            result["classes"] = exp_config.get("classes", [])

            ds = meta.get("dataset", {})
            if ds:
                windows = ds.get("windows_saved", 0)
                raw = ds.get("raw_samples", 0)
                result["details"] = f"{windows} windows, {raw:,} samples"
            else:
                result["details"] = "events only"

        except Exception as e:
            logger.debug(f"Failed to read metadata from {meta_path}: {e}")

        return result

    def _count_events(self, events_path: Path) -> int:
        """Count the number of trial events in events.csv."""
        if not events_path.exists():
            return 0
        try:
            with open(events_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                next(reader, None)  # skip header
                return sum(1 for _ in reader)
        except Exception:
            return 0

    def _on_item_clicked(self, item: QTreeWidgetItem, column: int) -> None:
        """Show details of clicked session."""
        session_dir = item.data(0, Qt.ItemDataRole.UserRole)
        if not session_dir:
            self._details_text.clear()
            return

        session_path = Path(session_dir)
        lines = [f"Path: {session_path}\n"]

        # Metadata
        meta_path = session_path / "metadata.json"
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)

                exp = meta.get("experiment_config", {})
                lines.append("Experiment Config:")
                lines.append(f"  Classes: {exp.get('classes', [])}")
                lines.append(f"  Trial Duration: {exp.get('trial_duration', '?')}s")
                lines.append(f"  Rest Duration: {exp.get('rest_duration', '?')}s")
                lines.append(f"  Trials/Class: {exp.get('trials_per_class', '?')}")
                lines.append("")

                hw = meta.get("hardware_config", {})
                if hw:
                    lines.append("Hardware:")
                    lines.append(f"  {hw.get('name', '?')} @ {hw.get('sample_rate', '?')} Hz")
                    lines.append("")

                ds = meta.get("dataset", {})
                if ds:
                    lines.append("Dataset:")
                    lines.append(f"  Windows saved: {ds.get('windows_saved', 0)}")
                    lines.append(f"  Windows empty: {ds.get('windows_empty', 0)}")
                    lines.append(f"  Raw samples: {ds.get('raw_samples', 0):,}")
            except Exception as e:
                lines.append(f"Error reading metadata: {e}")

        # Files
        lines.append("\nFiles:")
        for f in sorted(session_path.rglob("*")):
            if f.is_file():
                size_kb = f.stat().st_size / 1024
                rel = f.relative_to(session_path)
                lines.append(f"  {rel}  ({size_kb:.1f} KB)")

        self._details_text.setPlainText("\n".join(lines))
