"""Tests for the ML pipeline — loader, preprocessor, features, trainer."""

import csv
import json
import numpy as np
from pathlib import Path

from autolabeler.config import MLConfig
from autolabeler.ml.dataset_loader import load_dataset
from autolabeler.ml.preprocessor import preprocess, remove_dc_offset, normalize_zscore
from autolabeler.ml.features import extract_features, list_available_features
from autolabeler.ml.trainer import train, TrainingResults


def _create_synthetic_session(
    tmp_path: Path,
    n_trials_per_class: int = 10,
    n_samples: int = 500,
    n_channels: int = 1,
    classes: list[str] | None = None,
) -> Path:
    """Create a synthetic session directory with trial windows."""
    if classes is None:
        classes = ["A", "B"]

    session_dir = tmp_path / "subject_test" / "session_test"
    windows_dir = session_dir / "windows"
    windows_dir.mkdir(parents=True)

    # Write events.csv
    events_path = session_dir / "events.csv"
    with open(events_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["trial_number", "class_label", "block_number",
                          "start_time", "end_time", "duration"])
        trial = 0
        for block in range(n_trials_per_class):
            for cls in classes:
                trial += 1
                writer.writerow([trial, cls, block + 1, 0.0, 1.0, 1.0])

    # Write synthetic trial windows
    rng = np.random.RandomState(42)
    trial = 0
    for block in range(n_trials_per_class):
        for cls_idx, cls in enumerate(classes):
            trial += 1
            # Make classes distinguishable:
            # Class A: centered around 0, Class B: centered around 5
            mean = cls_idx * 5.0
            signal = rng.randn(n_samples, n_channels) + mean
            timestamps = np.linspace(0, 1, n_samples)
            data = np.column_stack([timestamps, signal])

            header_cols = ["timestamp"] + [f"ch{i+1}" for i in range(n_channels)]
            header = ",".join(header_cols)

            path = windows_dir / f"trial_{trial:03d}_{cls}.csv"
            np.savetxt(path, data, delimiter=",", header=header,
                       comments="", fmt="%.6f")

    # Write metadata
    meta = {
        "experiment_config": {
            "classes": classes,
            "trial_duration": 1.0,
        },
        "dataset": {
            "windows_saved": trial,
        },
    }
    with open(session_dir / "metadata.json", "w") as f:
        json.dump(meta, f)

    return session_dir


class TestDatasetLoader:
    """Test loading trial windows into arrays."""

    def test_loads_correct_shape(self, tmp_path):
        session = _create_synthetic_session(tmp_path, n_trials_per_class=5)
        X, y, labels = load_dataset(session, expected_samples=500)

        assert X.shape == (10, 500, 1)  # 5 per class × 2 classes
        assert y.shape == (10,)
        assert len(labels) == 2

    def test_labels_are_sorted(self, tmp_path):
        session = _create_synthetic_session(
            tmp_path, classes=["Zebra", "Apple"]
        )
        _, _, labels = load_dataset(session, expected_samples=500)
        assert labels == ["Apple", "Zebra"]

    def test_balanced_classes(self, tmp_path):
        session = _create_synthetic_session(tmp_path, n_trials_per_class=8)
        _, y, _ = load_dataset(session, expected_samples=500)
        unique, counts = np.unique(y, return_counts=True)
        assert len(unique) == 2
        assert all(c == 8 for c in counts)

    def test_multichannel(self, tmp_path):
        session = _create_synthetic_session(
            tmp_path, n_channels=3, n_trials_per_class=3
        )
        X, _, _ = load_dataset(session, expected_samples=500)
        assert X.shape[2] == 3

    def test_missing_windows_raises(self, tmp_path):
        try:
            load_dataset(tmp_path, expected_samples=100)
            assert False, "Expected FileNotFoundError"
        except FileNotFoundError:
            pass


class TestPreprocessor:
    """Test signal preprocessing."""

    def test_dc_removal(self):
        X = np.ones((5, 100, 1)) * 50.0
        X_proc = remove_dc_offset(X)
        assert np.allclose(X_proc.mean(axis=1), 0, atol=1e-10)

    def test_normalization(self):
        rng = np.random.RandomState(42)
        X = rng.randn(10, 200, 1) * 100 + 500
        X_norm = normalize_zscore(X)
        # After z-score, mean ≈ 0, std ≈ 1 per window
        for i in range(10):
            assert abs(X_norm[i, :, 0].mean()) < 0.01
            assert abs(X_norm[i, :, 0].std() - 1.0) < 0.01

    def test_preprocess_full(self):
        rng = np.random.RandomState(42)
        X = rng.randn(5, 1000, 1) * 10 + 100
        config = MLConfig(
            dc_offset_removal=True,
            bandpass_filter=False,
            normalize=True,
        )
        X_proc = preprocess(X, config)
        assert X_proc.shape == X.shape
        # Mean should be ~0 after DC removal + normalization
        for i in range(5):
            assert abs(X_proc[i, :, 0].mean()) < 0.01


class TestFeatures:
    """Test feature extraction."""

    def test_available_features(self):
        features = list_available_features()
        assert "rms" in features
        assert "peak_to_peak" in features
        assert "std" in features

    def test_feature_count_single_channel(self):
        X = np.random.randn(10, 100, 1)
        config = MLConfig(features=["rms", "std", "mean_abs"])
        X_feat, names = extract_features(X, config)
        assert X_feat.shape == (10, 3)
        assert len(names) == 3

    def test_feature_count_multichannel(self):
        X = np.random.randn(10, 100, 3)
        config = MLConfig(features=["rms", "std"])
        X_feat, names = extract_features(X, config)
        assert X_feat.shape == (10, 6)  # 2 features × 3 channels
        assert "rms_ch1" in names
        assert "std_ch3" in names

    def test_rms_known_signal(self):
        # Constant signal of 3.0 → RMS = 3.0
        X = np.full((1, 100, 1), 3.0)
        config = MLConfig(features=["rms"])
        X_feat, _ = extract_features(X, config)
        assert abs(X_feat[0, 0] - 3.0) < 0.01

    def test_peak_to_peak_known_signal(self):
        X = np.zeros((1, 100, 1))
        X[0, 10, 0] = -5.0
        X[0, 50, 0] = 10.0
        config = MLConfig(features=["peak_to_peak"])
        X_feat, _ = extract_features(X, config)
        assert abs(X_feat[0, 0] - 15.0) < 0.01


class TestTrainer:
    """Test model training pipeline."""

    def test_trains_multiple_models(self):
        rng = np.random.RandomState(42)
        # Two clearly separable classes
        X = np.vstack([rng.randn(20, 5), rng.randn(20, 5) + 5])
        y = np.array([0] * 20 + [1] * 20)
        label_names = ["A", "B"]
        feat_names = ["f1", "f2", "f3", "f4", "f5"]

        config = MLConfig(
            models=["random_forest", "logistic_regression"],
            cv_folds=3,
        )
        results, best_model = train(X, y, label_names, feat_names, config)

        assert len(results.model_results) == 2
        assert results.best_accuracy > 0.5
        assert best_model is not None

    def test_results_have_confusion_matrix(self):
        rng = np.random.RandomState(42)
        X = np.vstack([rng.randn(15, 3), rng.randn(15, 3) + 3])
        y = np.array([0] * 15 + [1] * 15)

        config = MLConfig(models=["random_forest"], cv_folds=3)
        results, _ = train(X, y, ["A", "B"], ["f1", "f2", "f3"], config)

        cm = results.model_results[0].confusion_matrix
        assert len(cm) == 2
        assert len(cm[0]) == 2

    def test_full_pipeline(self, tmp_path):
        """Integration: synthetic dataset → load → preprocess → features → train."""
        session = _create_synthetic_session(
            tmp_path, n_trials_per_class=15, n_samples=200
        )

        X_raw, y, labels = load_dataset(session, expected_samples=200)
        X_proc = preprocess(X_raw)
        X_feat, feat_names = extract_features(X_proc)
        results, model = train(X_feat, y, labels, feat_names)

        # With clearly separable classes, should get above chance (50% for 2 classes)
        assert results.best_accuracy > 0.5
        assert len(results.model_results) >= 1
        assert model is not None
