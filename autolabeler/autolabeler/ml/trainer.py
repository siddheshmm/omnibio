"""Model trainer — train classifiers with cross-validation and export.

Trains multiple sklearn classifiers using stratified K-fold cross-validation.
Returns per-model results including accuracy, confusion matrix, and the
trained model instances. Best model can be saved as .joblib.
"""

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from autolabeler.config import MLConfig

logger = logging.getLogger(__name__)

# Model registry
MODEL_REGISTRY = {
    "random_forest": lambda rs: RandomForestClassifier(
        n_estimators=100, random_state=rs, n_jobs=-1,
    ),
    "logistic_regression": lambda rs: LogisticRegression(
        max_iter=1000, random_state=rs,
    ),
    "gradient_boosting": lambda rs: GradientBoostingClassifier(
        n_estimators=100, random_state=rs,
    ),
}


@dataclass
class ModelResult:
    """Result from training a single model."""
    model_name: str
    accuracy: float
    std_accuracy: float
    confusion_matrix: list[list[int]]
    classification_report: str
    train_time: float
    cv_folds: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TrainingResults:
    """Results from the full training pipeline."""
    model_results: list[ModelResult] = field(default_factory=list)
    best_model_name: str = ""
    best_accuracy: float = 0.0
    label_names: list[str] = field(default_factory=list)
    n_samples: int = 0
    n_features: int = 0
    feature_names: list[str] = field(default_factory=list)

    @property
    def best_result(self) -> Optional[ModelResult]:
        for r in self.model_results:
            if r.model_name == self.best_model_name:
                return r
        return None

    def to_dict(self) -> dict:
        return {
            "best_model": self.best_model_name,
            "best_accuracy": self.best_accuracy,
            "label_names": self.label_names,
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "feature_names": self.feature_names,
            "models": [r.to_dict() for r in self.model_results],
        }


def train(
    X: np.ndarray,
    y: np.ndarray,
    label_names: list[str],
    feature_names: list[str],
    config: Optional[MLConfig] = None,
    progress_callback: Optional[callable] = None,
) -> tuple[TrainingResults, dict]:
    """Train multiple models with cross-validation.

    Args:
        X: Feature matrix, shape (n_samples, n_features).
        y: Labels, shape (n_samples,).
        label_names: Class name strings.
        feature_names: Feature name strings.
        config: ML configuration.
        progress_callback: Optional callback(model_name, status_str).

    Returns:
        Tuple of (TrainingResults, best_model_instance).
    """
    if config is None:
        config = MLConfig()

    results = TrainingResults(
        label_names=label_names,
        n_samples=X.shape[0],
        n_features=X.shape[1],
        feature_names=feature_names,
    )

    best_model = None
    best_acc = -1.0

    cv = StratifiedKFold(
        n_splits=min(config.cv_folds, len(y)),
        shuffle=True,
        random_state=config.random_state,
    )

    for model_name in config.models:
        if model_name not in MODEL_REGISTRY:
            logger.warning(f"Unknown model '{model_name}', skipping.")
            continue

        if progress_callback:
            progress_callback(model_name, "Training...")

        t0 = time.time()
        model = MODEL_REGISTRY[model_name](config.random_state)

        # Cross-validated predictions
        try:
            y_pred = cross_val_predict(model, X, y, cv=cv)
        except Exception as e:
            logger.error(f"Failed to train {model_name}: {e}")
            continue

        acc = accuracy_score(y, y_pred)
        cm = confusion_matrix(y, y_pred).tolist()
        report = classification_report(
            y, y_pred, target_names=label_names, zero_division=0,
        )

        # Compute per-fold accuracies for std
        fold_accs = []
        for train_idx, val_idx in cv.split(X, y):
            fold_model = MODEL_REGISTRY[model_name](config.random_state)
            fold_model.fit(X[train_idx], y[train_idx])
            fold_acc = accuracy_score(y[val_idx], fold_model.predict(X[val_idx]))
            fold_accs.append(fold_acc)

        train_time = time.time() - t0

        result = ModelResult(
            model_name=model_name,
            accuracy=acc,
            std_accuracy=float(np.std(fold_accs)),
            confusion_matrix=cm,
            classification_report=report,
            train_time=train_time,
            cv_folds=cv.n_splits,
        )
        results.model_results.append(result)

        logger.info(
            f"{model_name}: accuracy={acc:.3f} ± {result.std_accuracy:.3f} "
            f"({train_time:.1f}s)"
        )

        # Track best
        if acc > best_acc:
            best_acc = acc
            best_model_name = model_name
            # Fit on full dataset for export
            best_model = MODEL_REGISTRY[model_name](config.random_state)
            best_model.fit(X, y)

    results.best_model_name = best_model_name if best_model else ""
    results.best_accuracy = best_acc

    if progress_callback:
        progress_callback("done", f"Best: {best_model_name} ({best_acc:.1%})")

    return results, best_model


def save_model(
    model,
    results: TrainingResults,
    output_dir: Path,
) -> Path:
    """Save the best model and results to disk.

    Args:
        model: The trained sklearn model instance.
        results: Training results.
        output_dir: Directory to save model.joblib + results.json.

    Returns:
        Path to saved model file.
    """
    import joblib

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "model.joblib"
    joblib.dump(model, model_path)

    results_path = output_dir / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results.to_dict(), f, indent=2)

    logger.info(f"Model saved to {model_path}")
    return model_path
