"""Shared utilities for the dosing NARX twin."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR / "config.yaml"
DEFAULT_DATASET_DIR = SCRIPT_DIR / "artifacts" / "dataset"

INPUT_COLUMNS = (
    "add_media_ml",
    "add_alt_media_ml",
    "remove_waste_ml",
    "dose_total_ml",
    "volume_ml",
    "cum_add_media_ml",
    "cum_add_alt_media_ml",
    "cum_remove_waste_ml",
    "cum_dose_total_ml",
)
TARGET_PREFIXES = ("norm_od", "growth_rate", "co2_ppm", "od_", "nm_")


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_tables(dataset_dir: Path) -> pd.DataFrame:
    tables = [pd.read_csv(path, parse_dates=["timestamp"]) for path in sorted((dataset_dir / "runs").glob("*.csv"))]
    if not tables:
        raise FileNotFoundError(f"No run tables in {dataset_dir / 'runs'}; run build_dataset.py first.")
    return pd.concat(tables, ignore_index=True).sort_values(["run_key", "timestamp"])


def state_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column.startswith(TARGET_PREFIXES)]


def observed_column(target: str) -> str:
    return f"observed_{target}"


def build_supervised_frame(frame: pd.DataFrame, lags: int, horizons: list[int]) -> tuple[pd.DataFrame, list[str]]:
    """Build lagged ARX features and masked one-step targets."""
    states = state_columns(frame)
    grouped = frame.groupby("run_key", group_keys=False)
    blocks: list[pd.DataFrame] = []

    for lag in range(lags + 1):
        lagged_inputs = grouped[list(INPUT_COLUMNS)].shift(lag)
        lagged_inputs.columns = [f"{column}_lag{lag}" for column in INPUT_COLUMNS]
        blocks.append(lagged_inputs)

        lagged_states = grouped[states].shift(lag)
        lagged_states.columns = [f"{column}_lag{lag}" for column in states]
        blocks.append(lagged_states)

    design = pd.concat(blocks, axis=1)
    design["condition"] = frame["condition"].astype(str)
    design["modality"] = frame["modality"].astype(str)
    design["run_key"] = frame["run_key"].astype(str)
    design["timestamp"] = frame["timestamp"]

    target_blocks: dict[str, pd.Series] = {}
    for horizon in horizons:
        for target in states:
            future = grouped[target].shift(-horizon)
            current = frame[target]
            future_obs = grouped[observed_column(target)].shift(-horizon).fillna(False).astype(bool)
            current_obs = frame[observed_column(target)].fillna(False).astype(bool)
            target_blocks[f"target__{target}__h{horizon}"] = future
            target_blocks[f"persistence__{target}__h{horizon}"] = current
            target_blocks[f"valid__{target}__h{horizon}"] = future_obs & current_obs

    design = pd.concat([design, pd.DataFrame(target_blocks)], axis=1)
    return design, states


def lag_feature_columns(design: pd.DataFrame, lags: int) -> list[str]:
    suffixes = tuple(f"_lag{lag}" for lag in range(lags + 1))
    return [column for column in design.columns if column.endswith(suffixes)]


def make_preprocessor(feature_columns_: list[str]) -> ColumnTransformer:
    categorical = ["condition", "modality"]
    return ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                feature_columns_,
            ),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )


def grouped_cv(groups: pd.Series, seed: int) -> GroupKFold:
    splits = max(2, min(5, groups.nunique() - 1))
    return GroupKFold(n_splits=splits)


def metrics_dict(y_true: pd.Series, y_pred: np.ndarray, persistence: pd.Series) -> dict[str, float]:
    return {
        "n_test_bins": int(len(y_true)),
        "mae": float(np.mean(np.abs(y_true - y_pred))),
        "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "r2": float(1.0 - np.sum((y_true - y_pred) ** 2) / max(np.sum((y_true - y_true.mean()) ** 2), 1e-12)),
        "n_persistence_bins": int(len(persistence)),
        "persistence_mae": float(np.mean(np.abs(y_true - persistence))),
        "persistence_rmse": float(np.sqrt(np.mean((y_true - persistence) ** 2))),
    }
