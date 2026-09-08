"""Train grouped, one-step regularized NARX sensor models (Ridge baseline)."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, GroupShuffleSplit

from narx_common import (
    DEFAULT_CONFIG,
    DEFAULT_DATASET_DIR,
    build_supervised_frame,
    grouped_cv,
    lag_feature_columns,
    load_config,
    load_tables,
    make_preprocessor,
    metrics_dict,
)


SCRIPT_DIR = Path(__file__).resolve().parent
ALPHAS = np.logspace(-4, 4, 25)


def fit_ridge(features: np.ndarray, target: np.ndarray, groups: np.ndarray, seed: int) -> Ridge:
    cv = grouped_cv(pd.Series(groups), seed)
    search = GridSearchCV(
        Ridge(),
        {"alpha": ALPHAS},
        cv=cv,
        scoring="neg_mean_absolute_error",
        n_jobs=1,
    )
    search.fit(features, target, groups=groups)
    return search.best_estimator_


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "artifacts" / "model")
    args = parser.parse_args()
    config = load_config(args.config)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw = load_tables(args.dataset_dir)
    lags = int(config.get("lags", 12))
    horizons = [int(value) for value in config.get("horizons", [1, 6, 12, 24])]
    design, targets = build_supervised_frame(raw, lags, horizons)
    feature_cols = lag_feature_columns(design, lags)

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=float(config.get("test_size", 0.2)),
        random_state=int(config.get("seed", 42)),
    )
    train_idx, test_idx = next(splitter.split(design, groups=design["run_key"]))
    train, test = design.iloc[train_idx], design.iloc[test_idx]

    preprocessor = make_preprocessor(feature_cols)
    preprocessor.fit(train[feature_cols + ["condition", "modality"]])
    train_features = preprocessor.transform(train[feature_cols + ["condition", "modality"]])
    test_features = preprocessor.transform(test[feature_cols + ["condition", "modality"]])

    train_targets = targets
    if str(config.get("training_targets", "all")).lower() == "core":
        train_targets = [target for target in config.get("core_targets", []) if target in targets]

    report = []
    models: dict[str, Ridge] = {}
    for horizon in horizons:
        for target in train_targets:
            target_col = f"target__{target}__h{horizon}"
            valid_col = f"valid__{target}__h{horizon}"
            persistence_col = f"persistence__{target}__h{horizon}"

            train_valid = train[valid_col].fillna(False)
            test_valid = test[valid_col].fillna(False)
            if train_valid.sum() < int(config.get("min_train_samples", 30)):
                continue
            if test_valid.sum() < int(config.get("min_test_samples", 10)):
                continue

            model = fit_ridge(
                train_features[train_valid.to_numpy()],
                train.loc[train_valid, target_col].to_numpy(),
                train.loc[train_valid, "run_key"].to_numpy(),
                int(config.get("seed", 42)),
            )
            prediction = model.predict(test_features[test_valid.to_numpy()])
            result = metrics_dict(
                test.loc[test_valid, target_col],
                prediction,
                test.loc[test_valid, persistence_col],
            )
            result.update(
                {
                    "target": target,
                    "horizon_bins": horizon,
                    "horizon_minutes": horizon * 5,
                    "alpha": float(model.alpha),
                }
            )
            report.append(result)
            models[f"{target}__h{horizon}"] = model

    pd.DataFrame(report).sort_values(["horizon_bins", "target"]).to_csv(args.output_dir / "metrics.csv", index=False)
    pd.DataFrame(
        {
            "partition": ["train", "test"],
            "run_key": [";".join(sorted(train.run_key.unique())), ";".join(sorted(test.run_key.unique()))],
        }
    ).to_csv(args.output_dir / "split.csv", index=False)
    joblib.dump(
        {
            "models": models,
            "preprocessor": preprocessor,
            "lags": lags,
            "feature_columns": feature_cols,
            "state_columns": targets,
            "horizons": horizons,
            "model_type": "ridge_narx",
        },
        args.output_dir / "narx_models.joblib",
    )
    print(f"trained {len(models)} Ridge NARX targets; metrics: {args.output_dir / 'metrics.csv'}")


if __name__ == "__main__":
    main()
