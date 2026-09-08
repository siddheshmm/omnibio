"""Recursively simulate held-out runs with one-step Ridge NARX models."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from narx_common import INPUT_COLUMNS, load_tables


SCRIPT_DIR = Path(__file__).resolve().parent


def feature_row(
    run: pd.DataFrame,
    simulated: pd.DataFrame,
    position: int,
    lags: int,
    states: list[str],
    feature_columns: list[str],
) -> pd.DataFrame:
    values: dict[str, object] = {
        "condition": run.iloc[0]["condition"],
        "modality": run.iloc[0]["modality"],
    }
    for lag in range(lags + 1):
        source = run.iloc[position - lag]
        for input_column in INPUT_COLUMNS:
            values[f"{input_column}_lag{lag}"] = source[input_column]
        for state in states:
            if pd.notna(simulated.iloc[position - lag][state]):
                values[f"{state}_lag{lag}"] = simulated.iloc[position - lag][state]
            else:
                values[f"{state}_lag{lag}"] = source[state]
    return pd.DataFrame([{column: values.get(column, np.nan) for column in feature_columns + ["condition", "modality"]}])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=SCRIPT_DIR / "artifacts" / "dataset")
    parser.add_argument("--model-path", type=Path, default=SCRIPT_DIR / "artifacts" / "model" / "narx_models.joblib")
    parser.add_argument("--split-path", type=Path, default=SCRIPT_DIR / "artifacts" / "model" / "split.csv")
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "artifacts" / "rollouts")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    bundle = joblib.load(args.model_path)
    models = bundle["models"]
    preprocessor = bundle["preprocessor"]
    lags = bundle["lags"]
    states = bundle["state_columns"]
    feature_columns = bundle["feature_columns"] + ["condition", "modality"]

    missing = [state for state in states if f"{state}__h1" not in models]
    if missing:
        raise ValueError(f"No one-step model for: {missing}")

    split = pd.read_csv(args.split_path)
    test_runs = split.loc[split["partition"] == "test", "run_key"].iloc[0].split(";")
    data = load_tables(args.dataset_dir)

    summary = []
    for run_key in test_runs:
        run = data.loc[data["run_key"] == run_key].reset_index(drop=True)
        simulated = run[states].copy()
        simulated.iloc[lags + 1 :] = np.nan

        for position in range(lags, len(run) - 1):
            row = feature_row(run, simulated, position, lags, states, feature_columns)
            encoded = preprocessor.transform(row)
            for state in states:
                model_key = f"{state}__h1"
                if model_key not in models:
                    continue
                simulated.loc[position + 1, state] = models[model_key].predict(encoded)[0]

        output = pd.concat([run[["timestamp", "run_key"] + states], simulated.add_prefix("predicted_")], axis=1)
        output.to_csv(args.output_dir / f"{run_key.replace(':', '__')}.csv", index=False)

        for state in states:
            actual = run[state].iloc[lags + 1 :]
            predicted = simulated[state].iloc[lags + 1 :]
            valid = actual.notna() & predicted.notna()
            if valid.sum() < 10:
                continue
            y = actual[valid]
            yhat = predicted[valid]
            persistence = run[state].iloc[lags:-1].reset_index(drop=True)[valid.reset_index(drop=True)]
            summary.append(
                {
                    "run_key": run_key,
                    "target": state,
                    "n_bins": int(valid.sum()),
                    "mae": mean_absolute_error(y, yhat),
                    "rmse": mean_squared_error(y, yhat) ** 0.5,
                    "r2": r2_score(y, yhat),
                    "persistence_mae": mean_absolute_error(y, persistence),
                }
            )

    pd.DataFrame(summary).to_csv(args.output_dir / "rollout_metrics.csv", index=False)
    print(f"wrote recursive rollouts for {len(test_runs)} held-out runs to {args.output_dir}")


if __name__ == "__main__":
    main()
