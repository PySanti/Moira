from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

matplotlib.use("Agg")
import matplotlib.pyplot as plt


TARGET_COL = "t_max_x+1"
DATE_COLS = {"date", "date_str"}
CATEGORICAL_COLS = ["season", "ciudad"]
LOGGER = logging.getLogger("moira.train_sprint2")

MODEL_PARAMS = {
    "loss": "absolute_error",
    "learning_rate": 0.025,
    "max_iter": 1200,
    "max_leaf_nodes": 63,
    "min_samples_leaf": 25,
    "l2_regularization": 0.03,
    "early_stopping": False,
    "random_state": 42,
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    root = repo_root()
    here = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Train sprint2 Tmax x+1 model with leakage-safe temporal backtesting."
    )
    parser.add_argument("--dataset", default=str(root / "dataset" / "sprint2.csv"))
    parser.add_argument("--report-output", default=str(here / "report.json"))
    parser.add_argument(
        "--val-predictions-output",
        default=str(here / "validation_predictions.csv"),
    )
    parser.add_argument(
        "--test-predictions-output",
        default=str(here / "test_predictions.csv"),
    )
    parser.add_argument("--model-output", default=str(here / "best_model.joblib"))
    parser.add_argument("--plots-dir", default=str(here / "plots"))
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    parser.add_argument(
        "--test-start-year",
        type=int,
        default=2021,
        help="Rows with date year >= this value are held out for final test.",
    )
    return parser.parse_args()


def setup_logging(log_level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_dataset(path: Path) -> pd.DataFrame:
    LOGGER.info("Loading dataset: %s", path)
    df = pd.read_csv(path)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COL}")
    if "date" not in df.columns:
        raise ValueError("Missing date column.")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", TARGET_COL]).copy()
    df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    df["year"] = df["date"].dt.year.astype(int)

    LOGGER.info(
        "Dataset loaded | rows=%s | columns=%s | date_min=%s | date_max=%s",
        len(df),
        len(df.columns),
        df["date"].min().date().isoformat(),
        df["date"].max().date().isoformat(),
    )

    return df.reset_index(drop=True)


def feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = set(DATE_COLS) | {TARGET_COL, "year"}
    return [col for col in df.columns if col not in excluded]


def split_feature_types(df: pd.DataFrame, features: list[str]) -> tuple[list[str], list[str]]:
    categorical = [col for col in CATEGORICAL_COLS if col in features]
    numeric = [col for col in features if col not in categorical]
    return numeric, categorical


def make_pipeline(numeric_cols: list[str], categorical_cols: list[str]) -> Pipeline:
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    model = HistGradientBoostingRegressor(**MODEL_PARAMS)

    return Pipeline([
        ("preprocess", preprocessor),
        ("model", model),
    ])


def fit_predict(
    train_df: pd.DataFrame,
    predict_df: pd.DataFrame,
    features: list[str],
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> tuple[np.ndarray, Pipeline]:
    fit_start = time.perf_counter()
    pipe = make_pipeline(numeric_cols, categorical_cols)
    pipe.fit(train_df[features], train_df[TARGET_COL])
    predictions = pipe.predict(predict_df[features])
    LOGGER.debug(
        "Fit/predict done | train_rows=%s | predict_rows=%s | elapsed_sec=%.2f",
        len(train_df),
        len(predict_df),
        time.perf_counter() - fit_start,
    )
    return predictions, pipe


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    errors = y_pred - y_true
    abs_errors = np.abs(errors)
    ss_res = float(np.sum(errors ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))

    return {
        "n": int(len(y_true)),
        "mae": float(np.mean(abs_errors)),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "median_ae": float(np.median(abs_errors)),
        "p90_ae": float(np.quantile(abs_errors, 0.90)),
        "bias": float(np.mean(errors)),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0,
    }


def grouped_mae(df: pd.DataFrame, group_col: str) -> dict[str, dict[str, float | int]]:
    out = {}

    for value, group in df.groupby(group_col, sort=True):
        y_true = group[TARGET_COL].to_numpy(dtype=float)
        y_pred = group["prediction"].to_numpy(dtype=float)
        out[str(value)] = {
            "n": int(len(group)),
            "mae": float(np.mean(np.abs(y_pred - y_true))),
        }

    return out


def baseline_metrics(df: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    baselines = [
        "Tmax_so_far_23h_x",
        "MA_Tmax_3d_asof_23h",
        "climatology_tmax_doy",
        "tmean_ma7",
    ]
    out = {}

    for col in baselines:
        if col not in df.columns:
            continue

        mask = df[col].notna()
        if not mask.any():
            continue

        y_true = df.loc[mask, TARGET_COL].to_numpy(dtype=float)
        y_pred = df.loc[mask, col].to_numpy(dtype=float)
        out[col] = regression_metrics(y_true, y_pred)

    return out


def run_backtest(
    trainval_df: pd.DataFrame,
    features: list[str],
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    years = sorted(trainval_df["year"].unique().tolist())
    prediction_frames = []
    folds = []

    LOGGER.info(
        "Starting expanding-window backtest | first_year=%s | last_year=%s | folds=%s",
        years[0],
        years[-1],
        max(0, len(years) - 1),
    )

    for fold_number, val_year in enumerate(years[1:], start=1):
        fold_train = trainval_df[trainval_df["year"] < val_year]
        fold_val = trainval_df[trainval_df["year"] == val_year]

        if fold_train.empty or fold_val.empty:
            continue

        fold_start = time.perf_counter()
        LOGGER.info(
            "Fold %02d/%02d | train_years=%s-%s | val_year=%s | train_rows=%s | val_rows=%s",
            fold_number,
            len(years) - 1,
            int(fold_train["year"].min()),
            int(fold_train["year"].max()),
            int(val_year),
            len(fold_train),
            len(fold_val),
        )

        pred, _ = fit_predict(
            train_df=fold_train,
            predict_df=fold_val,
            features=features,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
        )

        fold_predictions = fold_val[[
            "date",
            "date_str",
            "year",
            "season",
            TARGET_COL,
        ]].copy()
        fold_predictions["prediction"] = pred
        fold_predictions["train_start_year"] = int(fold_train["year"].min())
        fold_predictions["train_end_year"] = int(fold_train["year"].max())
        prediction_frames.append(fold_predictions)

        y_val = fold_val[TARGET_COL].to_numpy(dtype=float)
        fold_metrics = regression_metrics(y_val, pred)
        folds.append({
            "validation_year": int(val_year),
            "train_start_year": int(fold_train["year"].min()),
            "train_end_year": int(fold_train["year"].max()),
            "train_rows": int(len(fold_train)),
            "validation_rows": int(len(fold_val)),
            "metrics": fold_metrics,
        })
        LOGGER.info(
            "Fold %02d done | val_year=%s | mae=%.4f | rmse=%.4f | bias=%.4f | elapsed_sec=%.2f",
            fold_number,
            int(val_year),
            fold_metrics["mae"],
            fold_metrics["rmse"],
            fold_metrics["bias"],
            time.perf_counter() - fold_start,
        )

    if not prediction_frames:
        return pd.DataFrame(), folds

    LOGGER.info("Backtest complete | folds_completed=%s", len(folds))
    return pd.concat(prediction_frames, ignore_index=True), folds


def write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def save_figure(fig: plt.Figure, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Plot written: %s", path)
    return str(path)


def plot_validation_mae_by_year(folds: list[dict[str, Any]], plots_dir: Path) -> str:
    years = [fold["validation_year"] for fold in folds]
    maes = [fold["metrics"]["mae"] for fold in folds]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(years, maes, marker="o", linewidth=1.8)
    ax.axhline(np.mean(maes), color="tab:red", linestyle="--", linewidth=1.4, label="MAE promedio")
    ax.set_title("MAE de validacion por anio")
    ax.set_xlabel("Anio validado")
    ax.set_ylabel("MAE (C)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    return save_figure(fig, plots_dir / "validation_mae_by_year.png")


def plot_mae_by_season(
    validation_predictions: pd.DataFrame,
    test_predictions_df: pd.DataFrame,
    plots_dir: Path,
) -> str:
    def seasonal_mae(df: pd.DataFrame) -> pd.Series:
        return df.groupby("season").apply(
            lambda group: np.mean(np.abs(group["prediction"] - group[TARGET_COL])),
            include_groups=False,
        )

    seasons = ["winter", "spring", "summer", "autumn"]
    val_mae = seasonal_mae(validation_predictions).reindex(seasons)
    test_mae = seasonal_mae(test_predictions_df).reindex(seasons)
    x = np.arange(len(seasons))
    width = 0.38

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, val_mae, width=width, label="Validacion")
    ax.bar(x + width / 2, test_mae, width=width, label="Test")
    ax.set_title("MAE por temporada")
    ax.set_xlabel("Temporada")
    ax.set_ylabel("MAE (C)")
    ax.set_xticks(x, seasons)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    return save_figure(fig, plots_dir / "mae_by_season.png")


def plot_test_actual_vs_predicted(test_predictions_df: pd.DataFrame, plots_dir: Path) -> str:
    y_true = test_predictions_df[TARGET_COL].to_numpy(dtype=float)
    y_pred = test_predictions_df["prediction"].to_numpy(dtype=float)
    lower = min(float(y_true.min()), float(y_pred.min()))
    upper = max(float(y_true.max()), float(y_pred.max()))

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_true, y_pred, alpha=0.55, s=18)
    ax.plot([lower, upper], [lower, upper], color="tab:red", linestyle="--", linewidth=1.5)
    ax.set_title("Test: real vs prediccion")
    ax.set_xlabel("Tmax real x+1 (C)")
    ax.set_ylabel("Tmax predicha x+1 (C)")
    ax.grid(True, alpha=0.3)

    return save_figure(fig, plots_dir / "test_actual_vs_predicted.png")


def plot_test_timeseries(test_predictions_df: pd.DataFrame, plots_dir: Path) -> str:
    ordered = test_predictions_df.sort_values("date")

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(ordered["date"], ordered[TARGET_COL], label="Real", linewidth=1.3)
    ax.plot(ordered["date"], ordered["prediction"], label="Prediccion", linewidth=1.2, alpha=0.85)
    ax.set_title("Test: Tmax real vs predicha en el tiempo")
    ax.set_xlabel("Fecha X")
    ax.set_ylabel("Tmax x+1 (C)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    return save_figure(fig, plots_dir / "test_timeseries_actual_vs_predicted.png")


def plot_error_distribution(
    validation_predictions: pd.DataFrame,
    test_predictions_df: pd.DataFrame,
    plots_dir: Path,
) -> str:
    val_errors = validation_predictions["prediction"] - validation_predictions[TARGET_COL]
    test_errors = test_predictions_df["prediction"] - test_predictions_df[TARGET_COL]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(val_errors, bins=45, alpha=0.55, label="Validacion")
    ax.hist(test_errors, bins=35, alpha=0.65, label="Test")
    ax.axvline(0, color="black", linewidth=1.2)
    ax.set_title("Distribucion de errores")
    ax.set_xlabel("Error prediccion - real (C)")
    ax.set_ylabel("Frecuencia")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    return save_figure(fig, plots_dir / "error_distribution.png")


def generate_plots(
    folds: list[dict[str, Any]],
    validation_predictions: pd.DataFrame,
    test_predictions_df: pd.DataFrame,
    plots_dir: Path,
) -> dict[str, str]:
    LOGGER.info("Generating plots in: %s", plots_dir)
    return {
        "validation_mae_by_year": plot_validation_mae_by_year(folds, plots_dir),
        "mae_by_season": plot_mae_by_season(
            validation_predictions,
            test_predictions_df,
            plots_dir,
        ),
        "test_actual_vs_predicted": plot_test_actual_vs_predicted(
            test_predictions_df,
            plots_dir,
        ),
        "test_timeseries_actual_vs_predicted": plot_test_timeseries(
            test_predictions_df,
            plots_dir,
        ),
        "error_distribution": plot_error_distribution(
            validation_predictions,
            test_predictions_df,
            plots_dir,
        ),
    }


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)

    run_start = time.perf_counter()
    dataset_path = Path(args.dataset)
    report_output = Path(args.report_output)
    val_predictions_output = Path(args.val_predictions_output)
    test_predictions_output = Path(args.test_predictions_output)
    model_output = Path(args.model_output)
    plots_dir = Path(args.plots_dir)

    df = load_dataset(dataset_path)
    features = feature_columns(df)
    numeric_cols, categorical_cols = split_feature_types(df, features)

    LOGGER.info(
        "Feature contract | total_features=%s | numeric=%s | categorical=%s | target=%s",
        len(features),
        len(numeric_cols),
        len(categorical_cols),
        TARGET_COL,
    )
    LOGGER.info("Model params: %s", MODEL_PARAMS)

    trainval_df = df[df["year"] < args.test_start_year].copy()
    test_df = df[df["year"] >= args.test_start_year].copy()

    if trainval_df.empty:
        raise ValueError("No rows available before test_start_year.")
    if test_df.empty:
        raise ValueError("No rows available for test split.")

    LOGGER.info(
        "Temporal split | trainval_rows=%s | trainval_years=%s-%s | test_rows=%s | test_years=%s-%s",
        len(trainval_df),
        int(trainval_df["year"].min()),
        int(trainval_df["year"].max()),
        len(test_df),
        int(test_df["year"].min()),
        int(test_df["year"].max()),
    )

    validation_predictions, folds = run_backtest(
        trainval_df=trainval_df,
        features=features,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
    )

    if validation_predictions.empty:
        raise ValueError("Backtesting produced no validation predictions.")

    LOGGER.info("Training final model on all pre-test rows.")
    final_fit_start = time.perf_counter()
    test_predictions, final_model = fit_predict(
        train_df=trainval_df,
        predict_df=test_df,
        features=features,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
    )
    LOGGER.info("Final model trained | elapsed_sec=%.2f", time.perf_counter() - final_fit_start)

    test_predictions_df = test_df[["date", "date_str", "year", "season", TARGET_COL]].copy()
    test_predictions_df["prediction"] = test_predictions

    val_predictions_output.parent.mkdir(parents=True, exist_ok=True)
    test_predictions_output.parent.mkdir(parents=True, exist_ok=True)
    model_output.parent.mkdir(parents=True, exist_ok=True)

    validation_predictions.to_csv(val_predictions_output, index=False)
    test_predictions_df.to_csv(test_predictions_output, index=False)
    joblib.dump(final_model, model_output)

    validation_y = validation_predictions[TARGET_COL].to_numpy(dtype=float)
    validation_pred = validation_predictions["prediction"].to_numpy(dtype=float)
    test_y = test_predictions_df[TARGET_COL].to_numpy(dtype=float)
    validation_metrics = regression_metrics(validation_y, validation_pred)
    test_metrics = regression_metrics(test_y, test_predictions)

    validation_source = validation_predictions.merge(
        df,
        on=["date", "date_str", "year", "season", TARGET_COL],
        how="left",
        suffixes=("", "_source"),
    )
    plots = generate_plots(
        folds=folds,
        validation_predictions=validation_predictions,
        test_predictions_df=test_predictions_df,
        plots_dir=plots_dir,
    )

    report = {
        "dataset": str(dataset_path),
        "target": TARGET_COL,
        "algorithm": {
            "name": "HistGradientBoostingRegressor",
            "library": "scikit-learn",
            "params": MODEL_PARAMS,
            "why": (
                "Gradient boosting handles nonlinear tabular interactions between "
                "temperature, humidity, pressure, wind and seasonality without using "
                "future information. The loss is absolute_error to align training "
                "with MAE."
            ),
        },
        "split": {
            "strategy": "temporal_holdout_plus_expanding_window_backtest",
            "test_rule": f"date.year >= {args.test_start_year}",
            "validation_rule": (
                "For each validation year y before test_start_year, train on all "
                "rows with year < y and validate on rows from y."
            ),
            "trainval_rows": int(len(trainval_df)),
            "test_rows": int(len(test_df)),
            "date_min": df["date"].min().date().isoformat(),
            "date_max": df["date"].max().date().isoformat(),
            "trainval_year_min": int(trainval_df["year"].min()),
            "trainval_year_max": int(trainval_df["year"].max()),
            "test_year_min": int(test_df["year"].min()),
            "test_year_max": int(test_df["year"].max()),
        },
        "columns": {
            "features": features,
            "numeric_features": numeric_cols,
            "categorical_features": categorical_cols,
            "dropped": sorted(list(DATE_COLS | {TARGET_COL, "year"})),
        },
        "validation": {
            "global_metrics": validation_metrics,
            "mae_by_year": {
                str(fold["validation_year"]): {
                    "n": fold["metrics"]["n"],
                    "mae": fold["metrics"]["mae"],
                    "rmse": fold["metrics"]["rmse"],
                    "bias": fold["metrics"]["bias"],
                    "train_rows": fold["train_rows"],
                }
                for fold in folds
            },
            "mae_by_season": grouped_mae(validation_predictions, "season"),
            "folds": folds,
            "baseline_metrics": baseline_metrics(validation_source),
        },
        "test": {
            "metrics": test_metrics,
            "mae_by_year": grouped_mae(test_predictions_df, "year"),
            "mae_by_season": grouped_mae(test_predictions_df, "season"),
            "baseline_metrics": baseline_metrics(test_df),
        },
        "artifacts": {
            "report": str(report_output),
            "validation_predictions": str(val_predictions_output),
            "test_predictions": str(test_predictions_output),
            "model": str(model_output),
            "plots": plots,
        },
        "leakage_controls": [
            "Rows after 2020 are never used in validation.",
            "The final test set contains only rows with year >= 2021.",
            "Each validation fold trains only with dates from previous years.",
            "Imputation and one-hot encoding are fitted inside each fold train set.",
            "date/date_str and the target are excluded from features.",
            "Model hyperparameters are fixed before backtesting, so validation metrics are not tuned on validation years.",
        ],
    }

    write_json(report, report_output)

    elapsed = time.perf_counter() - run_start
    LOGGER.info(
        "Training complete | validation_mae=%.4f | test_mae=%.4f | elapsed_sec=%.2f",
        validation_metrics["mae"],
        test_metrics["mae"],
        elapsed,
    )
    LOGGER.info("Report written: %s", report_output)
    LOGGER.info("Model written: %s", model_output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
