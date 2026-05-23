from __future__ import annotations

import argparse
import json
import logging
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt


try:
    from xgboost import XGBRegressor

    XGBOOST_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - optional dependency path
    XGBRegressor = None
    XGBOOST_IMPORT_ERROR = str(exc)

try:
    from lightgbm import LGBMRegressor

    LIGHTGBM_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - optional dependency path
    LGBMRegressor = None
    LIGHTGBM_IMPORT_ERROR = str(exc)


TARGET_COL = "t_max_x+1"
DATE_COLS = {"date", "date_str"}
CATEGORICAL_COLS = ["season", "ciudad"]
LOGGER = logging.getLogger("moira.train_sprint3")


@dataclass(frozen=True)
class ModelSpec:
    name: str
    library: str
    family: str
    scale_numeric: bool
    params: dict[str, Any]
    why: str
    factory: Callable[[], Any]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    root = repo_root()
    here = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description=(
            "Train sprint3 Tmax x+1 models with leakage-safe temporal backtesting "
            "and model selection by validation MAE."
        )
    )
    parser.add_argument("--dataset", default=str(root / "data" / "processed" / "sprint3.csv"))
    parser.add_argument("--report-output", default=str(here / "report.json"))
    parser.add_argument("--model-comparison-output", default=str(here / "model_comparison.csv"))
    parser.add_argument(
        "--val-predictions-output",
        default=str(here / "validation_predictions.csv"),
    )
    parser.add_argument(
        "--test-predictions-output",
        default=str(here / "test_predictions.csv"),
    )
    parser.add_argument("--model-output", default=str(here / "models" / "best_model.joblib"))
    parser.add_argument("--plots-dir", default=str(here / "plots"))
    parser.add_argument(
        "--models",
        default="all",
        help=(
            "Comma-separated model names to run. Use 'all' for all available models. "
            "Example: hist_gbr,xgboost,lightgbm"
        ),
    )
    parser.add_argument(
        "--test-start-year",
        type=int,
        default=2021,
        help="Rows with date year >= this value are held out for final test.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
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


def available_model_specs() -> tuple[dict[str, ModelSpec], dict[str, str]]:
    specs: dict[str, ModelSpec] = {}
    unavailable: dict[str, str] = {}

    hist_params = {
        "loss": "absolute_error",
        "learning_rate": 0.025,
        "max_iter": 1200,
        "max_leaf_nodes": 63,
        "min_samples_leaf": 25,
        "l2_regularization": 0.03,
        "early_stopping": False,
        "random_state": 42,
    }
    specs["hist_gbr"] = ModelSpec(
        name="hist_gbr",
        library="scikit-learn",
        family="boosting",
        scale_numeric=False,
        params=hist_params,
        why="Strong tabular boosting baseline used in sprint2.",
        factory=lambda: HistGradientBoostingRegressor(**hist_params),
    )

    rf_params = {
        "n_estimators": 220,
        "max_depth": None,
        "min_samples_leaf": 2,
        "n_jobs": -1,
        "random_state": 42,
    }
    specs["random_forest"] = ModelSpec(
        name="random_forest",
        library="scikit-learn",
        family="bagging",
        scale_numeric=False,
        params=rf_params,
        why="Robust nonlinear ensemble baseline.",
        factory=lambda: RandomForestRegressor(**rf_params),
    )

    et_params = {
        "n_estimators": 180,
        "max_depth": None,
        "min_samples_leaf": 2,
        "n_jobs": -1,
        "random_state": 42,
    }
    specs["extra_trees"] = ModelSpec(
        name="extra_trees",
        library="scikit-learn",
        family="bagging",
        scale_numeric=False,
        params=et_params,
        why="High-variance randomization can improve tabular generalization.",
        factory=lambda: ExtraTreesRegressor(**et_params),
    )

    gbr_params = {
        "loss": "absolute_error",
        "learning_rate": 0.03,
        "n_estimators": 350,
        "max_depth": 3,
        "subsample": 0.9,
        "random_state": 42,
    }
    specs["gbr"] = ModelSpec(
        name="gbr",
        library="scikit-learn",
        family="boosting",
        scale_numeric=False,
        params=gbr_params,
        why="Classical boosting baseline optimized for MAE.",
        factory=lambda: GradientBoostingRegressor(**gbr_params),
    )

    ridge_params = {
        "alpha": 1.5,
        "random_state": 42,
    }
    specs["ridge"] = ModelSpec(
        name="ridge",
        library="scikit-learn",
        family="linear",
        scale_numeric=True,
        params=ridge_params,
        why="Regularized linear baseline for calibration.",
        factory=lambda: Ridge(**ridge_params),
    )

    enet_params = {
        "alpha": 0.002,
        "l1_ratio": 0.2,
        "max_iter": 7000,
        "random_state": 42,
    }
    specs["elasticnet"] = ModelSpec(
        name="elasticnet",
        library="scikit-learn",
        family="linear",
        scale_numeric=True,
        params=enet_params,
        why="Linear sparse baseline with mixed L1/L2 regularization.",
        factory=lambda: ElasticNet(**enet_params),
    )

    mlp_params = {
        "hidden_layer_sizes": (96, 48),
        "activation": "relu",
        "solver": "adam",
        "learning_rate_init": 0.001,
        "max_iter": 260,
        "alpha": 0.0008,
        "early_stopping": True,
        "n_iter_no_change": 16,
        "random_state": 42,
    }
    specs["mlp"] = ModelSpec(
        name="mlp",
        library="scikit-learn",
        family="neural_network",
        scale_numeric=True,
        params=mlp_params,
        why="Neural baseline without external deep learning frameworks.",
        factory=lambda: MLPRegressor(**mlp_params),
    )

    if XGBRegressor is None:
        unavailable["xgboost"] = XGBOOST_IMPORT_ERROR or "not installed"
    else:
        xgb_params = {
            "objective": "reg:absoluteerror",
            "n_estimators": 450,
            "learning_rate": 0.03,
            "max_depth": 6,
            "subsample": 0.9,
            "colsample_bytree": 0.8,
            "min_child_weight": 5,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
            "random_state": 42,
            "n_jobs": -1,
            "tree_method": "hist",
        }
        specs["xgboost"] = ModelSpec(
            name="xgboost",
            library="xgboost",
            family="boosting",
            scale_numeric=False,
            params=xgb_params,
            why="State-of-the-art gradient boosting for tabular regression.",
            factory=lambda: XGBRegressor(**xgb_params),
        )

    if LGBMRegressor is None:
        unavailable["lightgbm"] = LIGHTGBM_IMPORT_ERROR or "not installed"
    else:
        lgbm_params = {
            "objective": "mae",
            "n_estimators": 450,
            "learning_rate": 0.03,
            "num_leaves": 63,
            "min_child_samples": 25,
            "subsample": 0.9,
            "colsample_bytree": 0.8,
            "reg_lambda": 0.1,
            "n_jobs": -1,
            "random_state": 42,
            "verbosity": -1,
        }
        specs["lightgbm"] = ModelSpec(
            name="lightgbm",
            library="lightgbm",
            family="boosting",
            scale_numeric=False,
            params=lgbm_params,
            why="Efficient histogram boosting for large tabular datasets.",
            factory=lambda: LGBMRegressor(**lgbm_params),
        )

    return specs, unavailable


def resolve_selected_models(
    model_arg: str,
    available_specs: dict[str, ModelSpec],
    unavailable_specs: dict[str, str],
) -> list[str]:
    if model_arg.strip().lower() == "all":
        return sorted(available_specs.keys())

    selected = [item.strip().lower() for item in model_arg.split(",") if item.strip()]
    if not selected:
        raise ValueError("No valid model names provided in --models.")

    unknown = [name for name in selected if name not in available_specs and name not in unavailable_specs]
    if unknown:
        raise ValueError(
            "Unknown model(s): "
            + ", ".join(unknown)
            + ". Available: "
            + ", ".join(sorted(list(available_specs.keys()) + list(unavailable_specs.keys())))
        )

    unavailable_selected = [name for name in selected if name in unavailable_specs]
    if unavailable_selected:
        messages = [f"{name}: {unavailable_specs[name]}" for name in unavailable_selected]
        raise RuntimeError(
            "Requested model(s) unavailable. Install dependencies or adjust --models. "
            + " | ".join(messages)
        )

    return selected


def make_pipeline(spec: ModelSpec, numeric_cols: list[str], categorical_cols: list[str]) -> Pipeline:
    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if spec.scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipe = Pipeline(numeric_steps)

    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return Pipeline(
        [
            ("preprocess", preprocessor),
            ("model", spec.factory()),
        ]
    )


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    errors = y_pred - y_true
    abs_errors = np.abs(errors)
    ss_res = float(np.sum(errors**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))

    return {
        "n": int(len(y_true)),
        "mae": float(np.mean(abs_errors)),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "median_ae": float(np.median(abs_errors)),
        "p90_ae": float(np.quantile(abs_errors, 0.90)),
        "bias": float(np.mean(errors)),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0,
    }


def grouped_mae(df: pd.DataFrame, group_col: str) -> dict[str, dict[str, float | int]]:
    out: dict[str, dict[str, float | int]] = {}
    for value, group in df.groupby(group_col, sort=True):
        abs_error = (group["prediction"] - group[TARGET_COL]).abs()
        out[str(value)] = {
            "n": int(len(group)),
            "mae": float(abs_error.mean()),
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


def fit_predict(
    spec: ModelSpec,
    train_df: pd.DataFrame,
    predict_df: pd.DataFrame,
    features: list[str],
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> tuple[np.ndarray, Pipeline, float]:
    fit_start = time.perf_counter()
    pipe = make_pipeline(spec, numeric_cols=numeric_cols, categorical_cols=categorical_cols)
    pipe.fit(train_df[features], train_df[TARGET_COL])
    predictions = pipe.predict(predict_df[features])
    elapsed = time.perf_counter() - fit_start
    return predictions, pipe, elapsed


def run_backtest(
    spec: ModelSpec,
    trainval_df: pd.DataFrame,
    features: list[str],
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> tuple[pd.DataFrame, list[dict[str, Any]], float]:
    years = sorted(trainval_df["year"].unique().tolist())
    prediction_frames = []
    folds = []
    total_elapsed = 0.0

    LOGGER.info(
        "Backtest start | model=%s | first_year=%s | last_year=%s | folds=%s",
        spec.name,
        years[0],
        years[-1],
        max(0, len(years) - 1),
    )

    for fold_number, val_year in enumerate(years[1:], start=1):
        fold_train = trainval_df[trainval_df["year"] < val_year]
        fold_val = trainval_df[trainval_df["year"] == val_year]
        if fold_train.empty or fold_val.empty:
            continue

        pred, _, fold_elapsed = fit_predict(
            spec=spec,
            train_df=fold_train,
            predict_df=fold_val,
            features=features,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
        )
        total_elapsed += fold_elapsed

        fold_predictions = fold_val[["date", "date_str", "year", "season", TARGET_COL]].copy()
        fold_predictions["prediction"] = pred
        fold_predictions["model_name"] = spec.name
        fold_predictions["train_start_year"] = int(fold_train["year"].min())
        fold_predictions["train_end_year"] = int(fold_train["year"].max())
        prediction_frames.append(fold_predictions)

        fold_metrics = regression_metrics(fold_val[TARGET_COL].to_numpy(dtype=float), pred)
        folds.append(
            {
                "validation_year": int(val_year),
                "train_start_year": int(fold_train["year"].min()),
                "train_end_year": int(fold_train["year"].max()),
                "train_rows": int(len(fold_train)),
                "validation_rows": int(len(fold_val)),
                "fit_predict_elapsed_sec": float(fold_elapsed),
                "metrics": fold_metrics,
            }
        )
        LOGGER.info(
            "Fold %02d/%02d | model=%s | val_year=%s | mae=%.4f | rmse=%.4f | elapsed_sec=%.2f",
            fold_number,
            len(years) - 1,
            spec.name,
            int(val_year),
            fold_metrics["mae"],
            fold_metrics["rmse"],
            fold_elapsed,
        )

    if not prediction_frames:
        return pd.DataFrame(), folds, total_elapsed
    return pd.concat(prediction_frames, ignore_index=True), folds, total_elapsed


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


def plot_model_comparison(mae_table: pd.DataFrame, plots_dir: Path) -> str:
    ordered = mae_table.sort_values("validation_mae", ascending=True)
    x = np.arange(len(ordered))
    width = 0.38

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width / 2, ordered["validation_mae"], width=width, label="Validation MAE")
    ax.bar(x + width / 2, ordered["test_mae"], width=width, label="Test MAE")
    ax.set_title("Model comparison by MAE")
    ax.set_xlabel("Model")
    ax.set_ylabel("MAE (C)")
    ax.set_xticks(x, ordered["model_name"], rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    return save_figure(fig, plots_dir / "model_comparison_mae.png")


def plot_validation_mae_by_year(folds: list[dict[str, Any]], plots_dir: Path) -> str:
    years = [fold["validation_year"] for fold in folds]
    maes = [fold["metrics"]["mae"] for fold in folds]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(years, maes, marker="o", linewidth=1.8)
    ax.axhline(np.mean(maes), color="tab:red", linestyle="--", linewidth=1.4, label="Mean MAE")
    ax.set_title("Validation MAE by year")
    ax.set_xlabel("Validation year")
    ax.set_ylabel("MAE (C)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    return save_figure(fig, plots_dir / "validation_mae_by_year.png")


def plot_test_mae_by_year(test_predictions_df: pd.DataFrame, plots_dir: Path) -> str:
    year_mae = (
        test_predictions_df.assign(abs_error=lambda d: (d["prediction"] - d[TARGET_COL]).abs())
        .groupby("year", sort=True)["abs_error"]
        .mean()
    )

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(year_mae.index.astype(str), year_mae.values)
    ax.set_title("Test MAE by year")
    ax.set_xlabel("Year")
    ax.set_ylabel("MAE (C)")
    ax.grid(axis="y", alpha=0.3)

    return save_figure(fig, plots_dir / "test_mae_by_year.png")


def plot_mae_by_season(
    validation_predictions: pd.DataFrame,
    test_predictions_df: pd.DataFrame,
    plots_dir: Path,
) -> str:
    seasons = ["winter", "spring", "summer", "autumn"]

    val_mae = (
        validation_predictions.assign(abs_error=lambda d: (d["prediction"] - d[TARGET_COL]).abs())
        .groupby("season", sort=True)["abs_error"]
        .mean()
        .reindex(seasons)
    )
    test_mae = (
        test_predictions_df.assign(abs_error=lambda d: (d["prediction"] - d[TARGET_COL]).abs())
        .groupby("season", sort=True)["abs_error"]
        .mean()
        .reindex(seasons)
    )

    x = np.arange(len(seasons))
    width = 0.38

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, val_mae.values, width=width, label="Validation")
    ax.bar(x + width / 2, test_mae.values, width=width, label="Test")
    ax.set_title("MAE by season")
    ax.set_xlabel("Season")
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
    ax.set_title("Test actual vs predicted")
    ax.set_xlabel("Actual Tmax x+1 (C)")
    ax.set_ylabel("Predicted Tmax x+1 (C)")
    ax.grid(True, alpha=0.3)

    return save_figure(fig, plots_dir / "test_actual_vs_predicted.png")


def plot_test_timeseries(test_predictions_df: pd.DataFrame, plots_dir: Path) -> str:
    ordered = test_predictions_df.sort_values("date")

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(ordered["date"], ordered[TARGET_COL], label="Actual", linewidth=1.3)
    ax.plot(ordered["date"], ordered["prediction"], label="Prediction", linewidth=1.2, alpha=0.85)
    ax.set_title("Test timeseries: actual vs predicted")
    ax.set_xlabel("Date X")
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
    ax.hist(val_errors, bins=45, alpha=0.55, label="Validation")
    ax.hist(test_errors, bins=35, alpha=0.65, label="Test")
    ax.axvline(0, color="black", linewidth=1.2)
    ax.set_title("Error distribution")
    ax.set_xlabel("Prediction error (C)")
    ax.set_ylabel("Frequency")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    return save_figure(fig, plots_dir / "error_distribution.png")


def generate_plots(
    comparison_df: pd.DataFrame,
    winner_folds: list[dict[str, Any]],
    winner_validation_predictions: pd.DataFrame,
    winner_test_predictions: pd.DataFrame,
    plots_dir: Path,
) -> dict[str, str]:
    LOGGER.info("Generating plots in: %s", plots_dir)
    return {
        "model_comparison_mae": plot_model_comparison(comparison_df, plots_dir),
        "validation_mae_by_year": plot_validation_mae_by_year(winner_folds, plots_dir),
        "test_mae_by_year": plot_test_mae_by_year(winner_test_predictions, plots_dir),
        "mae_by_season": plot_mae_by_season(
            winner_validation_predictions,
            winner_test_predictions,
            plots_dir,
        ),
        "test_actual_vs_predicted": plot_test_actual_vs_predicted(
            winner_test_predictions,
            plots_dir,
        ),
        "test_timeseries_actual_vs_predicted": plot_test_timeseries(
            winner_test_predictions,
            plots_dir,
        ),
        "error_distribution": plot_error_distribution(
            winner_validation_predictions,
            winner_test_predictions,
            plots_dir,
        ),
    }


def main() -> int:
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    args = parse_args()
    setup_logging(args.log_level)

    run_start = time.perf_counter()
    dataset_path = Path(args.dataset)
    report_output = Path(args.report_output)
    comparison_output = Path(args.model_comparison_output)
    val_predictions_output = Path(args.val_predictions_output)
    test_predictions_output = Path(args.test_predictions_output)
    model_output = Path(args.model_output)
    plots_dir = Path(args.plots_dir)

    available_specs, unavailable_specs = available_model_specs()
    selected_models = resolve_selected_models(args.models, available_specs, unavailable_specs)

    LOGGER.info("Models selected: %s", ", ".join(selected_models))
    if unavailable_specs:
        LOGGER.info("Unavailable optional models: %s", unavailable_specs)

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

    comparison_rows: list[dict[str, Any]] = []
    model_results: dict[str, dict[str, Any]] = {}

    for model_name in selected_models:
        spec = available_specs[model_name]
        model_start = time.perf_counter()
        LOGGER.info("Model start | %s", model_name)

        validation_predictions, folds, backtest_elapsed = run_backtest(
            spec=spec,
            trainval_df=trainval_df,
            features=features,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
        )
        if validation_predictions.empty:
            raise ValueError(f"Backtesting produced no validation predictions for model: {model_name}")

        validation_metrics = regression_metrics(
            validation_predictions[TARGET_COL].to_numpy(dtype=float),
            validation_predictions["prediction"].to_numpy(dtype=float),
        )

        test_predictions, final_model, final_fit_elapsed = fit_predict(
            spec=spec,
            train_df=trainval_df,
            predict_df=test_df,
            features=features,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
        )
        test_predictions_df = test_df[["date", "date_str", "year", "season", TARGET_COL]].copy()
        test_predictions_df["prediction"] = test_predictions
        test_predictions_df["model_name"] = model_name

        test_metrics = regression_metrics(
            test_predictions_df[TARGET_COL].to_numpy(dtype=float),
            test_predictions_df["prediction"].to_numpy(dtype=float),
        )

        elapsed = time.perf_counter() - model_start
        comparison_row = {
            "model_name": model_name,
            "library": spec.library,
            "family": spec.family,
            "validation_n": validation_metrics["n"],
            "validation_mae": validation_metrics["mae"],
            "validation_rmse": validation_metrics["rmse"],
            "validation_bias": validation_metrics["bias"],
            "test_n": test_metrics["n"],
            "test_mae": test_metrics["mae"],
            "test_rmse": test_metrics["rmse"],
            "test_bias": test_metrics["bias"],
            "backtest_elapsed_sec": float(backtest_elapsed),
            "final_fit_elapsed_sec": float(final_fit_elapsed),
            "total_elapsed_sec": float(elapsed),
        }
        comparison_rows.append(comparison_row)
        model_results[model_name] = {
            "spec": spec,
            "validation_predictions": validation_predictions,
            "folds": folds,
            "validation_metrics": validation_metrics,
            "test_predictions": test_predictions_df,
            "test_metrics": test_metrics,
            "final_model": final_model,
        }

        LOGGER.info(
            "Model done | %s | validation_mae=%.4f | test_mae=%.4f | elapsed_sec=%.2f",
            model_name,
            validation_metrics["mae"],
            test_metrics["mae"],
            elapsed,
        )

    comparison_df = pd.DataFrame(comparison_rows).sort_values("validation_mae", ascending=True)
    winner_name = str(comparison_df.iloc[0]["model_name"])
    winner = model_results[winner_name]
    winner_spec: ModelSpec = winner["spec"]
    winner_validation_predictions: pd.DataFrame = winner["validation_predictions"]
    winner_folds: list[dict[str, Any]] = winner["folds"]
    winner_test_predictions: pd.DataFrame = winner["test_predictions"]
    winner_model = winner["final_model"]

    comparison_output.parent.mkdir(parents=True, exist_ok=True)
    val_predictions_output.parent.mkdir(parents=True, exist_ok=True)
    test_predictions_output.parent.mkdir(parents=True, exist_ok=True)
    model_output.parent.mkdir(parents=True, exist_ok=True)

    comparison_df.to_csv(comparison_output, index=False)
    winner_validation_predictions.to_csv(val_predictions_output, index=False)
    winner_test_predictions.to_csv(test_predictions_output, index=False)
    joblib.dump(winner_model, model_output)

    validation_source = winner_validation_predictions.merge(
        df,
        on=["date", "date_str", "year", "season", TARGET_COL],
        how="left",
        suffixes=("", "_source"),
    )

    plots = generate_plots(
        comparison_df=comparison_df[["model_name", "validation_mae", "test_mae"]].copy(),
        winner_folds=winner_folds,
        winner_validation_predictions=winner_validation_predictions,
        winner_test_predictions=winner_test_predictions,
        plots_dir=plots_dir,
    )

    report = {
        "dataset": str(dataset_path),
        "target": TARGET_COL,
        "model_selection": {
            "criterion": "lowest_validation_mae",
            "winner": winner_name,
            "candidate_models": selected_models,
            "comparison_output": str(comparison_output),
        },
        "split": {
            "strategy": "temporal_holdout_plus_expanding_window_backtest",
            "test_rule": f"date.year >= {args.test_start_year}",
            "validation_rule": (
                "For each validation year y before test_start_year, train on all rows "
                "with year < y and validate on rows from y."
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
        "models": {
            name: {
                "library": model_results[name]["spec"].library,
                "family": model_results[name]["spec"].family,
                "scale_numeric": model_results[name]["spec"].scale_numeric,
                "params": model_results[name]["spec"].params,
                "why": model_results[name]["spec"].why,
                "validation_metrics": model_results[name]["validation_metrics"],
                "test_metrics": model_results[name]["test_metrics"],
            }
            for name in selected_models
        },
        "winner": {
            "model_name": winner_name,
            "library": winner_spec.library,
            "family": winner_spec.family,
            "params": winner_spec.params,
            "why": winner_spec.why,
            "validation": {
                "global_metrics": winner["validation_metrics"],
                "mae_by_year": {
                    str(fold["validation_year"]): {
                        "n": fold["metrics"]["n"],
                        "mae": fold["metrics"]["mae"],
                        "rmse": fold["metrics"]["rmse"],
                        "bias": fold["metrics"]["bias"],
                        "train_rows": fold["train_rows"],
                    }
                    for fold in winner_folds
                },
                "mae_by_season": grouped_mae(winner_validation_predictions, "season"),
                "folds": winner_folds,
                "baseline_metrics": baseline_metrics(validation_source),
            },
            "test": {
                "metrics": winner["test_metrics"],
                "mae_by_year": grouped_mae(winner_test_predictions, "year"),
                "mae_by_season": grouped_mae(winner_test_predictions, "season"),
                "baseline_metrics": baseline_metrics(test_df),
            },
        },
        "availability": {
            "available_models": sorted(list(available_specs.keys())),
            "unavailable_models": unavailable_specs,
        },
        "artifacts": {
            "report": str(report_output),
            "model_comparison": str(comparison_output),
            "validation_predictions": str(val_predictions_output),
            "test_predictions": str(test_predictions_output),
            "model": str(model_output),
            "plots": plots,
        },
        "data_quality_notes": [
            "Training uses sprint3.csv with known missing daily rows in 2026.",
            "The missing rows are outside train/validation years and do not leak future labels.",
        ],
        "leakage_controls": [
            "Rows after 2020 are never used in validation folds.",
            "The final test set contains only rows with year >= 2021.",
            "Each validation fold trains only with dates from previous years.",
            "Imputation, scaling and one-hot encoding are fitted inside each fold train set.",
            "date/date_str and the target are excluded from features.",
            "Winner selection uses validation MAE only.",
        ],
    }

    write_json(report, report_output)

    elapsed = time.perf_counter() - run_start
    LOGGER.info(
        "Training complete | winner=%s | validation_mae=%.4f | test_mae=%.4f | elapsed_sec=%.2f",
        winner_name,
        winner["validation_metrics"]["mae"],
        winner["test_metrics"]["mae"],
        elapsed,
    )
    LOGGER.info("Report written: %s", report_output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
