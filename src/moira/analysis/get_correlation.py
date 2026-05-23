#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
get_correlation.py

Uso:
  PYTHONPATH=src python -m moira.analysis.get_correlation \
    --dataset ./data/processed/sprint1/original_dataset.csv \
    --out ./reports/correlations \
    --target "t_max_x+1"

Salida:
  - correlation_summary.csv
  - plots/*.png
  - plots/spearman_feature_target_matrix.png
"""

from __future__ import annotations

import argparse
import math
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Helpers
# -----------------------------

def safe_filename(name: str, max_len: int = 120) -> str:
    name = str(name)
    name = unicodedata.normalize("NFKD", name)
    name = "".join(ch for ch in name if not unicodedata.combining(ch))
    name = name.lower()
    name = re.sub(r"[^a-z0-9_+-]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name[:max_len] or "feature"


def is_datetime_like(series: pd.Series, min_success_ratio: float = 0.80) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True

    if not pd.api.types.is_object_dtype(series):
        return False

    non_null = series.dropna()
    if len(non_null) == 0:
        return False

    parsed = pd.to_datetime(non_null, errors="coerce")
    return parsed.notna().mean() >= min_success_ratio


def correlation_ratio(categories: pd.Series, values: pd.Series) -> float:
    """
    Correlation ratio / Eta.
    Mide asociación entre una variable categórica y un target numérico.
    Rango: 0 a 1.
    """
    df = pd.DataFrame({"cat": categories, "y": values}).dropna()
    if df.empty:
        return np.nan

    y = df["y"].astype(float)
    grand_mean = y.mean()

    numerator = 0.0
    denominator = ((y - grand_mean) ** 2).sum()

    if denominator == 0:
        return np.nan

    for _, group in df.groupby("cat"):
        group_y = group["y"].astype(float)
        numerator += len(group_y) * ((group_y.mean() - grand_mean) ** 2)

    eta_squared = numerator / denominator
    return float(math.sqrt(max(0.0, eta_squared)))


def get_feature_type(series: pd.Series) -> str:
    if is_datetime_like(series):
        return "datetime"

    if pd.api.types.is_numeric_dtype(series):
        return "numeric"

    unique_count = series.nunique(dropna=True)

    if unique_count <= 50:
        return "categorical"

    return "high_cardinality_categorical"


def downsample_df(df: pd.DataFrame, max_points: int, seed: int = 42) -> pd.DataFrame:
    if len(df) <= max_points:
        return df
    return df.sample(n=max_points, random_state=seed)


# -----------------------------
# Plots feature vs target
# -----------------------------

def plot_numeric_feature(
    df: pd.DataFrame,
    feature: str,
    target: str,
    output_path: Path,
    max_points: int,
):
    data = df[[feature, target]].dropna()
    if data.empty:
        return

    data = downsample_df(data, max_points=max_points)

    x = data[feature].astype(float)
    y = data[target].astype(float)

    plt.figure(figsize=(9, 6))
    plt.scatter(x, y, alpha=0.35, s=12)

    if x.nunique() > 1:
        coeffs = np.polyfit(x, y, deg=1)
        x_line = np.linspace(x.min(), x.max(), 200)
        y_line = coeffs[0] * x_line + coeffs[1]
        plt.plot(x_line, y_line, linewidth=2)

    plt.title(f"{feature} vs {target}")
    plt.xlabel(feature)
    plt.ylabel(target)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_categorical_feature(
    df: pd.DataFrame,
    feature: str,
    target: str,
    output_path: Path,
    max_categories: int,
):
    data = df[[feature, target]].dropna()
    if data.empty:
        return

    counts = data[feature].astype(str).value_counts()
    top_categories = counts.head(max_categories).index

    data = data[data[feature].astype(str).isin(top_categories)].copy()
    data[feature] = data[feature].astype(str)

    grouped = [
        data.loc[data[feature] == cat, target].astype(float).values
        for cat in top_categories
    ]

    plt.figure(figsize=(max(10, len(top_categories) * 0.55), 6))
    plt.boxplot(grouped, labels=top_categories, showfliers=False)
    plt.title(f"{feature} vs {target}")
    plt.xlabel(feature)
    plt.ylabel(target)
    plt.xticks(rotation=45, ha="right")
    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_datetime_feature(
    df: pd.DataFrame,
    feature: str,
    target: str,
    output_path: Path,
    max_points: int,
):
    data = df[[feature, target]].dropna().copy()
    if data.empty:
        return

    data[feature] = pd.to_datetime(data[feature], errors="coerce")
    data = data.dropna()

    if data.empty:
        return

    data = data.sort_values(feature)
    data = downsample_df(data, max_points=max_points).sort_values(feature)

    plt.figure(figsize=(11, 6))
    plt.plot(data[feature], data[target].astype(float), linewidth=1)
    plt.title(f"{target} over {feature}")
    plt.xlabel(feature)
    plt.ylabel(target)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


# -----------------------------
# Spearman matrix
# -----------------------------

def get_numeric_columns_for_spearman(df: pd.DataFrame, target: str) -> list[str]:
    """
    Para una matriz Spearman común se incluyen columnas numéricas.
    Excluye columnas tipo fecha o texto no numérico.
    Incluye el target.
    """
    numeric_cols = []

    for col in df.columns:
        if is_datetime_like(df[col]):
            continue

        converted = pd.to_numeric(df[col], errors="coerce")
        non_null_ratio = converted.notna().mean()
        unique_count = converted.nunique(dropna=True)

        if non_null_ratio >= 0.80 and unique_count > 1:
            df[col] = converted
            numeric_cols.append(col)

    if target not in numeric_cols:
        raise ValueError(
            f"El target '{target}' no quedó incluido en la matriz Spearman. "
            "Verifica que sea numérico."
        )

    ordered_cols = [col for col in numeric_cols if col != target] + [target]
    return ordered_cols


def plot_spearman_matrix(
    df: pd.DataFrame,
    target: str,
    output_path: Path,
    annotate_limit: int = 25,
):
    cols = get_numeric_columns_for_spearman(df, target)

    if len(cols) < 2:
        print("[WARN] No hay suficientes columnas numéricas para generar matriz Spearman.")
        return

    corr_matrix = df[cols].corr(method="spearman")

    n_features = len(corr_matrix.columns)
    fig_size = max(10, n_features * 0.65)

    plt.figure(figsize=(fig_size, fig_size))

    im = plt.imshow(corr_matrix.values, vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, fraction=0.046, pad=0.04)

    plt.title("Spearman Correlation Matrix - Features + Target")

    plt.xticks(
        ticks=np.arange(n_features),
        labels=corr_matrix.columns,
        rotation=90,
    )

    plt.yticks(
        ticks=np.arange(n_features),
        labels=corr_matrix.index,
    )

    if n_features <= annotate_limit:
        for i in range(n_features):
            for j in range(n_features):
                value = corr_matrix.iloc[i, j]
                if pd.notna(value):
                    plt.text(
                        j,
                        i,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        fontsize=8,
                    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()

    print(f"Matriz Spearman guardada en: {output_path.resolve()}")


# -----------------------------
# Feature-target analysis
# -----------------------------

def analyze_feature(df: pd.DataFrame, feature: str, target: str) -> dict:
    x = df[feature]
    y = df[target]

    result = {
        "feature": feature,
        "type": get_feature_type(x),
        "rows_total": int(len(df)),
        "non_null_feature": int(x.notna().sum()),
        "non_null_target": int(y.notna().sum()),
        "non_null_pair": int(df[[feature, target]].dropna().shape[0]),
        "missing_pct_feature": float(x.isna().mean() * 100),
        "unique_values": int(x.nunique(dropna=True)),
        "pearson": np.nan,
        "spearman": np.nan,
        "eta": np.nan,
        "abs_pearson": np.nan,
        "abs_spearman": np.nan,
        "association_score": np.nan,
    }

    pair = df[[feature, target]].dropna()

    if pair.empty or pair[target].nunique() <= 1 or pair[feature].nunique() <= 1:
        return result

    if result["type"] == "numeric":
        x_num = pair[feature].astype(float)
        y_num = pair[target].astype(float)

        result["pearson"] = float(x_num.corr(y_num, method="pearson"))
        result["spearman"] = float(x_num.corr(y_num, method="spearman"))
        result["abs_pearson"] = abs(result["pearson"])
        result["abs_spearman"] = abs(result["spearman"])
        result["association_score"] = result["abs_spearman"]

    elif result["type"] in ["categorical", "high_cardinality_categorical"]:
        eta = correlation_ratio(pair[feature].astype(str), pair[target].astype(float))
        result["eta"] = eta
        result["association_score"] = eta

    elif result["type"] == "datetime":
        dt = pd.to_datetime(pair[feature], errors="coerce")
        valid = dt.notna()

        if valid.sum() > 2:
            ordinal = dt[valid].map(pd.Timestamp.toordinal).astype(float)
            y_num = pair.loc[valid, target].astype(float)

            result["pearson"] = float(ordinal.corr(y_num, method="pearson"))
            result["spearman"] = float(ordinal.corr(y_num, method="spearman"))
            result["abs_pearson"] = abs(result["pearson"])
            result["abs_spearman"] = abs(result["spearman"])
            result["association_score"] = result["abs_spearman"]

    return result


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Genera correlaciones feature-target y matriz Spearman features+target."
    )

    parser.add_argument(
        "--dataset",
        required=True,
        help="Ruta del CSV del dataset."
    )

    parser.add_argument(
        "--out",
        required=True,
        help="Carpeta de salida para reportes y gráficos."
    )

    parser.add_argument(
        "--target",
        required=True,
        help="Nombre de la columna target."
    )

    parser.add_argument(
        "--max-points",
        type=int,
        default=8000,
        help="Máximo de puntos por gráfico numérico. Default: 8000."
    )

    parser.add_argument(
        "--max-categories",
        type=int,
        default=25,
        help="Máximo de categorías a mostrar en gráficos categóricos. Default: 25."
    )

    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    output_dir = Path(args.out)
    plots_dir = output_dir / "plots"

    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    if not dataset_path.exists():
        raise FileNotFoundError(f"No existe el dataset: {dataset_path}")

    df = pd.read_csv(dataset_path)

    if args.target not in df.columns:
        raise ValueError(
            f"El target '{args.target}' no existe en el dataset. "
            f"Columnas disponibles: {list(df.columns)}"
        )

    df[args.target] = pd.to_numeric(df[args.target], errors="coerce")

    if df[args.target].isna().all():
        raise ValueError(f"El target '{args.target}' no pudo convertirse a numérico.")

    features = [col for col in df.columns if col != args.target]

    print(f"\nDataset: {dataset_path}")
    print(f"Filas: {len(df):,}")
    print(f"Columnas: {len(df.columns):,}")
    print(f"Target: {args.target}")
    print(f"Features a analizar: {len(features):,}")
    print(f"Salida: {output_dir.resolve()}\n")

    results = []

    # 1 y 2) CSV feature-target + gráficos individuales
    for i, feature in enumerate(features, start=1):
        print(f"[{i}/{len(features)}] Analizando: {feature}")

        result = analyze_feature(df, feature, args.target)
        results.append(result)

        feature_type = result["type"]
        plot_name = f"{safe_filename(feature)}__vs__{safe_filename(args.target)}.png"
        plot_path = plots_dir / plot_name

        try:
            if feature_type == "numeric":
                plot_numeric_feature(
                    df=df,
                    feature=feature,
                    target=args.target,
                    output_path=plot_path,
                    max_points=args.max_points,
                )

            elif feature_type in ["categorical", "high_cardinality_categorical"]:
                plot_categorical_feature(
                    df=df,
                    feature=feature,
                    target=args.target,
                    output_path=plot_path,
                    max_categories=args.max_categories,
                )

            elif feature_type == "datetime":
                plot_datetime_feature(
                    df=df,
                    feature=feature,
                    target=args.target,
                    output_path=plot_path,
                    max_points=args.max_points,
                )

            result["plot_path"] = str(plot_path)

        except Exception as e:
            result["plot_path"] = None
            result["plot_error"] = str(e)

    summary = pd.DataFrame(results)

    summary = summary.sort_values(
        by="association_score",
        ascending=False,
        na_position="last",
    )

    csv_path = output_dir / "correlation_summary.csv"
    summary.to_csv(csv_path, index=False)

    # 3) Gráfico Spearman entre features numéricas + target
    spearman_plot_path = plots_dir / "spearman_feature_target_matrix.png"

    plot_spearman_matrix(
        df=df.copy(),
        target=args.target,
        output_path=spearman_plot_path,
    )

    print("\n=== Listo ===")
    print(f"CSV correlaciones feature-target: {csv_path.resolve()}")
    print(f"Gráficos feature-target:          {plots_dir.resolve()}")
    print(f"Matriz Spearman:                  {spearman_plot_path.resolve()}")


if __name__ == "__main__":
    main()
