#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
get_correlation.py

Uso:
  python analyze_feature_correlations.py \
    --dataset ./dataset/original_dataset.csv \
    --out ./reports/correlations \
    --target "t_max_x+1"

Salida:
  - correlation_summary.csv
  - correlation_summary.md
  - plots/*.png
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
    success_ratio = parsed.notna().mean()
    return success_ratio >= min_success_ratio


def correlation_ratio(categories: pd.Series, values: pd.Series) -> float:
    """
    Correlation ratio / Eta.
    Sirve para medir asociación entre una variable categórica y un target numérico.
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
# Plotters
# -----------------------------

def plot_numeric_feature(df: pd.DataFrame, feature: str, target: str, output_path: Path, max_points: int):
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


def plot_categorical_feature(df: pd.DataFrame, feature: str, target: str, output_path: Path, max_categories: int = 25):
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


def plot_datetime_feature(df: pd.DataFrame, feature: str, target: str, output_path: Path, max_points: int):
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
# Analysis
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


def main():
    parser = argparse.ArgumentParser(
        description="Analiza correlación/asociación entre cada feature y un target numérico."
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
        help="Máximo de puntos por gráfico numérico para evitar imágenes muy pesadas. Default: 8000."
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

    if not pd.api.types.is_numeric_dtype(df[args.target]):
        df[args.target] = pd.to_numeric(df[args.target], errors="coerce")

    if df[args.target].isna().all():
        raise ValueError(f"El target '{args.target}' no pudo convertirse a numérico.")

    features = [col for col in df.columns if col != args.target]

    results = []

    print(f"\nDataset: {dataset_path}")
    print(f"Filas: {len(df):,}")
    print(f"Columnas: {len(df.columns):,}")
    print(f"Target: {args.target}")
    print(f"Features a analizar: {len(features):,}")
    print(f"Salida: {output_dir.resolve()}\n")

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
        na_position="last"
    )

    csv_path = output_dir / "correlation_summary.csv"
    md_path = output_dir / "correlation_summary.md"

    summary.to_csv(csv_path, index=False)

    md_cols = [
        "feature",
        "type",
        "association_score",
        "pearson",
        "spearman",
        "eta",
        "missing_pct_feature",
        "unique_values",
        "non_null_pair",
        "plot_path",
    ]

    md = "# Feature Correlation / Association Report\n\n"
    md += f"Dataset: `{dataset_path}`\n\n"
    md += f"Target: `{args.target}`\n\n"
    md += "Ordenado por `association_score` descendente.\n\n"
    md += summary[md_cols].to_markdown(index=False)

    md_path.write_text(md, encoding="utf-8")

    print("\n=== Listo ===")
    print(f"Resumen CSV: {csv_path.resolve()}")
    print(f"Resumen MD:  {md_path.resolve()}")
    print(f"Gráficos:    {plots_dir.resolve()}")


if __name__ == "__main__":
    main()