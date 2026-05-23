from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy.stats import kendalltau, ks_2samp, pearsonr, spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import mutual_info_regression


TARGET_COL = "t_max_x+1"
DATE_COL = "date"
EXCLUDE_COLS = {TARGET_COL, DATE_COL, "date_str", "ciudad"}
TRAIN_END_YEAR = 2020
TEST_END_YEAR = 2025
OUTPUT_DIR = Path(__file__).resolve().parent
PLOTS_DIR = OUTPUT_DIR / "plots"


def _safe_corr(method_name: str, x: pd.Series, y: pd.Series) -> float:
    df = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(df) < 3:
        return np.nan
    if df["x"].nunique() <= 1:
        return np.nan
    if method_name == "pearson":
        return float(pearsonr(df["x"], df["y"]).statistic)
    if method_name == "spearman":
        return float(spearmanr(df["x"], df["y"]).statistic)
    if method_name == "kendall":
        return float(kendalltau(df["x"], df["y"]).statistic)
    raise ValueError(f"Unsupported correlation method: {method_name}")


def _normalize_minmax(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    finite = s.replace([np.inf, -np.inf], np.nan)
    min_v = finite.min(skipna=True)
    max_v = finite.max(skipna=True)
    if pd.isna(min_v) or pd.isna(max_v) or min_v == max_v:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (finite - min_v) / (max_v - min_v)


def _feature_family(name: str) -> str:
    n = name.lower()
    if "climatology" in n or "anomaly" in n:
        return "climatology"
    if "lag" in n:
        return "temperature_lags"
    if "ma" in n or "trend" in n or "std" in n:
        return "temperature_rolling_trend"
    if "temp" in n or "tmax" in n or "tmin" in n or "tmean" in n or "dtr" in n:
        return "temperature_current"
    if "hr" in n or "td" in n or "vapor" in n:
        return "humidity_dewpoint"
    if "slp" in n or "pressure" in n:
        return "pressure"
    if "wind" in n:
        return "wind"
    if "precip" in n:
        return "precipitation"
    if "month" in n or "doy" in n or "season" in n:
        return "seasonality"
    if "daylight" in n:
        return "astronomical"
    if "flag" in n or "extreme" in n:
        return "flags"
    return "other"


def _make_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def _prepare_dataset() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str], List[str]]:
    repo_root = Path(__file__).resolve().parents[3]
    csv_path = repo_root / "data" / "processed" / "sprint3.csv"
    df = pd.read_csv(csv_path)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df["year"] = df[DATE_COL].dt.year

    train_df = df[df["year"] <= TRAIN_END_YEAR].copy()
    test_df = df[(df["year"] > TRAIN_END_YEAR) & (df["year"] <= TEST_END_YEAR)].copy()
    holdout_2026_df = df[df["year"] > TEST_END_YEAR].copy()

    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS and c != "year"]
    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]

    return train_df, test_df, holdout_2026_df, numeric_cols, categorical_cols


def _compute_univariate_metrics(train_df: pd.DataFrame, test_df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []

    baseline_pred = np.full(len(test_df), train_df[TARGET_COL].median())
    baseline_mae = mean_absolute_error(test_df[TARGET_COL], baseline_pred)

    yearly_stats = (
        train_df[["year", TARGET_COL]]
        .groupby("year", as_index=False)
        .agg(target_mean=(TARGET_COL, "mean"))
    )

    for col in numeric_cols:
        pearson = _safe_corr("pearson", train_df[col], train_df[TARGET_COL])
        spearman = _safe_corr("spearman", train_df[col], train_df[TARGET_COL])
        kendall = _safe_corr("kendall", train_df[col], train_df[TARGET_COL])

        x = train_df[[col, TARGET_COL]].dropna()
        mi = np.nan
        if len(x) > 10 and x[col].nunique() > 1:
            mi = float(mutual_info_regression(x[[col]], x[TARGET_COL], random_state=42)[0])

        missing_rate = float(train_df[col].isna().mean())
        near_zero_var = float(train_df[col].std(skipna=True) < 1e-8)

        # Univariate predictive signal (MAE lift over baseline)
        uni_mae = np.nan
        uni_lift = np.nan
        x_train = train_df[[col]].copy()
        y_train = train_df[TARGET_COL].copy()
        x_test = test_df[[col]].copy()
        y_test = test_df[TARGET_COL].copy()

        pipe = Pipeline(
            steps=[
                ("imp", SimpleImputer(strategy="median")),
                ("model", Ridge(alpha=1.0, random_state=42)),
            ]
        )
        try:
            pipe.fit(x_train, y_train)
            pred = pipe.predict(x_test)
            uni_mae = float(mean_absolute_error(y_test, pred))
            uni_lift = float((baseline_mae - uni_mae) / baseline_mae)
        except Exception:
            pass

        # Temporal stability: variability of yearly Pearson
        year_corrs: List[float] = []
        for y in sorted(train_df["year"].unique()):
            year_slice = train_df[train_df["year"] == y]
            year_corr = _safe_corr("pearson", year_slice[col], year_slice[TARGET_COL])
            if not np.isnan(year_corr):
                year_corrs.append(year_corr)

        if len(year_corrs) >= 3:
            stability = float(1.0 / (1.0 + np.std(year_corrs)))
        else:
            stability = 0.0

        # Drift by KS statistic and correlation shift
        drift_ks = np.nan
        corr_train = _safe_corr("pearson", train_df[col], train_df[TARGET_COL])
        corr_test = _safe_corr("pearson", test_df[col], test_df[TARGET_COL])
        corr_shift = np.nan
        a = train_df[col].dropna()
        b = test_df[col].dropna()
        if len(a) >= 30 and len(b) >= 30 and a.nunique() > 1 and b.nunique() > 1:
            drift_ks = float(ks_2samp(a, b).statistic)
        if not np.isnan(corr_train) and not np.isnan(corr_test):
            corr_shift = float(abs(corr_train - corr_test))

        rows.append(
            {
                "feature": col,
                "family": _feature_family(col),
                "pearson": pearson,
                "spearman": spearman,
                "kendall": kendall,
                "mutual_info": mi,
                "missing_rate": missing_rate,
                "near_zero_variance": near_zero_var,
                "univariate_mae": uni_mae,
                "univariate_lift": uni_lift,
                "temporal_stability": stability,
                "drift_ks": drift_ks,
                "corr_shift": corr_shift,
            }
        )

    metrics = pd.DataFrame(rows)
    return metrics


def _compute_model_importances(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    numeric_cols: List[str],
    categorical_cols: List[str],
) -> pd.DataFrame:
    feature_cols = numeric_cols + categorical_cols
    x_train = train_df[feature_cols]
    y_train = train_df[TARGET_COL]
    x_test = test_df[feature_cols]
    y_test = test_df[TARGET_COL]

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), numeric_cols),
            (
                "cat",
                Pipeline(
                    steps=[("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore"))]
                ),
                categorical_cols,
            ),
        ]
    )

    hgb = Pipeline(steps=[("pre", pre), ("model", HistGradientBoostingRegressor(random_state=42))])
    lgbm = Pipeline(
        steps=[
            ("pre", pre),
            (
                "model",
                LGBMRegressor(
                    random_state=42,
                    n_estimators=300,
                    learning_rate=0.05,
                    num_leaves=31,
                    subsample=0.9,
                    colsample_bytree=0.9,
                ),
            ),
        ]
    )

    models = {"hist_gbr": hgb, "lightgbm": lgbm}
    importance_rows: List[Dict[str, float]] = []

    for model_name, pipe in models.items():
        pipe.fit(x_train, y_train)

        perm = permutation_importance(
            pipe,
            x_test,
            y_test,
            n_repeats=5,
            random_state=42,
            scoring="neg_mean_absolute_error",
        )

        for i, f in enumerate(feature_cols):
            importance_rows.append(
                {
                    "feature": f,
                    f"perm_{model_name}": float(perm.importances_mean[i]),
                }
            )

        model_obj = pipe.named_steps["model"]
        if hasattr(model_obj, "feature_importances_"):
            fi = model_obj.feature_importances_
            if len(fi) == len(feature_cols):
                for i, f in enumerate(feature_cols):
                    importance_rows.append(
                        {
                            "feature": f,
                            f"gain_{model_name}": float(fi[i]),
                        }
                    )

    imp_df = pd.DataFrame(importance_rows)
    imp_df = imp_df.groupby("feature", as_index=False).sum(numeric_only=True)
    return imp_df


def _compute_redundancy(train_df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    corr = train_df[numeric_cols].corr(method="pearson").abs().copy()
    diag_mask = np.eye(len(corr), dtype=bool)
    corr = corr.mask(diag_mask)
    max_corr = corr.max(axis=1).rename("max_peer_corr").reset_index().rename(columns={"index": "feature"})
    return max_corr


def _plot_results(rank_df: pd.DataFrame, corr_top: pd.DataFrame) -> None:
    top10 = rank_df.head(10).copy()
    bottom10 = rank_df.tail(10).copy()

    plt.figure(figsize=(11, 6))
    plt.barh(top10["feature"].iloc[::-1], top10["utility_score"].iloc[::-1], color="#2f855a")
    plt.title("Top 10 features por utility score")
    plt.xlabel("Utility score")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "top_10_utility_score.png", dpi=150)
    plt.close()

    plt.figure(figsize=(11, 6))
    plt.barh(bottom10["feature"].iloc[::-1], bottom10["utility_score"].iloc[::-1], color="#c53030")
    plt.title("Bottom 10 features por utility score")
    plt.xlabel("Utility score")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "bottom_10_utility_score.png", dpi=150)
    plt.close()

    plt.figure(figsize=(9, 7))
    plt.scatter(rank_df["abs_pearson"], rank_df["abs_spearman"], alpha=0.5)
    plt.title("Abs Pearson vs Abs Spearman")
    plt.xlabel("Abs Pearson")
    plt.ylabel("Abs Spearman")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "pearson_vs_spearman.png", dpi=150)
    plt.close()

    mi_top = rank_df.sort_values("mutual_info", ascending=False).head(30)
    plt.figure(figsize=(12, 8))
    plt.barh(mi_top["feature"].iloc[::-1], mi_top["mutual_info"].iloc[::-1], color="#2b6cb0")
    plt.title("Top 30 Mutual Information")
    plt.xlabel("Mutual Information")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "mutual_information_top_30.png", dpi=150)
    plt.close()

    perm_cols = [c for c in rank_df.columns if c.startswith("perm_")]
    perm_avg = rank_df[["feature"] + perm_cols].copy()
    perm_avg["perm_avg"] = perm_avg[perm_cols].mean(axis=1)
    perm_top = perm_avg.sort_values("perm_avg", ascending=False).head(30)
    plt.figure(figsize=(12, 8))
    plt.barh(perm_top["feature"].iloc[::-1], perm_top["perm_avg"].iloc[::-1], color="#805ad5")
    plt.title("Top 30 Permutation Importance promedio")
    plt.xlabel("Permutation importance")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "permutation_importance_top_30.png", dpi=150)
    plt.close()

    miss = rank_df.sort_values("missing_rate", ascending=False).head(30)
    plt.figure(figsize=(12, 8))
    plt.barh(miss["feature"].iloc[::-1], miss["missing_rate"].iloc[::-1], color="#dd6b20")
    plt.title("Top 30 Missingness")
    plt.xlabel("Missing rate")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "missingness_by_feature.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 8))
    im = plt.imshow(corr_top.values, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im)
    plt.xticks(range(len(corr_top.columns)), corr_top.columns, rotation=90)
    plt.yticks(range(len(corr_top.index)), corr_top.index)
    plt.title("Correlacion entre top features")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "top_features_correlation_heatmap.png", dpi=150)
    plt.close()

    fam = rank_df.groupby("family", as_index=False)["utility_score"].mean().sort_values("utility_score", ascending=False)
    plt.figure(figsize=(10, 6))
    plt.barh(fam["family"].iloc[::-1], fam["utility_score"].iloc[::-1], color="#319795")
    plt.title("Utility score promedio por familia")
    plt.xlabel("Utility score promedio")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "family_utility_summary.png", dpi=150)
    plt.close()


def _build_readme_snippet(rank_df: pd.DataFrame, fam_df: pd.DataFrame) -> str:
    top10 = rank_df.head(10)
    bottom10 = rank_df.tail(10)

    lines: List[str] = []
    lines.append("# Sprint 4 - V0: Estudio de utilidad predictiva de features\n")
    lines.append("Se analizo `data/processed/sprint3.csv` con foco en explicar por que el rendimiento fuera de muestra no siempre cumple expectativas.\n")
    lines.append("\n")
    lines.append("Metodologia:\n")
    lines.append("- Split del estudio: train `1980-2020`, evaluacion externa `2021-2025`, y nota separada para 2026.\n")
    lines.append("- Ranking por `utility_score` compuesto (correlaciones, mutual information, importancia por permutacion, estabilidad temporal y penalizaciones por missingness, redundancia y drift).\n")
    lines.append("- Criterio principal: utilidad predictiva practica sobre modelo, no solo correlacion lineal.\n")
    lines.append("\n")

    lines.append("Top 10 features mas utiles:\n")
    lines.append("\n")
    lines.append("| Rank | Feature | Utility score | Abs Pearson | Abs Spearman | Mutual Info | Drift KS |\n")
    lines.append("| ---: | ------- | ------------: | ----------: | -----------: | ----------: | -------: |\n")
    for i, r in enumerate(top10.itertuples(index=False), start=1):
        lines.append(
            f"| {i} | `{r.feature}` | `{r.utility_score:.4f}` | `{r.abs_pearson:.4f}` | `{r.abs_spearman:.4f}` | `{r.mutual_info:.4f}` | `{r.drift_ks:.4f}` |\n"
        )

    lines.append("\n")
    lines.append("Top 10 features menos utiles:\n")
    lines.append("\n")
    lines.append("| Rank | Feature | Utility score | Missing rate | Redundancy max corr | Drift KS |\n")
    lines.append("| ---: | ------- | ------------: | -----------: | ------------------: | -------: |\n")
    for i, r in enumerate(bottom10.itertuples(index=False), start=1):
        lines.append(
            f"| {i} | `{r.feature}` | `{r.utility_score:.4f}` | `{r.missing_rate:.4f}` | `{r.max_peer_corr:.4f}` | `{r.drift_ks:.4f}` |\n"
        )

    lines.append("\n")
    lines.append("Lecturas clave:\n")
    top_fam = fam_df.iloc[0]["family"] if not fam_df.empty else "n/a"
    worst_fam = fam_df.iloc[-1]["family"] if not fam_df.empty else "n/a"
    lines.append(f"- La familia con mayor utilidad promedio fue `{top_fam}`.\n")
    lines.append(f"- La familia con menor utilidad promedio fue `{worst_fam}`.\n")
    lines.append("- Las features con baja utilidad suelen combinar baja senal, alta redundancia o drift temporal significativo.\n")
    lines.append("- El deterioro fuera de muestra se asocia a cambio de distribucion y menor estabilidad temporal en parte del set de variables.\n")
    lines.append("\n")
    lines.append("Graficos:\n")
    lines.append("\n")
    lines.append("![Top 10 utility score](./experiments/v0-sprint4/correlation_1.0/plots/top_10_utility_score.png)\n")
    lines.append("![Bottom 10 utility score](./experiments/v0-sprint4/correlation_1.0/plots/bottom_10_utility_score.png)\n")
    lines.append("![Pearson vs Spearman](./experiments/v0-sprint4/correlation_1.0/plots/pearson_vs_spearman.png)\n")
    lines.append("![Mutual information top 30](./experiments/v0-sprint4/correlation_1.0/plots/mutual_information_top_30.png)\n")
    lines.append("![Permutation importance top 30](./experiments/v0-sprint4/correlation_1.0/plots/permutation_importance_top_30.png)\n")
    lines.append("![Missingness by feature](./experiments/v0-sprint4/correlation_1.0/plots/missingness_by_feature.png)\n")
    lines.append("![Top features heatmap](./experiments/v0-sprint4/correlation_1.0/plots/top_features_correlation_heatmap.png)\n")
    lines.append("![Utility por familia](./experiments/v0-sprint4/correlation_1.0/plots/family_utility_summary.png)\n")

    return "".join(lines)


def main() -> None:
    warnings.filterwarnings(
        "ignore",
        message="X does not have valid feature names, but LGBMRegressor was fitted with feature names",
    )
    _make_dirs()
    train_df, test_df, holdout_2026_df, numeric_cols, categorical_cols = _prepare_dataset()

    metrics_df = _compute_univariate_metrics(train_df, test_df, numeric_cols)
    imp_df = _compute_model_importances(train_df, test_df, numeric_cols, categorical_cols)
    red_df = _compute_redundancy(train_df, numeric_cols)

    rank_df = metrics_df.merge(imp_df, on="feature", how="left").merge(red_df, on="feature", how="left")

    for c in rank_df.columns:
        if c.startswith("perm_") or c.startswith("gain_"):
            rank_df[c] = rank_df[c].fillna(0.0)

    rank_df["abs_pearson"] = rank_df["pearson"].abs()
    rank_df["abs_spearman"] = rank_df["spearman"].abs()
    rank_df["abs_kendall"] = rank_df["kendall"].abs()
    rank_df["perm_avg"] = rank_df[[c for c in rank_df.columns if c.startswith("perm_")]].mean(axis=1)
    rank_df["gain_avg"] = rank_df[[c for c in rank_df.columns if c.startswith("gain_")]].mean(axis=1)

    # Components normalized
    rank_df["n_abs_pearson"] = _normalize_minmax(rank_df["abs_pearson"])
    rank_df["n_abs_spearman"] = _normalize_minmax(rank_df["abs_spearman"])
    rank_df["n_abs_kendall"] = _normalize_minmax(rank_df["abs_kendall"])
    rank_df["n_mutual_info"] = _normalize_minmax(rank_df["mutual_info"])
    rank_df["n_perm_avg"] = _normalize_minmax(rank_df["perm_avg"])
    rank_df["n_gain_avg"] = _normalize_minmax(rank_df["gain_avg"])
    rank_df["n_univariate_lift"] = _normalize_minmax(rank_df["univariate_lift"])
    rank_df["n_temporal_stability"] = _normalize_minmax(rank_df["temporal_stability"])
    rank_df["n_missing_rate"] = _normalize_minmax(rank_df["missing_rate"])
    rank_df["n_max_peer_corr"] = _normalize_minmax(rank_df["max_peer_corr"])
    rank_df["n_drift_ks"] = _normalize_minmax(rank_df["drift_ks"])
    rank_df["n_corr_shift"] = _normalize_minmax(rank_df["corr_shift"])

    rank_df["redundancy_penalty"] = 0.6 * rank_df["n_max_peer_corr"] + 0.4 * rank_df["n_gain_avg"]
    rank_df["drift_penalty"] = 0.6 * rank_df["n_drift_ks"] + 0.4 * rank_df["n_corr_shift"]

    rank_df["utility_score"] = (
        0.15 * rank_df["n_abs_pearson"]
        + 0.15 * rank_df["n_abs_spearman"]
        + 0.05 * rank_df["n_abs_kendall"]
        + 0.20 * rank_df["n_mutual_info"]
        + 0.15 * rank_df["n_perm_avg"]
        + 0.10 * rank_df["n_gain_avg"]
        + 0.10 * rank_df["n_univariate_lift"]
        + 0.10 * rank_df["n_temporal_stability"]
        - 0.10 * rank_df["n_missing_rate"]
        - 0.10 * rank_df["redundancy_penalty"]
        - 0.10 * rank_df["drift_penalty"]
    )

    rank_df = rank_df.sort_values("utility_score", ascending=False).reset_index(drop=True)

    top_features = rank_df.head(10)["feature"].tolist()
    corr_top = train_df[top_features].corr(method="pearson")

    _plot_results(rank_df, corr_top)

    fam_df = (
        rank_df.groupby("family", as_index=False)
        .agg(
            n_features=("feature", "count"),
            utility_score_mean=("utility_score", "mean"),
            missing_rate_mean=("missing_rate", "mean"),
            drift_ks_mean=("drift_ks", "mean"),
        )
        .sort_values("utility_score_mean", ascending=False)
    )

    rank_df.to_csv(OUTPUT_DIR / "feature_utility_ranking.csv", index=False)
    rank_df.head(10).to_csv(OUTPUT_DIR / "top_10_features.csv", index=False)
    rank_df.tail(10).to_csv(OUTPUT_DIR / "bottom_10_features.csv", index=False)
    fam_df.to_csv(OUTPUT_DIR / "feature_family_summary.csv", index=False)
    corr_top.to_csv(OUTPUT_DIR / "correlation_matrix_top_features.csv")

    red_pairs = []
    corr_full = train_df[numeric_cols].corr(method="pearson")
    for i, a in enumerate(numeric_cols):
        for b in numeric_cols[i + 1 :]:
            v = corr_full.loc[a, b]
            if pd.notna(v) and abs(v) >= 0.9:
                red_pairs.append({"feature_a": a, "feature_b": b, "pearson_corr": float(v)})
    red_pairs_df = pd.DataFrame(red_pairs).sort_values("pearson_corr", ascending=False)
    red_pairs_df.to_csv(OUTPUT_DIR / "redundancy_clusters.csv", index=False)

    report = {
        "dataset": "data/processed/sprint3.csv",
        "target": TARGET_COL,
        "train_period": "1980-2020",
        "external_eval_period": "2021-2025",
        "holdout_note_period": "2026+",
        "rows": {
            "train": int(len(train_df)),
            "external_eval": int(len(test_df)),
            "holdout_2026_plus": int(len(holdout_2026_df)),
        },
        "features": {
            "numeric_analyzed": len(numeric_cols),
            "categorical_present": categorical_cols,
        },
        "top_10": rank_df.head(10)[
            ["feature", "family", "utility_score", "abs_pearson", "abs_spearman", "mutual_info", "drift_ks"]
        ].to_dict(orient="records"),
        "bottom_10": rank_df.tail(10)[
            ["feature", "family", "utility_score", "missing_rate", "max_peer_corr", "drift_ks"]
        ].to_dict(orient="records"),
        "family_summary": fam_df.to_dict(orient="records"),
        "utility_score_formula": "0.15 pearson + 0.15 spearman + 0.05 kendall + 0.20 MI + 0.15 permutation + 0.10 gain + 0.10 univariate_lift + 0.10 temporal_stability - 0.10 missing - 0.10 redundancy - 0.10 drift",
        "notes": [
            "Analisis orientado a utilidad predictiva practica.",
            "Bottom features pueden ser debiles por baja senal, drift, redundancia o missingness.",
            "Las conclusiones deben cruzarse con evaluacion de modelos para decisiones finales de feature engineering.",
        ],
    }

    (OUTPUT_DIR / "correlation_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    (OUTPUT_DIR / "README_SNIPPET.md").write_text(_build_readme_snippet(rank_df, fam_df), encoding="utf-8")

    print("Study completed")
    print(f"Output dir: {OUTPUT_DIR}")
    print("Top 10 features:")
    print(rank_df.head(10)[["feature", "utility_score"]].to_string(index=False))


if __name__ == "__main__":
    main()
