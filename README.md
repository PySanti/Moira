# Moira


El objetivo de este proyecto es crear un bot que se conectara con polymarket para apostar contra la temperatura maxima de una ciudad en un dia especifico.

trello : https://trello.com/b/R37KNQzR/moira

# Desarrollo de V0

![Version 0 image](./images/V0.png)

# Definicion modular de requisitos para V0

* Funcion de consumo de data 
* Generacion de dataset para entrenamiento
* Entrenamiento de modelo
* Fusion de piezas para flujo

## Desarrollo de funcion para consulta a API

Esta funcion consultara las siguientes fuentes:

`NCEI/GHCND - LaGuardia Airport (USW00014732)`: para Tmax y Tmin (mismo que polymarket).

NOAA = National Oceanic and Atmospheric Administration (EE. UU.).

Es una agencia del gobierno estadounidense encargada de meteorología (pronósticos, alertas)

NCEI = National Centers for Environmental Information.

Es un centro dentro de NOAA que funciona como archivo oficial y repositorio de datos ambientales (clima, océanos, geofísica)

GHCND = Global Historical Climatology Network – Daily

Es un gran dataset global de observaciones diarias de estaciones meteorológicas (sitios físicos con sensores) que NOAA/NCEI recopila y “normaliza” para que lo puedas consultar de forma consistente.

En resumen, es una interfaz de consumo de data relacionada con clima y meteorología que en este proyecto usaremos para tener las mismas referencias que polymarket.


`Open-Meteo`: para el resto de datos.

La mayoría de sus datos vienen de modelos numéricos y reanálisis (datos “en grilla”), por ejemplo ERA5/ERA5-Land e IFS, y te devuelve valores para una celda cercana a las coordenadas que pides (no una medición puntual exacta).

### Definicion y refinamiento de features

Nota: es importante tener en cuenta las horas de ejecucion del bot, esto por que el bot se entrenara con data conseguida al final de los dias, entonces el bot mientras mas hacia el final del dia se ejecute, mas preciso sera por que mas se ajustara a su contexto de entrenamiento.

En esta seccion definire las features que se utilizaran para predecir la temperatura de un dia X + 1 a partir de data del dia X.

Empezare con una cantidad reducida de features para ampliar posiblemente en el futuro, mientras mas features, mas complicado construir la funcion.


| Nombre de feature                  |         Unidad | Rango de valores (típico) | Significado (incluye cálculo)                                                                              |
| ---------------------------------- | -------------: | ------------------------- | ---------------------------------------------------------------------------------------------------------- |
| **Tmax_día_x**                     |             °C | ~ -50 a 55                | **Cálculo:** `Tmax[x]`. Máxima del día *x* (persistencia térmica).                                         |
| **Tmin_día_x**                     |             °C | ~ -60 a 35                | **Cálculo:** `Tmin[x]`. Mínima del día *x* (masa de aire/enfriamiento nocturno).                           |
| **Tmedia_día_x**                   |             °C | ~ -55 a 45                | **Cálculo:** `(Tmax[x] + Tmin[x]) / 2` (o `Tmed[x]`). Estado térmico general.                              |
| **ΔTmax_1d**                       |             °C | ~ -20 a 20                | **Cálculo:** `Tmax[x] − Tmax[x−1]`. Tendencia/cambio reciente.                                             |
| **MA_Tmax_3d**                     |             °C | ~ -50 a 55                | **Cálculo:** `(Tmax[x] + Tmax[x−1] + Tmax[x−2]) / 3`. Inercia térmica de corto plazo.                      |
| **DTR_x**                          |             °C | ~ 0 a 25 (puede >30)      | **Cálculo:** `Tmax[x] − Tmin[x]`. Amplitud térmica; proxy nubosidad/humedad.                               |
| **HR_media_día_x**                 |              % | 0 a 100                   | **Cálculo:** `HR_mean[x]`. Humedad relativa media diaria.                                                  |
| **Punto_de_rocío_día_x (Td)**      |             °C | ~ -60 a 30+               | **Cálculo:** `Td[x]` (preferible si viene en el dataset). Contenido real de vapor de agua.                 |
| **Presión_media_día_x (SLP)**      |            hPa | ~ 870 a 1085              | **Cálculo:** `SLP_mean[x]`. Señal sinótica (altas/bajas).                                                  |
| **ΔPresión_24h**                   |            hPa | ~ -20 a 20                | **Cálculo:** `SLP_mean[x] − SLP_mean[x−1]`. Cambio sinótico rápido.                                        |
| **Viento_vel_media_día_x**         |            m/s | 0 a 30 (rachas mayores)   | **Cálculo:** `wind_speed_mean[x]`. Mezcla/advección.                                                       |
| **Viento_dir_sin(x)**              |              — | -1 a 1                    | **Cálculo:** `sin(2π * wind_dir_deg[x] / 360)`. Codificación circular de dirección.                        |
| **Viento_dir_cos(x)**              |              — | -1 a 1                    | **Cálculo:** `cos(2π * wind_dir_deg[x] / 360)`. Codificación circular de dirección.                        |
| **Nubosidad_media_día_x**          | % (o fracción) | 0–100 (o 0–1)             | **Cálculo:** `cloud_cover_mean[x]`. Control de radiación entrante.                                         |
| **Precipitación_acum_día_x**       |         mm/día | 0 a 300+                  | **Cálculo:** `precip_sum[x]`. Efecto de lluvia/nubosidad/evaporación.                                      |
| **t_max_x+1 (si está disponible)** |             °C | ~ -50 a 55                | **Cálculo:** `Tmax[x+1]`. **Label/objetivo** para entrenamiento; **no usar como feature** en inferencia.   |
| **ciudad**                         |      categoría | N categorías              | **Cálculo:** ID/nombre. Se codifica (one-hot/target encoding/embeddings) para capturar climatología local. |
| **doy_sin**                        |              — | -1 a 1                    | **Cálculo:** `sin(2π * doy / 365)`. Estacionalidad en forma cíclica (diciembre cerca de enero).            |
| **doy_cos**                        |              — | -1 a 1                    | **Cálculo:** `cos(2π * doy / 365)`. Complementa `doy_sin` para representar el ciclo anual.                 |

**Nota**: `dia`, `mes` y `año` , son redundantes teniendo `doy_sin` y `doy_cos`, dejarlo unicamente para identificar registros, descartarlos para entrenamiento e inferencia.

### Creacion de funcion para consulta a api

Version inicial :

```python
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ---------------- CONFIG ----------------
NCEI_DATA_URL = "https://www.ncei.noaa.gov/access/services/data/v1"
LGA_GHCND_STATION = "USW00014732"  # LaGuardia Airport (GHCND)

CITY_COORDS = {
    "new york": {"lat": 40.7128, "lon": -74.0060, "tz": "America/New_York"},
    "chicago":  {"lat": 41.8781, "lon": -87.6298, "tz": "America/Chicago"},
    "atlanta":  {"lat": 33.7490, "lon": -84.3880, "tz": "America/New_York"},
    "seul":     {"lat": 37.5665, "lon": 126.9780, "tz": "Asia/Seoul"},
    "londres":  {"lat": 51.5074, "lon": -0.1278, "tz": "Europe/London"},
}

# ---------------- HELPERS ----------------
def _fetch_ncei_daily(station: str, start_date: str, end_date: str, data_types: list[str]) -> pd.DataFrame:
    """
    Descarga Daily Summaries (GHCND) de NCEI para un rango [start_date, end_date] (YYYY-MM-DD),
    devolviendo un df indexado por fecha con columnas por dataTypes (ej: TMAX, TMIN).

    Unidades: metric => °C para TMAX/TMIN.
    """
    params = {
        "dataset": "daily-summaries",
        "stations": station,
        "startDate": start_date,
        "endDate": end_date,
        "dataTypes": ",".join(data_types),
        "format": "json",
        "units": "metric",  # <-- °C
        "includeStationName": "false",
        "includeStationLocation": "false",
        "includeAttributes": "false",
    }
    headers = {"User-Agent": "tmax-bot/1.0 (contact: you@example.com)"}

    r = requests.get(NCEI_DATA_URL, params=params, headers=headers, timeout=25)
    if r.status_code != 200:
        raise ConnectionError(f"Error NCEI: {r.status_code} - {r.text[:300]}")

    rows = r.json()
    if not rows:
        return pd.DataFrame()

    def _find_key(d: dict, candidates: list[str]) -> str | None:
        keys = {k.lower(): k for k in d.keys()}
        for c in candidates:
            if c.lower() in keys:
                return keys[c.lower()]
        return None

    date_key = _find_key(rows[0], ["DATE", "date"])
    if not date_key:
        raise ValueError(f"NCEI: no encontré campo de fecha. Keys={list(rows[0].keys())}")

    out = []
    for row in rows:
        rec = {"time": pd.to_datetime(row[date_key]).normalize()}
        for dt in data_types:
            val = None
            for k in row.keys():
                if k.upper() == dt.upper():
                    val = row[k]
                    break
            if val in (None, "", "NaN"):
                rec[dt.upper()] = np.nan
            else:
                try:
                    rec[dt.upper()] = float(val)
                except ValueError:
                    rec[dt.upper()] = np.nan
        out.append(rec)

    df = pd.DataFrame(out).drop_duplicates(subset=["time"], keep="last").set_index("time").sort_index()
    return df


def _fetch_open_meteo_daily(lat: float, lon: float, start_date: str, end_date: str, tz: str) -> pd.DataFrame:
    """
    Open-Meteo Archive para el resto de features.
    Unidades explícitas: °C y m/s.
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": [
            "temperature_2m_min",
            "temperature_2m_mean",
            "precipitation_sum",
            "wind_speed_10m_mean",
            "wind_direction_10m_dominant",
            "surface_pressure_mean",
            "relative_humidity_2m_mean",
            "cloud_cover_mean",
            "dew_point_2m_mean",
        ],
        "temperature_unit": "celsius",
        "wind_speed_unit": "ms",
        "precipitation_unit": "mm",
        "timezone": tz,
    }

    r = requests.get(url, params=params, timeout=25)
    if r.status_code != 200:
        raise ConnectionError(f"Error Open-Meteo: {r.status_code} - {r.text[:300]}")

    daily = r.json().get("daily", {})
    if "time" not in daily:
        raise ValueError(f"Open-Meteo: respuesta inesperada. Keys daily={list(daily.keys())}")

    df = pd.DataFrame(daily)
    df["time"] = pd.to_datetime(df["time"]).dt.normalize()
    df = df.drop_duplicates(subset=["time"], keep="last").set_index("time").sort_index()

    df = df.rename(columns={
        "temperature_2m_min": "Tmin_model",      # °C (modelo)
        "temperature_2m_mean": "Tmean_model",    # °C (modelo)
        "relative_humidity_2m_mean": "HR",       # %
        "dew_point_2m_mean": "Td",               # °C
        "surface_pressure_mean": "SLP",          # hPa (presión a superficie)
        "wind_speed_10m_mean": "WindSpd",        # m/s
        "wind_direction_10m_dominant": "WindDir",# grados
        "cloud_cover_mean": "Cloud",             # %
        "precipitation_sum": "Precip",           # mm
    })
    return df


def _missing_mask(series_or_df: pd.Series | pd.DataFrame) -> bool:
    """True si el valor está missing (NaN)."""
    if isinstance(series_or_df, pd.DataFrame):
        return series_or_df.isna().any().any()
    return pd.isna(series_or_df)


# ---------------- MAIN ----------------
def get_weather_features(city: str, date_str: str, strict: bool = True) -> dict:
    """
    strict=True:
      - si faltan datos previos necesarios (no solo Tmax), retorna {}.
    """
    city_key = city.lower().strip()
    if city_key not in CITY_COORDS:
        raise ValueError(f"Ciudad no soportada. Use: {list(CITY_COORDS.keys())}")

    target_date = datetime.strptime(date_str, "%d-%m-%y")
    next_day = target_date + timedelta(days=1)
    start_date = target_date - timedelta(days=5)

    api_start = start_date.strftime("%Y-%m-%d")
    api_end = next_day.strftime("%Y-%m-%d")

    expected_idx = pd.date_range(api_start, api_end, freq="D")
    target_ts = pd.to_datetime(target_date.strftime("%Y-%m-%d"))
    next_ts = pd.to_datetime(next_day.strftime("%Y-%m-%d"))

    # 1) Open-Meteo
    om = _fetch_open_meteo_daily(
        lat=CITY_COORDS[city_key]["lat"],
        lon=CITY_COORDS[city_key]["lon"],
        start_date=api_start,
        end_date=api_end,
        tz=CITY_COORDS[city_key]["tz"],
    ).reindex(expected_idx)

    # 2) NCEI (NYC)
    if city_key == "new york":
        ncei = _fetch_ncei_daily(
            station=LGA_GHCND_STATION,
            start_date=api_start,
            end_date=api_end,
            data_types=["TMAX", "TMIN"],
        ).reindex(expected_idx)
    else:
        raise ValueError("Esta versión solo implementa NCEI para New York (LaGuardia USW00014732).")

    # DataFrame unificado
    df = pd.DataFrame(index=expected_idx)
    df.index.name = "time"
    df["Tmax"] = ncei.get("TMAX")
    df["Tmin"] = ncei.get("TMIN")
    df["Tmean"] = (df["Tmax"] + df["Tmin"]) / 2.0

    for col in ["HR", "Td", "SLP", "WindSpd", "WindDir", "Cloud", "Precip"]:
        df[col] = om.get(col)

    # Derivadas
    df["Delta_Tmax_1d"] = df["Tmax"] - df["Tmax"].shift(1)
    df["MA_Tmax_3d"] = df["Tmax"].rolling(window=3, min_periods=3).mean()
    df["DTR"] = df["Tmax"] - df["Tmin"]
    df["Delta_Presion"] = df["SLP"] - df["SLP"].shift(1)

    wind_rad = np.deg2rad(df["WindDir"])
    df["Wind_sin"] = np.sin(wind_rad)
    df["Wind_cos"] = np.cos(wind_rad)

    df["doy"] = df.index.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * df["doy"] / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["doy"] / 365.25)

    if target_ts not in df.index:
        raise ValueError(f"No se encontró el día objetivo {target_ts.date()} en el índice reindexado.")

    # ---------------- VALIDACIÓN NUEVA (FULL) ----------------
    # Días previos necesarios:
    # - Para ΔTmax_1d: x-1
    # - Para MA_Tmax_3d: x-1, x-2
    # - Para label: x+1
    # - Para ΔPresión_24h: x-1
    needed_days = {
        "x-2": target_ts - pd.Timedelta(days=2),
        "x-1": target_ts - pd.Timedelta(days=1),
        "x": target_ts,
        "x+1": next_ts,
    }

    # Variables necesarias por día:
    # - Tmax: x-2, x-1, x, x+1
    # - Tmin: x (para DTR y Tmean)
    # - Open-Meteo: HR/Td/SLP/Wind/Cloud/Precip en x
    # - SLP también en x-1 para Delta_Presion
    requirements = []

    # Tmax requeridos (para rolling/diff/label)
    for tag, day in needed_days.items():
        requirements.append((f"NCEI.Tmax({tag})", day, "Tmax"))

    # Tmin requerido al menos en x
    requirements.append(("NCEI.Tmin(x)", needed_days["x"], "Tmin"))

    # Open-Meteo variables para día x
    for var in ["HR", "Td", "SLP", "WindSpd", "WindDir", "Cloud", "Precip"]:
        requirements.append((f"OM.{var}(x)", needed_days["x"], var))

    # SLP de x-1 para Delta_Presion
    requirements.append(("OM.SLP(x-1)", needed_days["x-1"], "SLP"))

    missing_items = []
    for name, day, col in requirements:
        if day not in df.index:
            missing_items.append(f"{name}: day_out_of_range({day.date()})")
            continue
        if pd.isna(df.loc[day, col]):
            missing_items.append(f"{name}: NaN")

    if strict and missing_items:
        print("Error: No se pudieron obtener TODOS los datos previos necesarios.")
        for it in missing_items[:30]:
            print(" -", it)
        if len(missing_items) > 30:
            print(f" - ... y {len(missing_items) - 30} más")
        return {}
    # ----------------------------------------------------------

    target_row = df.loc[target_ts]

    def _safe(v):
        return None if (v is None or (isinstance(v, float) and np.isnan(v))) else float(v)

    features = {
        "Tmax_día_x": _safe(target_row["Tmax"]),
        "Tmin_día_x": _safe(target_row["Tmin"]),
        "Tmedia_día_x": _safe(target_row["Tmean"]),
        "ΔTmax_1d": _safe(target_row["Delta_Tmax_1d"]),
        "MA_Tmax_3d": _safe(target_row["MA_Tmax_3d"]),
        "DTR_x": _safe(target_row["DTR"]),
        "HR_media_día_x": _safe(target_row["HR"]),
        "Punto_de_rocío_día_x (Td)": _safe(target_row["Td"]),
        "Presión_media_día_x (SLP)": _safe(target_row["SLP"]),
        "ΔPresión_24h": _safe(target_row["Delta_Presion"]),
        "Viento_vel_media_día_x": _safe(target_row["WindSpd"]),
        "Viento_dir_sin(x)": _safe(target_row["Wind_sin"]),
        "Viento_dir_cos(x)": _safe(target_row["Wind_cos"]),
        "Nubosidad_media_día_x": _safe(target_row["Cloud"]),
        "Precipitación_acum_día_x": _safe(target_row["Precip"]),
        "t_max_x+1": _safe(df.loc[next_ts, "Tmax"]),
        "ciudad": city,
        "doy_sin": _safe(target_row["doy_sin"]),
        "doy_cos": _safe(target_row["doy_cos"]),
    }

    return features
```

### Testeo de funcion

La funcion fue testeada por:

* Posibles bloqueos por rate limiting 
* Velocidad de respuesta promedio
* Alcance de fechas
* Null values
* Acierto en valores dados
* Consistencia de unidades
* Correspondencia con valores historicos de polymarket para todos los tmax y tmin

### Generacion de dataset

#### Version de dataset 1.0

Luego de minar datos de 1980 a 2026 utilizando el script `./utils/miner.py`, obtuvimos los siguientes resultados:

```
Dimensiones de dataset 15118 x  21
```

Posteriormente, se genero el valset utilizando registros de otono a traves del script `./utils/split_dataset.py`, obteniendo los siguientes resultados:

```
Dimensiones de train 11392 x  21
Dimensiones de train 3726 x  21
```

La lista de features es:

```

date
date_str
Tmax_día_x
Tmin_día_x
Tmedia_día_x
ΔTmax_1d
MA_Tmax_3d
DTR_x
HR_media_día_x
Punto_de_rocío_día_x (Td)
Presión_media_día_x (SLP)
ΔPresión_24h
Viento_vel_media_día_x
Viento_dir_sin(x)
Viento_dir_cos(x)
Nubosidad_media_día_x
Precipitación_acum_día_x
t_max_x+1
ciudad
doy_sin
doy_cos
```




## Entrenamiento de modelo

### Entrenamiento 1.0

Algoritmo : HistGradientBoostingRegressor
Version de dataset: 1.0
Valset : otono


Se selecciono el algoritmo `HistGradientBoostingRegressor` y en conjunto con el dataset anteriormente mencionado y un proceso de seleccion de hiperparametros, se obtuvieron los siguientes resultados:

**Configuarcion**:

```python

# main.py
import argparse
import json
import unicodedata
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.svm import SVR

import joblib


# -------------------- Helpers --------------------

def norm(s: str) -> str:
    s = str(s).strip()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    out = []
    prev_us = False
    for ch in s:
        if ch.isalnum():
            out.append(ch)
            prev_us = False
        else:
            if not prev_us:
                out.append("_")
                prev_us = True
    s2 = "".join(out).strip("_")
    while "__" in s2:
        s2 = s2.replace("__", "_")
    return s2


def split_xy(df: pd.DataFrame, target_col: str, drop_cols: List[str]) -> Tuple[pd.DataFrame, pd.Series]:
    if target_col not in df.columns:
        raise ValueError(f"No existe target_col='{target_col}' en el dataframe.")
    X = df.drop(columns=[target_col] + drop_cols, errors="ignore")
    y = df[target_col]
    return X, y


# -------------------- Main --------------------

def main():
    train_path = Path("./dataset/trainset.csv")
    val_path = Path("./dataset/valset.csv")

    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)

    # Target
    target = "t_max_x+1"

    if target not in df_val.columns:
        raise SystemExit(f"[ERROR] El target '{target}' no existe en valset.csv.")

    # Excluir columnas de fecha si existen (porque ya tienes doy_sin/cos)
    drop_cols = ["date","date_str"]
    X_train, y_train = split_xy(df_train, target, drop_cols)
    X_val, y_val = split_xy(df_val, target, drop_cols)

    # Columnas categóricas vs numéricas (simple y robusto)
    cat_cols = [c for c in X_train.columns if X_train[c].dtype in ["object", "str"] or str(X_train[c].dtype).startswith("category")]

    num_cols = [c for c in X_train.columns if c not in cat_cols]

    # Preprocesamiento básico: imputación + normalización
    # Nota: para modelos tipo boosting no es estrictamente necesario escalar,
    # pero lo incluyo porque lo pediste y no hace daño.
    numeric_pipe = Pipeline(steps=[
        ("scaler", StandardScaler()),
    ])

    categorical_pipe = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
        ],
        remainder="drop",
    )

    model = HistGradientBoostingRegressor(
        loss="absolute_error",
        random_state=42,
        early_stopping=True,
    )

    pipe = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", model),
    ])

    # Usar tu split train/val tal cual (sin CV aleatorio)
    # -> combinamos datasets y marcamos val como fold=0, train como fold=-1
    X_all = pd.concat([X_train, X_val], axis=0, ignore_index=True)
    y_all = pd.concat([y_train, y_val], axis=0, ignore_index=True)
    test_fold = np.array([-1] * len(X_train) + [0] * len(X_val))  # -1=entrena, 0=val
    ps = PredefinedSplit(test_fold=test_fold)

    # Espacio de búsqueda (práctico para tabular time-series con features engineered)
    param_dist = {
        "model__max_iter": [300, 600, 1000, 1500],
        "model__learning_rate": [0.02, 0.03, 0.05, 0.08, 0.1],
        "model__max_depth": [None, 3, 5, 7, 9],
        "model__max_leaf_nodes": [15, 31, 63, 127],
        "model__min_samples_leaf": [10, 20, 50, 100],
        "model__l2_regularization": [0.0, 0.01, 0.1, 1.0, 5.0],
        "model__max_bins": [64, 127, 255],
    }

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dist,
        n_iter=15_000,
        scoring="neg_mean_absolute_error",  # minimiza MAE
        cv=ps,
        refit=False, # que hace esto?
        random_state=42,
        n_jobs=15,
        verbose=2,
    )

    print("\n=== Entrenando + Hypertuning (MAE) ===")
    print(f"Train rows: {len(X_train):,} | Val rows: {len(X_val):,}")
    print(f"Target: {target}")
    if drop_cols:
        print(f"Drop cols (fecha detectada): {drop_cols}")
    print(f"Num cols: {len(num_cols)} | Cat cols: {len(cat_cols)}")
    if cat_cols:
        print(f"Categorical: {cat_cols}")

    search.fit(X_all, y_all)

    best_params = search.best_params_
    best_cv_mae = -float(search.best_score_)  # neg -> mae

    print("\n=== RESULTADOS ===")
    print(f"Best CV MAE (en tu valset): {best_cv_mae:.4f}")

    print("\nBest params:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")


    # Guardar reporte
    report = {
        "target": target,
        "rows": {"train": int(len(X_train)), "val": int(len(X_val))},
        "cols": {"num": num_cols, "cat": cat_cols, "dropped": drop_cols},
        "best_params": best_params,
        "best_cv_mae": best_cv_mae
    }
    report_name = "report.json"
    Path(report_name).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Reporte guardado en: {Path(report_name).resolve()}")


if __name__ == "__main__":
    main()

```

**Resultados**:

```
{
  "target": "t_max_x+1",
  "rows": {
    "train": 11392,
    "val": 3726
  },
  "cols": {
    "num": [
      "Tmax_día_x",
      "Tmin_día_x",
      "Tmedia_día_x",
      "ΔTmax_1d",
      "MA_Tmax_3d",
      "DTR_x",
      "HR_media_día_x",
      "Punto_de_rocío_día_x (Td)",
      "Presión_media_día_x (SLP)",
      "ΔPresión_24h",
      "Viento_vel_media_día_x",
      "Viento_dir_sin(x)",
      "Viento_dir_cos(x)",
      "Nubosidad_media_día_x",
      "Precipitación_acum_día_x",
      "doy_sin",
      "doy_cos"
    ],
    "cat": [
      "ciudad"
    ],
    "dropped": [
      "date",
      "date_str"
    ]
  },
  "best_params": {
    "model__min_samples_leaf": 10,
    "model__max_leaf_nodes": 127,
    "model__max_iter": 300,
    "model__max_depth": null,
    "model__max_bins": 255,
    "model__learning_rate": 0.03,
    "model__l2_regularization": 0.1
  },
  "best_cv_mae": 2.3821435019630037
}
```

## Entrenamiento 1.1


Algoritmo : HistGradientBoostingRegressor
Version de dataset: 1.0
Valset : random split

**Configuracion**

```python
import argparse
import json
import unicodedata
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.svm import SVR

import joblib


# -------------------- Helpers --------------------

def norm(s: str) -> str:
    s = str(s).strip()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    out = []
    prev_us = False
    for ch in s:
        if ch.isalnum():
            out.append(ch)
            prev_us = False
        else:
            if not prev_us:
                out.append("_")
                prev_us = True
    s2 = "".join(out).strip("_")
    while "__" in s2:
        s2 = s2.replace("__", "_")
    return s2


def split_xy(df: pd.DataFrame, target_col: str, drop_cols: List[str]) -> Tuple[pd.DataFrame, pd.Series]:
    if target_col not in df.columns:
        raise ValueError(f"No existe target_col='{target_col}' en el dataframe.")
    X = df.drop(columns=[target_col] + drop_cols, errors="ignore")
    y = df[target_col]
    return X, y


# -------------------- Main --------------------

def main():
    dataset = Path("../../dataset/original_dataset.csv")

    # ✅ Leer dataset original
    df = pd.read_csv(dataset)

    # Target
    target = "t_max_x+1"

    if target not in df.columns:
        raise SystemExit(f"[ERROR] El target '{target}' no existe en el dataset original.")

    # ✅ Split RANDOM: train / val / test (70/15/15)
    # Primero sacamos test (15%), luego val (15% del total ≈ 0.17647 del restante)
    df_trainval, df_test = train_test_split(df, test_size=0.15, random_state=42, shuffle=True)
    df_train, df_val = train_test_split(df_trainval, test_size=0.20, random_state=42, shuffle=True)

    # Excluir columnas de fecha si existen (porque ya tienes doy_sin/cos)
    drop_cols = ["date", "date_str"]
    X_train, y_train = split_xy(df_train, target, drop_cols)
    X_val, y_val = split_xy(df_val, target, drop_cols)
    X_test, y_test = split_xy(df_test, target, drop_cols)

    # Columnas categóricas vs numéricas (simple y robusto)
    cat_cols = [c for c in X_train.columns if X_train[c].dtype in ["object", "str"] or str(X_train[c].dtype).startswith("category")]
    num_cols = [c for c in X_train.columns if c not in cat_cols]

    # Preprocesamiento básico: imputación + normalización
    numeric_pipe = Pipeline(steps=[
        ("scaler", StandardScaler()),
    ])

    categorical_pipe = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
        ],
        remainder="drop",
    )

    model = HistGradientBoostingRegressor(
        loss="absolute_error",
        random_state=42,
        early_stopping=True,
    )

    pipe = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", model),
    ])

    # ✅ Hypertuning SOLO con train/val (PredefinedSplit)
    X_all = pd.concat([X_train, X_val], axis=0, ignore_index=True)
    y_all = pd.concat([y_train, y_val], axis=0, ignore_index=True)
    test_fold = np.array([-1] * len(X_train) + [0] * len(X_val))  # -1=entrena, 0=val
    ps = PredefinedSplit(test_fold=test_fold)

    # Espacio de búsqueda
    param_dist = {
        "model__max_iter": [300, 600, 1000, 1500],
        "model__learning_rate": [0.02, 0.03, 0.05, 0.08, 0.1],
        "model__max_depth": [None, 3, 5, 7, 9],
        "model__max_leaf_nodes": [15, 31, 63, 127],
        "model__min_samples_leaf": [10, 20, 50, 100],
        "model__l2_regularization": [0.0, 0.01, 0.1, 1.0, 5.0],
        "model__max_bins": [64, 127, 255],
    }

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dist,
        n_iter=15_000,
        scoring="neg_mean_absolute_error",  # minimiza MAE
        cv=ps,
        refit=False,
        random_state=42,
        n_jobs=15,
        verbose=2,
    )

    print("\n=== Entrenando + Hypertuning (MAE) ===")
    print(f"Train rows: {len(X_train):,} | Val rows: {len(X_val):,} | Test rows: {len(X_test):,}")
    print(f"Target: {target}")
    if drop_cols:
        print(f"Drop cols (fecha detectada): {drop_cols}")
    print(f"Num cols: {len(num_cols)} | Cat cols: {len(cat_cols)}")
    if cat_cols:
        print(f"Categorical: {cat_cols}")

    search.fit(X_all, y_all)

    best_params = search.best_params_
    best_cv_mae = -float(search.best_score_)  # neg -> mae

    print("\n=== RESULTADOS ===")
    print(f"Best CV MAE (en tu valset): {best_cv_mae:.4f}")

    print("\nBest params:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    # Guardar reporte
    report = {
        "target": target,
        "rows": {"train": int(len(X_train)), "val": int(len(X_val)), "test": int(len(X_test))},
        "cols": {"num": num_cols, "cat": cat_cols, "dropped": drop_cols},
        "best_params": best_params,
        "best_cv_mae": best_cv_mae
    }
    report_name = "report.json"
    Path(report_name).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Reporte guardado en: {Path(report_name).resolve()}")


if __name__ == "__main__":
    main()

```

**Resultados**:

```python
{
  "target": "t_max_x+1",
  "rows": {
    "train": 10280,
    "val": 2570,
    "test": 2268
  },
  "cols": {
    "num": [
      "Tmax_día_x",
      "Tmin_día_x",
      "Tmedia_día_x",
      "ΔTmax_1d",
      "MA_Tmax_3d",
      "DTR_x",
      "HR_media_día_x",
      "Punto_de_rocío_día_x (Td)",
      "Presión_media_día_x (SLP)",
      "ΔPresión_24h",
      "Viento_vel_media_día_x",
      "Viento_dir_sin(x)",
      "Viento_dir_cos(x)",
      "Nubosidad_media_día_x",
      "Precipitación_acum_día_x",
      "doy_sin",
      "doy_cos"
    ],
    "cat": [
      "ciudad"
    ],
    "dropped": [
      "date",
      "date_str"
    ]
  },
  "best_params": {
    "model__min_samples_leaf": 50,
    "model__max_leaf_nodes": 31,
    "model__max_iter": 600,
    "model__max_depth": null,
    "model__max_bins": 127,
    "model__learning_rate": 0.03,
    "model__l2_regularization": 0.0
  },
  "best_cv_mae": 2.4033782931158805
}
```

## Entrenamiento 1.2

Algoritmo : XGBoostRegressor
Version de dataset: 1.0
Valset : random split

**Configuracion**:

```python

import argparse
import json
import unicodedata
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ✅ Cambiado: HistGradientBoostingRegressor -> XGBoost
# Requiere: pip install xgboost
from xgboost import XGBRegressor

from sklearn.svm import SVR

import joblib


# -------------------- Helpers --------------------

def norm(s: str) -> str:
    s = str(s).strip()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    out = []
    prev_us = False
    for ch in s:
        if ch.isalnum():
            out.append(ch)
            prev_us = False
        else:
            if not prev_us:
                out.append("_")
                prev_us = True
    s2 = "".join(out).strip("_")
    while "__" in s2:
        s2 = s2.replace("__", "_")
    return s2


def split_xy(df: pd.DataFrame, target_col: str, drop_cols: List[str]) -> Tuple[pd.DataFrame, pd.Series]:
    if target_col not in df.columns:
        raise ValueError(f"No existe target_col='{target_col}' en el dataframe.")
    X = df.drop(columns=[target_col] + drop_cols, errors="ignore")
    y = df[target_col]
    return X, y


# -------------------- Main --------------------

def main():
    dataset = Path("../../dataset/original_dataset.csv")

    # ✅ Leer dataset original
    df = pd.read_csv(dataset)

    # Target
    target = "t_max_x+1"

    if target not in df.columns:
        raise SystemExit(f"[ERROR] El target '{target}' no existe en el dataset original.")

    # ✅ Split RANDOM: train / val / test (70/15/15)
    df_trainval, df_test = train_test_split(df, test_size=0.15, random_state=42, shuffle=True)
    df_train, df_val = train_test_split(df_trainval, test_size=0.20, random_state=42, shuffle=True)

    # Excluir columnas de fecha si existen (porque ya tienes doy_sin/cos)
    drop_cols = ["date", "date_str"]
    X_train, y_train = split_xy(df_train, target, drop_cols)
    X_val, y_val = split_xy(df_val, target, drop_cols)
    X_test, y_test = split_xy(df_test, target, drop_cols)

    # Columnas categóricas vs numéricas (simple y robusto)
    cat_cols = [c for c in X_train.columns if X_train[c].dtype in ["object", "str"] or str(X_train[c].dtype).startswith("category")]
    num_cols = [c for c in X_train.columns if c not in cat_cols]

    # Preprocesamiento básico: normalización (se mantiene igual)
    numeric_pipe = Pipeline(steps=[
        ("scaler", StandardScaler()),
    ])

    categorical_pipe = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
        ],
        remainder="drop",
    )

    # ✅ Modelo XGBoost orientado a MAE (L1)
    model = XGBRegressor(
        objective="reg:absoluteerror",   # optimiza MAE (L1)
        eval_metric="mae",
        tree_method="hist",              # más rápido en tabular
        random_state=42,
        n_jobs=15,
        verbosity=0,
    )

    pipe = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", model),
    ])

    # ✅ Hypertuning SOLO con train/val (PredefinedSplit) (se mantiene igual)
    X_all = pd.concat([X_train, X_val], axis=0, ignore_index=True)
    y_all = pd.concat([y_train, y_val], axis=0, ignore_index=True)
    test_fold = np.array([-1] * len(X_train) + [0] * len(X_val))  # -1=entrena, 0=val
    ps = PredefinedSplit(test_fold=test_fold)

    # ✅ Espacio de búsqueda adaptado a XGBoost
    param_dist = {
        "model__n_estimators": [300, 600, 1000, 1500],
        "model__learning_rate": [0.02, 0.03, 0.05, 0.08, 0.1],
        "model__max_depth": [3, 4, 5, 6, 8, 10],
        "model__min_child_weight": [1, 5, 10, 20],
        "model__subsample": [0.6, 0.8, 1.0],
        "model__colsample_bytree": [0.6, 0.8, 1.0],
        "model__reg_lambda": [0.0, 0.01, 0.1, 1.0, 5.0, 10.0],
        "model__reg_alpha": [0.0, 0.01, 0.1, 1.0],
        "model__gamma": [0.0, 0.1, 0.5, 1.0],
        "model__max_bin": [64, 127, 255],
    }

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dist,
        n_iter=2_500,
        scoring="neg_mean_absolute_error",  # minimiza MAE
        cv=ps,
        refit=False,
        random_state=42,
        n_jobs=15,
        verbose=2,
    )

    print("\n=== Entrenando + Hypertuning (MAE) | XGBoost ===")
    print(f"Train rows: {len(X_train):,} | Val rows: {len(X_val):,} | Test rows: {len(X_test):,}")
    print(f"Target: {target}")
    if drop_cols:
        print(f"Drop cols (fecha detectada): {drop_cols}")
    print(f"Num cols: {len(num_cols)} | Cat cols: {len(cat_cols)}")
    if cat_cols:
        print(f"Categorical: {cat_cols}")

    search.fit(X_all, y_all)

    best_params = search.best_params_
    best_cv_mae = -float(search.best_score_)  # neg -> mae

    print("\n=== RESULTADOS ===")
    print(f"Best CV MAE (en tu valset): {best_cv_mae:.4f}")

    print("\nBest params:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    # Guardar reporte
    report = {
        "target": target,
        "rows": {"train": int(len(X_train)), "val": int(len(X_val)), "test": int(len(X_test))},
        "cols": {"num": num_cols, "cat": cat_cols, "dropped": drop_cols},
        "best_params": best_params,
        "best_cv_mae": best_cv_mae
    }
    report_name = "report.json"
    Path(report_name).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Reporte guardado en: {Path(report_name).resolve()}")


if __name__ == "__main__":
    main()
```


**Resultados**:

```python
{
  "target": "t_max_x+1",
  "rows": {
    "train": 10280,
    "val": 2570,
    "test": 2268
  },
  "cols": {
    "num": [
      "Tmax_día_x",
      "Tmin_día_x",
      "Tmedia_día_x",
      "ΔTmax_1d",
      "MA_Tmax_3d",
      "DTR_x",
      "HR_media_día_x",
      "Punto_de_rocío_día_x (Td)",
      "Presión_media_día_x (SLP)",
      "ΔPresión_24h",
      "Viento_vel_media_día_x",
      "Viento_dir_sin(x)",
      "Viento_dir_cos(x)",
      "Nubosidad_media_día_x",
      "Precipitación_acum_día_x",
      "doy_sin",
      "doy_cos"
    ],
    "cat": [
      "ciudad"
    ],
    "dropped": [
      "date",
      "date_str"
    ]
  },
  "best_params": {
    "model__subsample": 0.6,
    "model__reg_lambda": 5.0,
    "model__reg_alpha": 0.01,
    "model__n_estimators": 1000,
    "model__min_child_weight": 20,
    "model__max_depth": 6,
    "model__max_bin": 127,
    "model__learning_rate": 0.03,
    "model__gamma": 1.0,
    "model__colsample_bytree": 1.0
  },
  "best_cv_mae": 2.381992741876177
}
```
# Desarrollo de V1

![Version 1 image](./images/V1.png)

# Desarrollo de V2

![Version 2 image](./images/V2.png)

# Desarrollo de V3

![Version 3 image](./images/V3.png)

