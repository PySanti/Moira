"""
build_climate_data.py

Versión optimizada con cache para construir features as-of 23h del día X.

Cambios principales:
- Cache por año para NCEI/GHCND Daily Summaries.
- Cache por año para ISD-Lite ya procesado/escalado.
- Cache por año para tabla diaria 23h de ISD-Lite.
- Cache persistente opcional en disco para evitar redescargar/reprocesar entre ejecuciones.
- td_anomaly_x ya no reconstruye todo el histórico horario en cada llamada.

Uso:
    from build_climate_data import get_weather_features, preload_weather_cache

    preload_weather_cache("new york", "1980-01-01", "2025-12-31")
    row = get_weather_features("new york", "10-05-24", strict=True)
"""

from __future__ import annotations

import gzip
import hashlib
import os
import re
from io import StringIO
from pathlib import Path
from datetime import datetime

import requests
import pandas as pd
import numpy as np


# ---------------- CONFIG ----------------

NCEI_DATA_URL = "https://www.ncei.noaa.gov/access/services/data/v1"

# Daily Summaries / GHCND
LGA_GHCND_STATION = "USW00014732"  # LaGuardia Airport - GHCND

# ISD-Lite / Global Hourly
# Formato ISD: USAF-WBAN
LGA_ISD_LITE_STATION = "725030-14732"
ISD_LITE_BASE_URL = "https://www.ncei.noaa.gov/pub/data/noaa/isd-lite"

CITY_COORDS = {
    "new york": {"lat": 40.7128, "lon": -74.0060, "tz": "America/New_York"},
    "chicago":  {"lat": 41.8781, "lon": -87.6298, "tz": "America/Chicago"},
    "atlanta":  {"lat": 33.7490, "lon": -84.3880, "tz": "America/New_York"},
    "seul":     {"lat": 37.5665, "lon": 126.9780, "tz": "Asia/Seoul"},
    "londres":  {"lat": 51.5074, "lon": -0.1278, "tz": "Europe/London"},
}

DEFAULT_HISTORY_START_DATE = "1980-01-01"

# Cache persistente.
# Puedes cambiar la carpeta con:
#   WEATHER_CACHE_DIR=/ruta/cache python test.py
CACHE_DIR = Path(os.getenv("WEATHER_CACHE_DIR", ".weather_cache"))
ENABLE_DISK_CACHE = os.getenv("WEATHER_DISABLE_DISK_CACHE", "0").strip() not in {"1", "true", "TRUE", "yes", "YES"}


# ---------------- CACHES EN MEMORIA ----------------

_NCEI_DAILY_YEAR_CACHE: dict[tuple, pd.DataFrame] = {}
_ISD_LITE_RAW_YEAR_CACHE: dict[tuple[str, int], pd.DataFrame] = {}
_ISD_LITE_PROCESSED_YEAR_CACHE: dict[tuple[str, int, str], pd.DataFrame] = {}
_ISD_DAILY_23H_YEAR_CACHE: dict[tuple[str, int, str, int, int], pd.DataFrame] = {}


# ---------------- HELPERS CACHE DISCO ----------------

def _safe_key_part(value: object) -> str:
    s = str(value)
    s = re.sub(r"[^a-zA-Z0-9_.-]+", "_", s)
    return s[:80]


def _disk_cache_path(prefix: str, key: tuple) -> Path:
    raw = repr(key).encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()[:20]
    return CACHE_DIR / prefix / f"{digest}.pkl"


def _read_df_from_disk_cache(prefix: str, key: tuple) -> pd.DataFrame | None:
    if not ENABLE_DISK_CACHE:
        return None

    path = _disk_cache_path(prefix, key)

    if not path.exists():
        return None

    try:
        return pd.read_pickle(path)
    except Exception:
        # Si el cache quedó corrupto, se ignora y se reconstruye.
        return None


def _write_df_to_disk_cache(prefix: str, key: tuple, df: pd.DataFrame) -> None:
    if not ENABLE_DISK_CACHE:
        return

    path = _disk_cache_path(prefix, key)
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = path.with_suffix(".tmp.pkl")

    try:
        df.to_pickle(tmp_path)
        tmp_path.replace(path)
    except Exception:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass


def clear_memory_cache() -> None:
    """
    Limpia únicamente el cache en memoria.
    No borra el cache persistente en disco.
    """
    _NCEI_DAILY_YEAR_CACHE.clear()
    _ISD_LITE_RAW_YEAR_CACHE.clear()
    _ISD_LITE_PROCESSED_YEAR_CACHE.clear()
    _ISD_DAILY_23H_YEAR_CACHE.clear()


def cache_info() -> dict:
    """
    Devuelve conteos rápidos de cache para debugging.
    """
    return {
        "ncei_daily_year_cache": len(_NCEI_DAILY_YEAR_CACHE),
        "isd_lite_raw_year_cache": len(_ISD_LITE_RAW_YEAR_CACHE),
        "isd_lite_processed_year_cache": len(_ISD_LITE_PROCESSED_YEAR_CACHE),
        "isd_daily_23h_year_cache": len(_ISD_DAILY_23H_YEAR_CACHE),
        "disk_cache_enabled": ENABLE_DISK_CACHE,
        "disk_cache_dir": str(CACHE_DIR),
    }


# ---------------- HELPERS GENERALES ----------------

def _safe(v):
    return None if (v is None or pd.isna(v)) else float(v)


def _df_value(df: pd.DataFrame, ts: pd.Timestamp, col: str) -> float:
    ts = pd.to_datetime(ts).normalize()

    if ts not in df.index or col not in df.columns:
        return np.nan

    return df.loc[ts, col]


def _linear_slope(values: np.ndarray | list[float]) -> float:
    values = np.asarray(values, dtype=float)

    if len(values) < 2 or np.isnan(values).any():
        return np.nan

    x = np.arange(len(values), dtype=float)

    try:
        return float(np.polyfit(x, values, deg=1)[0])
    except Exception:
        return np.nan


def _mean_if_enough(values: np.ndarray | list[float], min_count: int) -> float:
    values = np.asarray(values, dtype=float)
    valid = values[np.isfinite(values)]

    if len(valid) < min_count:
        return np.nan

    return float(valid.mean())


def _season_from_month(month: int) -> str:
    if month in [12, 1, 2]:
        return "winter"
    if month in [3, 4, 5]:
        return "spring"
    if month in [6, 7, 8]:
        return "summer"
    return "autumn"


def _circular_doy_distance(doy_values: np.ndarray, target_doy: int) -> np.ndarray:
    raw = np.abs(doy_values - target_doy)
    return np.minimum(raw, 366 - raw)


def _climatology_stats(
    series: pd.Series,
    target_ts: pd.Timestamp,
    available_until_ts: pd.Timestamp,
    window_days: int = 7,
    min_records: int = 30,
) -> dict:
    """
    Calcula climatología usando solo registros anteriores a available_until_ts.

    Esto evita leakage temporal.
    """
    target_ts = pd.to_datetime(target_ts).normalize()
    available_until_ts = pd.to_datetime(available_until_ts).normalize()

    hist = series[series.index < available_until_ts].dropna()

    if hist.empty:
        return {
            "mean": np.nan,
            "p10": np.nan,
            "p90": np.nan,
            "n": 0,
        }

    target_doy = int(target_ts.dayofyear)
    hist_doy = hist.index.dayofyear.to_numpy()
    distances = _circular_doy_distance(hist_doy, target_doy)

    selected = hist[distances <= window_days]

    if len(selected) < min_records:
        return {
            "mean": np.nan,
            "p10": np.nan,
            "p90": np.nan,
            "n": int(len(selected)),
        }

    return {
        "mean": float(selected.mean()),
        "p10": float(selected.quantile(0.10)),
        "p90": float(selected.quantile(0.90)),
        "n": int(len(selected)),
    }


def _rows_from_local_day_until(
    hourly: pd.DataFrame,
    day_ts: pd.Timestamp,
    execution_dt: pd.Timestamp,
) -> pd.DataFrame:
    """
    Devuelve observaciones del día local desde 00:00 hasta execution_dt.
    """
    start_day = day_ts.normalize()

    return hourly[
        (hourly["datetime_local"] >= start_day)
        & (hourly["datetime_local"] <= execution_dt)
    ].copy()


def _precip_sum_until_execution(day_rows: pd.DataFrame) -> float:
    """
    Precipitación acumulada desde 00:00 hasta la hora de ejecución.

    Preferencia:
    1. Sumar precipitación 1h si hay suficientes observaciones.
    2. Si no, usar precipitación 6h como fallback.
    """
    if day_rows.empty:
        return np.nan

    p1 = day_rows["Precip_1h"].dropna()

    if len(p1) >= 6:
        return float(p1.sum())

    p6 = day_rows["Precip_6h"].dropna()

    if len(p6) >= 1:
        return float(p6.sum())

    return np.nan


# ---------------- HELPERS NCEI DAILY SUMMARIES / GHCND ----------------

def _fetch_ncei_daily_year(
    station: str,
    year: int,
    data_types: list[str],
) -> pd.DataFrame:
    """
    Descarga y cachea un año de Daily Summaries de NCEI/GHCND.

    Este cache por año evita que cada llamada a get_weather_features vuelva a
    descargar rangos históricos enormes con keys diferentes.
    """
    normalized_types = tuple(sorted([dt.upper() for dt in data_types]))
    cache_key = (station, int(year), normalized_types)

    if cache_key in _NCEI_DAILY_YEAR_CACHE:
        return _NCEI_DAILY_YEAR_CACHE[cache_key].copy()

    disk_df = _read_df_from_disk_cache("ncei_daily_year", cache_key)
    if disk_df is not None:
        _NCEI_DAILY_YEAR_CACHE[cache_key] = disk_df.copy()
        return disk_df.copy()

    start_date = f"{year}-01-01"

    # Evita pedir datos demasiado futuros en el año actual.
    today = pd.Timestamp.today().normalize()
    year_end = pd.Timestamp(year=year, month=12, day=31)

    if year_end > today and year >= today.year:
        end_ts = today
    else:
        end_ts = year_end

    end_date = end_ts.strftime("%Y-%m-%d")

    params = {
        "dataset": "daily-summaries",
        "stations": station,
        "startDate": start_date,
        "endDate": end_date,
        "dataTypes": ",".join(normalized_types),
        "format": "json",
        "units": "metric",
        "includeStationName": "false",
        "includeStationLocation": "false",
        "includeAttributes": "false",
    }

    headers = {"User-Agent": "tmax-bot/1.0"}

    r = requests.get(NCEI_DATA_URL, params=params, headers=headers, timeout=30)

    if r.status_code != 200:
        raise ConnectionError(
            f"Error NCEI Daily Summaries {year}: {r.status_code} - {r.text[:300]}"
        )

    rows = r.json()

    if not rows:
        df_empty = pd.DataFrame(columns=list(normalized_types))
        df_empty.index = pd.DatetimeIndex([], name="time")
        _NCEI_DAILY_YEAR_CACHE[cache_key] = df_empty.copy()
        _write_df_to_disk_cache("ncei_daily_year", cache_key, df_empty)
        return df_empty.copy()

    def _find_key(d: dict, candidates: list[str]) -> str | None:
        keys = {k.lower(): k for k in d.keys()}

        for c in candidates:
            if c.lower() in keys:
                return keys[c.lower()]

        return None

    date_key = _find_key(rows[0], ["DATE", "date"])

    if not date_key:
        raise ValueError(
            f"NCEI: no encontré campo de fecha. Keys={list(rows[0].keys())}"
        )

    out = []

    for row in rows:
        rec = {"time": pd.to_datetime(row[date_key]).normalize()}

        for dt in normalized_types:
            val = None

            for k in row.keys():
                if k.upper() == dt.upper():
                    val = row[k]
                    break

            if val in (None, "", "NaN"):
                rec[dt] = np.nan
            else:
                try:
                    rec[dt] = float(val)
                except ValueError:
                    rec[dt] = np.nan

        out.append(rec)

    df = (
        pd.DataFrame(out)
        .drop_duplicates(subset=["time"], keep="last")
        .set_index("time")
        .sort_index()
    )

    for dt in normalized_types:
        if dt not in df.columns:
            df[dt] = np.nan

    _NCEI_DAILY_YEAR_CACHE[cache_key] = df.copy()
    _write_df_to_disk_cache("ncei_daily_year", cache_key, df)

    return df.copy()


def _fetch_ncei_daily_cached(
    station: str,
    start_date: str,
    end_date: str,
    data_types: list[str],
) -> pd.DataFrame:
    """
    Retorna Daily Summaries para [start_date, end_date] usando cache por año.
    """
    start_ts = pd.to_datetime(start_date).normalize()
    end_ts = pd.to_datetime(end_date).normalize()

    frames = []

    for year in range(start_ts.year, end_ts.year + 1):
        df_year = _fetch_ncei_daily_year(
            station=station,
            year=year,
            data_types=data_types,
        )

        if not df_year.empty:
            frames.append(df_year)

    if not frames:
        df_empty = pd.DataFrame(columns=[dt.upper() for dt in data_types])
        df_empty.index = pd.DatetimeIndex([], name="time")
        return df_empty

    df = pd.concat(frames, axis=0)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df[(df.index >= start_ts) & (df.index <= end_ts)].copy()

    return df


# Compatibilidad con versiones anteriores.
def _fetch_ncei_daily(
    station: str,
    start_date: str,
    end_date: str,
    data_types: list[str],
) -> pd.DataFrame:
    return _fetch_ncei_daily_cached(
        station=station,
        start_date=start_date,
        end_date=end_date,
        data_types=data_types,
    )


# ---------------- HELPERS ISD-LITE ----------------

def _to_scaled_value(series: pd.Series, scale: float = 10.0) -> pd.Series:
    """
    ISD-Lite usa -9999 como missing.
    Muchas variables vienen escaladas por 10.
    """
    s = pd.to_numeric(series, errors="coerce")
    s = s.replace(-9999, np.nan)
    return s / scale


def _to_unscaled_value(series: pd.Series) -> pd.Series:
    """
    Para variables sin escala decimal, como wind direction o sky condition.
    """
    s = pd.to_numeric(series, errors="coerce")
    s = s.replace(-9999, np.nan)
    return s


def _precip_scaled(series: pd.Series) -> pd.Series:
    """
    Precipitación ISD-Lite:
    - -9999 = missing
    - -1 = trace precipitation

    Para ML diario, trace se toma como 0.0 mm.
    """
    s = pd.to_numeric(series, errors="coerce")
    s = s.replace(-9999, np.nan)
    s = s.replace(-1, 0.0)
    return s / 10.0


def _sky_condition_to_cloud_pct(code: float) -> float:
    """
    Convierte sky condition code de ISD-Lite a porcentaje aproximado de nubosidad.

    Códigos principales:
    0-8 representan oktas.
    9 y 10 se tratan como missing/indeterminado.
    11-19 son variantes textuales; se aproximan a porcentajes.
    """
    if pd.isna(code):
        return np.nan

    code = int(code)

    if 0 <= code <= 8:
        return (code / 8.0) * 100.0

    if code in [9, 10]:
        return np.nan

    if code in [11, 12, 13]:
        return 37.5

    if code in [14, 15, 16]:
        return 75.0

    if code in [17, 18, 19]:
        return 100.0

    return np.nan


def _relative_humidity_from_temp_dewpoint(
    temp_c: pd.Series,
    dewpoint_c: pd.Series,
) -> pd.Series:
    """
    Calcula humedad relativa aproximada usando temperatura y punto de rocío.

    Fórmula Magnus:
    RH = 100 * e(Td) / e(T)
    """
    temp_c = pd.to_numeric(temp_c, errors="coerce")
    dewpoint_c = pd.to_numeric(dewpoint_c, errors="coerce")

    a = 17.625
    b = 243.04

    es_td = np.exp((a * dewpoint_c) / (b + dewpoint_c))
    es_t = np.exp((a * temp_c) / (b + temp_c))

    rh = 100.0 * (es_td / es_t)
    rh = rh.clip(lower=0.0, upper=100.0)

    return rh


def _fetch_isd_lite_year_raw(station: str, year: int) -> pd.DataFrame:
    """
    Descarga un año crudo de ISD-Lite para una estación.
    """
    cache_key = (station, int(year))

    if cache_key in _ISD_LITE_RAW_YEAR_CACHE:
        return _ISD_LITE_RAW_YEAR_CACHE[cache_key].copy()

    disk_df = _read_df_from_disk_cache("isd_lite_raw_year", cache_key)
    if disk_df is not None:
        _ISD_LITE_RAW_YEAR_CACHE[cache_key] = disk_df.copy()
        return disk_df.copy()

    url = f"{ISD_LITE_BASE_URL}/{year}/{station}-{year}.gz"

    r = requests.get(url, timeout=40)

    if r.status_code == 404:
        df_empty = pd.DataFrame()
        _ISD_LITE_RAW_YEAR_CACHE[cache_key] = df_empty.copy()
        _write_df_to_disk_cache("isd_lite_raw_year", cache_key, df_empty)
        return df_empty.copy()

    if r.status_code != 200:
        raise ConnectionError(
            f"Error ISD-Lite {year}: {r.status_code} - {r.text[:200]}"
        )

    try:
        text = gzip.decompress(r.content).decode("utf-8", errors="replace")
    except OSError:
        text = r.content.decode("utf-8", errors="replace")

    if not text.strip():
        df_empty = pd.DataFrame()
        _ISD_LITE_RAW_YEAR_CACHE[cache_key] = df_empty.copy()
        _write_df_to_disk_cache("isd_lite_raw_year", cache_key, df_empty)
        return df_empty.copy()

    cols = [
        "year",
        "month",
        "day",
        "hour",
        "air_temperature",
        "dew_point_temperature",
        "sea_level_pressure",
        "wind_direction",
        "wind_speed",
        "sky_condition",
        "precip_1h",
        "precip_6h",
    ]

    df = pd.read_csv(
        StringIO(text),
        sep=r"\s+",
        names=cols,
        engine="python",
    )

    _ISD_LITE_RAW_YEAR_CACHE[cache_key] = df.copy()
    _write_df_to_disk_cache("isd_lite_raw_year", cache_key, df)

    return df.copy()


# Compatibilidad con versiones anteriores.
def _fetch_isd_lite_year(station: str, year: int) -> pd.DataFrame:
    return _fetch_isd_lite_year_raw(station=station, year=year)


def _fetch_isd_lite_year_processed(
    station: str,
    year: int,
    tz: str,
) -> pd.DataFrame:
    """
    Descarga, convierte y cachea un año de ISD-Lite ya procesado.

    En la versión anterior, cada llamada volvía a convertir columnas,
    escalar unidades, calcular humedad relativa y filtrar fechas.
    Esta versión lo hace una sola vez por año.
    """
    cache_key = (station, int(year), tz)

    if cache_key in _ISD_LITE_PROCESSED_YEAR_CACHE:
        return _ISD_LITE_PROCESSED_YEAR_CACHE[cache_key].copy()

    disk_df = _read_df_from_disk_cache("isd_lite_processed_year", cache_key)
    if disk_df is not None:
        _ISD_LITE_PROCESSED_YEAR_CACHE[cache_key] = disk_df.copy()
        return disk_df.copy()

    raw = _fetch_isd_lite_year_raw(station=station, year=year)

    if raw.empty:
        df_empty = pd.DataFrame()
        _ISD_LITE_PROCESSED_YEAR_CACHE[cache_key] = df_empty.copy()
        _write_df_to_disk_cache("isd_lite_processed_year", cache_key, df_empty)
        return df_empty.copy()

    raw = raw.copy()

    raw["datetime_utc"] = pd.to_datetime(
        dict(
            year=raw["year"],
            month=raw["month"],
            day=raw["day"],
            hour=raw["hour"],
        ),
        errors="coerce",
        utc=True,
    )

    raw = raw.dropna(subset=["datetime_utc"]).copy()

    raw["datetime_local"] = (
        raw["datetime_utc"]
        .dt.tz_convert(tz)
        .dt.tz_localize(None)
    )

    raw["time"] = raw["datetime_local"].dt.normalize()

    raw["Temp_C"] = _to_scaled_value(raw["air_temperature"], scale=10.0)
    raw["Td"] = _to_scaled_value(raw["dew_point_temperature"], scale=10.0)
    raw["SLP"] = _to_scaled_value(raw["sea_level_pressure"], scale=10.0)
    raw["WindDir"] = _to_unscaled_value(raw["wind_direction"])
    raw["WindSpd"] = _to_scaled_value(raw["wind_speed"], scale=10.0)
    raw["Cloud"] = (
        _to_unscaled_value(raw["sky_condition"])
        .apply(_sky_condition_to_cloud_pct)
    )

    raw["Precip_1h"] = _precip_scaled(raw["precip_1h"])
    raw["Precip_6h"] = _precip_scaled(raw["precip_6h"])

    raw["HR"] = _relative_humidity_from_temp_dewpoint(
        temp_c=raw["Temp_C"],
        dewpoint_c=raw["Td"],
    )

    keep_cols = [
        "datetime_utc",
        "datetime_local",
        "time",
        "Temp_C",
        "HR",
        "Td",
        "SLP",
        "WindSpd",
        "WindDir",
        "Cloud",
        "Precip_1h",
        "Precip_6h",
    ]

    df = raw[keep_cols].sort_values("datetime_local").reset_index(drop=True)

    _ISD_LITE_PROCESSED_YEAR_CACHE[cache_key] = df.copy()
    _write_df_to_disk_cache("isd_lite_processed_year", cache_key, df)

    return df.copy()


def _fetch_ncei_isd_lite_hourly(
    station: str,
    start_date: str,
    end_date: str,
    tz: str,
) -> pd.DataFrame:
    """
    Retorna observaciones horarias locales en [start_date 00:00, end_date 23:59].

    Optimización:
    - Usa cache por año ya procesado.
    - Incluye buffer de años alrededor del rango local para no perder horas por
      conversión UTC -> America/New_York cerca de Año Nuevo.
    """
    start_ts = pd.to_datetime(start_date).normalize()
    end_ts = pd.to_datetime(end_date).normalize() + pd.Timedelta(
        hours=23,
        minutes=59,
    )

    # Buffer para cubrir conversiones UTC/local alrededor del cambio de año.
    first_year = (start_ts - pd.Timedelta(days=2)).year
    last_year = (end_ts + pd.Timedelta(days=2)).year

    frames = []

    for year in range(first_year, last_year + 1):
        df_year = _fetch_isd_lite_year_processed(
            station=station,
            year=year,
            tz=tz,
        )

        if not df_year.empty:
            frames.append(df_year)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df = df[
        (df["datetime_local"] >= start_ts)
        & (df["datetime_local"] <= end_ts)
    ].copy()

    df = df.sort_values("datetime_local").reset_index(drop=True)

    return df


def _build_23h_daily_from_hourly(
    hourly: pd.DataFrame,
    days: pd.DatetimeIndex,
    execution_hour: int,
    tolerance_hours: int,
) -> pd.DataFrame:
    """
    Construye una tabla diaria tomando la observación más cercana a execution_hour,
    sin usar observaciones futuras.

    Implementación vectorizada con merge_asof.
    Evita hacer un filtro completo del dataframe horario por cada día.
    """
    output_cols = [
        "HR_23h",
        "Td_23h",
        "SLP_23h",
        "WindSpd_23h",
        "WindDir_23h",
        "Cloud_23h",
    ]

    days = pd.DatetimeIndex(pd.to_datetime(days).normalize()).sort_values()

    if len(days) == 0:
        df_empty = pd.DataFrame(columns=output_cols)
        df_empty.index = pd.DatetimeIndex([], name="time")
        return df_empty

    targets = pd.DataFrame({
        "time": days,
        "target_dt": days + pd.Timedelta(hours=execution_hour),
    }).sort_values("target_dt")

    if hourly.empty:
        out = targets[["time"]].copy()
        for col in output_cols:
            out[col] = np.nan
        return out.set_index("time").sort_index()

    obs_cols = [
        "datetime_local",
        "HR",
        "Td",
        "SLP",
        "WindSpd",
        "WindDir",
        "Cloud",
    ]

    obs = (
        hourly[obs_cols]
        .dropna(subset=["datetime_local"])
        .sort_values("datetime_local")
        .copy()
    )

    if obs.empty:
        out = targets[["time"]].copy()
        for col in output_cols:
            out[col] = np.nan
        return out.set_index("time").sort_index()

    merged = pd.merge_asof(
        targets,
        obs,
        left_on="target_dt",
        right_on="datetime_local",
        direction="backward",
        tolerance=pd.Timedelta(hours=tolerance_hours),
    )

    out = pd.DataFrame({
        "time": merged["time"],
        "HR_23h": merged["HR"],
        "Td_23h": merged["Td"],
        "SLP_23h": merged["SLP"],
        "WindSpd_23h": merged["WindSpd"],
        "WindDir_23h": merged["WindDir"],
        "Cloud_23h": merged["Cloud"],
    })

    return out.set_index("time").sort_index()


def _fetch_isd_daily_23h_year(
    station: str,
    year: int,
    tz: str,
    execution_hour: int,
    tolerance_hours: int,
) -> pd.DataFrame:
    """
    Construye y cachea la tabla diaria 23h de un año local.

    Esto es clave para td_anomaly_x:
    antes se reconstruía todo el histórico 1980->X en cada llamada.
    ahora se calcula una vez por año y se reutiliza.
    """
    cache_key = (station, int(year), tz, int(execution_hour), int(tolerance_hours))

    if cache_key in _ISD_DAILY_23H_YEAR_CACHE:
        return _ISD_DAILY_23H_YEAR_CACHE[cache_key].copy()

    disk_df = _read_df_from_disk_cache("isd_daily_23h_year", cache_key)
    if disk_df is not None:
        _ISD_DAILY_23H_YEAR_CACHE[cache_key] = disk_df.copy()
        return disk_df.copy()

    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"

    hourly = _fetch_ncei_isd_lite_hourly(
        station=station,
        start_date=start_date,
        end_date=end_date,
        tz=tz,
    )

    days = pd.date_range(start_date, end_date, freq="D")

    df = _build_23h_daily_from_hourly(
        hourly=hourly,
        days=days,
        execution_hour=execution_hour,
        tolerance_hours=tolerance_hours,
    )

    _ISD_DAILY_23H_YEAR_CACHE[cache_key] = df.copy()
    _write_df_to_disk_cache("isd_daily_23h_year", cache_key, df)

    return df.copy()


def _fetch_isd_daily_23h_cached(
    station: str,
    start_date: str,
    end_date: str,
    tz: str,
    execution_hour: int,
    tolerance_hours: int,
) -> pd.DataFrame:
    """
    Retorna tabla diaria 23h en [start_date, end_date] usando cache por año.
    """
    start_ts = pd.to_datetime(start_date).normalize()
    end_ts = pd.to_datetime(end_date).normalize()

    frames = []

    for year in range(start_ts.year, end_ts.year + 1):
        df_year = _fetch_isd_daily_23h_year(
            station=station,
            year=year,
            tz=tz,
            execution_hour=execution_hour,
            tolerance_hours=tolerance_hours,
        )

        if not df_year.empty:
            frames.append(df_year)

    if not frames:
        df_empty = pd.DataFrame(
            columns=[
                "HR_23h",
                "Td_23h",
                "SLP_23h",
                "WindSpd_23h",
                "WindDir_23h",
                "Cloud_23h",
            ]
        )
        df_empty.index = pd.DatetimeIndex([], name="time")
        return df_empty

    df = pd.concat(frames, axis=0)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df[(df.index >= start_ts) & (df.index <= end_ts)].copy()

    return df


# ---------------- PRELOAD OPCIONAL ----------------

def preload_weather_cache(
    city: str = "new york",
    start_date: str = DEFAULT_HISTORY_START_DATE,
    end_date: str = "2025-12-31",
    execution_hour: int = 23,
    nearest_tolerance_hours: int = 2,
    include_td_daily_cache: bool = True,
) -> dict:
    """
    Pre-carga caches útiles para minar dataset histórico.

    Recomendado antes de loops largos:
        preload_weather_cache("new york", "1980-01-01", "2025-12-31")

    Esto no cambia el resultado de get_weather_features; solo evita que la primera
    llamada tenga que construir todo el cache bajo demanda.
    """
    city_key = city.lower().strip()

    if city_key != "new york":
        raise ValueError("Esta versión solo implementa fuentes NCEI/LaGuardia para New York.")

    tz = CITY_COORDS[city_key]["tz"]

    start_ts = pd.to_datetime(start_date).normalize()
    end_ts = pd.to_datetime(end_date).normalize()

    # Daily Summaries para climatología, lags y target.
    _fetch_ncei_daily_cached(
        station=LGA_GHCND_STATION,
        start_date=start_ts.strftime("%Y-%m-%d"),
        end_date=end_ts.strftime("%Y-%m-%d"),
        data_types=["TMAX", "TMIN"],
    )

    # ISD-Lite procesado por año.
    _fetch_ncei_isd_lite_hourly(
        station=LGA_ISD_LITE_STATION,
        start_date=start_ts.strftime("%Y-%m-%d"),
        end_date=end_ts.strftime("%Y-%m-%d"),
        tz=tz,
    )

    # Tabla diaria 23h para td_anomaly_x.
    if include_td_daily_cache:
        _fetch_isd_daily_23h_cached(
            station=LGA_ISD_LITE_STATION,
            start_date=start_ts.strftime("%Y-%m-%d"),
            end_date=end_ts.strftime("%Y-%m-%d"),
            tz=tz,
            execution_hour=execution_hour,
            tolerance_hours=nearest_tolerance_hours,
        )

    return cache_info()


# ---------------- MAIN ----------------

def get_weather_features(
    city: str,
    date_str: str,
    strict: bool = True,
    execution_hour: int = 23,
    nearest_tolerance_hours: int = 2,
    history_start_date: str = DEFAULT_HISTORY_START_DATE,
    climatology_window_days: int = 7,
    min_climatology_records: int = 30,
    compute_td_anomaly: bool = True,
) -> dict:
    """
    Genera features usando únicamente información disponible hasta las 23:00
    del día X, hora local de New York.

    date_str:
      formato esperado: "%d-%m-%y"

    Target:
      - t_max_x+1 sale de NCEI/GHCND Daily Summaries.

    Features:
      - Variables horarias vienen de ISD-Lite / sensores de LaGuardia.
      - Tmax/Tmin/Tmean del día X se calculan desde 00:00 hasta 23:00.
      - No incluye features de forecast.
    """
    city_key = city.lower().strip()

    if city_key not in CITY_COORDS:
        raise ValueError(f"Ciudad no soportada. Use: {list(CITY_COORDS.keys())}")

    if city_key != "new york":
        raise ValueError(
            "Esta versión solo implementa fuentes NCEI/LaGuardia para New York."
        )

    target_date = datetime.strptime(date_str, "%d-%m-%y")

    target_ts = pd.to_datetime(target_date.strftime("%Y-%m-%d")).normalize()
    next_ts = target_ts + pd.Timedelta(days=1)

    execution_dt = target_ts + pd.Timedelta(hours=execution_hour)

    history_start_ts = pd.to_datetime(history_start_date).normalize()

    # Daily Summaries:
    # - histórico completo para climatología Tmax.
    # - x-7 para lags y medias.
    # - x+1 para target.
    daily_start = min(history_start_ts, target_ts - pd.Timedelta(days=7))
    daily_end = next_ts

    daily_idx = pd.date_range(daily_start, daily_end, freq="D")

    ncei_daily = _fetch_ncei_daily_cached(
        station=LGA_GHCND_STATION,
        start_date=daily_start.strftime("%Y-%m-%d"),
        end_date=daily_end.strftime("%Y-%m-%d"),
        data_types=["TMAX", "TMIN"],
    ).reindex(daily_idx)

    ncei_daily["TMEAN"] = (ncei_daily["TMAX"] + ncei_daily["TMIN"]) / 2.0
    ncei_daily["DTR"] = ncei_daily["TMAX"] - ncei_daily["TMIN"]

    # ISD-Lite horario:
    # - x-2 para td_ma3 y pressure_trend_3d.
    # - x-1 para SLP 23h anterior.
    # - x para features as-of 23h.
    hourly_start = target_ts - pd.Timedelta(days=2)
    hourly_end = target_ts

    hourly = _fetch_ncei_isd_lite_hourly(
        station=LGA_ISD_LITE_STATION,
        start_date=hourly_start.strftime("%Y-%m-%d"),
        end_date=hourly_end.strftime("%Y-%m-%d"),
        tz=CITY_COORDS[city_key]["tz"],
    )

    if hourly.empty:
        if strict:
            print("Error: ISD-Lite no devolvió observaciones horarias.")
            return {}

    day_rows = _rows_from_local_day_until(
        hourly=hourly,
        day_ts=target_ts,
        execution_dt=execution_dt,
    )

    if day_rows.empty and strict:
        print(
            f"Error: no hay observaciones para {target_ts.date()} "
            f"hasta {execution_hour}:00."
        )
        return {}

    days_for_23h = pd.date_range(
        target_ts - pd.Timedelta(days=2),
        target_ts,
        freq="D",
    )

    # Se mantiene a partir del hourly de x-2:x para evitar depender del cache
    # anual cuando solo se necesitan tres días.
    daily_23h = _build_23h_daily_from_hourly(
        hourly=hourly,
        days=days_for_23h,
        execution_hour=execution_hour,
        tolerance_hours=nearest_tolerance_hours,
    )

    obs_23 = daily_23h.loc[target_ts]
    obs_prev_23 = daily_23h.loc[target_ts - pd.Timedelta(days=1)]

    # ---------------- FEATURES BASE AS-OF 23H ----------------

    tmax_so_far_23h_x = day_rows["Temp_C"].max()
    tmin_so_far_23h_x = day_rows["Temp_C"].min()
    tmean_so_far_23h_x = day_rows["Temp_C"].mean()

    tmax_x_minus_1 = _df_value(
        ncei_daily,
        target_ts - pd.Timedelta(days=1),
        "TMAX",
    )

    tmax_x_minus_2 = _df_value(
        ncei_daily,
        target_ts - pd.Timedelta(days=2),
        "TMAX",
    )

    delta_tmax_so_far_1d_23h = (
        tmax_so_far_23h_x - tmax_x_minus_1
        if pd.notna(tmax_so_far_23h_x) and pd.notna(tmax_x_minus_1)
        else np.nan
    )

    ma_tmax_3d_asof_23h = (
        (tmax_so_far_23h_x + tmax_x_minus_1 + tmax_x_minus_2) / 3.0
        if pd.notna(tmax_so_far_23h_x)
        and pd.notna(tmax_x_minus_1)
        and pd.notna(tmax_x_minus_2)
        else np.nan
    )

    dtr_so_far_23h_x = (
        tmax_so_far_23h_x - tmin_so_far_23h_x
        if pd.notna(tmax_so_far_23h_x) and pd.notna(tmin_so_far_23h_x)
        else np.nan
    )

    hr_23h_x = obs_23.get("HR_23h", np.nan)
    td_23h_x = obs_23.get("Td_23h", np.nan)
    slp_23h_x = obs_23.get("SLP_23h", np.nan)
    wind_spd_23h_x = obs_23.get("WindSpd_23h", np.nan)
    wind_dir_23h_x = obs_23.get("WindDir_23h", np.nan)
    cloud_23h_x = obs_23.get("Cloud_23h", np.nan)

    slp_23h_x_minus_1 = obs_prev_23.get("SLP_23h", np.nan)

    delta_slp_24h_23h = (
        slp_23h_x - slp_23h_x_minus_1
        if pd.notna(slp_23h_x) and pd.notna(slp_23h_x_minus_1)
        else np.nan
    )

    wind_rad = np.deg2rad(wind_dir_23h_x) if pd.notna(wind_dir_23h_x) else np.nan
    wind_dir_sin_23h_x = np.sin(wind_rad) if pd.notna(wind_rad) else np.nan
    wind_dir_cos_23h_x = np.cos(wind_rad) if pd.notna(wind_rad) else np.nan

    precip_sum_00_23h_x = _precip_sum_until_execution(day_rows)

    # ---------------- NUEVAS FEATURES NO-FORECAST: GHCND ----------------

    clim_tmax_x = _climatology_stats(
        series=ncei_daily["TMAX"],
        target_ts=target_ts,
        available_until_ts=target_ts,
        window_days=climatology_window_days,
        min_records=min_climatology_records,
    )

    climatology_tmax_doy = clim_tmax_x["mean"]

    tmax_anomaly_x = (
        tmax_so_far_23h_x - climatology_tmax_doy
        if pd.notna(tmax_so_far_23h_x) and pd.notna(climatology_tmax_doy)
        else np.nan
    )

    extreme_heat_flag = (
        int(tmax_so_far_23h_x >= clim_tmax_x["p90"])
        if pd.notna(tmax_so_far_23h_x) and pd.notna(clim_tmax_x["p90"])
        else np.nan
    )

    extreme_cold_flag = (
        int(tmax_so_far_23h_x <= clim_tmax_x["p10"])
        if pd.notna(tmax_so_far_23h_x) and pd.notna(clim_tmax_x["p10"])
        else np.nan
    )

    tmax_lag2 = _df_value(ncei_daily, target_ts - pd.Timedelta(days=2), "TMAX")
    tmax_lag3 = _df_value(ncei_daily, target_ts - pd.Timedelta(days=3), "TMAX")
    tmax_lag7 = _df_value(ncei_daily, target_ts - pd.Timedelta(days=7), "TMAX")

    tmin_lag1 = _df_value(ncei_daily, target_ts - pd.Timedelta(days=1), "TMIN")

    tmean_last_7 = [
        _df_value(ncei_daily, target_ts - pd.Timedelta(days=i), "TMEAN")
        for i in range(7, 0, -1)
    ]

    tmean_ma7 = _mean_if_enough(tmean_last_7, min_count=7)

    tmax_trend_3d = _linear_slope([
        _df_value(ncei_daily, target_ts - pd.Timedelta(days=2), "TMAX"),
        _df_value(ncei_daily, target_ts - pd.Timedelta(days=1), "TMAX"),
        tmax_so_far_23h_x,
    ])

    tmax_trend_7d = _linear_slope([
        _df_value(ncei_daily, target_ts - pd.Timedelta(days=6), "TMAX"),
        _df_value(ncei_daily, target_ts - pd.Timedelta(days=5), "TMAX"),
        _df_value(ncei_daily, target_ts - pd.Timedelta(days=4), "TMAX"),
        _df_value(ncei_daily, target_ts - pd.Timedelta(days=3), "TMAX"),
        _df_value(ncei_daily, target_ts - pd.Timedelta(days=2), "TMAX"),
        _df_value(ncei_daily, target_ts - pd.Timedelta(days=1), "TMAX"),
        tmax_so_far_23h_x,
    ])

    dtr_ma3 = _mean_if_enough([
        _df_value(ncei_daily, target_ts - pd.Timedelta(days=2), "DTR"),
        _df_value(ncei_daily, target_ts - pd.Timedelta(days=1), "DTR"),
        dtr_so_far_23h_x,
    ], min_count=3)

    # ---------------- NUEVAS FEATURES NO-FORECAST: ISD-LITE ----------------

    td_values_3d = daily_23h.loc[
        target_ts - pd.Timedelta(days=2): target_ts,
        "Td_23h",
    ].to_numpy(dtype=float)

    td_ma3 = _mean_if_enough(td_values_3d, min_count=3)

    slp_values_3d = daily_23h.loc[
        target_ts - pd.Timedelta(days=2): target_ts,
        "SLP_23h",
    ].to_numpy(dtype=float)

    pressure_trend_3d = _linear_slope(slp_values_3d)

    wind_u = (
        wind_spd_23h_x * np.sin(wind_rad)
        if pd.notna(wind_spd_23h_x) and pd.notna(wind_rad)
        else np.nan
    )

    wind_v = (
        wind_spd_23h_x * np.cos(wind_rad)
        if pd.notna(wind_spd_23h_x) and pd.notna(wind_rad)
        else np.nan
    )

    # td_anomaly_x requiere climatología histórica de Td_23h.
    # Optimización:
    # - Antes: reconstruía hourly_history desde 1980 hasta X en cada llamada.
    # - Ahora: usa daily_23h por año cacheado en memoria/disco.
    td_anomaly_x = np.nan

    if compute_td_anomaly:
        td_daily_23h = _fetch_isd_daily_23h_cached(
            station=LGA_ISD_LITE_STATION,
            start_date=history_start_ts.strftime("%Y-%m-%d"),
            end_date=target_ts.strftime("%Y-%m-%d"),
            tz=CITY_COORDS[city_key]["tz"],
            execution_hour=execution_hour,
            tolerance_hours=nearest_tolerance_hours,
        )

        if not td_daily_23h.empty:
            clim_td_x = _climatology_stats(
                series=td_daily_23h["Td_23h"],
                target_ts=target_ts,
                available_until_ts=target_ts,
                window_days=climatology_window_days,
                min_records=min_climatology_records,
            )

            td_climatology_doy = clim_td_x["mean"]

            td_anomaly_x = (
                td_23h_x - td_climatology_doy
                if pd.notna(td_23h_x) and pd.notna(td_climatology_doy)
                else np.nan
            )

    # ---------------- ESTACIONALIDAD ----------------

    doy = target_ts.dayofyear
    month = target_ts.month
    season = _season_from_month(month)

    doy_sin = np.sin(2 * np.pi * doy / 365.25)
    doy_cos = np.cos(2 * np.pi * doy / 365.25)

    # ---------------- TARGET ----------------

    target_x_plus_1 = _df_value(ncei_daily, next_ts, "TMAX")

    features = {
        # Base as-of 23h
        "Tmax_so_far_23h_x": _safe(tmax_so_far_23h_x),
        "Tmin_so_far_23h_x": _safe(tmin_so_far_23h_x),
        "Tmean_so_far_23h_x": _safe(tmean_so_far_23h_x),
        "Delta_Tmax_so_far_1d_23h": _safe(delta_tmax_so_far_1d_23h),
        "MA_Tmax_3d_asof_23h": _safe(ma_tmax_3d_asof_23h),
        "DTR_so_far_23h_x": _safe(dtr_so_far_23h_x),

        "HR_23h_x": _safe(hr_23h_x),
        "Td_23h_x": _safe(td_23h_x),
        "SLP_23h_x": _safe(slp_23h_x),
        "Delta_SLP_24h_23h": _safe(delta_slp_24h_23h),
        "WindSpd_23h_x": _safe(wind_spd_23h_x),
        "WindDir_sin_23h_x": _safe(wind_dir_sin_23h_x),
        "WindDir_cos_23h_x": _safe(wind_dir_cos_23h_x),
        "Cloud_23h_x": _safe(cloud_23h_x),
        "Precip_sum_00_23h_x": _safe(precip_sum_00_23h_x),

        # Nuevas features no-forecast
        "climatology_tmax_doy": _safe(climatology_tmax_doy),
        "tmax_anomaly_x": _safe(tmax_anomaly_x),
        "tmax_lag2": _safe(tmax_lag2),
        "tmax_lag3": _safe(tmax_lag3),
        "tmax_lag7": _safe(tmax_lag7),
        "tmin_lag1": _safe(tmin_lag1),
        "tmean_ma7": _safe(tmean_ma7),
        "tmax_trend_3d": _safe(tmax_trend_3d),
        "tmax_trend_7d": _safe(tmax_trend_7d),
        "dtr_ma3": _safe(dtr_ma3),
        "td_anomaly_x": _safe(td_anomaly_x),
        "td_ma3": _safe(td_ma3),
        "wind_u": _safe(wind_u),
        "wind_v": _safe(wind_v),
        "pressure_trend_3d": _safe(pressure_trend_3d),
        "month": _safe(month),
        "season": season,
        "extreme_heat_flag": _safe(extreme_heat_flag),
        "extreme_cold_flag": _safe(extreme_cold_flag),

        # Target oficial
        "t_max_x+1": _safe(target_x_plus_1),

        # Metadata / estacionalidad
        "ciudad": city,
        "doy_sin": _safe(doy_sin),
        "doy_cos": _safe(doy_cos),
    }

    if strict:
        allowed_non_numeric = {"ciudad", "season"}

        # td_anomaly_x puede faltar al inicio del histórico si todavía no hay
        # suficientes registros para construir climatología de Td_23h.
        allowed_missing = {
            "td_anomaly_x",
        }

        missing_features = [
            k for k, v in features.items()
            if k not in allowed_non_numeric
            and k not in allowed_missing
            and v is None
        ]

        if missing_features:
            print("Error: faltan features necesarias para construir la fila as-of 23:00.")
            for k in missing_features[:40]:
                print(" -", k)

            if len(missing_features) > 40:
                print(f" - ... y {len(missing_features) - 40} más")

            return {}

    return features
