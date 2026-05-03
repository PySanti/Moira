"""
Consulta APIs climáticas para construir features as-of 23h del día X.

Versión:
- Mantiene features base as-of 23h.
- Agrega únicamente nuevas features NO-FORECAST:
    - climatology_tmax_doy
    - tmax_anomaly_x
    - tmax_lag2 / tmax_lag3 / tmax_lag7
    - tmin_lag1
    - tmean_ma7
    - tmax_trend_3d / tmax_trend_7d
    - dtr_ma3
    - td_anomaly_x
    - td_ma3
    - wind_u / wind_v
    - pressure_trend_3d
    - month / season
    - extreme_heat_flag / extreme_cold_flag

No incluye:
- forecast_tmax_x+1
- forecast_error_tmax_lag1
- forecast_error_tmax_ma3
- forecast_anomaly_x+1
"""

from __future__ import annotations

import gzip
from io import StringIO
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
# LaGuardia suele usarse como 725030-14732.
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


# ---------------- CACHES ----------------

_NCEI_DAILY_CACHE: dict[tuple, pd.DataFrame] = {}
_ISD_LITE_YEAR_CACHE: dict[tuple[str, int], pd.DataFrame] = {}


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


def _nearest_observation_at_or_before(
    hourly: pd.DataFrame,
    target_dt: pd.Timestamp,
    tolerance_hours: int = 2,
) -> pd.Series:
    """
    Busca la observación más cercana a una hora objetivo, sin usar futuro.

    Ejemplo:
    target_dt = 2025-07-10 23:00
    Puede tomar 23:00, 22:00 o 21:00 si están dentro de la tolerancia.
    """
    if hourly.empty:
        return pd.Series(dtype=float)

    eligible = hourly[hourly["datetime_local"] <= target_dt].copy()

    if eligible.empty:
        return pd.Series(dtype=float)

    eligible["diff_hours"] = (
        target_dt - eligible["datetime_local"]
    ).dt.total_seconds() / 3600.0

    eligible = eligible[eligible["diff_hours"] <= tolerance_hours]

    if eligible.empty:
        return pd.Series(dtype=float)

    return eligible.sort_values("diff_hours").iloc[0]


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

def _fetch_ncei_daily(
    station: str,
    start_date: str,
    end_date: str,
    data_types: list[str],
) -> pd.DataFrame:
    """
    Descarga Daily Summaries de NCEI/GHCND.

    Unidades con units=metric:
    - TMAX/TMIN: °C
    - PRCP: mm, si se solicita.
    """
    cache_key = (
        station,
        start_date,
        end_date,
        tuple(sorted([dt.upper() for dt in data_types])),
    )

    if cache_key in _NCEI_DAILY_CACHE:
        return _NCEI_DAILY_CACHE[cache_key].copy()

    params = {
        "dataset": "daily-summaries",
        "stations": station,
        "startDate": start_date,
        "endDate": end_date,
        "dataTypes": ",".join(data_types),
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
            f"Error NCEI Daily Summaries: {r.status_code} - {r.text[:300]}"
        )

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
        raise ValueError(
            f"NCEI: no encontré campo de fecha. Keys={list(rows[0].keys())}"
        )

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

    df = (
        pd.DataFrame(out)
        .drop_duplicates(subset=["time"], keep="last")
        .set_index("time")
        .sort_index()
    )

    _NCEI_DAILY_CACHE[cache_key] = df.copy()

    return df


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


def _fetch_isd_lite_year(station: str, year: int) -> pd.DataFrame:
    """
    Descarga un año de ISD-Lite para una estación.

    URL esperada:
    https://www.ncei.noaa.gov/pub/data/noaa/isd-lite/YYYY/STATION-YYYY.gz
    """
    cache_key = (station, year)

    if cache_key in _ISD_LITE_YEAR_CACHE:
        return _ISD_LITE_YEAR_CACHE[cache_key].copy()

    url = f"{ISD_LITE_BASE_URL}/{year}/{station}-{year}.gz"

    r = requests.get(url, timeout=40)

    if r.status_code == 404:
        df_empty = pd.DataFrame()
        _ISD_LITE_YEAR_CACHE[cache_key] = df_empty
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
        _ISD_LITE_YEAR_CACHE[cache_key] = df_empty
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

    _ISD_LITE_YEAR_CACHE[cache_key] = df.copy()

    return df


def _fetch_ncei_isd_lite_hourly(
    station: str,
    start_date: str,
    end_date: str,
    tz: str,
) -> pd.DataFrame:
    """
    Descarga ISD-Lite horario y conserva observaciones por hora local.

    Retorna filas horarias con:
    - datetime_utc
    - datetime_local
    - time
    - Temp_C
    - HR
    - Td
    - SLP
    - WindSpd
    - WindDir
    - Cloud
    - Precip_1h
    - Precip_6h
    """
    start_ts = pd.to_datetime(start_date).normalize()
    end_ts = pd.to_datetime(end_date).normalize() + pd.Timedelta(
        hours=23,
        minutes=59,
    )

    frames = []

    for year in range(start_ts.year, end_ts.year + 1):
        yearly = _fetch_isd_lite_year(station=station, year=year)

        if not yearly.empty:
            frames.append(yearly)

    if not frames:
        return pd.DataFrame()

    raw = pd.concat(frames, ignore_index=True)

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

    raw = raw[
        (raw["datetime_local"] >= start_ts)
        & (raw["datetime_local"] <= end_ts)
    ].copy()

    raw = raw.sort_values("datetime_local")

    return raw


def _build_23h_daily_from_hourly(
    hourly: pd.DataFrame,
    days: pd.DatetimeIndex,
    execution_hour: int,
    tolerance_hours: int,
) -> pd.DataFrame:
    """
    Construye una tabla diaria tomando la observación más cercana a execution_hour,
    sin usar observaciones futuras.
    """
    rows = []

    for day_ts in days:
        day_ts = pd.to_datetime(day_ts).normalize()
        target_dt = day_ts + pd.Timedelta(hours=execution_hour)

        obs = _nearest_observation_at_or_before(
            hourly=hourly,
            target_dt=target_dt,
            tolerance_hours=tolerance_hours,
        )

        if obs.empty:
            rows.append({
                "time": day_ts,
                "HR_23h": np.nan,
                "Td_23h": np.nan,
                "SLP_23h": np.nan,
                "WindSpd_23h": np.nan,
                "WindDir_23h": np.nan,
                "Cloud_23h": np.nan,
            })
        else:
            rows.append({
                "time": day_ts,
                "HR_23h": obs.get("HR", np.nan),
                "Td_23h": obs.get("Td", np.nan),
                "SLP_23h": obs.get("SLP", np.nan),
                "WindSpd_23h": obs.get("WindSpd", np.nan),
                "WindDir_23h": obs.get("WindDir", np.nan),
                "Cloud_23h": obs.get("Cloud", np.nan),
            })

    return pd.DataFrame(rows).set_index("time").sort_index()


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

    ncei_daily = _fetch_ncei_daily(
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
    td_anomaly_x = np.nan

    if compute_td_anomaly:
        td_history_start = history_start_ts
        td_history_end = target_ts

        hourly_history = _fetch_ncei_isd_lite_hourly(
            station=LGA_ISD_LITE_STATION,
            start_date=td_history_start.strftime("%Y-%m-%d"),
            end_date=td_history_end.strftime("%Y-%m-%d"),
            tz=CITY_COORDS[city_key]["tz"],
        )

        if not hourly_history.empty:
            td_days = pd.date_range(td_history_start, td_history_end, freq="D")

            td_daily_23h = _build_23h_daily_from_hourly(
                hourly=hourly_history,
                days=td_days,
                execution_hour=execution_hour,
                tolerance_hours=nearest_tolerance_hours,
            )

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