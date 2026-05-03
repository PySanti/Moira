"""
Consulta las APIs climáticas necesarias para obtener datos históricos o recientes del día X.

Versión as-of 23h:
- NCEI/GHCND Daily Summaries:
    - TMAX[x-1], TMAX[x-2] para lags/MA.
    - TMAX[x+1] para target oficial.

- NCEI ISD-Lite / Global Hourly, estación LaGuardia:
    - Temperatura horaria desde 00:00 hasta 23:00 del día X.
    - HR, Td, SLP, viento y nubosidad cercanos a las 23:00.
    - Precipitación acumulada desde 00:00 hasta 23:00.

El objetivo es simular el momento real de ejecución del bot:
día X a las 23:00 hora local de New York.
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


# ---------------- HELPERS GENERALES ----------------

def _safe(v):
    return None if (v is None or pd.isna(v)) else float(v)


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
    - PRCP: mm, si se solicita
    """
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
    url = f"{ISD_LITE_BASE_URL}/{year}/{station}-{year}.gz"

    r = requests.get(url, timeout=40)

    if r.status_code == 404:
        return pd.DataFrame()

    if r.status_code != 200:
        raise ConnectionError(
            f"Error ISD-Lite {year}: {r.status_code} - {r.text[:200]}"
        )

    try:
        text = gzip.decompress(r.content).decode("utf-8", errors="replace")
    except OSError:
        text = r.content.decode("utf-8", errors="replace")

    if not text.strip():
        return pd.DataFrame()

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


# ---------------- MAIN ----------------

def get_weather_features(
    city: str,
    date_str: str,
    strict: bool = True,
    execution_hour: int = 23,
    nearest_tolerance_hours: int = 2,
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
    previous_execution_dt = execution_dt - pd.Timedelta(days=1)

    # Daily Summaries:
    # - x-2 y x-1 para lags/MA
    # - x+1 para target
    daily_start = target_ts - pd.Timedelta(days=2)
    daily_end = next_ts

    daily_idx = pd.date_range(daily_start, daily_end, freq="D")

    ncei_daily = _fetch_ncei_daily(
        station=LGA_GHCND_STATION,
        start_date=daily_start.strftime("%Y-%m-%d"),
        end_date=daily_end.strftime("%Y-%m-%d"),
        data_types=["TMAX", "TMIN"],
    ).reindex(daily_idx)

    # ISD-Lite horario:
    # - x-1 para presión 23h del día anterior
    # - x para features as-of 23h
    hourly_start = target_ts - pd.Timedelta(days=1)
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

    obs_23 = _nearest_observation_at_or_before(
        hourly=hourly,
        target_dt=execution_dt,
        tolerance_hours=nearest_tolerance_hours,
    )

    obs_prev_23 = _nearest_observation_at_or_before(
        hourly=hourly,
        target_dt=previous_execution_dt,
        tolerance_hours=nearest_tolerance_hours,
    )

    # ---------------- FEATURES TÉRMICAS AS-OF 23H ----------------

    tmax_so_far_23h_x = day_rows["Temp_C"].max()
    tmin_so_far_23h_x = day_rows["Temp_C"].min()
    tmean_so_far_23h_x = day_rows["Temp_C"].mean()

    tmax_x_minus_1 = ncei_daily.loc[target_ts - pd.Timedelta(days=1), "TMAX"]
    tmax_x_minus_2 = ncei_daily.loc[target_ts - pd.Timedelta(days=2), "TMAX"]

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

    # ---------------- FEATURES 23H ----------------

    hr_23h_x = obs_23.get("HR", np.nan)
    td_23h_x = obs_23.get("Td", np.nan)
    slp_23h_x = obs_23.get("SLP", np.nan)
    wind_spd_23h_x = obs_23.get("WindSpd", np.nan)
    wind_dir_23h_x = obs_23.get("WindDir", np.nan)
    cloud_23h_x = obs_23.get("Cloud", np.nan)

    slp_23h_x_minus_1 = obs_prev_23.get("SLP", np.nan)

    delta_slp_24h_23h = (
        slp_23h_x - slp_23h_x_minus_1
        if pd.notna(slp_23h_x) and pd.notna(slp_23h_x_minus_1)
        else np.nan
    )

    wind_rad = np.deg2rad(wind_dir_23h_x) if pd.notna(wind_dir_23h_x) else np.nan
    wind_dir_sin_23h_x = np.sin(wind_rad) if pd.notna(wind_rad) else np.nan
    wind_dir_cos_23h_x = np.cos(wind_rad) if pd.notna(wind_rad) else np.nan

    precip_sum_00_23h_x = _precip_sum_until_execution(day_rows)

    # ---------------- ESTACIONALIDAD ----------------

    doy = target_ts.dayofyear
    doy_sin = np.sin(2 * np.pi * doy / 365.25)
    doy_cos = np.cos(2 * np.pi * doy / 365.25)

    # ---------------- TARGET ----------------

    target_x_plus_1 = ncei_daily.loc[next_ts, "TMAX"]

    features = {
        # Térmicas as-of 23h
        "Tmax_so_far_23h_x": _safe(tmax_so_far_23h_x),
        "Tmin_so_far_23h_x": _safe(tmin_so_far_23h_x),
        "Tmean_so_far_23h_x": _safe(tmean_so_far_23h_x),
        "Delta_Tmax_so_far_1d_23h": _safe(delta_tmax_so_far_1d_23h),
        "MA_Tmax_3d_asof_23h": _safe(ma_tmax_3d_asof_23h),
        "DTR_so_far_23h_x": _safe(dtr_so_far_23h_x),

        # Variables de sensores de LaGuardia cercanas a 23h
        "HR_23h_x": _safe(hr_23h_x),
        "Td_23h_x": _safe(td_23h_x),
        "SLP_23h_x": _safe(slp_23h_x),
        "Delta_SLP_24h_23h": _safe(delta_slp_24h_23h),
        "WindSpd_23h_x": _safe(wind_spd_23h_x),
        "WindDir_sin_23h_x": _safe(wind_dir_sin_23h_x),
        "WindDir_cos_23h_x": _safe(wind_dir_cos_23h_x),
        "Cloud_23h_x": _safe(cloud_23h_x),

        # Acumulada desde 00:00 hasta 23h
        "Precip_sum_00_23h_x": _safe(precip_sum_00_23h_x),

        # Target oficial
        "t_max_x+1": _safe(target_x_plus_1),

        # Metadata / estacionalidad
        "ciudad": city,
        "doy_sin": _safe(doy_sin),
        "doy_cos": _safe(doy_cos),
    }

    if strict:
        allowed_non_numeric = {"ciudad"}

        missing_features = [
            k for k, v in features.items()
            if k not in allowed_non_numeric and v is None
        ]

        if missing_features:
            print("Error: faltan features necesarias para construir la fila as-of 23:00.")
            for k in missing_features[:40]:
                print(" -", k)

            if len(missing_features) > 40:
                print(f" - ... y {len(missing_features) - 40} más")

            return {}

    return features