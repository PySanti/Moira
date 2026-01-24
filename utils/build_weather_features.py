
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd
import requests


OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


CITY_CFG = {
    "new york": {"lat": 40.7128, "lon": -74.0060, "tz": "America/New_York"},
    "atlanta": {"lat": 33.7490, "lon": -84.3880, "tz": "America/New_York"},
    "chicago": {"lat": 41.8781, "lon": -87.6298, "tz": "America/Chicago"},
    "londres": {"lat": 51.5074, "lon": -0.1278, "tz": "Europe/London"},
    "seul": {"lat": 37.5665, "lon": 126.9780, "tz": "Asia/Seoul"},
}


COLUMNS_ORDER = [
    "city", "date_local", "tz", "lat", "lon",
    "T_00", "T_06", "T_12", "Tmin_X_sofar", "T_mean_0_12", "T_std_0_12",
    "Td_00", "Td_06", "Td_12", "RH_12", "q_12", "VPD_12",
    "cloud_total_12", "cloud_low_12", "cloud_mean_0_12",
    "GHI_sum_0_12", "DNI_sum_0_12", "DHI_sum_0_12", "Rn_sum_0_12", "sunshine_h_0_12",
    "U_12", "V_12", "wind_speed_mean_0_12", "gust_max_0_12",
    "SLP_12", "P_sfc_12", "dP_3h", "dP_6h", "dP_24h",
    "precip_sum_0_12", "precip_intensity_max_0_12",
    "visibility_12", "fog_flag_0_12", "storm_flag_0_12",
    "soil_temp_12", "soil_moist_12", "snow_depth", "snow_cover",
    "AOD_12", "PM25_12", "O3_12",
    "dT_06_12", "dTd_06_12", "dU_06_12", "dV_06_12", "partial_range",
    "Tmax_Xm1", "Tmax_Xm2", "T12_Xm1", "Td12_Xm1", "SLP12_Xm1",
    "roll_Tmax_3", "roll_Tmax_7", "precip_3d_to_Xm1", "precip_7d_to_Xm1",
    "DoY", "sin_DoY", "cos_DoY", "daylength",
    "T850_12Z", "q850_12Z", "U850_12Z", "V850_12Z", "Z500_12Z", "thickness_1000_500_12Z", "dT850_24h", "dZ500_24h",
    "Tmax_X",
]


def _parse_ddmmyy(s: str) -> date:
    s = s.strip()
    dt = datetime.strptime(s, "%d-%m-%y")
    y = dt.year
    if y < 1970:
        y += 100
    return date(y, dt.month, dt.day)


def _safe_float(x: Any) -> float:
    try:
        if x is None:
            return float("nan")
        v = float(x)
        if math.isfinite(v):
            return v
        return float("nan")
    except Exception:
        return float("nan")


def _c_to_f(tc: float) -> float:
    if not math.isfinite(tc):
        return float("nan")
    return tc * 9.0 / 5.0 + 32.0


def _hpa_to_inhg(hpa: float) -> float:
    if not math.isfinite(hpa):
        return float("nan")
    return hpa * 0.0295299830714


def _mm_to_in(mm: float) -> float:
    if not math.isfinite(mm):
        return float("nan")
    return mm * 0.03937007874015748


def _m_to_miles(m: float) -> float:
    if not math.isfinite(m):
        return float("nan")
    return m * 0.000621371192237334


def _kmh_to_ms(kmh: float) -> float:
    if not math.isfinite(kmh):
        return float("nan")
    return kmh / 3.6


def _dewpoint_c_from_t_rh(tc: float, rh: float) -> float:
    if not (math.isfinite(tc) and math.isfinite(rh) and rh > 0.0 and rh <= 100.0):
        return float("nan")
    a = 17.625
    b = 243.04
    gamma = math.log(rh / 100.0) + (a * tc) / (b + tc)
    denom = (a - gamma)
    if denom == 0.0 or not math.isfinite(denom):
        return float("nan")
    td = (b * gamma) / denom
    return td


def _svp_hpa_from_t_c(tc: float) -> float:
    if not math.isfinite(tc):
        return float("nan")
    return 6.112 * math.exp((17.67 * tc) / (tc + 243.5))


def _vpd_inhg_from_t_rh(tc: float, rh: float) -> float:
    if not (math.isfinite(tc) and math.isfinite(rh) and rh >= 0.0 and rh <= 100.0):
        return float("nan")
    es = _svp_hpa_from_t_c(tc)
    if not math.isfinite(es):
        return float("nan")
    ea = (rh / 100.0) * es
    vpd_hpa = es - ea
    return _hpa_to_inhg(vpd_hpa)


def _wind_uv_from_speed_dir_ms(speed_ms: float, dir_deg_met: float) -> tuple[float, float]:
    if not (math.isfinite(speed_ms) and math.isfinite(dir_deg_met)):
        return (float("nan"), float("nan"))
    dir_rad = math.radians(dir_deg_met)
    u = -speed_ms * math.sin(dir_rad)
    v = -speed_ms * math.cos(dir_rad)
    return (u, v)


def _request_json_with_retries(url: str, params: dict, timeout: float = 20.0, retries: int = 3, backoff: float = 1.5) -> dict:
    last_err = None
    for i in range(max(1, retries)):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code >= 400:
                raise requests.HTTPError(f"HTTP {r.status_code}: {r.text[:300]}")
            return r.json()
        except Exception as e:
            last_err = e
            if i < retries - 1:
                time.sleep(backoff ** i)
                continue
            raise last_err


def build_weather_features(city: str, date_ddmmyy: str) -> tuple[pd.DataFrame, dict]:
    missing_notes: list[str] = []
    data_sources: list[str] = []
    units: dict[str, str] = {}

    if not isinstance(city, str) or not city.strip():
        raise ValueError("city must be a non-empty string")
    city_key = city.strip().lower()
    if city_key not in CITY_CFG:
        raise ValueError(f"city must be one of {set(CITY_CFG.keys())}")

    cfg = CITY_CFG[city_key]
    lat = float(cfg["lat"])
    lon = float(cfg["lon"])
    tz = str(cfg["tz"])
    tzinfo = ZoneInfo(tz)

    x_date = _parse_ddmmyy(date_ddmmyy)
    date_local_str = x_date.isoformat()

    # Prepare default row
    row: dict[str, Any] = {c: np.nan for c in COLUMNS_ORDER}
    row["city"] = city_key
    row["date_local"] = date_local_str
    row["tz"] = tz
    row["lat"] = lat
    row["lon"] = lon

    # Units (set for all numeric feature columns)
    for c in COLUMNS_ORDER:
        if c in {"city", "date_local", "tz"}:
            continue
        units[c] = ""

    # Define units exactly as requested/used
    temp_cols = ["T_00", "T_06", "T_12", "Tmin_X_sofar", "T_mean_0_12", "T_std_0_12",
                 "Td_00", "Td_06", "Td_12", "dT_06_12", "dTd_06_12", "partial_range",
                 "Tmax_Xm1", "Tmax_Xm2", "T12_Xm1", "Td12_Xm1", "roll_Tmax_3", "roll_Tmax_7",
                 "T850_12Z", "dT850_24h", "Tmax_X"]
    for c in temp_cols:
        units[c] = "degF"

    for c in ["wind_speed_mean_0_12", "gust_max_0_12", "U_12", "V_12", "dU_06_12", "dV_06_12", "U850_12Z", "V850_12Z"]:
        units[c] = "m/s"

    for c in ["SLP_12", "P_sfc_12", "dP_3h", "dP_6h", "dP_24h", "VPD_12"]:
        units[c] = "inHg"

    for c in ["precip_sum_0_12", "precip_intensity_max_0_12", "precip_3d_to_Xm1", "precip_7d_to_Xm1"]:
        units[c] = "in"

    for c in ["RH_12", "cloud_total_12", "cloud_low_12", "cloud_mean_0_12"]:
        units[c] = "%"

    units["visibility_12"] = "mi"
    units["GHI_sum_0_12"] = "Wh/m^2"
    for c in ["DNI_sum_0_12", "DHI_sum_0_12", "Rn_sum_0_12"]:
        units[c] = ""
    units["sunshine_h_0_12"] = "h"
    units["DoY"] = "day"
    units["sin_DoY"] = ""
    units["cos_DoY"] = ""
    units["daylength"] = "h"
    units["q_12"] = ""
    units["fog_flag_0_12"] = ""
    units["storm_flag_0_12"] = ""
    units["soil_temp_12"] = ""
    units["soil_moist_12"] = ""
    units["snow_depth"] = ""
    units["snow_cover"] = ""
    units["AOD_12"] = ""
    units["PM25_12"] = ""
    units["O3_12"] = ""
    units["q850_12Z"] = ""
    units["Z500_12Z"] = ""
    units["thickness_1000_500_12Z"] = ""
    units["dZ500_24h"] = ""

    # Time/season features
    doy = int(x_date.timetuple().tm_yday)
    row["DoY"] = doy
    row["sin_DoY"] = math.sin(2.0 * math.pi * doy / 365.25)
    row["cos_DoY"] = math.cos(2.0 * math.pi * doy / 365.25)
    row["daylength"] = np.nan  # optional; left NaN

    # Open-Meteo query ranges
    start_daily = (x_date - timedelta(days=7)).isoformat()
    end_daily = x_date.isoformat()

    start_hourly = (x_date - timedelta(days=1)).isoformat()
    end_hourly = x_date.isoformat()

    hourly_vars = [
        "temperature_2m",
        "relative_humidity_2m",
        "dew_point_2m",
        "cloud_cover",
        "cloud_cover_low",
        "wind_speed_10m",
        "wind_gusts_10m",
        "wind_direction_10m",
        "surface_pressure",
        "precipitation",
        "visibility",
        "shortwave_radiation",
        # Optional extras if present (may be missing)
        "pressure_msl",
        "precipitation_rate",
        "weathercode",
        "weather_code",
    ]

    daily_vars = [
        "temperature_2m_max",
        "precipitation_sum",
    ]

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_daily,
        "end_date": end_daily,
        "hourly": ",".join(hourly_vars),
        "daily": ",".join(daily_vars),
        "timezone": tz,
    }

    data_sources.append(f"{OPEN_METEO_ARCHIVE_URL}?latitude={lat}&longitude={lon}&start_date={start_daily}&end_date={end_daily}&timezone={tz}")
    missing_reason = ""

    try:
        js = _request_json_with_retries(OPEN_METEO_ARCHIVE_URL, params=params, timeout=20.0, retries=3, backoff=1.6)
    except Exception as e:
        missing_reason = f"Open-Meteo request failed: {type(e).__name__}: {e}"
        df = pd.DataFrame([[row.get(c, np.nan) for c in COLUMNS_ORDER]], columns=COLUMNS_ORDER)
        meta = {"data_sources": data_sources, "units": units, "missing_reason": missing_reason}
        return df, meta

    # Build hourly dataframe
    hourly = js.get("hourly", {}) or {}
    hourly_units = js.get("hourly_units", {}) or {}
    times = hourly.get("time", None)
    if not times:
        missing_reason = "Missing hourly.time in response"
        df = pd.DataFrame([[row.get(c, np.nan) for c in COLUMNS_ORDER]], columns=COLUMNS_ORDER)
        meta = {"data_sources": data_sources, "units": units, "missing_reason": missing_reason}
        return df, meta

    dt_index = pd.to_datetime(pd.Series(times), errors="coerce")
    if getattr(dt_index.dt, "tz", None) is None:
        dt_index = dt_index.dt.tz_localize(tzinfo, ambiguous="NaT", nonexistent="shift_forward")
    else:
        dt_index = dt_index.dt.tz_convert(tzinfo)

    hdf = pd.DataFrame(index=dt_index.values)
    for v in hourly_vars:
        if v in hourly and hourly.get(v) is not None:
            arr = hourly.get(v)
            if isinstance(arr, list) and len(arr) == len(hdf.index):
                hdf[v] = pd.to_numeric(pd.Series(arr), errors="coerce").values
            else:
                hdf[v] = np.nan
                missing_notes.append(f"hourly.{v} wrong length")
        else:
            hdf[v] = np.nan
            missing_notes.append(f"hourly.{v} missing")

    # Determine unit conversions based on hourly_units
    def _unit_of(var: str) -> str:
        u = hourly_units.get(var, "")
        return str(u).strip()

    # Convert temperature (assumed degC from API)
    # If API returns degF, detect and convert to C for internal calcs
    temp_unit = _unit_of("temperature_2m").lower()
    t_is_f = ("°f" in temp_unit) or ("fahrenheit" in temp_unit)
    if t_is_f:
        # Convert F->C for internal
        hdf["temperature_2m"] = (hdf["temperature_2m"] - 32.0) * 5.0 / 9.0

    # Dew point could be missing; if present, ensure in C internally
    td_unit = _unit_of("dew_point_2m").lower()
    td_is_f = ("°f" in td_unit) or ("fahrenheit" in td_unit)
    if td_is_f:
        hdf["dew_point_2m"] = (hdf["dew_point_2m"] - 32.0) * 5.0 / 9.0

    # Wind speed conversions to m/s internal
    def _to_ms(series: pd.Series, unit_str: str) -> pd.Series:
        u = unit_str.lower()
        if "km/h" in u or "kmh" in u:
            return series.apply(_kmh_to_ms)
        if "m/s" in u or "ms" == u or "mps" in u:
            return series.astype(float)
        if "mph" in u:
            return series.astype(float) * 0.44704
        return series.astype(float)

    hdf["wind_speed_10m_ms"] = _to_ms(hdf["wind_speed_10m"], _unit_of("wind_speed_10m"))
    hdf["wind_gusts_10m_ms"] = _to_ms(hdf["wind_gusts_10m"], _unit_of("wind_gusts_10m"))

    # Pressure conversions: hPa -> inHg internal
    def _to_hpa(series: pd.Series, unit_str: str) -> pd.Series:
        u = unit_str.lower()
        if "hpa" in u or "mbar" in u:
            return series.astype(float)
        if "pa" == u:
            return series.astype(float) / 100.0
        if "inhg" in u:
            # convert inHg -> hPa
            return series.astype(float) / 0.0295299830714
        return series.astype(float)

    hdf["surface_pressure_hpa"] = _to_hpa(hdf["surface_pressure"], _unit_of("surface_pressure"))
    hdf["pressure_msl_hpa"] = _to_hpa(hdf["pressure_msl"], _unit_of("pressure_msl"))

    # Precip conversions: mm -> in internal
    def _to_mm(series: pd.Series, unit_str: str) -> pd.Series:
        u = unit_str.lower()
        if "mm" in u:
            return series.astype(float)
        if "inch" in u or "in" == u:
            return series.astype(float) / 0.03937007874015748
        return series.astype(float)

    hdf["precip_mm"] = _to_mm(hdf["precipitation"], _unit_of("precipitation"))
    hdf["precip_rate_mmph"] = _to_mm(hdf["precipitation_rate"], _unit_of("precipitation_rate"))

    # Visibility conversions: m -> miles internal
    def _to_m(series: pd.Series, unit_str: str) -> pd.Series:
        u = unit_str.lower()
        if u == "m" or "meter" in u:
            return series.astype(float)
        if "km" in u:
            return series.astype(float) * 1000.0
        if "mi" in u or "mile" in u:
            return series.astype(float) / 0.000621371192237334
        return series.astype(float)

    hdf["visibility_m"] = _to_m(hdf["visibility"], _unit_of("visibility"))

    # Extract helpers for exact local times
    def _ts_for(d: date, hh: int) -> pd.Timestamp:
        return pd.Timestamp(datetime(d.year, d.month, d.day, hh, 0, 0), tz=tzinfo)

    def _at(var: str, d: date, hh: int) -> float:
        ts = _ts_for(d, hh)
        if ts in hdf.index:
            return _safe_float(hdf.loc[ts, var])
        return float("nan")

    def _between(d: date, h0: int, h1: int, inclusive_end: bool = True) -> pd.DataFrame:
        t0 = _ts_for(d, h0)
        t1 = _ts_for(d, h1)
        if inclusive_end:
            return hdf.loc[(hdf.index >= t0) & (hdf.index <= t1)]
        return hdf.loc[(hdf.index >= t0) & (hdf.index < t1)]

    # Compute dew point series in C
    td_available = "dew_point_2m" in hourly and hourly.get("dew_point_2m") is not None and not hdf["dew_point_2m"].isna().all()
    if not td_available:
        # derive from T and RH
        tC = hdf["temperature_2m"].astype(float)
        rh = hdf["relative_humidity_2m"].astype(float)
        derived = []
        for tc, rhi in zip(tC.values.tolist(), rh.values.tolist()):
            derived.append(_dewpoint_c_from_t_rh(_safe_float(tc), _safe_float(rhi)))
        hdf["dew_point_2m"] = pd.Series(derived, index=hdf.index, dtype="float64")
        missing_notes.append("dew_point_2m derived via Magnus from temperature_2m and relative_humidity_2m")

    # Feature window for day X: [00:00, 12:00] local (inclusive)
    wX = _between(x_date, 0, 12, inclusive_end=True)

    # Temps at 00/06/12 (F)
    T00_c = _at("temperature_2m", x_date, 0)
    T06_c = _at("temperature_2m", x_date, 6)
    T12_c = _at("temperature_2m", x_date, 12)

    row["T_00"] = _c_to_f(T00_c)
    row["T_06"] = _c_to_f(T06_c)
    row["T_12"] = _c_to_f(T12_c)

    # Tmin so far, mean, std (F)
    tC_series = wX["temperature_2m"].astype(float)
    if tC_series.notna().any():
        row["Tmin_X_sofar"] = _c_to_f(float(np.nanmin(tC_series.values)))
        row["T_mean_0_12"] = _c_to_f(float(np.nanmean(tC_series.values)))
        row["T_std_0_12"] = float(np.nanstd((_c_to_f(tC_series)).values, ddof=0)) if len(tC_series) > 0 else np.nan
    else:
        missing_notes.append("temperature_2m missing in [00,12] window")

    # Dew points at 00/06/12 (F)
    Td00_c = _at("dew_point_2m", x_date, 0)
    Td06_c = _at("dew_point_2m", x_date, 6)
    Td12_c = _at("dew_point_2m", x_date, 12)

    row["Td_00"] = _c_to_f(Td00_c)
    row["Td_06"] = _c_to_f(Td06_c)
    row["Td_12"] = _c_to_f(Td12_c)

    # RH_12
    RH12 = _at("relative_humidity_2m", x_date, 12)
    row["RH_12"] = RH12

    # q_12 (optional; keep NaN by default)
    row["q_12"] = np.nan

    # VPD_12 in inHg using T_12 and RH_12
    row["VPD_12"] = _vpd_inhg_from_t_rh(T12_c, RH12)

    # Clouds
    row["cloud_total_12"] = _at("cloud_cover", x_date, 12)
    row["cloud_low_12"] = _at("cloud_cover_low", x_date, 12)

    cloud_series = wX["cloud_cover"].astype(float)
    row["cloud_mean_0_12"] = float(np.nanmean(cloud_series.values)) if cloud_series.notna().any() else np.nan

    # Radiation integration: use hours 00..11 (12 hours) for energy Wh/m^2
    wX_rad = _between(x_date, 0, 12, inclusive_end=False)  # [00,12)
    rad_series = wX_rad["shortwave_radiation"].astype(float)
    if rad_series.notna().any():
        row["GHI_sum_0_12"] = float(np.nansum(rad_series.values)) * 1.0  # Wh/m^2 approx (W/m^2 * hour)
    else:
        row["GHI_sum_0_12"] = np.nan
        if "shortwave_radiation" in missing_notes:
            pass

    # Not available from this source
    row["DNI_sum_0_12"] = np.nan
    row["DHI_sum_0_12"] = np.nan
    row["Rn_sum_0_12"] = np.nan
    row["sunshine_h_0_12"] = np.nan

    # Wind mean and gust max over [00,12]
    ws_series = wX["wind_speed_10m_ms"].astype(float)
    row["wind_speed_mean_0_12"] = float(np.nanmean(ws_series.values)) if ws_series.notna().any() else np.nan

    gust_series = wX["wind_gusts_10m_ms"].astype(float)
    row["gust_max_0_12"] = float(np.nanmax(gust_series.values)) if gust_series.notna().any() else np.nan

    # Wind components at 12
    ws12 = _at("wind_speed_10m_ms", x_date, 12)
    wd12 = _at("wind_direction_10m", x_date, 12)
    u12, v12 = _wind_uv_from_speed_dir_ms(ws12, wd12)
    row["U_12"] = u12
    row["V_12"] = v12

    # Pressure at 12: prefer pressure_msl if present else surface_pressure
    p12_hpa = _at("pressure_msl_hpa", x_date, 12)
    if not math.isfinite(p12_hpa):
        p12_hpa = _at("surface_pressure_hpa", x_date, 12)
    row["SLP_12"] = _hpa_to_inhg(p12_hpa)

    psfc12_hpa = _at("surface_pressure_hpa", x_date, 12)
    row["P_sfc_12"] = _hpa_to_inhg(psfc12_hpa)

    # Pressure differences
    psfc09_hpa = _at("surface_pressure_hpa", x_date, 9)
    psfc06_hpa = _at("surface_pressure_hpa", x_date, 6)
    row["dP_3h"] = _hpa_to_inhg(psfc12_hpa) - _hpa_to_inhg(psfc09_hpa) if (math.isfinite(psfc12_hpa) and math.isfinite(psfc09_hpa)) else np.nan
    row["dP_6h"] = _hpa_to_inhg(psfc12_hpa) - _hpa_to_inhg(psfc06_hpa) if (math.isfinite(psfc12_hpa) and math.isfinite(psfc06_hpa)) else np.nan

    # dP_24h: P(12 today) - P(12 yesterday)
    y_date = x_date - timedelta(days=1)
    psfc12_y_hpa = _at("surface_pressure_hpa", y_date, 12)
    row["dP_24h"] = _hpa_to_inhg(psfc12_hpa) - _hpa_to_inhg(psfc12_y_hpa) if (math.isfinite(psfc12_hpa) and math.isfinite(psfc12_y_hpa)) else np.nan

    # Precip sums in inches over [00,12]
    precip_series_mm = wX["precip_mm"].astype(float)
    row["precip_sum_0_12"] = _mm_to_in(float(np.nansum(precip_series_mm.values))) if precip_series_mm.notna().any() else np.nan

    # Precip intensity max (in/h) if precipitation_rate exists; else NaN
    pr_series_mmph = wX["precip_rate_mmph"].astype(float)
    row["precip_intensity_max_0_12"] = _mm_to_in(float(np.nanmax(pr_series_mmph.values))) if pr_series_mmph.notna().any() else np.nan

    # Visibility at 12 (miles)
    vis12_m = _at("visibility_m", x_date, 12)
    row["visibility_12"] = _m_to_miles(vis12_m)

    # Fog/storm flags: default NaN; if weather code present, simple derivation
    row["fog_flag_0_12"] = np.nan
    row["storm_flag_0_12"] = np.nan
    wc_col = None
    if "weathercode" in hdf.columns and not hdf["weathercode"].isna().all():
        wc_col = "weathercode"
    elif "weather_code" in hdf.columns and not hdf["weather_code"].isna().all():
        wc_col = "weather_code"
    if wc_col is not None:
        wc_win = wX[wc_col].astype(float)
        # Simple rules (WMO weather codes): fog often 45,48 ; thunderstorm 95,96,99
        fog = wc_win.isin([45.0, 48.0]).any()
        storm = wc_win.isin([95.0, 96.0, 99.0]).any()
        row["fog_flag_0_12"] = 1.0 if fog else 0.0
        row["storm_flag_0_12"] = 1.0 if storm else 0.0

    # Placeholders (not implemented / not available in this source)
    for c in ["soil_temp_12", "soil_moist_12", "snow_depth", "snow_cover", "AOD_12", "PM25_12", "O3_12"]:
        row[c] = np.nan

    # Derived deltas and ranges
    row["dT_06_12"] = row["T_12"] - row["T_06"] if (math.isfinite(row["T_12"]) and math.isfinite(row["T_06"])) else np.nan
    row["dTd_06_12"] = row["Td_12"] - row["Td_06"] if (math.isfinite(row["Td_12"]) and math.isfinite(row["Td_06"])) else np.nan

    # dU/dV: compute U_06/V_06 if possible
    ws06 = _at("wind_speed_10m_ms", x_date, 6)
    wd06 = _at("wind_direction_10m", x_date, 6)
    u06, v06 = _wind_uv_from_speed_dir_ms(ws06, wd06)
    row["dU_06_12"] = (row["U_12"] - u06) if (math.isfinite(row["U_12"]) and math.isfinite(u06)) else np.nan
    row["dV_06_12"] = (row["V_12"] - v06) if (math.isfinite(row["V_12"]) and math.isfinite(v06)) else np.nan

    row["partial_range"] = row["T_12"] - row["Tmin_X_sofar"] if (math.isfinite(row["T_12"]) and math.isfinite(row["Tmin_X_sofar"])) else np.nan

    # Daily data processing for Tmax and lag/roll features
    daily = js.get("daily", {}) or {}
    daily_units = js.get("daily_units", {}) or {}
    d_times = daily.get("time", None)

    ddf = None
    if d_times:
        d_idx = pd.to_datetime(pd.Series(d_times), errors="coerce")
        # daily times are dates; keep as date index
        ddf = pd.DataFrame(index=d_idx.dt.date.values)
        for v in daily_vars:
            if v in daily and daily.get(v) is not None:
                arr = daily.get(v)
                if isinstance(arr, list) and len(arr) == len(ddf.index):
                    ddf[v] = pd.to_numeric(pd.Series(arr), errors="coerce").values
                else:
                    ddf[v] = np.nan
                    missing_notes.append(f"daily.{v} wrong length")
            else:
                ddf[v] = np.nan
                missing_notes.append(f"daily.{v} missing")
    else:
        missing_notes.append("daily.time missing")

    # Helper to get daily var by date
    def _daily_at(var: str, d: date) -> float:
        if ddf is None or var not in ddf.columns:
            return float("nan")
        try:
            return _safe_float(ddf.loc[d, var])
        except Exception:
            return float("nan")

    # Detect daily temperature unit
    tmax_unit = str(daily_units.get("temperature_2m_max", "")).lower()
    tmax_is_f = ("°f" in tmax_unit) or ("fahrenheit" in tmax_unit)

    def _tmax_f(d: date) -> float:
        v = _daily_at("temperature_2m_max", d)
        if not math.isfinite(v):
            return float("nan")
        if tmax_is_f:
            return v
        return _c_to_f(v)

    # Target Tmax_X
    row["Tmax_X"] = _tmax_f(x_date)

    # Lags
    row["Tmax_Xm1"] = _tmax_f(x_date - timedelta(days=1))
    row["Tmax_Xm2"] = _tmax_f(x_date - timedelta(days=2))

    # Rolling means up to X-1
    def _roll_mean_tmax(days: int) -> float:
        vals = []
        for k in range(1, days + 1):
            vals.append(_tmax_f(x_date - timedelta(days=k)))
        vals = [v for v in vals if math.isfinite(v)]
        return float(np.mean(vals)) if len(vals) > 0 else np.nan

    row["roll_Tmax_3"] = _roll_mean_tmax(3)
    row["roll_Tmax_7"] = _roll_mean_tmax(7)

    # T12_Xm1 and Td12_Xm1
    T12_y_c = _at("temperature_2m", y_date, 12)
    Td12_y_c = _at("dew_point_2m", y_date, 12)
    row["T12_Xm1"] = _c_to_f(T12_y_c)
    row["Td12_Xm1"] = _c_to_f(Td12_y_c)

    # SLP12_Xm1 (use same logic as SLP_12 but for yesterday at 12)
    p12y_hpa = _at("pressure_msl_hpa", y_date, 12)
    if not math.isfinite(p12y_hpa):
        p12y_hpa = _at("surface_pressure_hpa", y_date, 12)
    row["SLP12_Xm1"] = _hpa_to_inhg(p12y_hpa)

    # Daily precipitation rolling sums (inches) if daily precipitation_sum exists
    def _daily_precip_in(d: date) -> float:
        if ddf is None:
            return float("nan")
        v = _daily_at("precipitation_sum", d)
        if not math.isfinite(v):
            return float("nan")
        u = str(daily_units.get("precipitation_sum", "")).lower()
        if "mm" in u:
            return _mm_to_in(v)
        if "inch" in u or "in" == u:
            return v
        return _mm_to_in(v)

    def _precip_roll_in(days: int) -> float:
        vals = []
        for k in range(1, days + 1):
            vals.append(_daily_precip_in(x_date - timedelta(days=k)))
        vals = [v for v in vals if math.isfinite(v)]
        return float(np.sum(vals)) if len(vals) > 0 else np.nan

    row["precip_3d_to_Xm1"] = _precip_roll_in(3)
    row["precip_7d_to_Xm1"] = _precip_roll_in(7)

    # Upper-air placeholders
    for c in ["T850_12Z", "q850_12Z", "U850_12Z", "V850_12Z", "Z500_12Z",
              "thickness_1000_500_12Z", "dT850_24h", "dZ500_24h"]:
        row[c] = np.nan

    # Ensure cloud_low_12 is NaN if missing in response
    if not math.isfinite(row["cloud_low_12"]) and ("hourly.cloud_cover_low missing" not in missing_notes):
        if "hourly.cloud_cover_low missing" in missing_notes:
            pass

    # Ensure exact column order and build dataframe
    df = pd.DataFrame([[row.get(c, np.nan) for c in COLUMNS_ORDER]], columns=COLUMNS_ORDER)

    # Missing reason summary
    if missing_notes:
        missing_reason = "; ".join(sorted(set(missing_notes)))
    else:
        missing_reason = ""

    meta = {
        "data_sources": data_sources,
        "units": units,
        "missing_reason": missing_reason,
    }
    return df, meta


