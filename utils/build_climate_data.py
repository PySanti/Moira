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
        "dataTypes": ",".join(data_types),  # e.g. "TMAX,TMIN"
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

    # Detecta key de fecha (a veces viene como DATE o date)
    def _find_key(d: dict, candidates: list[str]) -> str | None:
        keys = {k.lower(): k for k in d.keys()}
        for c in candidates:
            if c.lower() in keys:
                return keys[c.lower()]
        return None

    date_key = _find_key(rows[0], ["DATE", "date"])
    if not date_key:
        raise ValueError(f"NCEI: no encontré campo de fecha en la respuesta. Keys={list(rows[0].keys())}")

    out = []
    for row in rows:
        rec = {}
        rec["time"] = pd.to_datetime(row[date_key]).normalize()
        for dt in data_types:
            # dt puede venir como "TMAX" exactamente, pero lo busco case-insensitive
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
    Open-Meteo Archive para el resto de features (NO usaremos su Tmax si estamos en NYC),
    unidades explícitas: °C y m/s.
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
        "temperature_2m_min": "Tmin_model",     # °C (modelo)
        "temperature_2m_mean": "Tmean_model",   # °C (modelo)
        "relative_humidity_2m_mean": "HR",      # %
        "dew_point_2m_mean": "Td",              # °C
        "surface_pressure_mean": "SLP",         # hPa (ojo: es surface pressure, no SLP real)
        "wind_speed_10m_mean": "WindSpd",       # m/s
        "wind_direction_10m_dominant": "WindDir", # grados
        "cloud_cover_mean": "Cloud",            # %
        "precipitation_sum": "Precip",          # mm
    })

    return df


# ---------------- MAIN ----------------
def get_weather_features(city: str, date_str: str, strict: bool = True) -> dict:
    """
    Para NYC:
      - Tmax (x y previos, y x+1) viene de NCEI GHCND (USW00014732).
      - (Opcional, pero recomendado) Tmin también se toma de NCEI para coherencia.
      - El resto (HR, Td, Cloud, etc.) viene de Open-Meteo.

    strict=True:
      - si faltan Tmax necesarios para ΔTmax_1d / MA_Tmax_3d / label x+1 => retorna {}
    """
    city_key = city.lower().strip()
    if city_key not in CITY_COORDS:
        raise ValueError(f"Ciudad no soportada. Use: {list(CITY_COORDS.keys())}")

    target_date = datetime.strptime(date_str, "%d-%m-%y")
    next_day = target_date + timedelta(days=1)
    start_date = target_date - timedelta(days=5)

    api_start = start_date.strftime("%Y-%m-%d")
    api_end = next_day.strftime("%Y-%m-%d")

    # Rango diario esperado (incluye x-5 ... x ... x+1) => 7 días
    expected_idx = pd.date_range(api_start, api_end, freq="D")

    # 1) Open-Meteo para features NO Tmax
    om = _fetch_open_meteo_daily(
        lat=CITY_COORDS[city_key]["lat"],
        lon=CITY_COORDS[city_key]["lon"],
        start_date=api_start,
        end_date=api_end,
        tz=CITY_COORDS[city_key]["tz"],
    ).reindex(expected_idx)

    # 2) NCEI para Tmax (y Tmin recomendado)
    #    *** Esto cumple tu requerimiento: TODOS los Tmax (<=x y x+1) salen de NCEI ***
    if city_key == "new york":
        ncei = _fetch_ncei_daily(
            station=LGA_GHCND_STATION,
            start_date=api_start,
            end_date=api_end,
            data_types=["TMAX", "TMIN"],  # <-- si NO quieres Tmin, deja ["TMAX"]
        ).reindex(expected_idx)
    else:
        # Si quieres generalizar a otras ciudades, aquí deberías mapear estaciones.
        raise ValueError("Esta versión solo implementa Tmax por NCEI para New York (LaGuardia USW00014732).")

    df = pd.DataFrame(index=expected_idx)
    df.index.name = "time"

    # Tmax y Tmin de estación (°C)
    df["Tmax"] = ncei["TMAX"]
    df["Tmin"] = ncei["TMIN"]  # si no traes TMIN, quedará NaN

    # Para Tmean, si tenemos Tmax y Tmin de estación, mejor coherencia:
    df["Tmean"] = (df["Tmax"] + df["Tmin"]) / 2.0

    # Resto de variables desde Open-Meteo (modelo)
    for col in ["HR", "Td", "SLP", "WindSpd", "WindDir", "Cloud", "Precip"]:
        df[col] = om[col]

    # --- Derivadas (a prueba de huecos) ---
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

    target_ts = pd.to_datetime(target_date.strftime("%Y-%m-%d"))
    next_ts = pd.to_datetime(next_day.strftime("%Y-%m-%d"))

    if target_ts not in df.index:
        raise ValueError(f"No se encontró el día objetivo {target_ts.date()} en el índice reindexado.")

    target_row = df.loc[target_ts]

    # Validaciones mínimas para que no metas basura al dataset
    # (Tmax x, Tmax x-1, Tmax x-2, y Tmax x+1 para label)
    required_for_tmax_features = [
        ("Tmax_día_x", df.loc[target_ts, "Tmax"]),
        ("Tmax_día_x-1", df.loc[target_ts - pd.Timedelta(days=1), "Tmax"]),
        ("Tmax_día_x-2", df.loc[target_ts - pd.Timedelta(days=2), "Tmax"]),
        ("Tmax_día_x+1 (label)", df.loc[next_ts, "Tmax"]),
    ]
    missing = [name for name, val in required_for_tmax_features if pd.isna(val)]

    if strict and missing:
        print("Faltan Tmax de NCEI para calcular features/label correctamente:", missing)
        return {}

    def _safe(v):
        return None if (v is None or (isinstance(v, float) and np.isnan(v))) else float(v)

    features = {
        # Tmax/Tmin/Tmean vienen de NCEI (°C)
        "Tmax_día_x": _safe(target_row["Tmax"]),
        "Tmin_día_x": _safe(target_row["Tmin"]),
        "Tmedia_día_x": _safe(target_row["Tmean"]),

        # derivadas basadas en Tmax NCEI
        "ΔTmax_1d": _safe(target_row["Delta_Tmax_1d"]),
        "MA_Tmax_3d": _safe(target_row["MA_Tmax_3d"]),
        "DTR_x": _safe(target_row["DTR"]),

        # resto desde Open-Meteo
        "HR_media_día_x": _safe(target_row["HR"]),
        "Punto_de_rocío_día_x (Td)": _safe(target_row["Td"]),
        "Presión_media_día_x (SLP)": _safe(target_row["SLP"]),
        "ΔPresión_24h": _safe(target_row["Delta_Presion"]),
        "Viento_vel_media_día_x": _safe(target_row["WindSpd"]),
        "Viento_dir_sin(x)": _safe(target_row["Wind_sin"]),
        "Viento_dir_cos(x)": _safe(target_row["Wind_cos"]),
        "Nubosidad_media_día_x": _safe(target_row["Cloud"]),
        "Precipitación_acum_día_x": _safe(target_row["Precip"]),

        # label: Tmax(x+1) de NCEI (°C)
        "t_max_x+1": _safe(df.loc[next_ts, "Tmax"]),

        "ciudad": city,
        "doy_sin": _safe(target_row["doy_sin"]),
        "doy_cos": _safe(target_row["doy_cos"]),
    }

    return features
