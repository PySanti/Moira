import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import math

NCEI_DATA_URL = "https://www.ncei.noaa.gov/access/services/data/v1"
LGA_GHCND_STATION = "USW00014732"  # LaGuardia Airport (GHCND)

def _get_polymarket_like_tmax_f_klga(date_obj: datetime) -> int | None:
    """
    Devuelve Tmax diaria (°F entero) para KLGA/LaGuardia,
    aproximando el método de Polymarket (Wunderground KLGA, °F a enteros).
    Fuente programática: NCEI Daily Summaries (GHCND).
    """
    day = date_obj.strftime("%Y-%m-%d")

    params = {
        "dataset": "daily-summaries",
        "stations": LGA_GHCND_STATION,
        "startDate": day,
        "endDate": day,
        "dataTypes": "TMAX",
        "format": "json",
        "units": "standard",  # convierte a unidades US (°F, etc.) si aplica
        "includeStationName": "false",
        "includeStationLocation": "false",
        "includeAttributes": "false",
    }

    headers = {
        # NCEI/NWS suelen agradecer User-Agent identificable
        "User-Agent": "tmax-bot/1.0 (contact: you@example.com)"
    }

    r = requests.get(NCEI_DATA_URL, params=params, headers=headers, timeout=25)
    if r.status_code != 200:
        raise ConnectionError(f"Error NCEI: {r.status_code} - {r.text[:300]}")

    rows = r.json()
    if not rows:
        return None

    row = rows[0]
    # Campo típico: "TMAX"
    tmax_raw = None
    for k in row.keys():
        if k.upper() == "TMAX":
            tmax_raw = row[k]
            break

    if tmax_raw in (None, "", "NaN"):
        return None

    try:
        tmax_f = float(tmax_raw)
    except ValueError:
        return None

    # Algunos datasets usan flags de missing grandes (por si acaso)
    if tmax_f <= -9000:
        return None

    # Polymarket usa °F a enteros
    return tmax_f

def get_weather_features(city: str, date_str: str):
    """
    Obtiene features meteorológicas para una ciudad y fecha específicas.
    Label t_max_x+1 se ajusta para NYC al criterio Polymarket (KLGA, °F entero).
    """
    city_coords = {
        'new york': {'lat': 40.7128, 'lon': -74.0060},
        'chicago':  {'lat': 41.8781, 'lon': -87.6298},
        'atlanta':  {'lat': 33.7490, 'lon': -84.3880},
        'seul':     {'lat': 37.5665, 'lon': 126.9780},
        'londres':  {'lat': 51.5074, 'lon': -0.1278}
    }

    city_key = city.lower().strip()
    if city_key not in city_coords:
        raise ValueError(f"Ciudad no soportada. Use: {list(city_coords.keys())}")

    target_date = datetime.strptime(date_str, "%d-%m-%y")
    next_day_date = target_date + timedelta(days=1)
    start_date = target_date - timedelta(days=5)

    api_start = start_date.strftime("%Y-%m-%d")
    api_end = next_day_date.strftime("%Y-%m-%d")

    # Open-Meteo (para el resto de features)
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": city_coords[city_key]['lat'],
        "longitude": city_coords[city_key]['lon'],
        "start_date": api_start,
        "end_date": api_end,
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "temperature_2m_mean",
            "precipitation_sum",
            "wind_speed_10m_mean",
            "wind_direction_10m_dominant",
            "surface_pressure_mean",
            "relative_humidity_2m_mean",
            "cloud_cover_mean",
            "dew_point_2m_mean"
        ],
        "timezone": "America/New_York"
    }

    response = requests.get(url, params=params, timeout=25)
    if response.status_code != 200:
        raise ConnectionError(f"Error conectando con Open-Meteo: {response.text[:300]}")

    data = response.json()

    daily_data = data['daily']


    df = pd.DataFrame(daily_data)

    df_sorted = df.sort_values('time')
    deltas = df_sorted['time'].diff().dropna().dt.days
    if not (deltas == 1).all():
        # aquí hay huecos o saltos (2 días, etc.)
        print("⚠️ Hay gaps en fechas:", deltas.value_counts())
        return {}
    
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time')

    df = df.rename(columns={
        'temperature_2m_max': 'Tmax',
        'temperature_2m_min': 'Tmin',
        'temperature_2m_mean': 'Tmean',
        'relative_humidity_2m_mean': 'HR',
        'dew_point_2m_mean': 'Td',
        'surface_pressure_mean': 'SLP',
        'wind_speed_10m_mean': 'WindSpd',
        'wind_direction_10m_dominant': 'WindDir',
        'cloud_cover_mean': 'Cloud',
        'precipitation_sum': 'Precip'
    })



    df['Delta_Tmax_1d'] = df['Tmax'].diff()
    df['MA_Tmax_3d'] = df['Tmax'].rolling(window=3).mean()
    df['DTR'] = df['Tmax'] - df['Tmin']
    df['Delta_Presion'] = df['SLP'].diff()

    wind_rad = np.deg2rad(df['WindDir'])
    df['Wind_sin'] = np.sin(wind_rad)
    df['Wind_cos'] = np.cos(wind_rad)

    df['doy'] = df['time'].dt.dayofyear
    df['day'] = df['time'].dt.day
    df['month'] = df['time'].dt.month
    df['year'] = df['time'].dt.year
    df['doy_sin'] = np.sin(2 * np.pi * df['doy'] / 365.25)
    df['doy_cos'] = np.cos(2 * np.pi * df['doy'] / 365.25)

    target_rows = df[df['time'].dt.date == target_date.date()]
    if target_rows.empty:
        raise ValueError(f"No se encontraron datos para la fecha {date_str}")
    target_row = target_rows.iloc[0]

    t_max_next_polymarket_f = None
    if city_key == "new york":
        t_max_next_polymarket_f = _get_polymarket_like_tmax_f_klga(next_day_date)

    features = {
        "Tmax_día_x": target_row['Tmax'],
        "Tmin_día_x": target_row['Tmin'],
        "Tmedia_día_x": target_row['Tmean'],
        "ΔTmax_1d": target_row['Delta_Tmax_1d'],
        "MA_Tmax_3d": target_row['MA_Tmax_3d'],
        "DTR_x": target_row['DTR'],
        "HR_media_día_x": target_row['HR'],
        "Punto_de_rocío_día_x (Td)": target_row['Td'],
        "Presión_media_día_x (SLP)": target_row['SLP'],
        "ΔPresión_24h": target_row['Delta_Presion'],
        "Viento_vel_media_día_x": target_row['WindSpd'],
        "Viento_dir_sin(x)": target_row['Wind_sin'],
        "Viento_dir_cos(x)": target_row['Wind_cos'],
        "Nubosidad_media_día_x": target_row['Cloud'],
        "Precipitación_acum_día_x": target_row['Precip'],

        # NUEVO: label estilo Polymarket (solo NYC) -> °F entero (KLGA)
        "t_max_x+1": t_max_next_polymarket_f,
        "ciudad": city,
        "doy_sin": target_row['doy_sin'],
        "doy_cos": target_row['doy_cos']
    }

    return features
