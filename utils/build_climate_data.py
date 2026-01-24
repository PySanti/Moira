import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import math

def get_weather_features(city: str, date_str: str):
    """
    Obtiene features meteorológicas para una ciudad y fecha específicas.
    
    Args:
        city (str): 'new york', 'chicago', 'atlanta', 'seul', 'londres'
        date_str (str): Fecha en formato 'dd-mm-aa' (ej: '25-01-23')
        
    Returns:
        dict: Diccionario con todas las features calculadas.
    """
    
    # 1. Configuración de Coordenadas (Lat/Lon)
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

    # 2. Gestión de Fechas
    # Convertimos dd-mm-aa a objeto datetime.
    # Nota: %y asume el siglo actual o pasado según el estándar (20xx).
    target_date = datetime.strptime(date_str, "%d-%m-%y")
    
    # IMPORTANTE: Para calcular medias móviles de 3 días (MA_Tmax_3d) y deltas,
    # necesitamos datos de días anteriores. Pediremos 5 días atrás para tener margen.
    start_date = target_date - timedelta(days=5)
    
    # Formato para la API (YYYY-MM-DD)
    api_start = start_date.strftime("%Y-%m-%d")
    api_end = target_date.strftime("%Y-%m-%d")

    # 3. Llamada a la API (Open-Meteo Archive)
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": city_coords[city_key]['lat'],
        "longitude": city_coords[city_key]['lon'],
        "start_date": api_start,
        "end_date": api_end,
        # Solicitamos las variables base necesarias para tus cálculos
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "temperature_2m_mean",
            "precipitation_sum",
            "wind_speed_10m_mean",
            "wind_direction_10m_dominant",
            "surface_pressure_mean", # Presión a nivel de superficie (o sealevel si prefieres)
            "relative_humidity_2m_mean",
            "cloud_cover_mean",
            "dew_point_2m_mean"
        ],
        "timezone": "auto"
    }

    response = requests.get(url, params=params)
    
    if response.status_code != 200:
        raise ConnectionError(f"Error conectando con API: {response.text}")
        
    data = response.json()
    
    # 4. Procesamiento con Pandas
    daily_data = data['daily']
    df = pd.DataFrame(daily_data)
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time') # Aseguramos orden cronológico

    # --- CÁLCULO DE FEATURES ---
    
    # Renombrar columnas para facilitar manejo
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

    # Features derivadas (Calculadas sobre todo el vector de 5 días)
    
    # ΔTmax_1d: Tmax[x] - Tmax[x-1]
    df['Delta_Tmax_1d'] = df['Tmax'].diff()
    
    # MA_Tmax_3d: Media móvil de 3 días (incluyendo hoy)
    df['MA_Tmax_3d'] = df['Tmax'].rolling(window=3).mean()
    
    # DTR: Amplitud térmica
    df['DTR'] = df['Tmax'] - df['Tmin']
    
    # ΔPresión_24h
    df['Delta_Presion'] = df['SLP'].diff()
    
    # Viento Seno/Coseno
    # Convertimos grados a radianes primero
    wind_rad = np.deg2rad(df['WindDir'])
    df['Wind_sin'] = np.sin(wind_rad)
    df['Wind_cos'] = np.cos(wind_rad)

    # Features de Fecha (DOY)
    df['doy'] = df['time'].dt.dayofyear
    df['day'] = df['time'].dt.day
    df['month'] = df['time'].dt.month
    df['year'] = df['time'].dt.year
    
    # DOY ciclico
    df['doy_sin'] = np.sin(2 * np.pi * df['doy'] / 365.25)
    df['doy_cos'] = np.cos(2 * np.pi * df['doy'] / 365.25)

    # 5. Extracción de la fila objetivo
    # Seleccionamos solo la fila correspondiente a la fecha solicitada (la última)
    target_row = df[df['time'].dt.date == target_date.date()].iloc[0]

    # Construimos el diccionario final mapeando a tus nombres exactos
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
        # t_max_x+1: No podemos calcular el futuro real sin hacer trampa o pedir un día más. 
        # Lo dejo como None o podrías pedir un día extra en la API si es para entrenamiento histórico.
        "t_max_x+1 (Label)": None, 
        "día": target_row['day'],
        "mes": target_row['month'],
        "año": target_row['year'],
        "ciudad": city,
        "doy": target_row['doy'],
        "doy_sin": target_row['doy_sin'],
        "doy_cos": target_row['doy_cos']
    }
    
    return features


