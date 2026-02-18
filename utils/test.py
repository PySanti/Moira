import time
import random
import statistics
from datetime import date, timedelta
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

# ------------------------------------------------------------
# IMPORTA TU FUNCIÓN AQUÍ
# Ejemplo:
from build_climate_data import get_weather_features as get_feature_values
#
# Si se llama get_weather_features:
# from tu_modulo import get_weather_features as get_feature_values
# ------------------------------------------------------------

# CONFIG
CITY = "new york"
START = date(1980, 1, 1)
END = date(1982, 1, 1)

# Control de carga
SLEEP_BETWEEN_CALLS_SEC = 0.15   # reduce probabilidad de rate-limit
MAX_RETRIES_429 = 5             # reintentos cuando se detecta 429
BACKOFF_BASE_SEC = 1.0          # base de backoff
BACKOFF_JITTER_SEC = 0.5        # jitter para evitar sincronía

# Si tu get_feature_values acepta strict:
# - strict=True: si faltan dependencias, devuelve {} (menos "nulls", más descartes)
# - strict=False: devuelve features con None/NaN y un _integrity (mejor para auditar nulls)
STRICT_MODE = False

EXPECTED_FEATURE_KEYS = [
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
    "t_max_x+1",
    "ciudad",
    "doy_sin",
    "doy_cos",
]

def is_null(v) -> bool:
    if v is None:
        return True
    try:
        # np.nan, pandas NA, etc.
        return bool(pd.isna(v))
    except Exception:
        return False

def classify_error(e: Exception) -> str:
    msg = str(e)
    # tu función suele lanzar ConnectionError con texto que incluye status_code
    if " 429" in msg or "429" in msg or "Too Many Requests" in msg:
        return "RATE_LIMIT_429"
    if "Open-Meteo" in msg:
        return "OPEN_METEO_ERROR"
    if "NCEI" in msg or "NOAA" in msg:
        return "NCEI_ERROR"
    if "No se encontraron datos" in msg:
        return "NO_DATA_FOR_DATE"
    return "OTHER_ERROR"

def call_with_retry(func, *args, **kwargs):
    """
    Ejecuta func(*args, **kwargs) con reintentos cuando detecta 429.
    Retorna: (features, attempts, total_sleep_sec, last_error_type, last_error_msg)
    """
    attempts = 0
    slept = 0.0
    last_err_type = None
    last_err_msg = None

    for retry in range(MAX_RETRIES_429 + 1):
        attempts += 1
        try:
            return func(*args, **kwargs), attempts, slept, None, None
        except Exception as e:
            et = classify_error(e)
            last_err_type = et
            last_err_msg = str(e)[:300]

            if et == "RATE_LIMIT_429" and retry < MAX_RETRIES_429:
                backoff = BACKOFF_BASE_SEC * (2 ** retry) + random.random() * BACKOFF_JITTER_SEC
                time.sleep(backoff)
                slept += backoff
                continue

            # no reintentamos otros errores por defecto
            return None, attempts, slept, et, last_err_msg

    return None, attempts, slept, last_err_type, last_err_msg

def date_to_str(d: date) -> str:
    # tu función parsea "%d-%m-%y" (2 dígitos), 1980 -> "01-01-80"
    return d.strftime("%d-%m-%y")

def main():
    total_days = (END - START).days + 1
    print(f"Testing {CITY} from {START} to {END} -> {total_days} días")

    # Métricas
    durations = []
    durations_success = []
    attempts_list = []
    backoff_slept_list = []
    error_counts = Counter()
    rate_limit_hits = 0

    null_counts_by_key = Counter()
    missing_key_counts = Counter()
    null_total_per_day = []
    empty_returns = 0
    ok_returns = 0

    first_ok = None
    last_ok = None

    rows = []

    d = START
    i = 0
    while d <= END:
        print(d)
        i += 1
        ds = date_to_str(d)

        t0 = time.perf_counter()
        # NOTA: si tu función no acepta strict, elimina strict=...
        features, attempts, slept, err_type, err_msg = call_with_retry(
            get_feature_values, CITY, ds, strict=STRICT_MODE
        )
        t1 = time.perf_counter()
        dur = t1 - t0

        durations.append(dur)
        attempts_list.append(attempts)
        backoff_slept_list.append(slept)

        ok = False
        nulls_today = 0

        if err_type is not None:
            error_counts[err_type] += 1
            if err_type == "RATE_LIMIT_429":
                rate_limit_hits += 1
        else:
            # sin excepción
            if not features or (isinstance(features, dict) and len(features) == 0):
                empty_returns += 1
            else:
                ok = True
                ok_returns += 1
                durations_success.append(dur)
                if first_ok is None:
                    first_ok = d
                last_ok = d

                # Conteo de nulls / missing keys
                for k in EXPECTED_FEATURE_KEYS:
                    if k not in features:
                        missing_key_counts[k] += 1
                        nulls_today += 1
                        continue
                    v = features.get(k)
                    if is_null(v):
                        null_counts_by_key[k] += 1
                        nulls_today += 1

        null_total_per_day.append(nulls_today)

        rows.append({
            "date": d.isoformat(),
            "date_str": ds,
            "ok": ok,
            "duration_sec": dur,
            "attempts": attempts,
            "slept_backoff_sec": slept,
            "error_type": err_type,
            "error_msg": err_msg,
            "null_count_expected_keys": nulls_today,
        })

        # throttling entre llamadas
        time.sleep(SLEEP_BETWEEN_CALLS_SEC)
        d += timedelta(days=1)

        if i % 50 == 0:
            print(f"[{i}/{total_days}] ok={ok_returns} empty={empty_returns} 429={rate_limit_hits} last={ds} dur={dur:.2f}s")

    # --- Resumen ---
    def pct(x, p):
        if not x:
            return None
        xs = sorted(x)
        idx = int(round((p/100)*(len(xs)-1)))
        return xs[idx]

    summary = {
        "days_total": total_days,
        "ok_returns": ok_returns,
        "empty_returns": empty_returns,
        "errors_total": sum(error_counts.values()),
        "rate_limit_429_hits": rate_limit_hits,
        "first_ok_date": first_ok.isoformat() if first_ok else None,
        "last_ok_date": last_ok.isoformat() if last_ok else None,
        "duration_mean_sec": statistics.mean(durations) if durations else None,
        "duration_median_sec": statistics.median(durations) if durations else None,
        "duration_p95_sec": pct(durations, 95),
        "duration_success_mean_sec": statistics.mean(durations_success) if durations_success else None,
        "duration_success_p95_sec": pct(durations_success, 95),
        "attempts_mean": statistics.mean(attempts_list) if attempts_list else None,
        "backoff_slept_total_sec": sum(backoff_slept_list),
        "nulls_per_day_mean": statistics.mean(null_total_per_day) if null_total_per_day else None,
        "nulls_per_day_p95": pct(null_total_per_day, 95),
    }

    print("\n=== SUMMARY ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

    print("\n=== ERROR COUNTS ===")
    for k, v in error_counts.most_common():
        print(f"{k}: {v}")

    print("\n=== TOP NULL FEATURES (count) ===")
    for k, v in null_counts_by_key.most_common(15):
        print(f"{k}: {v}")

    print("\n=== TOP MISSING KEYS (count) ===")
    for k, v in missing_key_counts.most_common(15):
        print(f"{k}: {v}")

    # Guardar CSV para análisis
    df_out = pd.DataFrame(rows)
    df_out.to_csv("tmax_feature_test_1980_1981.csv", index=False)

    # Guardar resumen y conteos en archivos
    pd.DataFrame([summary]).to_json("tmax_feature_test_summary.json", orient="records", indent=2)
    pd.DataFrame(error_counts.most_common(), columns=["error_type", "count"]).to_csv("tmax_feature_test_errors.csv", index=False)

    null_df = pd.DataFrame(null_counts_by_key.most_common(), columns=["feature", "null_count"])
    null_df.to_csv("tmax_feature_test_nulls.csv", index=False)

    missing_df = pd.DataFrame(missing_key_counts.most_common(), columns=["feature", "missing_count"])
    missing_df.to_csv("tmax_feature_test_missing_keys.csv", index=False)

    print("\nArchivos generados:")
    print("- tmax_feature_test_1980_1981.csv")
    print("- tmax_feature_test_summary.json")
    print("- tmax_feature_test_errors.csv")
    print("- tmax_feature_test_nulls.csv")
    print("- tmax_feature_test_missing_keys.csv")

if __name__ == "__main__":
    main()
