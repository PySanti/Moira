"""
Recorre un rango histórico de fechas y genera registros tabulares para entrenamiento.
Guarda cada fila con sus features del día X y el target
"""

import os
import time
import random
from datetime import date, datetime, timedelta
from collections import Counter

import pandas as pd

# ------------------------------------------------------------
# IMPORTA TU FUNCIÓN AQUÍ
from build_climate_data import get_weather_features as get_feature_values
# ------------------------------------------------------------

# ---------------- CONFIG ----------------
CITY = "new york"
DEFAULT_START_DATE = date(1980, 1, 1)
END_DATE = date(2026, 2, 19)  # inclusive (19/02/2026)

OUTPUT_CSV = "dataset.csv"
SAVE_EVERY_N_RECORDS = 500  # cada 500 *nuevos* registros OK

# Control de carga
SLEEP_BETWEEN_CALLS_SEC = 0.15

# Reintentos
MAX_RETRIES = 2
BACKOFF_BASE_SEC = 1.0
BACKOFF_JITTER_SEC = 0.5

STRICT_MODE = True

# ---------------- HELPERS ----------------
def date_to_str(d: date) -> str:
    return d.strftime("%d-%m-%y")

def classify_error(e: Exception) -> str:
    msg = str(e)
    if " 429" in msg or "429" in msg or "Too Many Requests" in msg:
        return "RATE_LIMIT_429"

    timeout_markers = [
        "Read timed out",
        "ConnectTimeout",
        "ConnectionError",
        "Max retries exceeded",
        "timed out",
        "Temporary failure in name resolution",
        "NameResolutionError",
        "Remote end closed connection",
    ]
    if any(m in msg for m in timeout_markers):
        return "NETWORK_TIMEOUT"

    if "Open-Meteo" in msg:
        return "OPEN_METEO_ERROR"
    if "NCEI" in msg or "NOAA" in msg:
        return "NCEI_ERROR"

    return "OTHER_ERROR"

def call_with_retry(func, *args, **kwargs):
    attempts = 0
    slept = 0.0
    last_err_type = None
    last_err_msg = None

    for retry in range(MAX_RETRIES + 1):
        attempts += 1
        try:
            features = func(*args, **kwargs)
            return features, attempts, slept, None, None
        except Exception as e:
            et = classify_error(e)
            last_err_type = et
            last_err_msg = str(e)[:300]

            retryable = et in {"RATE_LIMIT_429", "NETWORK_TIMEOUT", "OPEN_METEO_ERROR", "NCEI_ERROR"}
            if retryable and retry < MAX_RETRIES:
                backoff = BACKOFF_BASE_SEC * (2 ** retry) + random.random() * BACKOFF_JITTER_SEC
                time.sleep(backoff)
                slept += backoff
                continue

            return None, attempts, slept, last_err_type, last_err_msg

    return None, attempts, slept, last_err_type, last_err_msg

def load_existing_dataset(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()

    df = pd.read_csv(path)
    if df.empty:
        return df

    if "date" not in df.columns:
        raise ValueError(f"El archivo {path} existe pero no tiene columna 'date'.")

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df = df[df["date"].notna()].copy()

    if df.empty:
        return pd.DataFrame()

    return df

def get_resume_start_date(df_existing: pd.DataFrame) -> date:
    if df_existing is None or df_existing.empty:
        return DEFAULT_START_DATE
    last_date = max(df_existing["date"])
    if not isinstance(last_date, date):
        last_date = pd.to_datetime(last_date).date()
    return last_date + timedelta(days=1)

def print_progress_line(processed_days: int, total_days: int, current_date: date, ok_new: int, empty: int, errors: int):
    remaining = total_days - processed_days
    line = (
        f"\r[PROGRESS] processed={processed_days}/{total_days} | remaining={remaining} | "
        f"date={current_date.isoformat()} | new_ok={ok_new} | empty={empty} | errors={errors}"
    )
    print(line, end="", flush=True)

# ---------------- MAIN ----------------
def main():
    df_existing = load_existing_dataset(OUTPUT_CSV)
    resume_start = get_resume_start_date(df_existing)

    if resume_start > END_DATE:
        last_date = max(df_existing["date"]) if not df_existing.empty else None
        print(f"dataset.csv ya está completo. Última fecha: {last_date} (>= {END_DATE}).")
        return

    existing_count = 0 if df_existing.empty else len(df_existing)
    print(f"Existing rows in {OUTPUT_CSV}: {existing_count}")
    print(f"Resuming from: {resume_start} to {END_DATE} (inclusive)")
    print(f"STRICT_MODE={STRICT_MODE}")
    print(f"Saving every {SAVE_EVERY_N_RECORDS} NEW ok records -> {OUTPUT_CSV}\n")

    total_days_to_process = (END_DATE - resume_start).days + 1

    processed_days = 0
    ok_new_records = 0
    empty_returns = 0
    error_counts = Counter()
    retry_attempts = []
    backoff_slept_total = 0.0

    new_rows_buffer = []
    t_start = time.perf_counter()

    d = resume_start
    while d <= END_DATE:
        processed_days += 1
        ds = date_to_str(d)

        # Mostrar progreso "en vivo" antes de la llamada
        print_progress_line(
            processed_days=processed_days,
            total_days=total_days_to_process,
            current_date=d,
            ok_new=ok_new_records,
            empty=empty_returns,
            errors=sum(error_counts.values()),
        )

        features, attempts, slept, err_type, err_msg = call_with_retry(
            get_feature_values, CITY, ds, strict=STRICT_MODE
        )

        retry_attempts.append(attempts)
        backoff_slept_total += slept

        if err_type is not None:
            error_counts[err_type] += 1
            # rompo la línea de progreso para loguear claro
            print()
            print(f"[ERR] {d.isoformat()} ({ds}) -> {err_type}: {err_msg}")
        else:
            if not features or (isinstance(features, dict) and len(features) == 0):
                empty_returns += 1
                print()
                print(f"[EMPTY] {d.isoformat()} ({ds}) -> returned {{}}")
            else:
                ok_new_records += 1
                row = {"date": d.isoformat(), "date_str": ds, **features}
                new_rows_buffer.append(row)

                if ok_new_records % SAVE_EVERY_N_RECORDS == 0:
                    df_new = pd.DataFrame(new_rows_buffer)
                    if not df_existing.empty:
                        df_out = pd.concat([df_existing, df_new], ignore_index=True)
                    else:
                        df_out = df_new

                    df_out.to_csv(OUTPUT_CSV, index=False)
                    df_existing = df_out
                    new_rows_buffer.clear()

                    elapsed = time.perf_counter() - t_start
                    print()
                    print(
                        f"[SAVE] new_ok={ok_new_records} total_rows={len(df_existing)} "
                        f"processed_days={processed_days}/{total_days_to_process} "
                        f"empty={empty_returns} errors={sum(error_counts.values())} "
                        f"elapsed={elapsed:.1f}s -> {OUTPUT_CSV}"
                    )

        time.sleep(SLEEP_BETWEEN_CALLS_SEC)
        d += timedelta(days=1)

    # Guardado final
    if new_rows_buffer:
        df_new = pd.DataFrame(new_rows_buffer)
        if not df_existing.empty:
            df_out = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_out = df_new
        df_out.to_csv(OUTPUT_CSV, index=False)
        df_existing = df_out
        new_rows_buffer.clear()

    elapsed = time.perf_counter() - t_start
    print("\n\n=== DONE ===")
    print(f"processed_days: {processed_days}")
    print(f"ok_new_records: {ok_new_records}")
    print(f"total_rows_in_dataset: {len(df_existing)}")
    print(f"empty_returns: {empty_returns}")
    print(f"errors_total: {sum(error_counts.values())}")
    print(f"backoff_slept_total_sec: {backoff_slept_total:.2f}")
    if retry_attempts:
        print(f"attempts_mean: {sum(retry_attempts)/len(retry_attempts):.2f}")
    print(f"elapsed_sec: {elapsed:.2f}")
    print(f"output: {OUTPUT_CSV}")

    if error_counts:
        print("\n=== ERROR COUNTS ===")
        for k, v in error_counts.most_common():
            print(f"{k}: {v}")

if __name__ == "__main__":
    main()
