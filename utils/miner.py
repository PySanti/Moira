#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
miner_v2.py

Minero robusto para construir dataset histórico con utils.build_climate_data.

Mejoras frente a miner.py:
- Usa import dinámico consistente: el preload y get_weather_features salen del MISMO módulo.
- Por defecto mina desde 1983-01-01, dejando 1980-01-01 como inicio histórico.
- Precarga cache con preload_weather_cache(...) si el módulo lo soporta.
- Reanuda sin depender solo del último día: procesa fechas que no existan en el dataset.
- Guarda errores/empties en CSV para auditoría.
- Guarda metadata reproducible del dataset.
- Valida estabilidad del esquema de features.
- Guarda de forma atómica para reducir riesgo de CSV corrupto.

Uso rápido:
  python miner_v2.py

Uso recomendado:
  python miner_v2.py \
    --module utils.build_climate_data \
    --city "new york" \
    --start 1983-01-01 \
    --end 2025-12-31 \
    --history-start 1980-01-01 \
    --output ./dataset/original_dataset.csv \
    --failed-output ./dataset/original_dataset_failed_rows.csv \
    --metadata-output ./dataset/original_dataset_metadata.json \
    --strict true \
    --preload true \
    --save-every 500

Si quieres probar pocas fechas:
  python miner_v2.py --start 1983-01-01 --end 1983-01-31 --output dataset_test.csv
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import os
import random
import shutil
import sys
import time
from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable

import pandas as pd


# ----------------------------
# CLI helpers
# ----------------------------

def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value

    v = str(value).strip().lower()

    if v in {"true", "1", "yes", "y", "si", "sí"}:
        return True

    if v in {"false", "0", "no", "n"}:
        return False

    raise argparse.ArgumentTypeError(f"Valor booleano inválido: {value}")


def parse_iso_date(value: str) -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Fecha inválida '{value}'. Usa formato YYYY-MM-DD."
        ) from exc


def date_to_function_str(d: date, date_format: str) -> str:
    if date_format.strip().lower() == "iso":
        return d.isoformat()
    return d.strftime(date_format)


def date_range_inclusive(start: date, end: date) -> list[date]:
    if start > end:
        raise ValueError("start no puede ser mayor que end.")

    days = []
    d = start

    while d <= end:
        days.append(d)
        d += timedelta(days=1)

    return days


# ----------------------------
# Import dinámico del módulo climático
# ----------------------------

def import_weather_module(module_name: str):
    try:
        return importlib.import_module(module_name)
    except Exception as exc:
        raise RuntimeError(
            f"No pude importar el módulo '{module_name}'. "
            "Ejecuta el script desde la raíz del proyecto o ajusta PYTHONPATH."
        ) from exc


def get_callable(module, function_name: str) -> Callable:
    if not hasattr(module, function_name):
        raise RuntimeError(
            f"El módulo '{module.__name__}' no contiene '{function_name}'."
        )

    fn = getattr(module, function_name)

    if not callable(fn):
        raise RuntimeError(
            f"'{module.__name__}.{function_name}' existe pero no es callable."
        )

    return fn


def accepts_kwarg(fn: Callable, kwarg_name: str) -> bool:
    try:
        sig = inspect.signature(fn)
    except Exception:
        return False

    for p in sig.parameters.values():
        if p.kind == inspect.Parameter.VAR_KEYWORD:
            return True

    return kwarg_name in sig.parameters


def filter_kwargs_for_function(fn: Callable, kwargs: dict[str, Any]) -> dict[str, Any]:
    return {
        k: v
        for k, v in kwargs.items()
        if accepts_kwarg(fn, k)
    }


# ----------------------------
# Dataset / archivos
# ----------------------------

def atomic_write_dataframe_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp_path, index=False)
    tmp_path.replace(path)


def atomic_write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    tmp_path.replace(path)


def maybe_backup_file(path: Path, backup_dir: Path | None) -> Path | None:
    if backup_dir is None:
        return None

    if not path.exists():
        return None

    backup_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"{path.stem}_backup_{ts}{path.suffix}"
    shutil.copy2(path, backup_path)

    return backup_path


def load_existing_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)

    if df.empty:
        return df

    if "date" not in df.columns:
        raise ValueError(f"El archivo {path} existe pero no tiene columna 'date'.")

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df = df[df["date"].notna()].copy()

    return df


def get_existing_dates(df: pd.DataFrame) -> set[date]:
    if df is None or df.empty or "date" not in df.columns:
        return set()

    return set(pd.to_datetime(df["date"], errors="coerce").dropna().dt.date)


def append_failed_rows(failed_rows: list[dict[str, Any]], failed_output: Path) -> None:
    if not failed_rows:
        return

    failed_output.parent.mkdir(parents=True, exist_ok=True)

    df_new = pd.DataFrame(failed_rows)

    if failed_output.exists():
        df_old = pd.read_csv(failed_output)
        df_out = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_out = df_new

    atomic_write_dataframe_csv(df_out, failed_output)


# ----------------------------
# Esquema
# ----------------------------

def schema_signature(keys: list[str]) -> str:
    raw = "\n".join(sorted(map(str, keys)))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def infer_existing_feature_schema(df_existing: pd.DataFrame) -> set[str] | None:
    if df_existing is None or df_existing.empty:
        return None

    reserved = {"date", "date_str"}
    return set(c for c in df_existing.columns if c not in reserved)


def validate_feature_schema(
    features: dict[str, Any],
    expected_schema: set[str] | None,
    allow_schema_variance: bool,
) -> tuple[bool, str | None]:
    current_schema = set(map(str, features.keys()))

    if expected_schema is None:
        return True, None

    if current_schema == expected_schema:
        return True, None

    missing = sorted(expected_schema - current_schema)
    extra = sorted(current_schema - expected_schema)

    msg = (
        "SCHEMA_MISMATCH | "
        f"missing={missing[:20]} extra={extra[:20]}"
    )

    if allow_schema_variance:
        return True, msg

    return False, msg


# ----------------------------
# Errores / retries
# ----------------------------

def classify_error(exc: Exception) -> str:
    msg = str(exc)

    if "429" in msg or "Too Many Requests" in msg:
        return "RATE_LIMIT_429"

    network_markers = [
        "Read timed out",
        "ConnectTimeout",
        "ConnectionError",
        "Max retries exceeded",
        "timed out",
        "Temporary failure in name resolution",
        "NameResolutionError",
        "Remote end closed connection",
        "Connection reset",
    ]

    if any(m in msg for m in network_markers):
        return "NETWORK_ERROR"

    service_markers = [
        "Open-Meteo",
        "NCEI",
        "NOAA",
        "ISD-Lite",
        "Internal Server Error",
        "Bad Gateway",
        "Service Unavailable",
        "Gateway Timeout",
    ]

    if any(m in msg for m in service_markers):
        return "SERVICE_ERROR"

    if "No se encontraron datos" in msg or "no data" in msg.lower():
        return "NO_DATA_FOR_DATE"

    return "OTHER_ERROR"


def call_with_retry(
    fn: Callable,
    city: str,
    date_str: str,
    kwargs: dict[str, Any],
    max_retries: int,
    backoff_base_sec: float,
    backoff_jitter_sec: float,
) -> tuple[Any, int, float, str | None, str | None]:
    attempts = 0
    slept = 0.0
    last_error_type = None
    last_error_msg = None

    for retry in range(max_retries + 1):
        attempts += 1

        try:
            return fn(city, date_str, **kwargs), attempts, slept, None, None

        except Exception as exc:
            error_type = classify_error(exc)
            last_error_type = error_type
            last_error_msg = str(exc)[:800]

            retryable = error_type in {
                "RATE_LIMIT_429",
                "NETWORK_ERROR",
                "SERVICE_ERROR",
            }

            if retryable and retry < max_retries:
                backoff = backoff_base_sec * (2 ** retry)
                backoff += random.random() * backoff_jitter_sec
                time.sleep(backoff)
                slept += backoff
                continue

            return None, attempts, slept, last_error_type, last_error_msg

    return None, attempts, slept, last_error_type, last_error_msg


# ----------------------------
# Preload
# ----------------------------

def run_preload_if_available(
    module,
    city: str,
    history_start: date,
    end: date,
    execution_hour: int,
    nearest_tolerance_hours: int,
    include_target: bool,
    preload: bool,
) -> dict[str, Any] | None:
    if not preload:
        return None

    if not hasattr(module, "preload_weather_cache"):
        print("[WARN] El módulo no tiene preload_weather_cache(...). Se omite preload.")
        return None

    preload_fn = getattr(module, "preload_weather_cache")

    if not callable(preload_fn):
        print("[WARN] preload_weather_cache existe pero no es callable. Se omite preload.")
        return None

    preload_end = end + timedelta(days=1) if include_target else end

    kwargs = {
        "city": city,
        "start_date": history_start.isoformat(),
        "end_date": preload_end.isoformat(),
        "execution_hour": execution_hour,
        "nearest_tolerance_hours": nearest_tolerance_hours,
    }

    kwargs = filter_kwargs_for_function(preload_fn, kwargs)

    print("\n=== PRELOAD CACHE ===")
    print(f"module: {module.__name__}")
    print(f"start_date: {history_start.isoformat()}")
    print(f"end_date: {preload_end.isoformat()}")
    print(f"kwargs: {kwargs}")

    t0 = time.perf_counter()
    result = preload_fn(**kwargs)
    elapsed = time.perf_counter() - t0

    print(f"[PRELOAD DONE] elapsed={elapsed:.2f}s result={result}")

    return {
        "elapsed_sec": elapsed,
        "result": result,
    }


# ----------------------------
# Progreso
# ----------------------------

def print_progress(
    processed: int,
    total: int,
    current_date: date,
    ok: int,
    empty: int,
    errors: int,
    skipped: int,
    elapsed: float,
) -> None:
    remaining = total - processed
    rate = processed / elapsed if elapsed > 0 else 0.0
    eta_sec = remaining / rate if rate > 0 else None

    eta_text = f"{eta_sec/60:.1f}m" if eta_sec is not None else "?"

    line = (
        f"\r[PROGRESS] {processed}/{total} | remaining={remaining} | "
        f"date={current_date.isoformat()} | ok={ok} | empty={empty} | "
        f"errors={errors} | skipped={skipped} | rate={rate:.2f} d/s | ETA={eta_text}"
    )

    print(line, end="", flush=True)


# ----------------------------
# Main
# ----------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Minero robusto para build_climate_data.get_weather_features."
    )

    ap.add_argument("--module", default="utils.build_climate_data")
    ap.add_argument("--function", default="get_weather_features")

    ap.add_argument("--city", default="new york")
    ap.add_argument("--start", type=parse_iso_date, default=parse_iso_date("1983-01-01"))
    ap.add_argument("--end", type=parse_iso_date, default=parse_iso_date("2025-12-31"))
    ap.add_argument("--history-start", type=parse_iso_date, default=parse_iso_date("1980-01-01"))

    ap.add_argument("--date-format", default="%d-%m-%y")

    ap.add_argument("--output", default="./dataset/original_dataset.csv")
    ap.add_argument("--failed-output", default="./dataset/original_dataset_failed_rows.csv")
    ap.add_argument("--metadata-output", default="./dataset/original_dataset_metadata.json")
    ap.add_argument("--backup-dir", default="./dataset/backups")

    ap.add_argument("--strict", type=str2bool, default=True)
    ap.add_argument("--preload", type=str2bool, default=True)
    ap.add_argument("--skip-existing", type=str2bool, default=True)

    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--sleep-between-calls-sec", type=float, default=0.0)

    ap.add_argument("--max-retries", type=int, default=2)
    ap.add_argument("--backoff-base-sec", type=float, default=1.0)
    ap.add_argument("--backoff-jitter-sec", type=float, default=0.5)

    # Parámetros que se pasan a get_weather_features si la función los acepta.
    ap.add_argument("--execution-hour", type=int, default=23)
    ap.add_argument("--nearest-tolerance-hours", type=int, default=6)
    ap.add_argument("--min-climatology-records", type=int, default=30)
    ap.add_argument("--climatology-window-days", type=int, default=7)
    ap.add_argument("--compute-td-anomaly", type=str2bool, default=True)
    ap.add_argument("--include-target", type=str2bool, default=True)

    ap.add_argument("--allow-schema-variance", type=str2bool, default=False)

    # Auditoría/debug.
    ap.add_argument("--dry-run", type=str2bool, default=False)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--checkpoint-every", type=int, default=100)

    args = ap.parse_args()

    output_path = Path(args.output)
    failed_output_path = Path(args.failed_output)
    metadata_output_path = Path(args.metadata_output)
    backup_dir = Path(args.backup_dir) if args.backup_dir else None

    if args.history_start > args.start:
        raise SystemExit(
            "[ERROR] --history-start no puede ser mayor que --start. "
            "Ej: history-start=1980-01-01, start=1983-01-01."
        )

    module = import_weather_module(args.module)
    get_features_fn = get_callable(module, args.function)

    # Backup opcional antes de tocar archivos existentes.
    backup_dataset = maybe_backup_file(output_path, backup_dir)
    backup_failed = maybe_backup_file(failed_output_path, backup_dir)

    if backup_dataset:
        print(f"[BACKUP] dataset -> {backup_dataset}")

    if backup_failed:
        print(f"[BACKUP] failed rows -> {backup_failed}")

    existing_df = load_existing_dataset(output_path)
    existing_dates = get_existing_dates(existing_df)

    expected_schema = infer_existing_feature_schema(existing_df)

    all_dates = date_range_inclusive(args.start, args.end)

    if args.skip_existing:
        dates_to_process = [d for d in all_dates if d not in existing_dates]
    else:
        dates_to_process = all_dates

    if args.limit is not None:
        dates_to_process = dates_to_process[:args.limit]

    preload_info = run_preload_if_available(
        module=module,
        city=args.city,
        history_start=args.history_start,
        end=args.end,
        execution_hour=args.execution_hour,
        nearest_tolerance_hours=args.nearest_tolerance_hours,
        include_target=args.include_target,
        preload=args.preload,
    )

    get_features_kwargs_raw = {
        "strict": args.strict,
        "execution_hour": args.execution_hour,
        "nearest_tolerance_hours": args.nearest_tolerance_hours,
        "history_start_date": args.history_start.isoformat(),
        "climatology_window_days": args.climatology_window_days,
        "min_climatology_records": args.min_climatology_records,
        "compute_td_anomaly": args.compute_td_anomaly,
        "include_target": args.include_target,
    }

    get_features_kwargs = filter_kwargs_for_function(
        get_features_fn,
        get_features_kwargs_raw,
    )

    print("\n=== MINER CONFIG ===")
    print(f"module: {args.module}")
    print(f"function: {args.function}")
    print(f"city: {args.city}")
    print(f"range: {args.start.isoformat()} -> {args.end.isoformat()}")
    print(f"history_start: {args.history_start.isoformat()}")
    print(f"strict: {args.strict}")
    print(f"preload: {args.preload}")
    print(f"skip_existing: {args.skip_existing}")
    print(f"output: {output_path}")
    print(f"failed_output: {failed_output_path}")
    print(f"metadata_output: {metadata_output_path}")
    print(f"existing_rows: {len(existing_df)}")
    print(f"existing_dates: {len(existing_dates)}")
    print(f"total_dates_in_range: {len(all_dates)}")
    print(f"dates_to_process: {len(dates_to_process)}")
    print(f"get_weather_features kwargs: {get_features_kwargs}")
    print(f"dry_run: {args.dry_run}\n")

    if not dates_to_process:
        print("[DONE] No hay fechas nuevas por procesar.")
        return 0

    if args.dry_run:
        print("[DRY RUN] Fechas que se procesarían:")
        for d in dates_to_process[:20]:
            print(f"- {d.isoformat()} ({date_to_function_str(d, args.date_format)})")
        if len(dates_to_process) > 20:
            print(f"... y {len(dates_to_process) - 20} más")
        return 0

    # Estado.
    t_start = time.perf_counter()

    processed = 0
    ok_new = 0
    empty_returns = 0
    schema_mismatch_count = 0
    skipped_existing = len(all_dates) - len(dates_to_process) if args.skip_existing else 0

    error_counts = Counter()
    schema_counts = Counter()
    attempts_list = []
    backoff_slept_total = 0.0

    rows_buffer: list[dict[str, Any]] = []
    failed_buffer: list[dict[str, Any]] = []

    df_current = existing_df.copy()

    # Si no había dataset previo, el primer row exitoso fija el esquema esperado.
    if expected_schema is None:
        expected_schema = None

    for d in dates_to_process:
        processed += 1
        ds = date_to_function_str(d, args.date_format)

        elapsed = time.perf_counter() - t_start
        print_progress(
            processed=processed,
            total=len(dates_to_process),
            current_date=d,
            ok=ok_new,
            empty=empty_returns,
            errors=sum(error_counts.values()),
            skipped=skipped_existing,
            elapsed=elapsed,
        )

        features, attempts, slept, err_type, err_msg = call_with_retry(
            fn=get_features_fn,
            city=args.city,
            date_str=ds,
            kwargs=get_features_kwargs,
            max_retries=args.max_retries,
            backoff_base_sec=args.backoff_base_sec,
            backoff_jitter_sec=args.backoff_jitter_sec,
        )

        attempts_list.append(attempts)
        backoff_slept_total += slept

        if err_type is not None:
            error_counts[err_type] += 1

            failed_buffer.append({
                "date": d.isoformat(),
                "date_str": ds,
                "status": "ERROR",
                "error_type": err_type,
                "error_msg": err_msg,
                "attempts": attempts,
                "slept_backoff_sec": slept,
                "created_at": datetime.now().isoformat(timespec="seconds"),
            })

            print()
            print(f"[ERR] {d.isoformat()} ({ds}) -> {err_type}: {err_msg}")

        else:
            if not isinstance(features, dict):
                error_counts["INVALID_RETURN_TYPE"] += 1

                failed_buffer.append({
                    "date": d.isoformat(),
                    "date_str": ds,
                    "status": "ERROR",
                    "error_type": "INVALID_RETURN_TYPE",
                    "error_msg": f"Expected dict, got {type(features).__name__}",
                    "attempts": attempts,
                    "slept_backoff_sec": slept,
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                })

                print()
                print(f"[ERR] {d.isoformat()} ({ds}) -> INVALID_RETURN_TYPE")

            elif len(features) == 0:
                empty_returns += 1

                failed_buffer.append({
                    "date": d.isoformat(),
                    "date_str": ds,
                    "status": "EMPTY",
                    "error_type": "EMPTY_RETURN",
                    "error_msg": "get_weather_features returned {}",
                    "attempts": attempts,
                    "slept_backoff_sec": slept,
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                })

                print()
                print(f"[EMPTY] {d.isoformat()} ({ds}) -> returned {{}}")

            else:
                normalized_features = {str(k): v for k, v in features.items()}

                if expected_schema is None:
                    expected_schema = set(normalized_features.keys())

                schema_ok, schema_msg = validate_feature_schema(
                    features=normalized_features,
                    expected_schema=expected_schema,
                    allow_schema_variance=args.allow_schema_variance,
                )

                schema_hash = schema_signature(list(normalized_features.keys()))
                schema_counts[schema_hash] += 1

                if not schema_ok:
                    schema_mismatch_count += 1
                    error_counts["SCHEMA_MISMATCH"] += 1

                    failed_buffer.append({
                        "date": d.isoformat(),
                        "date_str": ds,
                        "status": "ERROR",
                        "error_type": "SCHEMA_MISMATCH",
                        "error_msg": schema_msg,
                        "attempts": attempts,
                        "slept_backoff_sec": slept,
                        "schema_hash": schema_hash,
                        "created_at": datetime.now().isoformat(timespec="seconds"),
                    })

                    print()
                    print(f"[SCHEMA_MISMATCH] {d.isoformat()} ({ds}) -> {schema_msg}")

                else:
                    if schema_msg:
                        # Se permite pero queda auditado.
                        schema_mismatch_count += 1
                        failed_buffer.append({
                            "date": d.isoformat(),
                            "date_str": ds,
                            "status": "SCHEMA_WARNING",
                            "error_type": "SCHEMA_MISMATCH_ALLOWED",
                            "error_msg": schema_msg,
                            "attempts": attempts,
                            "slept_backoff_sec": slept,
                            "schema_hash": schema_hash,
                            "created_at": datetime.now().isoformat(timespec="seconds"),
                        })

                    ok_new += 1

                    row = {
                        "date": d.isoformat(),
                        "date_str": ds,
                        **normalized_features,
                    }

                    rows_buffer.append(row)

        # Guardado periódico.
        should_save = (
            (ok_new > 0 and ok_new % args.save_every == 0)
            or (len(failed_buffer) >= args.save_every)
        )

        if should_save:
            if rows_buffer:
                df_new = pd.DataFrame(rows_buffer)
                df_current = pd.concat([df_current, df_new], ignore_index=True)
                atomic_write_dataframe_csv(df_current, output_path)
                rows_buffer.clear()

            if failed_buffer:
                append_failed_rows(failed_buffer, failed_output_path)
                failed_buffer.clear()

            elapsed = time.perf_counter() - t_start
            print()
            print(
                f"[SAVE] processed={processed}/{len(dates_to_process)} "
                f"ok_new={ok_new} total_rows={len(df_current)} "
                f"empty={empty_returns} errors={sum(error_counts.values())} "
                f"elapsed={elapsed:.1f}s"
            )

        if args.sleep_between_calls_sec > 0:
            time.sleep(args.sleep_between_calls_sec)

    # Guardado final.
    if rows_buffer:
        df_new = pd.DataFrame(rows_buffer)
        df_current = pd.concat([df_current, df_new], ignore_index=True)
        atomic_write_dataframe_csv(df_current, output_path)
        rows_buffer.clear()

    if failed_buffer:
        append_failed_rows(failed_buffer, failed_output_path)
        failed_buffer.clear()

    elapsed = time.perf_counter() - t_start

    summary = {
        "module": args.module,
        "function": args.function,
        "city": args.city,
        "start": args.start.isoformat(),
        "end": args.end.isoformat(),
        "history_start": args.history_start.isoformat(),
        "date_format": args.date_format,
        "strict": args.strict,
        "preload": args.preload,
        "preload_info": preload_info,
        "skip_existing": args.skip_existing,
        "output": str(output_path),
        "failed_output": str(failed_output_path),
        "existing_rows_before": int(len(existing_df)),
        "total_rows_after": int(len(df_current)),
        "total_dates_in_range": int(len(all_dates)),
        "dates_to_process": int(len(dates_to_process)),
        "processed": int(processed),
        "ok_new_records": int(ok_new),
        "empty_returns": int(empty_returns),
        "errors_total": int(sum(error_counts.values())),
        "schema_mismatch_count": int(schema_mismatch_count),
        "skipped_existing": int(skipped_existing),
        "error_counts": dict(error_counts),
        "schema_counts": dict(schema_counts),
        "attempts_mean": (
            sum(attempts_list) / len(attempts_list)
            if attempts_list
            else None
        ),
        "backoff_slept_total_sec": float(backoff_slept_total),
        "elapsed_sec": float(elapsed),
        "rows_per_second": (
            processed / elapsed
            if elapsed > 0
            else None
        ),
        "get_weather_features_kwargs": get_features_kwargs,
        "allow_schema_variance": args.allow_schema_variance,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }

    atomic_write_json(summary, metadata_output_path)

    print("\n\n=== DONE ===")
    print(f"processed: {processed}")
    print(f"ok_new_records: {ok_new}")
    print(f"total_rows_after: {len(df_current)}")
    print(f"empty_returns: {empty_returns}")
    print(f"errors_total: {sum(error_counts.values())}")
    print(f"schema_mismatch_count: {schema_mismatch_count}")
    print(f"skipped_existing: {skipped_existing}")
    print(f"attempts_mean: {summary['attempts_mean']}")
    print(f"backoff_slept_total_sec: {backoff_slept_total:.2f}")
    print(f"elapsed_sec: {elapsed:.2f}")
    print(f"rows_per_second: {summary['rows_per_second']}")
    print(f"output: {output_path}")
    print(f"failed_output: {failed_output_path}")
    print(f"metadata_output: {metadata_output_path}")

    if error_counts:
        print("\n=== ERROR COUNTS ===")
        for k, v in error_counts.most_common():
            print(f"{k}: {v}")

    if schema_counts:
        print("\n=== SCHEMA COUNTS ===")
        for k, v in schema_counts.most_common():
            print(f"{k}: {v}")

    # Código de salida:
    # - 0 si no hubo errores críticos.
    # - 1 si hubo errores/empties/schema mismatch para que puedas detectarlo en scripts.
    if sum(error_counts.values()) > 0 or empty_returns > 0:
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
