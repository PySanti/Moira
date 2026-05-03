#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
test.py

Test dinámico para build_climate_data.get_weather_features.

Objetivo:
- No hardcodea nombres de features.
- No depende de EXPECTED_FEATURE_KEYS.
- Si actualizas build_climate_data agregando, quitando o renombrando features,
  este test sigue funcionando sin cambios.
- Evalúa el contrato general del módulo:
    1. La función se puede importar.
    2. La función retorna dict no vacío para fechas válidas.
    3. El esquema retornado es estable entre fechas exitosas.
    4. Los valores son serializables.
    5. No hay numéricos infinitos.
    6. Reporta nulls, errores, tiempos, variantes de esquema y features tipo target.

Uso recomendado:
  python test.py \
    --module build_climate_data \
    --function get_weather_features \
    --city "new york" \
    --start 1980-01-01 \
    --end 2025-12-31 \
    --n-samples 1000 \
    --strict false \
    --out-dir ./reports/feature_contract_test

Uso rápido:
  python test.py --n-samples 50

Notas:
- Este test valida la forma general y estabilidad del output.
- No puede validar por sí solo que una feature venga de LaGuardia o sea "as-of 23h";
  para eso build_climate_data debería exponer metadata/contratos de fuente.


   python .\test.py --module build_climate_data --function get_weather_features --city "new york" --start 1980-01-01 --end 2025-12-12 --n-samples 50 --strict True
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import math
import random
import statistics
import sys
import time
from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ----------------------------
# Helpers CLI
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
    """
    Formatea la fecha según lo que espere get_weather_features.

    Default actual:
      %d-%m-%y  ->  01-02-25

    También puedes usar:
      --date-format iso
    """
    if date_format.strip().lower() == "iso":
        return d.isoformat()

    return d.strftime(date_format)


# ----------------------------
# Import dinámico
# ----------------------------

def load_function(module_name: str, function_name: str):
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        raise RuntimeError(
            f"No pude importar el módulo '{module_name}'. "
            f"Ejecuta este test desde la carpeta donde exista {module_name}.py "
            "o ajusta PYTHONPATH."
        ) from exc

    if not hasattr(module, function_name):
        raise RuntimeError(
            f"El módulo '{module_name}' no contiene la función '{function_name}'."
        )

    fn = getattr(module, function_name)

    if not callable(fn):
        raise RuntimeError(f"'{module_name}.{function_name}' existe pero no es callable.")

    return fn


def function_accepts_kwarg(fn, kwarg_name: str) -> bool:
    """
    Permite que el test no dependa rígidamente de la firma exacta.
    Si la función acepta **kwargs, también le pasamos el kwarg.
    """
    try:
        sig = inspect.signature(fn)
    except Exception:
        return False

    for p in sig.parameters.values():
        if p.kind == inspect.Parameter.VAR_KEYWORD:
            return True

    return kwarg_name in sig.parameters


# ----------------------------
# Muestreo de fechas
# ----------------------------

def random_dates(
    start: date,
    end: date,
    n: int,
    no_replacement: bool = True,
    seed: int | None = 123,
) -> list[date]:
    if start > end:
        raise ValueError("start no puede ser mayor que end.")

    if seed is not None:
        random.seed(seed)

    total_days = (end - start).days + 1

    if no_replacement and n > total_days:
        raise ValueError(
            f"n_samples={n} es mayor que total_days={total_days} "
            "(no_replacement=True)."
        )

    if no_replacement:
        offsets = random.sample(range(total_days), k=n)
    else:
        offsets = [random.randrange(total_days) for _ in range(n)]

    return sorted(start + timedelta(days=o) for o in offsets)


# ----------------------------
# Valores / esquema
# ----------------------------

def is_null(value: Any) -> bool:
    if value is None:
        return True

    # Evita ambigüedad de pd.isna con listas/dicts.
    if isinstance(value, (list, tuple, dict, set)):
        return False

    try:
        result = pd.isna(value)
        if isinstance(result, (bool, np.bool_)):
            return bool(result)
    except Exception:
        pass

    return False


def value_kind(value: Any) -> str:
    if is_null(value):
        return "null"

    if isinstance(value, (bool, np.bool_)):
        return "bool"

    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return "integer"

    if isinstance(value, (float, np.floating)):
        return "float"

    if isinstance(value, str):
        return "string"

    if isinstance(value, (datetime, date, pd.Timestamp)):
        return "datetime"

    if isinstance(value, (list, tuple)):
        return "array"

    if isinstance(value, dict):
        return "object"

    return type(value).__name__


def is_numeric_non_finite(value: Any) -> bool:
    if is_null(value):
        return False

    if isinstance(value, (bool, np.bool_)):
        return False

    if isinstance(value, (int, np.integer)):
        return False

    if isinstance(value, (float, np.floating)):
        return not math.isfinite(float(value))

    return False


def serialize_value(value: Any) -> Any:
    """
    Convierte valores a algo amigable para CSV/JSON.
    """
    if is_null(value):
        return None

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if isinstance(value, (datetime, date)):
        return value.isoformat()

    if isinstance(value, (dict, list, tuple, set)):
        try:
            return json.dumps(value, ensure_ascii=False, default=str)
        except Exception:
            return repr(value)

    return value


def schema_signature(keys: list[str]) -> str:
    raw = "\n".join(sorted(map(str, keys)))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def looks_like_target_key(key: str) -> bool:
    """
    Detección genérica, no vinculada a un nombre exacto.
    Sirve para alertar si la función de inferencia está retornando posible label.
    """
    k = str(key).lower().strip()

    patterns = [
        "target",
        "label",
        "y_true",
        "y_real",
        "x+1",
        "x_1",
        "xmas1",
        "next_day",
        "tomorrow",
    ]

    return any(p in k for p in patterns)


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
    fn,
    city: str,
    date_str: str,
    strict: bool,
    max_retries: int,
    backoff_base_sec: float,
    backoff_jitter_sec: float,
):
    attempts = 0
    slept = 0.0
    last_error_type = None
    last_error_msg = None

    kwargs = {}

    if function_accepts_kwarg(fn, "strict"):
        kwargs["strict"] = strict

    for retry in range(max_retries + 1):
        attempts += 1

        try:
            return fn(city, date_str, **kwargs), attempts, slept, None, None
        except Exception as exc:
            error_type = classify_error(exc)
            last_error_type = error_type
            last_error_msg = str(exc)[:500]

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
# Reportes
# ----------------------------

def pct(values: list[float], p: float):
    if not values:
        return None

    xs = sorted(values)
    idx = int(round((p / 100.0) * (len(xs) - 1)))
    return xs[idx]


def build_schema_report(success_feature_dicts: list[dict[str, Any]]) -> pd.DataFrame:
    if not success_feature_dicts:
        return pd.DataFrame(
            columns=[
                "feature",
                "presence_count",
                "missing_count",
                "presence_rate",
                "null_count",
                "non_null_count",
                "null_rate_when_present",
                "observed_types",
                "type_count",
                "numeric_non_finite_count",
                "looks_like_target",
            ]
        )

    ok_count = len(success_feature_dicts)
    all_keys = sorted({str(k) for row in success_feature_dicts for k in row.keys()})
    rows = []

    sentinel = object()

    for key in all_keys:
        presence_count = 0
        null_count = 0
        non_null_count = 0
        non_finite_count = 0
        type_counter = Counter()

        for features in success_feature_dicts:
            value = features.get(key, sentinel)

            if value is sentinel:
                continue

            presence_count += 1
            kind = value_kind(value)
            type_counter[kind] += 1

            if is_null(value):
                null_count += 1
            else:
                non_null_count += 1

            if is_numeric_non_finite(value):
                non_finite_count += 1

        missing_count = ok_count - presence_count

        rows.append({
            "feature": key,
            "presence_count": presence_count,
            "missing_count": missing_count,
            "presence_rate": presence_count / ok_count if ok_count else None,
            "null_count": null_count,
            "non_null_count": non_null_count,
            "null_rate_when_present": (
                null_count / presence_count if presence_count else None
            ),
            "observed_types": json.dumps(dict(type_counter), ensure_ascii=False),
            "type_count": len([t for t in type_counter.keys() if t != "null"]),
            "numeric_non_finite_count": non_finite_count,
            "looks_like_target": looks_like_target_key(key),
        })

    return pd.DataFrame(rows)


def flatten_rows(records: list[dict[str, Any]], all_feature_keys: list[str]) -> pd.DataFrame:
    flat_rows = []

    for rec in records:
        row = {
            "date": rec["date"],
            "date_str": rec["date_str"],
            "ok": rec["ok"],
            "empty": rec["empty"],
            "duration_sec": rec["duration_sec"],
            "attempts": rec["attempts"],
            "slept_backoff_sec": rec["slept_backoff_sec"],
            "error_type": rec["error_type"],
            "error_msg": rec["error_msg"],
            "schema_hash": rec["schema_hash"],
            "feature_count": rec["feature_count"],
            "null_count": rec["null_count"],
            "non_finite_numeric_count": rec["non_finite_numeric_count"],
        }

        features = rec.get("features") or {}

        for key in all_feature_keys:
            row[key] = serialize_value(features.get(key, None))

        flat_rows.append(row)

    return pd.DataFrame(flat_rows)


# ----------------------------
# Main
# ----------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Test dinámico de contrato para get_weather_features."
    )

    ap.add_argument("--module", default="build_climate_data")
    ap.add_argument("--function", default="get_weather_features")
    ap.add_argument("--city", default="new york")

    ap.add_argument("--start", type=parse_iso_date, default=parse_iso_date("1980-01-01"))
    ap.add_argument("--end", type=parse_iso_date, default=parse_iso_date("2025-12-31"))
    ap.add_argument("--n-samples", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--no-replacement", type=str2bool, default=True)

    ap.add_argument(
        "--date-format",
        default="%d-%m-%y",
        help="Formato que espera la función. Usa 'iso' para YYYY-MM-DD.",
    )

    ap.add_argument("--strict", type=str2bool, default=False)

    ap.add_argument("--sleep-between-calls-sec", type=float, default=0.15)
    ap.add_argument("--max-retries", type=int, default=5)
    ap.add_argument("--backoff-base-sec", type=float, default=1.0)
    ap.add_argument("--backoff-jitter-sec", type=float, default=0.5)

    ap.add_argument("--out-dir", default="./reports/feature_contract_test")
    ap.add_argument("--checkpoint-every", type=int, default=50)

    # Criterios de aprobación.
    ap.add_argument("--min-ok-ratio", type=float, default=0.90)
    ap.add_argument("--max-empty-ratio", type=float, default=0.05)
    ap.add_argument("--max-error-ratio", type=float, default=0.10)
    ap.add_argument("--allow-schema-variance", action="store_true")
    ap.add_argument("--allow-non-finite", action="store_true")

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fn = load_function(args.module, args.function)

    dates = random_dates(
        start=args.start,
        end=args.end,
        n=args.n_samples,
        no_replacement=args.no_replacement,
        seed=args.seed,
    )

    print("\n=== DYNAMIC FEATURE CONTRACT TEST ===")
    print(f"Function: {args.module}.{args.function}")
    print(f"City: {args.city}")
    print(f"Date range: {args.start.isoformat()} -> {args.end.isoformat()}")
    print(f"Samples: {len(dates)}")
    print(f"Date format: {args.date_format}")
    print(f"strict={args.strict}")
    print(f"Output dir: {out_dir.resolve()}")
    print("No hardcoded EXPECTED_FEATURE_KEYS.\n")

    records: list[dict[str, Any]] = []
    success_feature_dicts: list[dict[str, Any]] = []

    durations = []
    durations_success = []
    attempts_list = []
    backoff_slept_list = []

    error_counts = Counter()
    schema_counts = Counter()

    first_ok: date | None = None
    last_ok: date | None = None

    for i, d in enumerate(dates, start=1):
        ds = date_to_function_str(d, args.date_format)

        print(f"[{i}/{len(dates)}] {d.isoformat()} ({ds})")

        t0 = time.perf_counter()

        features, attempts, slept, error_type, error_msg = call_with_retry(
            fn=fn,
            city=args.city,
            date_str=ds,
            strict=args.strict,
            max_retries=args.max_retries,
            backoff_base_sec=args.backoff_base_sec,
            backoff_jitter_sec=args.backoff_jitter_sec,
        )

        duration = time.perf_counter() - t0

        durations.append(duration)
        attempts_list.append(attempts)
        backoff_slept_list.append(slept)

        ok = False
        empty = False
        feature_count = 0
        null_count = 0
        non_finite_numeric_count = 0
        schema_hash = None

        if error_type is not None:
            error_counts[error_type] += 1

        else:
            if not isinstance(features, dict):
                error_type = "INVALID_RETURN_TYPE"
                error_msg = f"Expected dict, got {type(features).__name__}"
                error_counts[error_type] += 1
                features = None

            elif len(features) == 0:
                empty = True

            else:
                ok = True
                durations_success.append(duration)

                if first_ok is None:
                    first_ok = d
                last_ok = d

                normalized_features = {str(k): v for k, v in features.items()}
                features = normalized_features

                success_feature_dicts.append(features)

                feature_count = len(features)
                null_count = sum(1 for v in features.values() if is_null(v))
                non_finite_numeric_count = sum(
                    1 for v in features.values() if is_numeric_non_finite(v)
                )

                schema_hash = schema_signature(list(features.keys()))
                schema_counts[schema_hash] += 1

        records.append({
            "date": d.isoformat(),
            "date_str": ds,
            "ok": ok,
            "empty": empty,
            "duration_sec": duration,
            "attempts": attempts,
            "slept_backoff_sec": slept,
            "error_type": error_type,
            "error_msg": error_msg,
            "schema_hash": schema_hash,
            "feature_count": feature_count,
            "null_count": null_count,
            "non_finite_numeric_count": non_finite_numeric_count,
            "features": features if isinstance(features, dict) else None,
        })

        if args.sleep_between_calls_sec > 0:
            time.sleep(args.sleep_between_calls_sec)

        if args.checkpoint_every > 0 and i % args.checkpoint_every == 0:
            ok_count = sum(1 for r in records if r["ok"])
            empty_count = sum(1 for r in records if r["empty"])
            err_count = sum(error_counts.values())
            print(
                f"[checkpoint] i={i} ok={ok_count} empty={empty_count} "
                f"errors={err_count} last={ds} dur={duration:.2f}s"
            )

    # ----------------------------
    # Construcción de reportes
    # ----------------------------

    ok_count = sum(1 for r in records if r["ok"])
    empty_count = sum(1 for r in records if r["empty"])
    error_total = sum(error_counts.values())
    total = len(records)

    ok_ratio = ok_count / total if total else 0.0
    empty_ratio = empty_count / total if total else 0.0
    error_ratio = error_total / total if total else 0.0

    all_feature_keys = sorted({str(k) for f in success_feature_dicts for k in f.keys()})

    schema_df = build_schema_report(success_feature_dicts)
    rows_df = flatten_rows(records, all_feature_keys)

    schema_variants_rows = []
    for schema_hash, count in schema_counts.most_common():
        example = next(
            (
                r for r in records
                if r["ok"] and r["schema_hash"] == schema_hash
            ),
            None,
        )

        keys = sorted((example.get("features") or {}).keys()) if example else []

        schema_variants_rows.append({
            "schema_hash": schema_hash,
            "count": count,
            "coverage_rate_among_success": count / ok_count if ok_count else None,
            "feature_count": len(keys),
            "features_json": json.dumps(keys, ensure_ascii=False),
        })

    schema_variants_df = pd.DataFrame(schema_variants_rows)

    errors_df = pd.DataFrame(
        error_counts.most_common(),
        columns=["error_type", "count"],
    )

    non_finite_rows = []
    for rec in records:
        if not rec["ok"]:
            continue

        features = rec.get("features") or {}

        for key, value in features.items():
            if is_numeric_non_finite(value):
                non_finite_rows.append({
                    "date": rec["date"],
                    "date_str": rec["date_str"],
                    "feature": key,
                    "value": repr(value),
                })

    non_finite_df = pd.DataFrame(non_finite_rows)

    target_like_df = (
        schema_df[schema_df["looks_like_target"] == True]
        .copy()
        .sort_values(["presence_count", "feature"], ascending=[False, True])
        if not schema_df.empty
        else pd.DataFrame()
    )

    summary = {
        "module": args.module,
        "function": args.function,
        "city": args.city,
        "date_range_start": args.start.isoformat(),
        "date_range_end": args.end.isoformat(),
        "date_format": args.date_format,
        "strict": args.strict,
        "samples_total": total,
        "ok_returns": ok_count,
        "empty_returns": empty_count,
        "errors_total": error_total,
        "ok_ratio": ok_ratio,
        "empty_ratio": empty_ratio,
        "error_ratio": error_ratio,
        "first_ok_date": first_ok.isoformat() if first_ok else None,
        "last_ok_date": last_ok.isoformat() if last_ok else None,
        "feature_union_count": len(all_feature_keys),
        "schema_variants_count": len(schema_counts),
        "dominant_schema_hash": schema_counts.most_common(1)[0][0] if schema_counts else None,
        "dominant_schema_count": schema_counts.most_common(1)[0][1] if schema_counts else 0,
        "duration_mean_sec": statistics.mean(durations) if durations else None,
        "duration_median_sec": statistics.median(durations) if durations else None,
        "duration_p95_sec": pct(durations, 95),
        "duration_success_mean_sec": statistics.mean(durations_success) if durations_success else None,
        "duration_success_p95_sec": pct(durations_success, 95),
        "attempts_mean": statistics.mean(attempts_list) if attempts_list else None,
        "backoff_slept_total_sec": sum(backoff_slept_list),
        "non_finite_numeric_total": int(len(non_finite_df)),
        "target_like_key_count": int(len(target_like_df)) if not target_like_df.empty else 0,
        "seed": args.seed,
        "sampling_no_replacement": args.no_replacement,
    }

    # ----------------------------
    # Guardado
    # ----------------------------

    rows_df.to_csv(out_dir / "feature_test_rows.csv", index=False)
    schema_df.to_csv(out_dir / "feature_test_schema.csv", index=False)
    schema_variants_df.to_csv(out_dir / "feature_test_schema_variants.csv", index=False)
    errors_df.to_csv(out_dir / "feature_test_errors.csv", index=False)
    non_finite_df.to_csv(out_dir / "feature_test_non_finite.csv", index=False)
    target_like_df.to_csv(out_dir / "feature_test_target_like_keys.csv", index=False)

    pd.DataFrame({
        "date": [d.isoformat() for d in dates],
        "date_str": [date_to_function_str(d, args.date_format) for d in dates],
    }).to_csv(out_dir / "feature_test_dates_used.csv", index=False)

    (out_dir / "feature_test_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # ----------------------------
    # Consola
    # ----------------------------

    print("\n=== SUMMARY ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

    print("\n=== ERROR COUNTS ===")
    if error_counts:
        for k, v in error_counts.most_common():
            print(f"{k}: {v}")
    else:
        print("No errors.")

    print("\n=== SCHEMA VARIANTS ===")
    if schema_counts:
        for schema_hash, count in schema_counts.most_common():
            print(f"{schema_hash}: {count}")
    else:
        print("No successful schema.")

    print("\n=== TOP NULL FEATURES ===")
    if not schema_df.empty:
        top_nulls = (
            schema_df.sort_values(["null_count", "feature"], ascending=[False, True])
            .head(15)
        )

        for _, row in top_nulls.iterrows():
            if int(row["null_count"]) > 0:
                print(
                    f"{row['feature']}: nulls={int(row['null_count'])} "
                    f"presence={int(row['presence_count'])}"
                )
    else:
        print("No schema report.")

    print("\n=== TARGET-LIKE KEYS ===")
    if not target_like_df.empty:
        for _, row in target_like_df.iterrows():
            print(f"{row['feature']} | presence={int(row['presence_count'])}")
    else:
        print("No target-like keys detected.")

    print("\nArchivos generados:")
    for filename in [
        "feature_test_rows.csv",
        "feature_test_summary.json",
        "feature_test_schema.csv",
        "feature_test_schema_variants.csv",
        "feature_test_errors.csv",
        "feature_test_non_finite.csv",
        "feature_test_target_like_keys.csv",
        "feature_test_dates_used.csv",
    ]:
        print(f"- {out_dir / filename}")

    # ----------------------------
    # Criterios de fallo
    # ----------------------------

    failures = []

    if ok_ratio < args.min_ok_ratio:
        failures.append(
            f"ok_ratio={ok_ratio:.3f} < min_ok_ratio={args.min_ok_ratio:.3f}"
        )

    if empty_ratio > args.max_empty_ratio:
        failures.append(
            f"empty_ratio={empty_ratio:.3f} > max_empty_ratio={args.max_empty_ratio:.3f}"
        )

    if error_ratio > args.max_error_ratio:
        failures.append(
            f"error_ratio={error_ratio:.3f} > max_error_ratio={args.max_error_ratio:.3f}"
        )

    if not args.allow_schema_variance and len(schema_counts) > 1:
        failures.append(
            f"schema_variants_count={len(schema_counts)} > 1. "
            "La función no retorna el mismo conjunto de keys para todas las fechas exitosas."
        )

    if not args.allow_non_finite and len(non_finite_df) > 0:
        failures.append(
            f"non_finite_numeric_total={len(non_finite_df)} > 0."
        )

    if failures:
        print("\n=== TEST FAILED ===")
        for f in failures:
            print(f"- {f}")
        return 1

    print("\n=== TEST PASSED ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
