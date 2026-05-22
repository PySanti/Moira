#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
test.py

Test dinámico para build_climate_data.get_weather_features.

Objetivo:
- Mantiene el test dinámico del contrato general.
- Puede endurecer validaciones específicas del contrato Sprint 3-v0.
- Evalúa el contrato general del módulo:
    1. La función se puede importar.
    2. La función retorna dict no vacío para fechas válidas.
    3. El esquema retornado es estable entre fechas exitosas.
    4. Los valores son serializables.
    5. No hay numéricos infinitos.
    6. Reporta nulls, errores, tiempos, variantes de esquema y features tipo target.
    7. Puede ejecutar train_mode/inference_mode y auditar que inference no
        devuelva el target oficial.
    8. Puede validar invariantes estrictas del output de build_climate_data.

Uso recomendado:
  python test.py \
    --module utils.build_climate_data \
    --function get_weather_features \
    --city "new york" \
    --start 1980-01-01 \
    --end 2025-12-31 \
    --n-samples 1000 \
    --strict false \
    --mode train_mode \
    --out-dir ./reports/feature_contract_test

Uso rápido:
  python test.py --n-samples 50

Notas:
- En modo estricto de contrato para Sprint 3-v0, el test sí valida varias
  invariantes observables del output y la paridad train/inference.
- No puede probar por sí solo el origen real de cada dato aguas arriba.


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


def optional_bool(value: str | bool | None) -> bool | None:
    if value is None:
        return None

    if isinstance(value, str) and value.strip().lower() in {"none", "null", "auto"}:
        return None

    return str2bool(value)


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

def ensure_project_root_on_path() -> None:
    root = Path(__file__).resolve().parents[1]
    root_str = str(root)

    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def load_function(module_name: str, function_name: str):
    ensure_project_root_on_path()

    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        if not module_name.startswith("utils."):
            try:
                module = importlib.import_module(f"utils.{module_name}")
            except Exception:
                raise RuntimeError(
                    f"No pude importar el módulo '{module_name}'. "
                    f"Ejecuta este test desde la raíz del proyecto o ajusta PYTHONPATH."
                ) from exc
        else:
            raise RuntimeError(
                f"No pude importar el módulo '{module_name}'. "
                f"Ejecuta este test desde la raíz del proyecto o ajusta PYTHONPATH."
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
    kwargs: dict[str, Any],
    max_retries: int,
    backoff_base_sec: float,
    backoff_jitter_sec: float,
):
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


def filter_kwargs_for_function(fn, kwargs: dict[str, Any]) -> dict[str, Any]:
    return {
        k: v
        for k, v in kwargs.items()
        if v is not None and function_accepts_kwarg(fn, k)
    }


def build_weather_kwargs(args, fn, mode: str, include_target: bool | None) -> dict[str, Any]:
    raw = {
        "strict": args.strict,
        "execution_hour": args.execution_hour,
        "nearest_tolerance_hours": args.nearest_tolerance_hours,
        "history_start_date": args.history_start.isoformat(),
        "climatology_window_days": args.climatology_window_days,
        "min_climatology_records": args.min_climatology_records,
        "compute_td_anomaly": args.compute_td_anomaly,
        "mode": mode,
        "include_target": include_target,
    }

    return filter_kwargs_for_function(fn, raw)


def run_inference_leakage_check(
    fn,
    args,
    dates: list[date],
) -> dict[str, Any]:
    if not args.run_inference_leakage_check:
        return {
            "checked": 0,
            "failures": 0,
            "target_like_keys": [],
            "errors": [],
        }

    check_dates = dates[:max(0, min(args.inference_check_samples, len(dates)))]
    kwargs = build_weather_kwargs(
        args=args,
        fn=fn,
        mode="inference_mode",
        include_target=False,
    )

    failures = []
    target_like_keys = set()
    errors = []

    for d in check_dates:
        ds = date_to_function_str(d, args.date_format)
        features, _, _, error_type, error_msg = call_with_retry(
            fn=fn,
            city=args.city,
            date_str=ds,
            kwargs=kwargs,
            max_retries=args.max_retries,
            backoff_base_sec=args.backoff_base_sec,
            backoff_jitter_sec=args.backoff_jitter_sec,
        )

        if error_type is not None:
            errors.append({
                "date": d.isoformat(),
                "error_type": error_type,
                "error_msg": error_msg,
            })
            continue

        if not isinstance(features, dict) or len(features) == 0:
            continue

        keys = {str(k) for k in features.keys()}
        leaked = sorted(keys & {"t_max_x+1"})

        if leaked:
            failures.append({
                "date": d.isoformat(),
                "leaked_keys": leaked,
            })

        target_like_keys.update(k for k in keys if looks_like_target_key(k))

    return {
        "checked": len(check_dates),
        "failures": len(failures),
        "failure_examples": failures[:10],
        "target_like_keys": sorted(target_like_keys),
        "errors": errors[:10],
    }


def normalize_feature_mode(mode: str, include_target: bool | None) -> str:
    normalized = str(mode).strip().lower().replace("-", "_")
    aliases = {
        "train": "train_mode",
        "training": "train_mode",
        "train_mode": "train_mode",
        "inference": "inference_mode",
        "infer": "inference_mode",
        "predict": "inference_mode",
        "prediction": "inference_mode",
        "inference_mode": "inference_mode",
    }

    resolved = aliases.get(normalized, normalized)

    if include_target is not None:
        resolved = "train_mode" if include_target else "inference_mode"

    return resolved


def is_build_climate_contract_target(args) -> bool:
    module_name = str(args.module).strip().lower()
    function_name = str(args.function).strip().lower()
    return (
        args.enforce_build_climate_contract
        and function_name == "get_weather_features"
        and module_name.endswith("build_climate_data")
    )


def strict_allowed_missing_keys() -> set[str]:
    return {
        "td_anomaly_x",
        "Temp_06h_x",
        "Temp_12h_x",
        "Temp_18h_x",
        "Temp_21h_x",
        "Temp_change_23h_minus_18h",
        "Temp_change_23h_minus_12h",
        "Temp_change_23h_minus_06h",
        "Temp_change_23h_1d",
        "Temp_change_last_3h",
        "Temp_change_last_6h",
        "Temp_change_last_12h",
        "Temp_slope_last_6h",
        "Temp_slope_last_12h",
        "Td_change_last_6h",
        "HR_change_last_6h",
        "SLP_change_last_3h",
        "SLP_change_last_6h",
        "SLP_change_last_12h",
        "WindSpd_change_last_6h",
        "Cloud_change_last_6h",
        "Precip_positive_hours_00_23h",
        "Precip_positive_hours_last_6h",
        "Temp_23h_ma3",
        "Temp_23h_trend_3d",
        "HR_23h_ma3",
        "WindSpd_23h_ma3",
    }


def sprint3_required_feature_keys(include_target: bool) -> set[str]:
    keys = {
        "Tmax_so_far_23h_x",
        "Tmin_so_far_23h_x",
        "Tmean_so_far_23h_x",
        "Temp_23h_x",
        "DTR_so_far_23h_x",
        "HR_23h_x",
        "Td_23h_x",
        "SLP_23h_x",
        "WindSpd_23h_x",
        "Cloud_23h_x",
        "Precip_sum_00_23h_x",
        "Temp_std_00_23h_x",
        "Td_mean_00_23h_x",
        "HR_mean_00_23h_x",
        "SLP_mean_00_23h_x",
        "WindSpd_mean_00_23h_x",
        "Cloud_mean_00_23h_x",
        "Precip_sum_last_6h",
        "Temp_dewpoint_spread_23h",
        "Vapor_pressure_23h",
        "climatology_tmax_doy",
        "climatology_tmax_doy_plus1",
        "climatology_tmax_delta_doy_plus1_minus_x",
        "tmax_anomaly_x",
        "tmax_anomaly_vs_doy_plus1",
        "tmax_lag1",
        "tmin_lag1",
        "tmean_lag1",
        "tmax_ma7_completed",
        "tmean_ma7",
        "dtr_ma7_completed",
        "Temp_23h_ma3",
        "HR_23h_ma3",
        "WindSpd_23h_ma3",
        "wind_u",
        "wind_v",
        "pressure_trend_3d",
        "month",
        "month_sin",
        "month_cos",
        "season",
        "extreme_heat_flag",
        "extreme_cold_flag",
        "daylight_hours_x",
        "daylight_hours_plus1",
        "daylight_delta_plus1_minus_x",
        "ciudad",
        "doy_sin",
        "doy_cos",
    }

    if include_target:
        keys.add("t_max_x+1")

    return keys


def approx_equal(a: Any, b: Any, tol: float = 1e-6) -> bool:
    if is_null(a) and is_null(b):
        return True

    if is_null(a) or is_null(b):
        return False

    if isinstance(a, (bool, np.bool_)) or isinstance(b, (bool, np.bool_)):
        return bool(a) == bool(b)

    if isinstance(a, (int, float, np.integer, np.floating)) and isinstance(
        b,
        (int, float, np.integer, np.floating),
    ):
        return math.isclose(float(a), float(b), rel_tol=tol, abs_tol=tol)

    return a == b


def compare_feature_values(a: Any, b: Any, tol: float = 1e-6) -> bool:
    if isinstance(a, np.generic):
        a = a.item()

    if isinstance(b, np.generic):
        b = b.item()

    return approx_equal(a, b, tol=tol)


def validate_build_climate_row(
    features: dict[str, Any],
    args,
    resolved_mode: str,
) -> list[str]:
    failures = []
    has_target = resolved_mode == "train_mode"
    expected_count = (
        args.expected_feature_count_train
        if has_target
        else args.expected_feature_count_inference
    )

    if expected_count is not None and len(features) != expected_count:
        failures.append(
            f"feature_count={len(features)} != expected_feature_count={expected_count}"
        )

    required_keys = sprint3_required_feature_keys(include_target=has_target)
    missing_keys = sorted(required_keys - set(features.keys()))

    if missing_keys:
        failures.append(
            "missing_required_keys=" + ", ".join(missing_keys[:25])
        )

    if has_target and "t_max_x+1" not in features:
        failures.append("train_mode missing t_max_x+1")

    if not has_target and "t_max_x+1" in features:
        failures.append("inference_mode must not include t_max_x+1")

    ciudad = features.get("ciudad")
    if not isinstance(ciudad, str) or ciudad.strip().lower() != args.city.strip().lower():
        failures.append(f"ciudad invalid: {ciudad!r}")

    season = features.get("season")
    month = features.get("month")
    month_to_season = {
        12: "winter",
        1: "winter",
        2: "winter",
        3: "spring",
        4: "spring",
        5: "spring",
        6: "summer",
        7: "summer",
        8: "summer",
        9: "autumn",
        10: "autumn",
        11: "autumn",
    }

    if is_null(month):
        failures.append(f"month invalid: {month!r}")
    else:
        month_int = int(month)
        if month_int not in month_to_season:
            failures.append(f"month invalid: {month!r}")
        elif season != month_to_season[month_int]:
            failures.append(f"season/month mismatch: season={season!r} month={month!r}")

    null_allowed = strict_allowed_missing_keys()
    if args.strict:
        unexpected_nulls = sorted(
            key
            for key, value in features.items()
            if is_null(value) and key not in {"season", "ciudad"} and key not in null_allowed
        )
        if unexpected_nulls:
            failures.append(
                "unexpected_nulls=" + ", ".join(unexpected_nulls[:25])
            )

    if not any(is_null(features.get(k)) for k in ["DTR_so_far_23h_x", "Tmax_so_far_23h_x", "Tmin_so_far_23h_x"]):
        expected = float(features["Tmax_so_far_23h_x"]) - float(features["Tmin_so_far_23h_x"])
        if not approx_equal(features["DTR_so_far_23h_x"], expected):
            failures.append("DTR_so_far_23h_x invariant failed")

    if not any(is_null(features.get(k)) for k in ["Temp_range_00_23h_x", "Tmax_so_far_23h_x", "Tmin_so_far_23h_x"]):
        expected = float(features["Tmax_so_far_23h_x"]) - float(features["Tmin_so_far_23h_x"])
        if not approx_equal(features["Temp_range_00_23h_x"], expected):
            failures.append("Temp_range_00_23h_x invariant failed")

    if not any(is_null(features.get(k)) for k in ["Temp_dewpoint_spread_23h", "Temp_23h_x", "Td_23h_x"]):
        expected = float(features["Temp_23h_x"]) - float(features["Td_23h_x"])
        if not approx_equal(features["Temp_dewpoint_spread_23h"], expected):
            failures.append("Temp_dewpoint_spread_23h invariant failed")

    if not any(is_null(features.get(k)) for k in ["tmax_anomaly_x", "Tmax_so_far_23h_x", "climatology_tmax_doy"]):
        expected = float(features["Tmax_so_far_23h_x"]) - float(features["climatology_tmax_doy"])
        if not approx_equal(features["tmax_anomaly_x"], expected):
            failures.append("tmax_anomaly_x invariant failed")

    if not any(is_null(features.get(k)) for k in ["tmax_anomaly_vs_doy_plus1", "Tmax_so_far_23h_x", "climatology_tmax_doy_plus1"]):
        expected = float(features["Tmax_so_far_23h_x"]) - float(features["climatology_tmax_doy_plus1"])
        if not approx_equal(features["tmax_anomaly_vs_doy_plus1"], expected):
            failures.append("tmax_anomaly_vs_doy_plus1 invariant failed")

    if not any(is_null(features.get(k)) for k in ["climatology_tmax_delta_doy_plus1_minus_x", "climatology_tmax_doy_plus1", "climatology_tmax_doy"]):
        expected = (
            float(features["climatology_tmax_doy_plus1"])
            - float(features["climatology_tmax_doy"])
        )
        if not approx_equal(features["climatology_tmax_delta_doy_plus1_minus_x"], expected):
            failures.append("climatology_tmax_delta_doy_plus1_minus_x invariant failed")

    if not any(is_null(features.get(k)) for k in ["wind_u", "WindSpd_23h_x", "WindDir_sin_23h_x"]):
        expected = float(features["WindSpd_23h_x"]) * float(features["WindDir_sin_23h_x"])
        if not approx_equal(features["wind_u"], expected):
            failures.append("wind_u invariant failed")

    if not any(is_null(features.get(k)) for k in ["wind_v", "WindSpd_23h_x", "WindDir_cos_23h_x"]):
        expected = float(features["WindSpd_23h_x"]) * float(features["WindDir_cos_23h_x"])
        if not approx_equal(features["wind_v"], expected):
            failures.append("wind_v invariant failed")

    precip_sum = features.get("Precip_sum_00_23h_x")
    precip_flag = features.get("Precip_flag_00_23h")
    if not is_null(precip_sum) and not is_null(precip_flag):
        expected = 1.0 if float(precip_sum) > 0 else 0.0
        if not approx_equal(precip_flag, expected):
            failures.append("Precip_flag_00_23h invariant failed")

    for cyclical_a, cyclical_b, label in [
        ("doy_sin", "doy_cos", "doy"),
        ("month_sin", "month_cos", "month"),
    ]:
        a = features.get(cyclical_a)
        b = features.get(cyclical_b)
        if not is_null(a) and not is_null(b):
            radius = (float(a) ** 2) + (float(b) ** 2)
            if not math.isclose(radius, 1.0, rel_tol=1e-5, abs_tol=1e-5):
                failures.append(f"{label}_sin_cos invariant failed")

    for binary_key in ["extreme_heat_flag", "extreme_cold_flag", "Precip_flag_00_23h"]:
        value = features.get(binary_key)
        if not is_null(value) and value not in {0, 0.0, 1, 1.0}:
            failures.append(f"{binary_key} must be binary, got {value!r}")

    return failures


def run_train_inference_parity_check(
    fn,
    args,
    dates: list[date],
) -> dict[str, Any]:
    if not args.run_train_inference_parity_check:
        return {
            "checked": 0,
            "failures": 0,
            "failure_examples": [],
            "errors": [],
        }

    check_dates = dates[:max(0, min(args.train_inference_check_samples, len(dates)))]
    train_kwargs = build_weather_kwargs(
        args=args,
        fn=fn,
        mode="train_mode",
        include_target=True,
    )
    inference_kwargs = build_weather_kwargs(
        args=args,
        fn=fn,
        mode="inference_mode",
        include_target=False,
    )

    failures = []
    errors = []

    for d in check_dates:
        ds = date_to_function_str(d, args.date_format)

        train_features, _, _, train_error_type, train_error_msg = call_with_retry(
            fn=fn,
            city=args.city,
            date_str=ds,
            kwargs=train_kwargs,
            max_retries=args.max_retries,
            backoff_base_sec=args.backoff_base_sec,
            backoff_jitter_sec=args.backoff_jitter_sec,
        )
        inference_features, _, _, inference_error_type, inference_error_msg = call_with_retry(
            fn=fn,
            city=args.city,
            date_str=ds,
            kwargs=inference_kwargs,
            max_retries=args.max_retries,
            backoff_base_sec=args.backoff_base_sec,
            backoff_jitter_sec=args.backoff_jitter_sec,
        )

        if train_error_type or inference_error_type:
            errors.append({
                "date": d.isoformat(),
                "train_error_type": train_error_type,
                "train_error_msg": train_error_msg,
                "inference_error_type": inference_error_type,
                "inference_error_msg": inference_error_msg,
            })
            continue

        if not isinstance(train_features, dict) or not isinstance(inference_features, dict):
            failures.append({
                "date": d.isoformat(),
                "reason": "non_dict_return",
            })
            continue

        if bool(train_features) != bool(inference_features):
            failures.append({
                "date": d.isoformat(),
                "reason": "train_inference_success_mismatch",
            })
            continue

        if not train_features and not inference_features:
            continue

        if is_build_climate_contract_target(args):
            train_contract_failures = validate_build_climate_row(
                features={str(k): v for k, v in train_features.items()},
                args=args,
                resolved_mode="train_mode",
            )
            inference_contract_failures = validate_build_climate_row(
                features={str(k): v for k, v in inference_features.items()},
                args=args,
                resolved_mode="inference_mode",
            )

            if train_contract_failures or inference_contract_failures:
                failures.append({
                    "date": d.isoformat(),
                    "reason": "mode_specific_contract_failure",
                    "train_failures": train_contract_failures[:20],
                    "inference_failures": inference_contract_failures[:20],
                })
                continue

        train_keys_without_target = set(train_features.keys()) - {"t_max_x+1"}
        inference_keys = set(inference_features.keys())

        if train_keys_without_target != inference_keys:
            failures.append({
                "date": d.isoformat(),
                "reason": "schema_mismatch_after_target_removal",
                "train_only": sorted(train_keys_without_target - inference_keys)[:20],
                "inference_only": sorted(inference_keys - train_keys_without_target)[:20],
            })
            continue

        value_mismatches = []
        for key in sorted(inference_keys):
            if not compare_feature_values(train_features.get(key), inference_features.get(key)):
                value_mismatches.append(key)
                if len(value_mismatches) >= 20:
                    break

        if value_mismatches:
            failures.append({
                "date": d.isoformat(),
                "reason": "value_mismatch",
                "keys": value_mismatches,
            })

    return {
        "checked": len(check_dates),
        "failures": len(failures),
        "failure_examples": failures[:10],
        "errors": errors[:10],
    }


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
            "strict_failure_count": rec.get("strict_failure_count", 0),
            "strict_failures_json": rec.get("strict_failures_json"),
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

    ap.add_argument("--module", default="utils.build_climate_data")
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
    ap.add_argument("--mode", default="train_mode")
    ap.add_argument("--include-target", type=optional_bool, default=None)
    ap.add_argument("--history-start", type=parse_iso_date, default=parse_iso_date("1980-01-01"))
    ap.add_argument("--execution-hour", type=int, default=23)
    ap.add_argument("--nearest-tolerance-hours", type=int, default=6)
    ap.add_argument("--min-climatology-records", type=int, default=30)
    ap.add_argument("--climatology-window-days", type=int, default=7)
    ap.add_argument("--compute-td-anomaly", type=str2bool, default=True)
    ap.add_argument("--run-inference-leakage-check", type=str2bool, default=True)
    ap.add_argument("--inference-check-samples", type=int, default=10)
    ap.add_argument("--run-train-inference-parity-check", type=str2bool, default=True)
    ap.add_argument("--train-inference-check-samples", type=int, default=10)

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
    ap.add_argument("--enforce-build-climate-contract", type=str2bool, default=True)
    ap.add_argument("--expected-feature-count-train", type=int, default=133)
    ap.add_argument("--expected-feature-count-inference", type=int, default=132)

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fn = load_function(args.module, args.function)
    function_kwargs = build_weather_kwargs(
        args=args,
        fn=fn,
        mode=args.mode,
        include_target=args.include_target,
    )

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
    print(f"mode={args.mode}")
    print(f"function kwargs={function_kwargs}")
    print(f"Output dir: {out_dir.resolve()}")
    print("No hardcoded EXPECTED_FEATURE_KEYS.\n")

    records: list[dict[str, Any]] = []
    success_feature_dicts: list[dict[str, Any]] = []
    strict_contract_failures: list[dict[str, Any]] = []

    durations = []
    durations_success = []
    attempts_list = []
    backoff_slept_list = []

    error_counts = Counter()
    schema_counts = Counter()
    resolved_mode = normalize_feature_mode(args.mode, args.include_target)
    build_contract_target = is_build_climate_contract_target(args)

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
            kwargs=function_kwargs,
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
        strict_row_failures: list[str] = []

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

                if build_contract_target:
                    strict_row_failures = validate_build_climate_row(
                        features=features,
                        args=args,
                        resolved_mode=resolved_mode,
                    )
                    if strict_row_failures:
                        strict_contract_failures.append({
                            "date": d.isoformat(),
                            "date_str": ds,
                            "failures": strict_row_failures,
                        })

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
            "strict_failure_count": len(strict_row_failures),
            "strict_failures_json": json.dumps(strict_row_failures, ensure_ascii=False),
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

    inference_leakage_check = run_inference_leakage_check(
        fn=fn,
        args=args,
        dates=dates,
    )
    train_inference_parity_check = run_train_inference_parity_check(
        fn=fn,
        args=args,
        dates=dates,
    )
    strict_contract_df = pd.DataFrame([
        {
            "date": rec["date"],
            "date_str": rec["date_str"],
            "failure_count": len(rec["failures"]),
            "failures_json": json.dumps(rec["failures"], ensure_ascii=False),
        }
        for rec in strict_contract_failures
    ])

    summary = {
        "module": args.module,
        "function": args.function,
        "city": args.city,
        "date_range_start": args.start.isoformat(),
        "date_range_end": args.end.isoformat(),
        "date_format": args.date_format,
        "strict": args.strict,
        "mode": args.mode,
        "resolved_mode": resolved_mode,
        "include_target": args.include_target,
        "history_start": args.history_start.isoformat(),
        "execution_hour": args.execution_hour,
        "nearest_tolerance_hours": args.nearest_tolerance_hours,
        "climatology_window_days": args.climatology_window_days,
        "min_climatology_records": args.min_climatology_records,
        "compute_td_anomaly": args.compute_td_anomaly,
        "function_kwargs": function_kwargs,
        "enforce_build_climate_contract": build_contract_target,
        "expected_feature_count_train": args.expected_feature_count_train,
        "expected_feature_count_inference": args.expected_feature_count_inference,
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
        "inference_leakage_check": inference_leakage_check,
        "train_inference_parity_check": train_inference_parity_check,
        "strict_contract_failure_count": len(strict_contract_failures),
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
    strict_contract_df.to_csv(out_dir / "feature_test_strict_contract_failures.csv", index=False)

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

    print("\n=== STRICT CONTRACT FAILURES ===")
    if not strict_contract_df.empty:
        for _, row in strict_contract_df.head(10).iterrows():
            print(f"{row['date']} | failures={row['failures_json']}")
    else:
        print("No strict contract failures.")

    print("\nArchivos generados:")
    for filename in [
        "feature_test_rows.csv",
        "feature_test_summary.json",
        "feature_test_schema.csv",
        "feature_test_schema_variants.csv",
        "feature_test_errors.csv",
        "feature_test_non_finite.csv",
        "feature_test_target_like_keys.csv",
        "feature_test_strict_contract_failures.csv",
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

    if inference_leakage_check.get("failures", 0) > 0:
        failures.append(
            "inference_mode devolvió el target oficial t_max_x+1."
        )

    if train_inference_parity_check.get("errors"):
        failures.append(
            "train_inference_parity_check no pudo completarse sin errores."
        )

    if train_inference_parity_check.get("failures", 0) > 0:
        failures.append(
            "train_inference_parity_check detectó diferencias entre train_mode e inference_mode."
        )

    if build_contract_target and strict_contract_failures:
        failures.append(
            f"strict_contract_failure_count={len(strict_contract_failures)} > 0."
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
