#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
split_dataset.py

Lee un dataset, extrae todos los registros de otono y los guarda en valset.csv.
Luego ELIMINA esos registros del dataset original (sobrescribe el dataset),
haciendo un backup automático antes de modificarlo.

Por defecto usa "otoño meteorológico" (NYC / hemisferio norte):
- Sep 1 a Nov 30  (meses 9,10,11)

Opcional: definición "astronomical" (aprox):
- Sep 22 a Dec 20 (por año)

Uso:
  PYTHONPATH=src python -m moira.data.split_dataset --csv ./data/processed/sprint1/original_dataset.csv --val ./data/processed/sprint1/autumn_split/valset.csv

Opcional:
  PYTHONPATH=src python -m moira.data.split_dataset --csv ./data/processed/sprint1/original_dataset.csv --date-col date
  PYTHONPATH=src python -m moira.data.split_dataset --csv ./data/processed/sprint1/original_dataset.csv --definition astronomical
  PYTHONPATH=src python -m moira.data.split_dataset --csv ./data/processed/sprint1/original_dataset.csv --city-col ciudad --city "New York"
"""

from __future__ import annotations

import argparse
from pathlib import Path
from datetime import datetime
import unicodedata
import pandas as pd


def normalize_colname(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    out = []
    prev_us = False
    for ch in s:
        if ch.isalnum():
            out.append(ch)
            prev_us = False
        else:
            if not prev_us:
                out.append("_")
                prev_us = True
    s2 = "".join(out).strip("_")
    while "__" in s2:
        s2 = s2.replace("__", "_")
    return s2


def detect_date_col(columns) -> str | None:
    norm_map = {normalize_colname(c): c for c in columns}
    for cand in ["date", "fecha", "datetime", "dt", "time", "day"]:
        key = normalize_colname(cand)
        if key in norm_map:
            return norm_map[key]
    return None


def autumn_mask_meteorological(dt: pd.Series) -> pd.Series:
    # Sep, Oct, Nov (Hemisferio norte)
    return dt.dt.month.isin([9, 10, 11])


def autumn_mask_astronomical_approx(dt: pd.Series) -> pd.Series:
    # Aproximación por año: Sep 22 a Dec 20 inclusive
    years = dt.dt.year
    start = pd.to_datetime(
        years.astype(str) + "-09-22",
        errors="coerce"
    )
    end = pd.to_datetime(
        years.astype(str) + "-12-20",
        errors="coerce"
    )
    return (dt >= start) & (dt <= end)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="./data/processed/sprint1/original_dataset.csv", help="Ruta del dataset.")
    ap.add_argument("--val", default="./data/processed/sprint1/autumn_split/valset.csv", help="Ruta del valset.")
    ap.add_argument("--date-col", default=None, help="Nombre columna fecha (si no se autodetecta)")
    ap.add_argument("--definition", choices=["meteorological", "astronomical"], default="meteorological",
                    help="Definición de otoño (default: meteorological)")
    ap.add_argument("--city-col", default=None, help="Nombre de columna ciudad (opcional)")
    ap.add_argument("--city", default=None, help="Valor exacto de ciudad a filtrar (opcional)")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f"[ERROR] No existe: {csv_path.resolve()}")

    df = pd.read_csv(csv_path)

    # Filtrado opcional por ciudad (solo para seleccionar qué filas entran al split)
    if args.city_col and args.city is not None:
        if args.city_col not in df.columns:
            raise SystemExit(f"[ERROR] city-col '{args.city_col}' no existe en el CSV.")
        df_scope = df[df[args.city_col].astype(str) == str(args.city)].copy()
        df_rest_scope = df[df[args.city_col].astype(str) != str(args.city)].copy()
    else:
        df_scope = df.copy()
        df_rest_scope = None

    # Detectar columna de fecha
    date_col = args.date_col if args.date_col else detect_date_col(df_scope.columns)
    if not date_col or date_col not in df_scope.columns:
        raise SystemExit(
            "[ERROR] No pude detectar la columna de fecha. "
            "Pasa el nombre con --date-col (ej: --date-col date)."
        )

    # Parsear fecha
    dt = pd.to_datetime(df_scope[date_col], errors="coerce")
    bad_dates = dt.isna() & df_scope[date_col].notna()
    bad_count = int(bad_dates.sum())

    if bad_count > 0:
        # Guardar filas con fecha mala para revisarlas
        bad_path = csv_path.parent / "bad_dates_rows.csv"
        df_scope.loc[bad_dates].to_csv(bad_path, index=False)
        print(f"[WARN] {bad_count} fechas no parseables. Guardé esas filas en: {bad_path.resolve()}")

    df_scope["_parsed_date"] = dt

    # Crear máscara de otoño
    if args.definition == "meteorological":
        mask_autumn = autumn_mask_meteorological(df_scope["_parsed_date"])
    else:
        mask_autumn = autumn_mask_astronomical_approx(df_scope["_parsed_date"])

    # No consideramos filas sin fecha parseada como otoño (se quedan en train)
    mask_autumn = mask_autumn.fillna(False)

    val_df = df_scope[mask_autumn].drop(columns=["_parsed_date"])
    train_df_scope = df_scope[~mask_autumn].drop(columns=["_parsed_date"])

    # Si hubo filtrado por ciudad, recomponemos dataset completo:
    # - para esa ciudad: usamos train_df_scope
    # - para el resto de ciudades: dejamos intacto
    if df_rest_scope is not None:
        new_dataset_df = pd.concat([train_df_scope, df_rest_scope], ignore_index=True)
    else:
        new_dataset_df = train_df_scope

    # Backup antes de sobrescribir dataset.csv
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = csv_path.with_name(f"{csv_path.stem}_backup_{ts}{csv_path.suffix}")
    df.to_csv(backup_path, index=False)

    # Guardar valset y sobrescribir dataset.csv sin el otoño
    val_path = Path(args.val)
    val_df.to_csv(val_path, index=False)
    new_dataset_df.to_csv(csv_path, index=False)

    # Resumen
    print("\n=== SPLIT OTOÑO → VALSET ===")
    print(f"Dataset original: {len(df):,} filas")
    if df_rest_scope is not None:
        print(f"Scope ciudad='{args.city}' ({args.city_col}): {len(df_scope):,} filas")
        print(f"Resto ciudades (sin tocar): {len(df_rest_scope):,} filas")
    print(f"Valset (otoño): {len(val_df):,} filas -> {val_path.resolve()}")
    print(f"Nuevo dataset.csv (sin otoño): {len(new_dataset_df):,} filas -> {csv_path.resolve()}")
    print(f"Backup creado: {backup_path.resolve()}")
    if len(val_df) > 0:
        # Mostrar rango de fechas del valset
        val_dt = pd.to_datetime(val_df[date_col], errors="coerce")
        print(f"Rango fechas valset: {val_dt.min()}  →  {val_dt.max()}")
    print(f"Definición usada: {args.definition}")
    if bad_count > 0:
        print(f"[WARN] Fechas no parseables: {bad_count} (quedaron en dataset.csv)")

if __name__ == "__main__":
    main()
