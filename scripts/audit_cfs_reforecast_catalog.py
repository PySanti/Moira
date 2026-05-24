#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
from datetime import date, timedelta
from pathlib import Path
from xml.etree import ElementTree as ET

import requests


CFS_REFOR_CATALOG_TEMPLATE = (
    "https://www.ncei.noaa.gov/thredds/catalog/"
    "model-cfs_refor_hp_ts_9m/{year}/{yearmonth}/catalog.xml"
)


def random_dates(start: date, end: date, n: int, seed: int) -> list[date]:
    random.seed(seed)
    total = (end - start).days + 1
    if n > total:
        raise ValueError("n-samples mayor que el rango de fechas")
    offsets = random.sample(range(total), n)
    return sorted(start + timedelta(days=o) for o in offsets)


def fetch_month_catalog(year: int, month: int, timeout: int) -> str | None:
    yyyymm = f"{year:04d}{month:02d}"
    url = CFS_REFOR_CATALOG_TEMPLATE.format(year=year, yearmonth=yyyymm)
    r = requests.get(url, timeout=timeout)
    if r.status_code != 200:
        return None
    return r.text


def has_any_init_for_day(xml_text: str, day: date) -> bool:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return False

    # Match e.g. z850.1982010100.time.grb2
    pat = re.compile(r"\.(\d{10})\.time\.grb2$")
    day_prefix = day.strftime("%Y%m%d")
    for elem in root.iter():
        name = elem.attrib.get("name", "")
        m = pat.search(name)
        if not m:
            continue
        init_yyyymmddhh = m.group(1)
        if init_yyyymmddhh.startswith(day_prefix):
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Audita disponibilidad CFS reforecast por catalogo THREDDS")
    parser.add_argument("--start", default="1982-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--out", default="reports/contract-tests/sprint4-v0/cfs_catalog_audit_summary.json")
    args = parser.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    dates = random_dates(start=start, end=end, n=args.n_samples, seed=args.seed)

    month_cache: dict[tuple[int, int], str | None] = {}
    rows = []
    for d in dates:
        key = (d.year, d.month)
        if key not in month_cache:
            month_cache[key] = fetch_month_catalog(d.year, d.month, timeout=args.timeout)
        xml_text = month_cache[key]
        available = False if xml_text is None else has_any_init_for_day(xml_text, d)
        rows.append({"date": d.isoformat(), "available": available})

    ok = sum(1 for r in rows if r["available"])
    summary = {
        "start": args.start,
        "end": args.end,
        "n_samples": args.n_samples,
        "ok": ok,
        "missing": len(rows) - ok,
        "ok_ratio": ok / len(rows) if rows else 0.0,
        "rows": rows,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
