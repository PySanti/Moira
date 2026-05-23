# Moira Agent Notes

## Repo shape
- This repo is Python-only and lightweight: no `pyproject.toml`, no pytest/lint/typecheck config, and no CI workflow are checked in.
- The active pipeline is Sprint 2: feature generation in `utils/build_climate_data.py`, dataset in `dataset/sprint2.csv`, and training entrypoint in `training/v0-sprint2/train_1.1/main.py`.
- `utils/build_climate_data.py` exposes multiple `CITY_COORDS`, but both `get_weather_features()` and `preload_weather_cache()` currently hard-fail for every city except `"new york"`.

## Environment
- Dependencies come from `requirements.txt`.
- `python-version.txt` is UTF-16LE and specifies `Python 3.13.12`.
- README documents PEP 668 blocking `pip install --user` on system Python; prefer a project virtualenv.

## Verified commands
- Contract test for the current feature builder from the repo root:
```bash
python3 utils/test.py \
  --module utils.build_climate_data \
  --function get_weather_features \
  --city "new york" \
  --start 1983-01-01 \
  --end 2025-12-31 \
  --n-samples 1000 \
  --strict true \
  --mode train_mode \
  --nearest-tolerance-hours 6 \
  --history-start 1980-01-01 \
  --out-dir ./test-reports/sprint3-v0/feature_contract_test
```
- Rebuild the checked-in Sprint 2 dataset by overriding `utils/miner.py` defaults:
```bash
python3 utils/miner.py \
  --module utils.build_climate_data \
  --city "new york" \
  --start 1983-01-01 \
  --end 2025-12-31 \
  --history-start 1980-01-01 \
  --output ./dataset/sprint2.csv \
  --failed-output ./dataset/sprint2_failed_rows.csv \
  --metadata-output ./dataset/sprint2_metadata.json \
  --strict true \
  --preload true \
  --execution-hour 23 \
  --nearest-tolerance-hours 6 \
  --climatology-window-days 7 \
  --min-climatology-records 30 \
  --compute-td-anomaly true \
  --include-target true
```
- Train the current model from the repo root with `python3 training/v0-sprint2/train_1.1/main.py`. This script resolves paths from `__file__`, so root launch is safe.

## Feature-pipeline gotchas
- `get_weather_features()` expects `date_str` in `%d-%m-%y`. The helper scripts accept ISO dates on the CLI and convert them for you.
- `get_weather_features()` defaults to `mode="train_mode"`, so it includes target `t_max_x+1`. Use `mode="inference_mode"` or `include_target=False` for leakage-safe inference checks.
- Reproducibility depends on passing tolerance explicitly: `get_weather_features()` defaults `nearest_tolerance_hours=12`, `preload_weather_cache()` defaults `nearest_tolerance_hours=2`, but the helper scripts and checked-in Sprint 2 dataset use `6`.
- Disk cache is enabled by default under `.weather_cache/`. Override with `WEATHER_CACHE_DIR=/path` or disable with `WEATHER_DISABLE_DISK_CACHE=1`.

## Training quirks
- `training/v0-sprint2/train_1.1/main.py` uses a temporal split, not a random one: rows with `year < 2021` are train/validation, and `year >= 2021` is the final test holdout.
- Sprint 1 training scripts (`training/v0-sprint1/train_1.0`, `train_1.1`, `train_1.2`) still read `../../dataset/original_dataset.csv` via cwd-relative paths, so they are not safe to launch from arbitrary directories.
- `training/v0-sprint1/train_1.2/main.py` imports `xgboost`, but `xgboost` is not listed in `requirements.txt`.
