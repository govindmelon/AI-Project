"""Core logic for the AI-Based Smart Energy Consumption Optimizer.

The public model is trained across a sample of household smart-meter series
from the Low Carbon London dataset. A separate personal model can be trained
from a user's own daily electricity readings.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable
import random
import re
import zipfile

import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATASET_URL = (
    "https://zenodo.org/records/4656091/files/"
    "london_smart_meters_dataset_without_missing_values.zip?download=1"
)
DATASET_NAME = "London Smart Meters Dataset (without Missing Values)"
DATASET_CITATION = (
    "Godahewa, R., Bergmeir, C., Webb, G., Hyndman, R., & Montero-Manso, P. "
    "(2021). London Smart Meters Dataset (without Missing Values) [Dataset]. "
    "Zenodo. https://doi.org/10.5281/zenodo.4656091"
)
DATASET_INFO = (
    "The Low Carbon London smart-meter collection contains half-hourly "
    "electricity readings from 5,560 London household time series between "
    "November 2011 and February 2014. The selected Zenodo version has missing "
    "readings filled using last-observation-carried-forward (LOCF). The app "
    "samples a subset of households to keep local training manageable."
)
DEFAULT_HOUSEHOLD_SAMPLE = 200
MIN_DAILY_HISTORY = 35
PERSONAL_MIN_DAYS = 90
FEATURE_COLUMNS = [
    "lag_1", "lag_2", "lag_7", "lag_14", "lag_28",
    "mean_7", "std_7", "mean_28",
    "dow_sin", "dow_cos", "doy_sin", "doy_cos",
]


def download_dataset(
    destination: str | Path,
    url: str = DATASET_URL,
    progress_callback: Callable[[int, int], None] | None = None,
) -> Path:
    """Download the public dataset once, using a temporary file safely."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and zipfile.is_zipfile(destination):
        return destination

    temporary = destination.with_suffix(destination.suffix + ".part")
    downloaded = 0
    try:
        with requests.get(url, stream=True, timeout=(20, 180)) as response:
            response.raise_for_status()
            total = int(response.headers.get("content-length", 0) or 0)
            content_type = response.headers.get("content-type", "").lower()
            if "text/html" in content_type:
                raise RuntimeError("The dataset host returned a web page instead of the ZIP file.")
            with temporary.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
                        downloaded += len(chunk)
                        if progress_callback:
                            progress_callback(downloaded, total)
        if not zipfile.is_zipfile(temporary):
            raise RuntimeError("The downloaded file is not a valid ZIP archive.")
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def _parse_tsf_timestamp(value: str) -> pd.Timestamp:
    value = value.strip()
    for fmt in ("%Y-%m-%d %H-%M-%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        parsed = pd.to_datetime(value, format=fmt, errors="coerce")
        if not pd.isna(parsed):
            return pd.Timestamp(parsed)
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"Could not interpret TSF start timestamp: {value!r}")
    return pd.Timestamp(parsed)


def _read_tsf_header_and_sample(
    text_stream,
    max_households: int,
    random_seed: int,
) -> tuple[list[str], str, list[str], int]:
    """Read a TSF file and reservoir-sample series lines without loading it all."""
    attributes: list[str] = []
    frequency = ""
    in_data = False
    rng = random.Random(random_seed)
    selected: list[str] = []
    seen = 0

    for line in text_stream:
        stripped = line.strip()
        if not stripped:
            continue
        lower = stripped.lower()
        if not in_data:
            if lower.startswith("@attribute"):
                match = re.match(r"@attribute\s+(\S+)\s+", stripped, flags=re.I)
                if match:
                    attributes.append(match.group(1))
            elif lower.startswith("@frequency"):
                frequency = stripped.split(maxsplit=1)[-1].strip()
            elif lower == "@data":
                in_data = True
            continue

        # One complete household time series is stored per line in TSF format.
        seen += 1
        if len(selected) < max_households:
            selected.append(stripped)
        else:
            replacement = rng.randrange(seen)
            if replacement < max_households:
                selected[replacement] = stripped

    if not in_data:
        raise ValueError("The ZIP archive did not contain a valid TSF @data section.")
    if not selected:
        raise ValueError("No household time series were found in the TSF file.")
    return attributes, frequency, selected, seen


def parse_london_tsf_archive(
    archive_path: str | Path,
    max_households: int = DEFAULT_HOUSEHOLD_SAMPLE,
    random_seed: int = 42,
) -> dict[str, pd.Series]:
    """Parse a sampled set of household series and aggregate half-hours to days.

    The source observations are kWh per half-hour. Daily kWh is the sum of
    available half-hour observations; LOCF-filled data are expected from the
    selected public archive.
    """
    if max_households < 10:
        raise ValueError("Use at least 10 households for a multi-household model.")
    archive_path = Path(archive_path)
    if not zipfile.is_zipfile(archive_path):
        raise ValueError(f"Dataset archive is missing or invalid: {archive_path}")

    with zipfile.ZipFile(archive_path) as archive:
        candidates = [
            name for name in archive.namelist()
            if name.lower().endswith((".tsf", ".txt")) and not name.endswith("/")
        ]
        if not candidates:
            raise ValueError("No .tsf or .txt time-series file was found in the dataset ZIP.")
        chosen = max(candidates, key=lambda name: archive.getinfo(name).file_size)
        with archive.open(chosen, "r") as raw:
            import io
            text_stream = io.TextIOWrapper(raw, encoding="utf-8", errors="replace")
            attributes, frequency, sampled_lines, total_series = _read_tsf_header_and_sample(
                text_stream, max_households=max_households, random_seed=random_seed
            )

    if frequency and frequency.lower() not in {"half_hourly", "halfhourly", "30min"}:
        raise ValueError(f"Expected half-hourly data, but the archive says {frequency!r}.")
    attr_lookup = {name.lower(): i for i, name in enumerate(attributes)}
    name_idx = attr_lookup.get("series_name", 0)
    start_idx = attr_lookup.get("start_timestamp")
    if start_idx is None:
        raise ValueError("The TSF data does not provide a start_timestamp attribute.")

    households: dict[str, pd.Series] = {}
    for line in sampled_lines:
        # TSF attributes are colon-separated; the values are the final field.
        fields = line.split(":", max(len(attributes), 1))
        if len(fields) < len(attributes) + 1:
            continue
        metadata = fields[:len(attributes)]
        values_text = ":".join(fields[len(attributes):])
        try:
            household_id = metadata[name_idx].strip()
            start = _parse_tsf_timestamp(metadata[start_idx])
            raw_values = values_text.split(",")
            values = np.fromiter(
                (float(v.strip()) if v.strip() not in {"", "?", "NaN", "nan"} else np.nan for v in raw_values),
                dtype=np.float64,
                count=len(raw_values),
            )
            if len(values) < 48 * MIN_DAILY_HISTORY:
                continue
            timestamps = pd.date_range(start=start, periods=len(values), freq="30min")
            interval_series = pd.Series(values, index=timestamps, name="interval_kwh")
            daily = interval_series.resample("D").sum(min_count=42)
            daily = daily.dropna().astype(float)
            if len(daily) >= MIN_DAILY_HISTORY and daily.index.is_monotonic_increasing:
                households[household_id] = daily
        except (ValueError, TypeError, IndexError):
            continue

    if len(households) < 10:
        raise ValueError(
            f"Only {len(households)} usable household series were parsed from the archive."
        )
    # Keep deterministic insertion order for repeatable train/test splits.
    households = dict(sorted(households.items()))
    return households


def download_and_load_london_households(
    cache_path: str | Path,
    max_households: int = DEFAULT_HOUSEHOLD_SAMPLE,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[str, pd.Series]:
    archive_path = download_dataset(cache_path, progress_callback=progress_callback)
    return parse_london_tsf_archive(archive_path, max_households=max_households)


def make_supervised_features(daily: pd.Series) -> pd.DataFrame:
    """Make leak-free one-day-ahead lag and calendar features for one home."""
    series = pd.to_numeric(daily, errors="coerce").dropna().sort_index().astype(float)
    if len(series) < MIN_DAILY_HISTORY + 1:
        raise ValueError(f"At least {MIN_DAILY_HISTORY + 1} valid daily readings are required.")
    frame = pd.DataFrame({"target": series})
    for lag in (1, 2, 7, 14, 28):
        frame[f"lag_{lag}"] = series.shift(lag)
    previous = series.shift(1)
    frame["mean_7"] = previous.rolling(7).mean()
    frame["std_7"] = previous.rolling(7).std(ddof=0)
    frame["mean_28"] = previous.rolling(28).mean()
    dow = frame.index.dayofweek.to_numpy()
    doy = frame.index.dayofyear.to_numpy()
    frame["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    frame["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    frame["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    frame["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return frame.dropna(subset=FEATURE_COLUMNS + ["target"])


def _metrics(actual: pd.Series | np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual_arr = np.asarray(actual, dtype=float)
    predicted_arr = np.asarray(predicted, dtype=float)
    return {
        "mae": float(mean_absolute_error(actual_arr, predicted_arr)),
        "rmse": float(np.sqrt(mean_squared_error(actual_arr, predicted_arr))),
        "r2": float(r2_score(actual_arr, predicted_arr)) if len(actual_arr) > 1 else float("nan"),
    }


def train_global_model(households: dict[str, pd.Series]) -> dict:
    """Train across households and test on unseen households' later periods.

    Household IDs are split before model training. The final 20% of dates from
    held-out households are used for evaluation; the first 80% remains available
    as their historical context for lag features.
    """
    household_ids = sorted(households)
    if len(household_ids) < 10:
        raise ValueError("At least 10 household series are required for evaluation.")
    rng = np.random.default_rng(42)
    shuffled = np.array(household_ids, dtype=object)
    rng.shuffle(shuffled)
    n_test = max(2, int(round(len(shuffled) * 0.20)))
    test_ids = set(shuffled[:n_test].tolist())
    train_ids = set(shuffled[n_test:].tolist())

    train_frames: list[pd.DataFrame] = []
    test_frames: list[pd.DataFrame] = []
    for household_id, series in households.items():
        features = make_supervised_features(series)
        split_date = series.index[int(len(series) * 0.80)]
        features = features.copy()
        features["household_id"] = household_id
        features["date"] = features.index
        if household_id in train_ids:
            train_part = features[features["date"] < split_date]
            if not train_part.empty:
                train_frames.append(train_part)
        else:
            test_part = features[features["date"] >= split_date]
            if not test_part.empty:
                test_frames.append(test_part)

    if not train_frames or not test_frames:
        raise ValueError("Could not create household-level train/test partitions.")
    train = pd.concat(train_frames, ignore_index=True)
    test = pd.concat(test_frames, ignore_index=True)
    if len(train) < 500 or len(test) < 50:
        raise ValueError("Not enough supervised records. Increase the household sample size.")

    model = RandomForestRegressor(
        n_estimators=140,
        max_depth=16,
        min_samples_leaf=3,
        max_features=0.9,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(train[FEATURE_COLUMNS], train["target"])
    predictions = model.predict(test[FEATURE_COLUMNS])
    baseline = test["lag_7"].to_numpy(dtype=float)

    evaluation = test[["household_id", "date", "target", "lag_7"]].copy()
    evaluation = evaluation.rename(columns={"target": "actual_kwh", "lag_7": "weekly_baseline_kwh"})
    evaluation["model_prediction_kwh"] = predictions
    metrics = _metrics(evaluation["actual_kwh"], predictions)
    baseline_metrics = _metrics(evaluation["actual_kwh"], baseline)

    return {
        "model": model,
        "metrics": metrics,
        "baseline_metrics": baseline_metrics,
        "evaluation": evaluation.sort_values(["household_id", "date"]).reset_index(drop=True),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_households": int(len(train_ids)),
        "test_households": int(len(test_ids)),
        "test_household_ids": sorted(test_ids),
        "feature_columns": FEATURE_COLUMNS,
    }


def _next_day_features(history: pd.Series, next_date: pd.Timestamp) -> pd.DataFrame:
    values = history.dropna().sort_index().astype(float).to_numpy()
    if len(values) < 28:
        raise ValueError("At least 28 daily readings are needed for forecasting.")
    date = pd.Timestamp(next_date)
    dow = date.dayofweek
    doy = date.dayofyear
    row = {
        "lag_1": values[-1],
        "lag_2": values[-2],
        "lag_7": values[-7],
        "lag_14": values[-14],
        "lag_28": values[-28],
        "mean_7": float(np.mean(values[-7:])),
        "std_7": float(np.std(values[-7:], ddof=0)),
        "mean_28": float(np.mean(values[-28:])),
        "dow_sin": float(np.sin(2 * np.pi * dow / 7.0)),
        "dow_cos": float(np.cos(2 * np.pi * dow / 7.0)),
        "doy_sin": float(np.sin(2 * np.pi * doy / 365.25)),
        "doy_cos": float(np.cos(2 * np.pi * doy / 365.25)),
    }
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def forecast_next_days(model, daily: pd.Series, days: int = 7) -> pd.DataFrame:
    if not 1 <= days <= 14:
        raise ValueError("Forecast horizon must be between 1 and 14 days.")
    history = daily.dropna().sort_index().astype(float).copy()
    rows = []
    for _ in range(days):
        next_date = history.index[-1] + pd.Timedelta(days=1)
        x = _next_day_features(history, next_date)
        prediction = max(0.0, float(model.predict(x)[0]))
        rows.append({"date": next_date, "predicted_kwh": prediction})
        history.loc[next_date] = prediction
    return pd.DataFrame(rows).set_index("date")


def prepare_personal_daily_data(uploaded: pd.DataFrame) -> pd.Series:
    """Read a user's CSV containing date + daily_kwh columns."""
    if uploaded.empty:
        raise ValueError("The uploaded CSV is empty.")
    lookup = {str(c).strip().lower(): c for c in uploaded.columns}
    date_col = next((lookup[c] for c in ("date", "datetime", "timestamp") if c in lookup), None)
    value_col = next((lookup[c] for c in ("daily_kwh", "kwh", "energy_kwh", "consumption_kwh") if c in lookup), None)
    if date_col is None or value_col is None:
        raise ValueError("CSV needs a date column and a daily_kwh column. See the downloadable template.")
    data = uploaded[[date_col, value_col]].copy()
    data.columns = ["date", "daily_kwh"]
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["daily_kwh"] = pd.to_numeric(data["daily_kwh"], errors="coerce")
    data = data.dropna(subset=["date", "daily_kwh"])
    data = data[data["daily_kwh"] >= 0]
    if data.empty:
        raise ValueError("No valid non-negative daily kWh readings were found.")
    data["date"] = data["date"].dt.normalize()
    daily = data.groupby("date")["daily_kwh"].sum().sort_index()
    full_index = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(full_index)
    if daily.isna().any():
        raise ValueError("Your CSV has missing calendar days. Add a daily value for each date before training.")
    daily.index.name = "date"
    return daily.astype(float)


def train_personal_model(daily: pd.Series) -> dict:
    """Evaluate on the latest 20% of a home history, then refit on all data."""
    daily = daily.dropna().sort_index().astype(float)
    if len(daily) < PERSONAL_MIN_DAYS:
        raise ValueError(f"Upload at least {PERSONAL_MIN_DAYS} consecutive days of daily readings.")
    features = make_supervised_features(daily)
    split_at = int(len(features) * 0.80)
    train = features.iloc[:split_at]
    test = features.iloc[split_at:]
    if len(train) < 40 or len(test) < 10:
        raise ValueError("Not enough records remain after creating the training and test sets.")
    evaluation_model = RandomForestRegressor(
        n_estimators=120,
        max_depth=12,
        min_samples_leaf=2,
        random_state=17,
        n_jobs=-1,
    )
    evaluation_model.fit(train[FEATURE_COLUMNS], train["target"])
    predictions = evaluation_model.predict(test[FEATURE_COLUMNS])
    metrics = _metrics(test["target"], predictions)
    baseline_metrics = _metrics(test["target"], test["lag_7"].to_numpy(dtype=float))
    evaluation = pd.DataFrame({
        "actual_kwh": test["target"],
        "model_prediction_kwh": predictions,
        "weekly_baseline_kwh": test["lag_7"],
    }, index=test.index)

    final_model = RandomForestRegressor(
        n_estimators=120,
        max_depth=12,
        min_samples_leaf=2,
        random_state=17,
        n_jobs=-1,
    )
    final_model.fit(features[FEATURE_COLUMNS], features["target"])
    return {
        "model": final_model,
        "metrics": metrics,
        "baseline_metrics": baseline_metrics,
        "evaluation": evaluation,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
    }


def make_personal_csv_template(days: int = 120) -> bytes:
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=days, freq="D")
    template = pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "daily_kwh": [""] * days})
    return template.to_csv(index=False).encode("utf-8")


def calculate_appliance_usage(appliances: pd.DataFrame) -> pd.DataFrame:
    required = ["Appliance", "Power (W)", "Hours/day", "Days/week"]
    missing = [column for column in required if column not in appliances.columns]
    if missing:
        raise ValueError("Missing appliance columns: " + ", ".join(missing))
    result = appliances[required].copy()
    for column in ("Power (W)", "Hours/day", "Days/week"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result = result.dropna(subset=["Power (W)", "Hours/day", "Days/week"])
    result = result[(result["Power (W)"] >= 0) & result["Hours/day"].between(0, 24) & result["Days/week"].between(0, 7)].copy()
    result["Monthly energy (kWh)"] = (
        result["Power (W)"] * result["Hours/day"] * result["Days/week"] * (30.44 / 7.0) / 1000.0
    )
    result["Potential saving at 10% less use (kWh)"] = result["Monthly energy (kWh)"] * 0.10
    return result.sort_values("Monthly energy (kWh)", ascending=False).reset_index(drop=True)


def make_demo_households(n_households: int = 16, days: int = 240, seed: int = 42) -> dict[str, pd.Series]:
    """Synthetic data used only for offline tests; never used as real training data."""
    rng = np.random.default_rng(seed)
    index = pd.date_range("2022-01-01", periods=days, freq="D")
    households: dict[str, pd.Series] = {}
    for i in range(n_households):
        base = rng.uniform(7.0, 28.0)
        weekday = np.array([1.02, 1.00, 0.99, 1.00, 1.04, 1.13, 1.16])[index.dayofweek]
        season = 1.0 + rng.uniform(0.04, 0.16) * np.sin(np.arange(days) * 2 * np.pi / 365.25 + rng.uniform(0, 2 * np.pi))
        daily_noise = rng.normal(0, base * 0.09, days)
        occasional_spikes = (rng.random(days) < 0.025) * rng.uniform(2, 10, days)
        values = np.maximum(0.5, base * weekday * season + daily_noise + occasional_spikes)
        households[f"DEMO-{i + 1:03d}"] = pd.Series(values, index=index, name="daily_kwh")
    return households
