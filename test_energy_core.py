"""Offline tests. These use synthetic fixtures and do not download real data."""
import io
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from energy_core import (
    calculate_appliance_usage,
    forecast_next_days,
    make_demo_households,
    make_personal_csv_template,
    make_supervised_features,
    parse_london_tsf_archive,
    prepare_personal_daily_data,
    train_global_model,
    train_personal_model,
)


def test_appliance_calculation():
    frame = pd.DataFrame([{
        "Appliance": "Test appliance", "Power (W)": 100,
        "Hours/day": 2, "Days/week": 7,
    }])
    result = calculate_appliance_usage(frame)
    assert abs(result.iloc[0]["Monthly energy (kWh)"] - 6.088) < 0.001


def test_supervised_features_are_lagged():
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    series = pd.Series(np.arange(100, dtype=float) + 10, index=dates)
    features = make_supervised_features(series)
    first_date = features.index[0]
    assert features.loc[first_date, "lag_1"] == series.loc[first_date - pd.Timedelta(days=1)]
    assert features.loc[first_date, "lag_7"] == series.loc[first_date - pd.Timedelta(days=7)]


def test_global_model_and_forecast():
    households = make_demo_households(n_households=16, days=210)
    result = train_global_model(households)
    assert result["train_households"] >= 10
    assert result["test_households"] >= 2
    assert result["test_rows"] > 0
    test_id = result["test_household_ids"][0]
    forecast = forecast_next_days(result["model"], households[test_id], days=7)
    assert len(forecast) == 7
    assert (forecast["predicted_kwh"] >= 0).all()
    assert set(result["evaluation"]["household_id"]).issubset(set(result["test_household_ids"]))


def test_personal_upload_and_model():
    dates = pd.date_range("2024-01-01", periods=180, freq="D")
    rng = np.random.default_rng(5)
    values = 14 + 1.8 * np.sin(np.arange(180) * 2 * np.pi / 7) + rng.normal(0, 0.5, 180)
    upload = pd.DataFrame({"date": dates, "daily_kwh": values})
    daily = prepare_personal_daily_data(upload)
    result = train_personal_model(daily)
    forecast = forecast_next_days(result["model"], daily, 5)
    assert len(forecast) == 5
    assert result["test_rows"] > 10
    assert make_personal_csv_template().startswith(b"date,daily_kwh")


def test_tsf_archive_parser():
    start = pd.Timestamp("2012-01-01")
    count_days = 90
    values = ",".join(["0.10"] * (48 * count_days))
    lines = [
        "@relation london_test",
        "@attribute series_name string",
        "@attribute start_timestamp date",
        "@frequency half_hourly",
        "@missing false",
        "@equallength true",
        "@data",
    ]
    for i in range(12):
        lines.append(f"HOME{i:03d}:{start:%Y-%m-%d %H-%M-%S}:{values}")
    payload = "\n".join(lines)
    with TemporaryDirectory() as tmp:
        archive_path = Path(tmp) / "sample.zip"
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("london_sample.tsf", payload)
        households = parse_london_tsf_archive(archive_path, max_households=10)
    assert len(households) >= 10
    first = next(iter(households.values()))
    assert len(first) == count_days
    assert abs(float(first.iloc[0]) - 4.8) < 1e-8


if __name__ == "__main__":
    tests = [
        test_appliance_calculation,
        test_supervised_features_are_lagged,
        test_global_model_and_forecast,
        test_personal_upload_and_model,
        test_tsf_archive_parser,
    ]
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")
    print(f"All {len(tests)} offline tests passed.")
