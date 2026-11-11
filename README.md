# AI-Based Smart Energy Consumption Optimizer — Multi-household edition

This version upgrades the original one-household demonstration into two separate forecasting workflows:

1. **Public multi-household model:** a Random Forest is trained on a reproducible sample of household smart-meter series from the Low Carbon London dataset.
2. **Personal household model:** a separate Random Forest can be trained from a user's own daily electricity readings uploaded as a CSV.

The appliance optimizer remains a transparent wattage-and-hours calculation; it is not presented as a machine-learning measurement.

## Files

- `app.py` — Streamlit interface.
- `energy_core.py` — dataset download/parser, multi-household training, personal training, forecasting and appliance calculations.
- `requirements.txt` — dependencies.
- `test_energy_core.py` — offline tests using synthetic data and a small generated TSF fixture.
- `run_linux.sh` — creates/activates `.venv`, installs requirements and starts the app.
- `run_windows.bat` — Windows setup and launch.

## Run on Zorin OS / Ubuntu

Install Python environment support once:

```bash
sudo apt update
sudo apt install python3-full python3-venv
```

Extract this folder, open a terminal inside it, and run:

```bash
./run_linux.sh
```

Or run manually:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The first app launch downloads the public dataset archive (about 220 MB) into the local `data/` folder. It then samples up to 200 household time series by default (adjustable from 100 to 300 in the sidebar) to keep processing and model training manageable. The archive is retained, so it is not downloaded again on each launch.

## Personalise the model for a home

The **Personalise my home** tab accepts a CSV with at least 90 consecutive daily readings in kWh:

```csv
date,daily_kwh
2026-01-01,12.4
2026-01-02,13.1
2026-01-03,11.8
```

The app evaluates the personal model on the latest 20% of the uploaded data, then refits on the complete history to forecast the next days. If the meter export contains half-hourly or 15-minute intervals, sum the interval kWh readings by date first. Do not enter kW values or monthly bill totals in the `daily_kwh` column.

The uploaded CSV is processed locally by the Streamlit application. The code does not send it to an external service.

## Model evaluation design

For the public model, household IDs are split before model training. The final 20% of dates from held-out households are evaluated, while their earlier dates provide lag history. This tests how the general model behaves on households it did not train on. Results are compared with a simple previous-week baseline (`lag_7`).

The displayed evaluation is rolling one-day-ahead: each test prediction uses the previous actual daily readings. Multi-day future forecasts are recursive and may accumulate errors. Do not treat the one-day test metrics as a guarantee of future performance.

## Dataset

**London Smart Meters Dataset (without Missing Values)**, from the Low Carbon London project. It contains 5,560 half-hourly household time series from November 2011 to February 2014. Each interval value is kWh for the preceding half-hour. In this processed version, missing readings were filled using last-observation-carried-forward (LOCF).

Dataset record: https://zenodo.org/records/4656091

Citation: Godahewa, R., Bergmeir, C., Webb, G., Hyndman, R., & Montero-Manso, P. (2021). *London Smart Meters Dataset (without Missing Values)* [Dataset]. Zenodo. https://doi.org/10.5281/zenodo.4656091

The dataset represents London homes in 2011–2014, not current Chennai homes. It does not provide appliance labels, so the public model forecasts total meter consumption only. A household's own uploaded meter data is the strongest route to personalisation.

## Run offline tests

After installing dependencies:

```bash
python test_energy_core.py
```

These tests do not download the 220 MB public dataset. They check the appliance formula, lag features, global-model training/evaluation, personal CSV workflow, future forecasting and TSF parser using small generated fixtures.
