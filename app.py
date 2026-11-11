from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from energy_core import (
    DATASET_CITATION,
    DATASET_INFO,
    DATASET_URL,
    DEFAULT_HOUSEHOLD_SAMPLE,
    calculate_appliance_usage,
    download_dataset,
    forecast_next_days,
    make_personal_csv_template,
    parse_london_tsf_archive,
    prepare_personal_daily_data,
    train_global_model,
    train_personal_model,
)

st.set_page_config(
    page_title="Smart Energy Consumption Optimizer",
    page_icon="⚡",
    layout="wide",
)
st.title("⚡ AI-Based Smart Energy Consumption Optimizer")
st.caption("A multi-household forecasting model with an option to personalise it using your own meter readings.")

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
ARCHIVE_PATH = DATA_DIR / "london_smart_meters_dataset_without_missing_values.zip"


def ensure_dataset_downloaded() -> None:
    if ARCHIVE_PATH.exists():
        # Validate an existing archive; download_dataset returns immediately if it is valid.
        download_dataset(ARCHIVE_PATH)
        return
    st.info("First run: downloading the public multi-household dataset (about 220 MB). It is saved locally and will not need to be downloaded again.")
    progress = st.progress(0.0, text="Starting dataset download…")

    def update_progress(done: int, total: int) -> None:
        if total > 0:
            progress.progress(min(done / total, 1.0), text=f"Downloading dataset: {done / 1_000_000:.0f} / {total / 1_000_000:.0f} MB")
        else:
            progress.progress(0.0, text=f"Downloaded {done / 1_000_000:.0f} MB…")

    download_dataset(ARCHIVE_PATH, progress_callback=update_progress)
    progress.progress(1.0, text="Download complete. Preparing household sample…")


@st.cache_data(show_spinner="Preparing the sampled household data…")
def cached_households(archive_path: str, sample_count: int) -> dict[str, pd.Series]:
    return parse_london_tsf_archive(archive_path, max_households=sample_count)


@st.cache_resource(show_spinner="Training the multi-household AI model…")
def cached_global_model(households: dict[str, pd.Series]) -> dict:
    return train_global_model(households)


with st.sidebar:
    st.header("Forecast settings")
    horizon = st.slider("Forecast horizon (days)", min_value=1, max_value=14, value=7)
    sample_count = st.select_slider("Households to sample", options=[100, 150, 200, 250, 300], value=DEFAULT_HOUSEHOLD_SAMPLE)
    st.markdown("---")
    st.caption("A larger sample gives the model more household histories but takes longer to prepare and train.")
    if st.button("Clear cached model and reload", use_container_width=True):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

try:
    ensure_dataset_downloaded()
    households = cached_households(str(ARCHIVE_PATH), int(sample_count))
    trained = cached_global_model(households)
except Exception as exc:
    st.error("The public dataset or model could not be loaded.")
    st.write("Check the internet connection and try again. The app does not substitute fake measurements if the public dataset fails.")
    st.exception(exc)
    st.stop()

metrics = trained["metrics"]
baseline_metrics = trained["baseline_metrics"]

m1, m2, m3, m4 = st.columns(4)
m1.metric("Household series sampled", f"{len(households):,}")
m2.metric("Training households", f"{trained['train_households']:,}")
m3.metric("Unseen test households", f"{trained['test_households']:,}")
m4.metric("Model test MAE", f"{metrics['mae']:.2f} kWh")

forecast_tab, personal_tab, appliance_tab, about_tab = st.tabs(
    ["Multi-household forecast", "Personalise my home", "Appliance optimizer", "About the project"]
)

with forecast_tab:
    st.subheader("Forecast for a selected household meter")
    test_ids = trained["test_household_ids"]
    selected_id = st.selectbox("Choose a held-out household series", test_ids, index=0)
    history = households[selected_id]
    forecast = forecast_next_days(trained["model"], history, days=horizon)
    evaluation = trained["evaluation"]
    selected_eval = evaluation[evaluation["household_id"] == selected_id].tail(60)

    left, right = st.columns([1.6, 1])
    with left:
        fig, ax = plt.subplots(figsize=(10, 4.2))
        ax.plot(selected_eval["date"], selected_eval["actual_kwh"], label="Actual use", linewidth=1.8)
        ax.plot(selected_eval["date"], selected_eval["model_prediction_kwh"], label="AI prediction", linewidth=1.6)
        ax.plot(selected_eval["date"], selected_eval["weekly_baseline_kwh"], label="Previous-week baseline", linewidth=1.2, alpha=0.8)
        ax.set_ylabel("Energy (kWh/day)")
        ax.set_xlabel("Date")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    with right:
        st.markdown("**Next-day estimate**")
        st.metric("Predicted daily use", f"{forecast.iloc[0]['predicted_kwh']:.2f} kWh")
        st.markdown("**Evaluation on unseen households**")
        st.metric("AI MAE", f"{metrics['mae']:.2f} kWh")
        st.metric("Weekly baseline MAE", f"{baseline_metrics['mae']:.2f} kWh", delta=f"{baseline_metrics['mae'] - metrics['mae']:.2f} kWh vs baseline", delta_color="normal")

    st.subheader(f"Next {horizon} days")
    st.dataframe(forecast.rename(columns={"predicted_kwh": "Predicted energy (kWh)"}), use_container_width=True)
    st.download_button("Download forecast CSV", forecast.to_csv().encode("utf-8"), "multi_household_forecast.csv", "text/csv")

    with st.expander("Full model evaluation"):
        st.write(f"Training samples: {trained['train_rows']:,} | Test samples: {trained['test_rows']:,}")
        st.write(f"Training household series: {trained['train_households']:,} | Held-out household series: {trained['test_households']:,}")
        st.write(f"AI MAE: {metrics['mae']:.3f} kWh")
        st.write(f"AI RMSE: {metrics['rmse']:.3f} kWh")
        st.write(f"AI R²: {metrics['r2']:.3f}")
        st.write(f"Previous-week baseline MAE: {baseline_metrics['mae']:.3f} kWh")
        st.write(f"Previous-week baseline RMSE: {baseline_metrics['rmse']:.3f} kWh")
        st.write(f"Previous-week baseline R²: {baseline_metrics['r2']:.3f}")
        st.caption("Households are split before training. The model is evaluated on different household IDs and their later dates. Within the test period, each one-day prediction uses the previous actual readings, so this is a rolling one-day-ahead test—not a 14-day recursive accuracy test.")

with personal_tab:
    st.subheader("Train a model for your own home")
    st.write("Upload a CSV containing at least 90 consecutive daily meter readings. When you run this app on your own computer, the file is processed by your local app; this code has no external upload step.")
    st.download_button("Download CSV template", make_personal_csv_template(), "household_daily_usage_template.csv", "text/csv")
    st.code("date,daily_kwh\n2026-01-01,12.4\n2026-01-02,13.1\n2026-01-03,11.8", language="csv")
    st.caption("Use daily total energy in kWh. Do not enter a monthly bill amount or a power value in kW. If your meter export is in 15- or 30-minute intervals, sum the interval kWh readings by date first.")
    uploaded = st.file_uploader("Upload your household CSV", type=["csv"], key="personal_csv")
    if uploaded is not None:
        try:
            personal_frame = pd.read_csv(uploaded)
            personal_daily = prepare_personal_daily_data(personal_frame)
            st.success(f"Loaded {len(personal_daily)} consecutive daily readings from {personal_daily.index.min().date()} to {personal_daily.index.max().date()}.")
            if len(personal_daily) < 90:
                st.warning(f"The personal model needs at least 90 days. Your file currently has {len(personal_daily)} days.")
            else:
                with st.spinner("Training and evaluating your household-specific model…"):
                    personal = train_personal_model(personal_daily)
                    personal_forecast = forecast_next_days(personal["model"], personal_daily, horizon)
                p1, p2, p3 = st.columns(3)
                p1.metric("Personal model MAE", f"{personal['metrics']['mae']:.2f} kWh")
                p2.metric("Personal model RMSE", f"{personal['metrics']['rmse']:.2f} kWh")
                p3.metric("Personal weekly-baseline MAE", f"{personal['baseline_metrics']['mae']:.2f} kWh")
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(personal["evaluation"].index, personal["evaluation"]["actual_kwh"], label="Actual household use")
                ax.plot(personal["evaluation"].index, personal["evaluation"]["model_prediction_kwh"], label="Personal model prediction")
                ax.plot(personal["evaluation"].index, personal["evaluation"]["weekly_baseline_kwh"], label="Previous-week baseline", alpha=0.8)
                ax.set_ylabel("Energy (kWh/day)")
                ax.set_xlabel("Date")
                ax.legend()
                ax.grid(alpha=0.25)
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
                st.subheader("Your household forecast")
                st.dataframe(personal_forecast.rename(columns={"predicted_kwh": "Predicted energy (kWh)"}), use_container_width=True)
                st.download_button("Download your forecast", personal_forecast.to_csv().encode("utf-8"), "my_household_forecast.csv", "text/csv")
                st.caption("The personal model is evaluated on the last 20% of the uploaded history, then retrained on all uploaded readings for the future forecast. Its accuracy is specific to this file and may change with seasons or household routines.")
        except Exception as exc:
            st.error(f"Could not use this CSV: {exc}")

with appliance_tab:
    st.subheader("Estimate energy use by appliance")
    st.write("Edit the example values or replace them with appliance nameplate wattage and estimated use. These are calculations, not smart-meter measurements.")
    starter = pd.DataFrame([
        {"Appliance": "Ceiling fan", "Power (W)": 75, "Hours/day": 8.0, "Days/week": 7},
        {"Appliance": "LED bulb", "Power (W)": 9, "Hours/day": 5.0, "Days/week": 7},
        {"Appliance": "Television", "Power (W)": 100, "Hours/day": 3.0, "Days/week": 6},
        {"Appliance": "Laptop", "Power (W)": 65, "Hours/day": 5.0, "Days/week": 6},
        {"Appliance": "Air conditioner", "Power (W)": 1500, "Hours/day": 4.0, "Days/week": 5},
    ])
    edited = st.data_editor(starter, num_rows="dynamic", use_container_width=True, column_config={
        "Power (W)": st.column_config.NumberColumn(min_value=0, step=1),
        "Hours/day": st.column_config.NumberColumn(min_value=0, max_value=24, step=0.5),
        "Days/week": st.column_config.NumberColumn(min_value=0, max_value=7, step=1),
    })
    try:
        estimates = calculate_appliance_usage(edited)
        if estimates.empty:
            st.info("Enter at least one valid appliance row to see estimates.")
        else:
            total = float(estimates["Monthly energy (kWh)"].sum())
            saving = float(estimates["Potential saving at 10% less use (kWh)"].sum())
            a, b = st.columns(2)
            a.metric("Estimated monthly appliance use", f"{total:.1f} kWh")
            b.metric("Illustrative 10% reduction", f"{saving:.1f} kWh/month")
            st.dataframe(estimates, use_container_width=True, hide_index=True)
            st.subheader("Usage suggestions")
            for _, row in estimates.head(3).iterrows():
                if row["Monthly energy (kWh)"] > 0:
                    st.write(f"**{row['Appliance']}** is among the largest estimates in your table. A 10% reduction in the entered operating time would save approximately {row['Potential saving at 10% less use (kWh)']:.2f} kWh per month.")
            st.download_button("Download appliance estimates", estimates.to_csv(index=False).encode("utf-8"), "appliance_energy_estimates.csv", "text/csv")
    except Exception as exc:
        st.warning(f"Please check the appliance table: {exc}")

with about_tab:
    st.subheader("Multi-household AI model")
    st.write("A Random Forest Regressor is trained on lag and calendar features from multiple household meter series. Household IDs are split before training, so the test set contains household series not used to fit the public model.")
    st.write(DATASET_INFO)
    st.write(DATASET_CITATION)
    st.markdown(f"[Dataset record and download details]({DATASET_URL.split('?')[0]})")
    st.subheader("Personalisation")
    st.write("The household-specific model is trained separately from a user's daily kWh history. The appliance calculator is another separate feature: appliance ratings and hours are used in a physical energy calculation and are not treated as smart-meter readings.")
    st.subheader("Important limitations")
    st.write("The public sample represents London households and historical usage from 2011–2014, not current Chennai households. It contains no detailed appliance-level labels. The selected archive uses LOCF to fill missing readings. Personal predictions are only as reliable as the user's own meter data and should be compared with a simple previous-week baseline.")

st.markdown("---")
st.caption("Class XII Artificial Intelligence project prototype | Python · Pandas · NumPy · Scikit-learn · Streamlit")
