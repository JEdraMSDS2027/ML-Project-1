from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Philippine Crop Forecast", layout="wide")
st.title("Next-quarter crop production forecast")
st.write(
    "Choose a province and crop to estimate next-quarter production in metric tons. "
    "The model uses four quarters of weather to estimate a change from the same quarter last year."
)

bundle_path = Path(__file__).with_name("crop_forecast_log_change_bundle.joblib")
if not bundle_path.is_file():
    st.error("Run the export cell in the log-change notebook first.")
    st.stop()


@st.cache_resource
def load_bundle(path, modified_at):
    return joblib.load(path)


def restore_defaults(default_inputs):
    for key, value in default_inputs.items():
        st.session_state[key] = value
    st.session_state.pop("log_change_forecast", None)


modified_at = bundle_path.stat().st_mtime_ns
bundle = load_bundle(str(bundle_path), modified_at)
st.caption(f"Model selected by validation MAE: {bundle['model_name']}")

weather_columns = bundle["weather_columns"]
choices = bundle["crop_choices"]
province = st.selectbox("Province", sorted(choices["province"].unique()), format_func=str.title)
crop = st.selectbox(
    "Crop", sorted(choices.loc[choices["province"] == province, "crop"].unique()),
    key=f"crop_{province}",
)

weather = bundle["weather"]
province_weather = weather.loc[weather["province"] == province].set_index("weather_quarter")
last_quarter = province_weather.index.max()
forecast_quarter = last_quarter + 1
baseline_quarter = forecast_quarter - 4
lag_quarters = [forecast_quarter - lag for lag in range(1, 5)]

if not all(quarter in province_weather.index for quarter in lag_quarters):
    st.error("This province needs four consecutive weather quarters for a forecast.")
    st.stop()

history = bundle["history"]
actual = history.loc[
    (history["province"] == province)
    & (history["crop"] == crop)
    & (history["target_quarter"] <= last_quarter)
].set_index("target_quarter")["production"].sort_index()
if baseline_quarter not in actual.index:
    st.error(f"Production from {baseline_quarter} is needed for this forecast.")
    st.stop()

previous_year_tons = float(actual.loc[baseline_quarter])
st.metric(f"Same quarter last year ({baseline_quarter})", f"{previous_year_tons:,.1f} metric tons")

weather_used = province_weather.loc[lag_quarters, weather_columns].copy()
labels = {
    "temperature_2m_mean": "Mean temperature (°C)",
    "temperature_2m_max": "Average daily high temperature (°C)",
    "temperature_2m_min": "Average daily low temperature (°C)",
    "precipitation_sum": "Total precipitation (mm)",
    "rain_sum": "Total rain (mm)",
    "sunshine_duration": "Sunshine duration (seconds)",
    "shortwave_radiation_sum": "Shortwave radiation (MJ/m²)",
    "et0_fao_evapotranspiration": "Reference evapotranspiration (mm)",
    "wind_speed_10m_max": "Average daily maximum wind speed (km/h)",
    "wind_gusts_10m_max": "Average daily maximum wind gust (km/h)",
}
widget_keys = {column: f"weather_{province}_{last_quarter}_{column}" for column in weather_columns}
default_inputs = {
    widget_keys[column]: float(weather_used.at[last_quarter, column])
    for column in weather_columns
}

st.subheader(f"Weather for {last_quarter}")
st.caption("Edit the latest quarter's weather values, then click Forecast.")
left, right = st.columns(2)
for index, column in enumerate(weather_columns):
    key = widget_keys[column]
    if key not in st.session_state:
        st.session_state[key] = default_inputs[key]
    with (left if index % 2 == 0 else right):
        weather_used.at[last_quarter, column] = st.number_input(
            labels[column], step=0.1, format="%.2f", key=key,
        )

with st.expander("Fixed weather inputs from the three earlier quarters"):
    older_weather = weather_used.iloc[1:].rename(columns=labels)
    older_weather.index = older_weather.index.astype(str)
    st.dataframe(older_weather.round(2))

scenario = (
    modified_at, province, crop, forecast_quarter, tuple(weather_used.loc[last_quarter])
)
forecast_button, reset_button = st.columns(2)
calculate = forecast_button.button("Forecast", key="forecast", type="primary")
reset_button.button(
    "Restore defaults", key="restore_defaults", on_click=restore_defaults, args=(default_inputs,),
)

if calculate:
    row = {
        "province": province,
        "crop": crop,
        "previous_year_tons": previous_year_tons,
        "quarter_of_year": str(forecast_quarter.quarter),
    }
    for lag, quarter in enumerate(lag_quarters, start=1):
        for column in weather_columns:
            row[f"{column}_lag{lag}"] = float(weather_used.at[quarter, column])

    features = pd.DataFrame([row], columns=bundle["feature_columns"])
    log_change = float(bundle["log_change_model"].predict(features)[0])
    production = np.expm1(np.log1p(previous_year_tons) + log_change)
    if np.isfinite(production):
        st.session_state["log_change_forecast"] = (scenario, max(0.0, float(production)))
    else:
        st.error("The weather inputs produced a forecast outside the model's numerical range.")

result = st.session_state.get("log_change_forecast")
forecast = result[1] if result and result[0] == scenario else None
if forecast is None:
    st.caption(f"Click Forecast to calculate production for {forecast_quarter}.")
else:
    st.metric(f"{crop} forecast for {forecast_quarter}", f"{forecast:,.1f} metric tons")

quarters = pd.period_range(actual.index.min(), last_quarter, freq="Q-DEC")
series = actual.reindex(quarters)
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(quarters.to_timestamp(), series, label="Reported production")

if forecast is not None:
    forecast_dates = [forecast_quarter.to_timestamp()]
    forecast_values = [forecast]
    if pd.notna(series.iloc[-1]):
        forecast_dates.insert(0, last_quarter.to_timestamp())
        forecast_values.insert(0, series.iloc[-1])
    ax.plot(forecast_dates, forecast_values, "o--", color="darkorange", label="Forecast")

ax.set_title(f"{crop} production in {province.title()}")
ax.set_ylabel("Production (metric tons)")
ax.set_xlabel("Year")
ax.grid(alpha=0.2)
ax.legend()
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)
