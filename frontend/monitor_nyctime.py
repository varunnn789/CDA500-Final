import sys
from pathlib import Path
import pytz
from datetime import datetime, timedelta

parent_dir = str(Path(__file__).parent.parent)
sys.path.append(parent_dir)
import os
import pandas as pd
import plotly.express as px
import streamlit as st

from src.inference import fetch_hourly_rides, get_feature_store, fetch_predictions

def convert_to_est(utc_time):
    est_tz = pytz.timezone('US/Eastern')
    if utc_time.tzinfo is None:
        utc_time = pytz.utc.localize(utc_time)
    return utc_time.astimezone(est_tz)

# Custom CSS for beautification (matching frontend_v3.py)
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    h1 {
        color: #1E3A8A;
        font-size: 36px;
        text-align: center;
    }
    h2 {
        color: #2563EB;
        font-size: 24px;
    }
    .stSlider {
        background-color: #FFFFFF;
        border-radius: 5px;
        padding: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# Get current time in EST for display
current_date = pd.Timestamp.now(tz="UTC")
current_date_est = convert_to_est(current_date)
st.title("Citi Bike Demand Prediction - MAE Monitoring")
st.header(f'Current Time: {current_date_est.strftime("%Y-%m-%d %H:%M:%S")} EST')

# Define the list of models and corresponding feature group names
models = [
    "baseline_previous_hour",
    "lightgbm_28days_lags",
    "lightgbm_top10_features",
    "gradient_boosting_temporal_features",
    "lightgbm_enhanced_lags_cyclic_temporal_interactions"
]

prediction_feature_groups = {
    "baseline_previous_hour": "predictions_model_baseline",
    "lightgbm_28days_lags": "predictions_model_lgbm_28days",
    "lightgbm_top10_features": "predictions_model_lgbm_top10",
    "gradient_boosting_temporal_features": "predictions_model_gbt",
    "lightgbm_enhanced_lags_cyclic_temporal_interactions": "predictions_model_lgbm_enhanced"
}

# Map technical model names to user-friendly display names
model_display_names = {
    "baseline_previous_hour": "Baseline Previous Hour",
    "lightgbm_28days_lags": "LightGBM 28 Days Lags",
    "lightgbm_top10_features": "LightGBM Top 10 Features",
    "gradient_boosting_temporal_features": "Gradient Boosting Temporal Features",
    "lightgbm_enhanced_lags_cyclic_temporal_interactions": "LightGBM Enhanced Lags Cyclic Temporal Interactions"
}

# Sidebar for user input
st.sidebar.header("Settings")
past_hours = st.sidebar.slider(
    "Number of Past Hours to Plot",
    min_value=12,
    max_value=24 * 28,
    value=12,
    step=1,
)

# Fetch actual rides data
st.write(f"Fetching actual rides data for the past {past_hours} hours...")
df_actual = fetch_hourly_rides(past_hours)
df_actual['pickup_hour'] = df_actual['pickup_hour'].apply(convert_to_est)

# Fetch predictions for all models and calculate MAE
mae_dataframes = []
for model_name in models:
    st.write(f"Fetching predictions for {model_display_names[model_name]} for the past {past_hours} hours...")
    df_pred = fetch_predictions(past_hours, prediction_feature_groups[model_name])
    df_pred['pickup_hour'] = df_pred['pickup_hour'].apply(convert_to_est)

    # Merge actual and predicted data
    merged_df = pd.merge(df_actual, df_pred, on=["start_station_name", "pickup_hour"], how="inner")

    # Calculate the absolute error
    merged_df["absolute_error"] = abs(merged_df["predicted_rides"] - merged_df["rides"])

    # Group by 'pickup_hour' and calculate the mean absolute error (MAE)
    mae_by_hour = merged_df.groupby("pickup_hour")["absolute_error"].mean().reset_index()
    mae_by_hour.rename(columns={"absolute_error": "MAE"}, inplace=True)
    mae_by_hour["Model"] = model_display_names[model_name]
    mae_dataframes.append(mae_by_hour)

# Concatenate MAE data for all models
if mae_dataframes:
    all_mae_df = pd.concat(mae_dataframes, ignore_index=True)

    # Create a Plotly plot with all models
    fig = px.line(
        all_mae_df,
        x="pickup_hour",
        y="MAE",
        color="Model",
        title=f"Mean Absolute Error (MAE) for All Models - Past {past_hours} Hours",
        labels={"pickup_hour": "Pickup Hour (EST)", "MAE": "Mean Absolute Error"},
        line_shape='linear',
        markers=True,
    )

    # Update x-axis to show date and time in EST
    fig.update_xaxes(tickformat="%Y-%m-%d %H:%M:%S")

    # Display the plot
    st.plotly_chart(fig)

    # Calculate and display average MAE for each model
    st.subheader("Average MAE for Each Model")
    avg_mae = all_mae_df.groupby("Model")["MAE"].mean().reset_index()
    avg_mae["MAE"] = avg_mae["MAE"].round(2)
    st.dataframe(avg_mae)
else:
    st.warning("No data available to display MAE trends.")