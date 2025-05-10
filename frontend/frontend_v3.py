import sys
from pathlib import Path
import os
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
import pytz
import plotly.express as px
import plotly.graph_objs as go

parent_dir = str(Path(__file__).parent.parent)
sys.path.append(parent_dir)

from src.config import DATA_DIR
from src.inference import get_feature_store, fetch_next_hour_predictions, fetch_hourly_rides, fetch_predictions

# Function to convert UTC time to EST
def convert_to_est(utc_time):
    est_tz = pytz.timezone('US/Eastern')
    if utc_time.tzinfo is None:
        utc_time = pytz.utc.localize(utc_time)
    return utc_time.astimezone(est_tz)

# Custom CSS for beautification
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
    .stSelectbox {
        background-color: #FFFFFF;
        border-radius: 5px;
        padding: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# Get current time in EST for display
current_date = pd.Timestamp.now(tz="Etc/UTC")
current_date_est = convert_to_est(current_date)

st.title("Citi Bike Demand Prediction - Next Hour")
st.header(f'Current Time: {current_date_est.strftime("%Y-%m-%d %H:%M:%S")} EST')

# Define the list of models and corresponding feature group names
model_names = [
    "baseline_previous_hour",
    "lightgbm_28days_lags",
    "lightgbm_top10_features",
    "gradient_boosting_temporal_features",
    "lightgbm_enhanced_lags_cyclic_temporal_interactions"
]

# Map technical model names to user-friendly display names
model_display_names = {
    "baseline_previous_hour": "Baseline Previous Hour",
    "lightgbm_28days_lags": "LightGBM 28 Days Lags",
    "lightgbm_top10_features": "LightGBM Top 10 Features",
    "gradient_boosting_temporal_features": "Gradient Boosting Temporal Features",
    "lightgbm_enhanced_lags_cyclic_temporal_interactions": "LightGBM Enhanced Lags Cyclic Temporal Interactions"
}

prediction_feature_groups = {
    "baseline_previous_hour": "predictions_model_baseline",
    "lightgbm_28days_lags": "predictions_model_lgbm_28days",
    "lightgbm_top10_features": "predictions_model_lgbm_top10",
    "gradient_boosting_temporal_features": "predictions_model_gbt",
    "lightgbm_enhanced_lags_cyclic_temporal_interactions": "predictions_model_lgbm_enhanced"
}

# Sidebar: Model and station selection
with st.sidebar:
    st.header("Settings")
    selected_model_display = st.selectbox("Select Model:", list(model_display_names.values()), index=0)
    # Map display name back to technical name
    selected_model = [k for k, v in model_display_names.items() if v == selected_model_display][0]
    feature_group_name = prediction_feature_groups[selected_model]

    # Fetch stations for selection (without coordinates, just names)
    with st.spinner("Fetching station data..."):
        fs = get_feature_store()
        fg = fs.get_feature_group(name="recent_time_series_hourly_feature_group", version=1)
        df = fg.read()
        stations = sorted(df["start_station_name"].unique())

    selected_station = st.selectbox("Select a Station:", ["All Stations"] + stations)

# Fetch next-hour predictions for the selected model
expected_next_hour = (current_date_est + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
with st.spinner(f"Fetching predictions for {selected_model_display}..."):
    predictions, prediction_time = fetch_next_hour_predictions(feature_group_name)
    if predictions is not None and not predictions.empty:
        predictions['pickup_hour'] = predictions['pickup_hour'].apply(convert_to_est)
        predictions['pickup_hour'] = predictions['pickup_hour'].dt.strftime('%Y-%m-%d %H:%M:%S')
        prediction_time = convert_to_est(prediction_time).strftime('%Y-%m-%d %H:%M:%S')
        if prediction_time != expected_next_hour.strftime('%Y-%m-%d %H:%M:%S'):
            st.warning(f"Predictions for the next hour ({expected_next_hour.strftime('%Y-%m-%d %H:%M:%S')} EST) are unavailable. Showing predictions for {prediction_time} EST.")
    else:
        st.warning(f"No predictions available in the feature store. Please ensure the inference pipeline is running correctly.")
        predictions = pd.DataFrame()
        prediction_time = None

# Fetch historical data (previous 2 weeks)
historical_hours = 24 * 14  # 2 weeks = 14 days = 336 hours
with st.spinner(f"Fetching historical data for the past 2 weeks..."):
    df_actual = fetch_hourly_rides(historical_hours)
    df_pred = fetch_predictions(historical_hours, feature_group_name)

    df_actual['pickup_hour'] = df_actual['pickup_hour'].apply(convert_to_est)
    df_pred['pickup_hour'] = df_pred['pickup_hour'].apply(convert_to_est)

# Merge historical actual and predicted data
historical_data = pd.merge(df_actual, df_pred, on=["start_station_name", "pickup_hour"], how="left")
historical_data['pickup_hour'] = historical_data['pickup_hour'].dt.strftime('%Y-%m-%d %H:%M:%S')

# Filter predictions and historical data based on the selected station
if selected_station != "All Stations":
    filtered_predictions = predictions[predictions['start_station_name'] == selected_station].copy()
    filtered_historical = historical_data[historical_data['start_station_name'] == selected_station].copy()
else:
    filtered_predictions = predictions.copy()
    filtered_historical = historical_data.copy()

# Display prediction statistics
if not filtered_predictions.empty:
    st.subheader("Prediction Statistics")
    st.write(f"Prediction Time: {prediction_time} EST")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Average Predicted Rides", f"{filtered_predictions['predicted_rides'].mean():.0f}")
    with col2:
        st.metric("Maximum Predicted Rides", f"{filtered_predictions['predicted_rides'].max():.0f}")
    with col3:
        st.metric("Minimum Predicted Rides", f"{filtered_predictions['predicted_rides'].min():.0f}")

    st.dataframe(filtered_predictions[["start_station_name", "predicted_rides"]])

    st.subheader("Predicted vs Actual Demand Over Past 2 Weeks")
    if selected_station == "All Stations":
        top_stations = filtered_predictions.sort_values("predicted_rides", ascending=False).head(3)
        for _, row in top_stations.iterrows():
            station_name = row["start_station_name"]
            location_historical = filtered_historical[filtered_historical["start_station_name"] == station_name].copy()
            location_pred = filtered_predictions[filtered_predictions["start_station_name"] == station_name].copy()
            
            # Plot historical actual and predicted rides
            fig = px.line(
                location_historical,
                x='pickup_hour',
                y=['rides', 'predicted_rides'],
                title=f"Station: {station_name}",
                labels={'pickup_hour': 'Time (EST)', 'value': 'Rides', 'variable': 'Data Type'},
                line_shape='linear'
            )
            # Add marker for the most recent prediction
            if not location_pred.empty:
                fig.add_trace(go.Scatter(
                    x=[location_pred['pickup_hour'].iloc[-1]],
                    y=[location_pred['predicted_rides'].iloc[-1]],
                    mode='markers',
                    marker=dict(color='red', symbol='x', size=10),
                    name='Next Hour Prediction'
                ))
            st.plotly_chart(fig, theme="streamlit", use_container_width=True)
    else:
        location_historical = filtered_historical[filtered_historical["start_station_name"] == selected_station].copy()
        location_pred = filtered_predictions[filtered_predictions["start_station_name"] == selected_station].copy()
        
        # Plot historical actual and predicted rides
        fig = px.line(
            location_historical,
            x='pickup_hour',
            y=['rides', 'predicted_rides'],
            title=f"Station: {selected_station}",
            labels={'pickup_hour': 'Time (EST)', 'value': 'Rides', 'variable': 'Data Type'},
            line_shape='linear'
        )
        # Add marker for the most recent prediction
        if not location_pred.empty:
            fig.add_trace(go.Scatter(
                x=[location_pred['pickup_hour'].iloc[-1]],
                y=[location_pred['predicted_rides'].iloc[-1]],
                mode='markers',
                marker=dict(color='red', symbol='x', size=10),
                name='Next Hour Prediction'
            ))
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)
else:
    st.warning("No predictions available to display.")