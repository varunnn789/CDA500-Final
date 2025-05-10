import sys
from pathlib import Path
import os
import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium
from datetime import datetime, timedelta
import pytz
import plotly.express as px
import plotly.graph_objs as go

parent_dir = str(Path(__file__).parent.parent)
sys.path.append(parent_dir)

from src.config import DATA_DIR
from src.inference import get_feature_store

# Function to convert UTC time to EST
def convert_to_est(utc_time):
    est_tz = pytz.timezone('US/Eastern')
    if utc_time.tzinfo is None:
        utc_time = pytz.utc.localize(utc_time)
    return utc_time.astimezone(est_tz)

# Function to fetch station coordinates (assumed to be in a static file or feature store)
def fetch_station_coordinates():
    # Placeholder: Fetch coordinates from a static file or feature store
    try:
        # Example: Load from a static CSV file in DATA_DIR
        stations_file = DATA_DIR / "citi_bike_stations.csv"
        if stations_file.exists():
            stations_df = pd.read_csv(stations_file)
            return stations_df[["start_station_name", "latitude", "longitude"]]
        else:
            # Fallback: Fetch from feature store
            fs = get_feature_store()
            fg = fs.get_feature_group(name="recent_time_series_hourly_feature_group", version=1)
            df = fg.read()
            if "latitude" in df.columns and "longitude" in df.columns:
                stations_df = df[["start_station_name", "latitude", "longitude"]].drop_duplicates()
                return stations_df
            else:
                raise ValueError("Station coordinates not found in feature store or static file.")
    except Exception as e:
        st.error(f"Error fetching station coordinates: {e}")
        return None

# Function to create a map with station markers colored by predicted rides
def create_bike_map(predictions, stations_df, selected_station=None):
    if stations_df is None or stations_df.empty:
        return None

    # Merge predictions with station coordinates
    merged_df = pd.merge(
        predictions[["start_station_name", "predicted_rides"]],
        stations_df,
        on="start_station_name",
        how="left"
    )
    merged_df = merged_df.dropna(subset=["latitude", "longitude"])  # Drop stations without coordinates
    merged_df["predicted_rides"] = merged_df["predicted_rides"].fillna(0)

    # Create a Folium map centered on NYC
    m = folium.Map(location=[40.7128, -74.0060], zoom_start=12, tiles="cartodbpositron")

    # Create a color scale for predicted rides
    min_rides = merged_df["predicted_rides"].min()
    max_rides = merged_df["predicted_rides"].max()
    if max_rides == min_rides:
        max_rides = min_rides + 1  # Avoid division by zero
    color_scale = lambda x: f"#{int(255 * (x - min_rides) / (max_rides - min_rides)):02x}00{int(255 * (1 - (x - min_rides) / (max_rides - min_rides))):02x}"

    # Add markers for each station
    for _, row in merged_df.iterrows():
        station_name = row["start_station_name"]
        predicted_rides = row["predicted_rides"]
        lat = row["latitude"]
        lon = row["longitude"]
        color = color_scale(predicted_rides)
        border_color = "green" if selected_station == station_name else "black"
        border_weight = 3 if selected_station == station_name else 1

        folium.CircleMarker(
            location=[lat, lon],
            radius=8,
            color=border_color,
            weight=border_weight,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            tooltip=f"Station: {station_name}<br>Predicted Rides: {predicted_rides:.0f}",
        ).add_to(m)

    return m

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
st.header(f'{current_date_est.strftime("%Y-%m-%d %H:%M:%S")} EST')

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

# Sidebar: Model and station selection
with st.sidebar:
    st.header("Settings")
    selected_model = st.selectbox("Select Model:", models, index=0)
    feature_group_name = prediction_feature_groups[selected_model]

    # Fetch stations for selection
    with st.spinner("Fetching station data..."):
        stations_df = fetch_station_coordinates()
        if stations_df is not None:
            stations = sorted(stations_df["start_station_name"].unique())
        else:
            stations = []

    selected_station = st.selectbox("Select a Station:", ["All Stations"] + stations)

    # Button to remove station filter
    if st.button("Remove Station Filter"):
        selected_station = "All Stations"

# Fetch predictions for the selected model
with st.spinner(f"Fetching predictions for {selected_model}..."):
    predictions = fetch_next_hour_predictions(feature_group_name)
    if predictions is not None and not predictions.empty:
        predictions['pickup_hour'] = predictions['pickup_hour'].apply(convert_to_est)
        predictions['pickup_hour'] = predictions['pickup_hour'].dt.strftime('%Y-%m-%d %H:%M:%S')
    else:
        st.warning("No predictions available for the next hour.")
        predictions = pd.DataFrame()

# Filter predictions based on the selected station
if selected_station != "All Stations":
    filtered_predictions = predictions[predictions['start_station_name'] == selected_station].copy()
else:
    filtered_predictions = predictions.copy()

# Display map
if stations_df is not None and not filtered_predictions.empty:
    st.subheader("Citi Bike Demand Predictions Map")
    map_obj = create_bike_map(filtered_predictions, stations_df, selected_station if selected_station != "All Stations" else None)
    if map_obj:
        st_folium(map_obj, width=800, height=600, returned_objects=[])
    else:
        st.warning("Unable to display map due to missing station coordinates.")
else:
    st.warning("Map cannot be displayed: No predictions or station coordinates available.")

# Display prediction statistics
if not filtered_predictions.empty:
    st.subheader("Prediction Statistics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Average Predicted Rides", f"{filtered_predictions['predicted_rides'].mean():.0f}")
    with col2:
        st.metric("Maximum Predicted Rides", f"{filtered_predictions['predicted_rides'].max():.0f}")
    with col3:
        st.metric("Minimum Predicted Rides", f"{filtered_predictions['predicted_rides'].min():.0f}")

    st.dataframe(filtered_predictions[["start_station_name", "predicted_rides"]])

    st.subheader("Predicted Demand for Top Stations")
    if selected_station == "All Stations":
        top_stations = filtered_predictions.sort_values("predicted_rides", ascending=False).head(2)
        for _, row in top_stations.iterrows():
            station_name = row["start_station_name"]
            location_data = filtered_predictions[filtered_predictions["start_station_name"] == station_name].copy()
            fig = px.line(
                location_data,
                x='pickup_hour',
                y='predicted_rides',
                title=f"Station: {station_name}, Pickup Hour: {location_data['pickup_hour'].iloc[-1]}",
                labels={'pickup_hour': 'Time (EST)', 'predicted_rides': 'Predicted Rides'}
            )
            fig.add_trace(go.Scatter(
                x=[location_data['pickup_hour'].iloc[-1]],
                y=[location_data['predicted_rides'].iloc[-1]],
                mode='markers',
                marker=dict(color='red', symbol='x', size=10),
                name='Prediction'
            ))
            st.plotly_chart(fig, theme="streamlit", use_container_width=True)
    else:
        location_data = filtered_predictions[filtered_predictions["start_station_name"] == selected_station].copy()
        fig = px.line(
            location_data,
            x='pickup_hour',
            y='predicted_rides',
            title=f"Station: {selected_station}, Pickup Hour: {location_data['pickup_hour'].iloc[-1]}",
            labels={'pickup_hour': 'Time (EST)', 'predicted_rides': 'Predicted Rides'}
        )
        fig.add_trace(go.Scatter(
            x=[location_data['pickup_hour'].iloc[-1]],
            y=[location_data['predicted_rides'].iloc[-1]],
            mode='markers',
            marker=dict(color='red', symbol='x', size=10),
            name='Prediction'
        ))
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)

    # Top 10 stations
    top10 = filtered_predictions.sort_values("predicted_rides", ascending=False).head(10)
    st.subheader("Top 10 Stations by Predicted Rides")
    st.dataframe(top10[["start_station_name", "predicted_rides"]])
else:
    st.warning("No predictions available to display.")