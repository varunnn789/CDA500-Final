import sys
from pathlib import Path
import zipfile
import os
import folium
import geopandas as gpd
import pandas as pd
import requests
import streamlit as st
from branca.colormap import LinearColormap
from streamlit_folium import st_folium
from datetime import datetime
import pytz
import plotly.graph_objs as go
import plotly.express as px

parent_dir = str(Path(__file__).parent.parent)
sys.path.append(parent_dir)
os.environ['SHAPE_RESTORE_SHX'] = 'YES'

from src.config import DATA_DIR
from src.inference import fetch_next_hour_predictions, load_batch_of_features_from_store
from src.plot_utils import plot_prediction

# Function to convert UTC time to EST
def convert_to_est(utc_time):
    est_tz = pytz.timezone('US/Eastern')
    if utc_time.tzinfo is None:
        utc_time = pytz.utc.localize(utc_time)
    return utc_time.astimezone(est_tz)

if "map_created" not in st.session_state:
    st.session_state.map_created = False

def create_taxi_map(shapefile_path, prediction_data, selected_location=None):
    nyc_zones = gpd.read_file(shapefile_path)
    nyc_zones = nyc_zones.merge(
        prediction_data[["pickup_location_id", "predicted_demand"]],
        left_on="LocationID",
        right_on="pickup_location_id",
        how="left"
    )
    nyc_zones["predicted_demand"] = nyc_zones["predicted_demand"].fillna(0)
    nyc_zones = nyc_zones.to_crs(epsg=4326)

    m = folium.Map(location=[40.7128, -74.0060], zoom_start=10, tiles="cartodbpositron")
    colormap = LinearColormap(
        colors=["#FFEDA0", "#FED976", "#FEB24C", "#FD8D3C", "#FC4E2A", "#E31A1C", "#BD0026"],
        vmin=nyc_zones["predicted_demand"].min(),
        vmax=nyc_zones["predicted_demand"].max()
    )
    colormap.add_to(m)

    def style_function(feature):
        location_id = feature['properties']['LocationID']
        fill_color = colormap(float(feature["properties"].get("predicted_demand", 0)))

        # Highlight the selected location
        if selected_location is not None and location_id == selected_location:
            return {
                "fillColor": fill_color,
                "color": "green",  # Highlight color
                "weight": 3,         # Thicker border
                "fillOpacity": 0.9,   # More opaque
            }
        else:
            return {
                "fillColor": fill_color,
                "color": "black",
                "weight": 1,
                "fillOpacity": 0.7,
            }


    zones_json = nyc_zones.to_json()
    folium.GeoJson(
        zones_json,
        style_function=style_function,
        tooltip=folium.GeoJsonTooltip(
            fields=["zone", "predicted_demand"],
            aliases=["Zone:", "Predicted Demand:"],
            style="background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;"
        )
    ).add_to(m)

    st.session_state.map_obj = m
    st.session_state.map_created = True
    return m

def load_shape_data_file(data_dir, url="https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip", log=True):
    """
    Downloads, extracts, and loads a shapefile as a GeoDataFrame.

    Parameters:
        data_dir (str or Path): Directory where the data will be stored.
        url (str): URL of the shapefile zip file.
        log (bool): Whether to log progress messages.

    Returns:
        GeoDataFrame: The loaded shapefile as a GeoDataFrame.
    """
    # Ensure data directory exists
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Define file paths
    zip_path = data_dir / "taxi_zones.zip"
    extract_path = data_dir / "taxi_zones"
    shapefile_path = extract_path / "taxi_zones.shp"

    # Download the file if it doesn't already exist
    if not zip_path.exists():
        if log:
            print(f"Downloading file from {url}...")
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()  # Raise an HTTPError for bad responses
            with open(zip_path, "wb") as f:
                f.write(response.content)
            if log:
                print(f"File downloaded and saved to {zip_path}")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to download file from {url}: {e}")
    else:
        if log:
            print(f"File already exists at {zip_path}, skipping download.")

    # Extract the zip file if the shapefile doesn't already exist
    if not shapefile_path.exists():
        if log:
            print(f"Extracting files to {extract_path}...")
        try:
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(extract_path)
            if log:
                print(f"Files extracted to {extract_path}")
        except zipfile.BadZipFile as e:
            raise Exception(f"Failed to extract zip file {zip_path}: {e}")
    else:
        if log:
            print(f"Shapefile already exists at {shapefile_path}, skipping extraction.")

    # Load and return the shapefile as a GeoDataFrame
    if log:
        print(f"Loading shapefile from {shapefile_path}...")
    try:
        gdf = gpd.read_file(shapefile_path).to_crs("epsg:4326")
        if log:
            print("Shapefile successfully loaded.")
        return gdf
    except Exception as e:
        raise Exception(f"Failed to load shapefile {shapefile_path}: {e}")

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

st.title(f"New York Yellow Taxi Cab Demand Next Hour")
st.header(f'{current_date_est.strftime("%Y-%m-%d %H:%M:%S")} EST')

with st.spinner(text="Download shape file for taxi zones"):
    geo_df = load_shape_data_file(DATA_DIR)

with st.spinner(text="Fetching batch of inference data"):
    features = load_batch_of_features_from_store(current_date)
    if 'pickup_hour' in features.columns:
        features['pickup_hour'] = features['pickup_hour'].apply(convert_to_est)
        features['pickup_hour'] = features['pickup_hour'].dt.strftime('%Y-%m-%d %H:%M:%S')

with st.spinner(text="Fetching predictions"):
    predictions = fetch_next_hour_predictions()
    if 'pickup_hour' in predictions.columns:
        predictions['pickup_hour'] = predictions['pickup_hour'].apply(convert_to_est)
        predictions['pickup_hour'] = predictions['pickup_hour'].dt.strftime('%Y-%m-%d %H:%M:%S')

    # Merge zone names with predictions
    predictions = pd.merge(
        predictions,
        geo_df[["LocationID", "zone"]],
        left_on="pickup_location_id",
        right_on="LocationID",
        how="left",
    )

shapefile_path = DATA_DIR / "taxi_zones" / "taxi_zones.shp"

# Sidebar: Location selection
with st.sidebar:
    locations = sorted(predictions['pickup_location_id'].unique())
    selected_location = st.selectbox("Select a location:", locations)
    
    # Button to remove dropdown filter
    if st.button("Remove Dropdown Filter"):
        selected_location = None

# Filter data based on the selected location
if selected_location:
    filtered_predictions = predictions[predictions['pickup_location_id'] == selected_location].copy()
    filtered_features = features[features['pickup_location_id'] == selected_location].copy()
else:
    filtered_predictions = predictions.copy()
    filtered_features = features.copy()

with st.spinner(text="Plot predicted rides demand"):
    if predictions is None or predictions.empty:
        st.warning("No prediction data available.")
    else:
        st.subheader("Taxi Ride Predictions Map")
        map_obj = create_taxi_map(shapefile_path, predictions, selected_location)

        if st.session_state.map_created:
            st_folium(st.session_state.map_obj, width=800, height=600, returned_objects=[])

        st.subheader("Prediction Statistics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average Rides", f"{filtered_predictions['predicted_demand'].mean():.0f}")
        with col2:
            st.metric("Maximum Rides", f"{filtered_predictions['predicted_demand'].max():.0f}")
        with col3:
            st.metric("Minimum Rides", f"{filtered_predictions['predicted_demand'].min():.0f}")

        st.dataframe(filtered_predictions[["pickup_location_id", "predicted_demand"]]) #Table with all predictions, removed Zone

        st.subheader("Time Series for Top Locations")

        # Default chart: Time series for the top 2 locations
        if not selected_location:
            top2_location_ids = filtered_predictions.sort_values("predicted_demand", ascending=False).head(2)["pickup_location_id"].to_list()
            for location_id in top2_location_ids:
                location_data = filtered_predictions[filtered_predictions["pickup_location_id"] == location_id].copy()

                fig = px.line(location_data, x='pickup_hour', y='predicted_demand',
                              title=f"Pickup Hour: {location_data['pickup_hour'].iloc[-1]}, Location ID: {location_id}",
                              labels={'pickup_hour': 'Time', 'predicted_demand': 'Ride Counts'})
                fig.add_trace(go.Scatter(x=[location_data['pickup_hour'].iloc[-1]],
                                         y=[location_data['predicted_demand'].iloc[-1]],
                                         mode='markers', marker=dict(color='red', symbol='x', size=10),
                                         name='Prediction'))

                st.plotly_chart(fig, theme="streamlit", use_container_width=True)
        else:
            # Time series plot for the selected location
            location_data = filtered_predictions[filtered_predictions["pickup_location_id"] == selected_location].copy()
            fig = px.line(location_data, x='pickup_hour', y='predicted_demand',
                          title=f"Pickup Hour: {location_data['pickup_hour'].iloc[-1]}, Location ID: {selected_location}",
                          labels={'pickup_hour': 'Time', 'predicted_demand': 'Ride Counts'})
            fig.add_trace(go.Scatter(x=[location_data['pickup_hour'].iloc[-1]],
                                     y=[location_data['predicted_demand'].iloc[-1]],
                                     mode='markers', marker=dict(color='red', symbol='x', size=10),
                                     name='Prediction'))

            st.plotly_chart(fig, theme="streamlit", use_container_width=True)

        # Add name column for top 10 pickup locations
        top10 = filtered_predictions.sort_values("predicted_demand", ascending=False).head(10)
        top10 = top10.rename(columns={"zone": "Location Name"})  # Rename 'zone' column to 'Location Name'
        st.subheader("Top 10 Pickup Locations")
        st.dataframe(top10[["pickup_location_id", "Location Name", "predicted_demand"]])
