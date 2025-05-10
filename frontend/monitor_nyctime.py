import sys
from pathlib import Path
import pytz
from datetime import datetime

parent_dir = str(Path(__file__).parent.parent)
sys.path.append(parent_dir)
import os
os.environ['SHAPE_RESTORE_SHX'] = 'YES'
import pandas as pd
import plotly.express as px
import streamlit as st

from src.inference import fetch_hourly_rides, fetch_predictions

def convert_to_est(utc_time):
    est_tz = pytz.timezone('US/Eastern')
    if utc_time.tzinfo is None:
        utc_time = pytz.utc.localize(utc_time)
    return utc_time.astimezone(est_tz)

st.title("Mean Absolute Error (MAE) by Pickup Hour")

# Get current time in EST for display
current_date = pd.Timestamp.now(tz="UTC")
current_date_est = convert_to_est(current_date)
st.header(f'{current_date_est.strftime("%Y-%m-%d %H:%M:%S")} EST')

# Sidebar for user input
st.sidebar.header("Settings")
past_hours = st.sidebar.slider(
    "Number of Past Hours to Plot",
    min_value=12,
    max_value=24 * 28,
    value=12,
    step=1,
)

# Fetch data
st.write("Fetching data for the past", past_hours, "hours...")
df1 = fetch_hourly_rides(past_hours)
df2 = fetch_predictions(past_hours)

# Convert pickup_hour to EST
df1['pickup_hour'] = df1['pickup_hour'].apply(convert_to_est)
df2['pickup_hour'] = df2['pickup_hour'].apply(convert_to_est)

# Merge the DataFrames on 'pickup_location_id' and 'pickup_hour'
merged_df = pd.merge(df1, df2, on=["pickup_location_id", "pickup_hour"])

# Calculate the absolute error
merged_df["absolute_error"] = abs(merged_df["predicted_demand"] - merged_df["rides"])

# Group by 'pickup_hour' and calculate the mean absolute error (MAE)
mae_by_hour = merged_df.groupby("pickup_hour")["absolute_error"].mean().reset_index()
mae_by_hour.rename(columns={"absolute_error": "MAE"}, inplace=True)

# Create a Plotly plot
fig = px.line(
    mae_by_hour,
    x="pickup_hour",
    y="MAE",
    title=f"Mean Absolute Error (MAE) for the Past {past_hours} Hours",
    labels={"pickup_hour": "Pickup Hour (EST)", "MAE": "Mean Absolute Error"},
    markers=True,
)

# Update x-axis to show date and time in EST
fig.update_xaxes(tickformat="%Y-%m-%d %H:%M:%S")

# Display the plot
st.plotly_chart(fig)
st.write(f'Average MAE: {mae_by_hour["MAE"].mean():.2f}')
