from datetime import datetime, timedelta

import pandas as pd

import src.config as config
from src.inference import (
    get_feature_store,
    get_model_predictions,
    load_model_from_registry,
)

# Get the current datetime
current_date = pd.Timestamp.now(tz="Etc/UTC")
feature_store = get_feature_store()

# Read time-series data from the feature store
fetch_data_to = current_date.floor('h')  # Include the current hour
fetch_data_from = current_date - timedelta(days=1 * 29)
print(f"Fetching data from {fetch_data_from} to {fetch_data_to}")

feature_view = feature_store.get_feature_view(
    name="citi_bike_recent_hourly_feature_view",
    version=1
)

ts_data = feature_view.get_batch_data(
    start_time=(fetch_data_from - timedelta(days=1)),
    end_time=(fetch_data_to + timedelta(days=1)),
)
ts_data = ts_data[ts_data.pickup_hour.between(fetch_data_from, fetch_data_to)]
ts_data.sort_values(["start_station_name", "pickup_hour"]).reset_index(drop=True)
ts_data["pickup_hour"] = ts_data["pickup_hour"].dt.tz_localize(None)

from src.data_utils import transform_ts_data_info_features

# Transform the data into features
features = transform_ts_data_info_features(ts_data, window_size=24 * 28, step_size=23)

# Sort features by pickup_hour in descending order to select the most recent hour
features.sort_values("pickup_hour", ascending=False, inplace=True)

# Filter features for the most recent hour
features_next_hour = features.groupby("start_station_name").first().reset_index()
recent_hour = features_next_hour["pickup_hour"].max()
print(f"Most recent hour in features_next_hour: {recent_hour}")

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

# Make predictions with each model and save to separate feature groups
for model_name in models:
    print(f"\nProcessing model: {model_name}")
    
    # Load the model
    model = load_model_from_registry(model_name=model_name)
    
    # Preprocess features: Drop non-numeric columns for models that require numeric input
    features_for_model = features_next_hour.copy()
    if model_name != "baseline_previous_hour":  # baseline_previous_hour handles features differently
        features_for_model = features_for_model.drop(columns=["pickup_hour", "start_station_name"])
    
    # Make predictions for the most recent hour
    predictions = get_model_predictions(model, features_for_model, model_name=model_name)
    predictions["pickup_hour"] = recent_hour  # Use the most recent actual hour
    
    # Create or retrieve the feature group for this model's predictions
    feature_group = feature_store.get_or_create_feature_group(
        name=prediction_feature_groups[model_name],
        version=1,
        description=f"Predictions from {model_name} model",
        primary_key=["start_station_name", "pickup_hour"],
        event_time="pickup_hour",
    )
    
    # Insert the predictions into the feature group
    feature_group.insert(predictions, write_options={"wait_for_job": False})
    print(f"Saved predictions for {model_name} to feature group: {prediction_feature_groups[model_name]}")