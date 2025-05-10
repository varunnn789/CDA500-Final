from datetime import datetime, timedelta, timezone

import hopsworks
import numpy as np
import pandas as pd
import pytz
from hsfs.feature_store import FeatureStore
from sklearn.feature_selection import SelectKBest, f_regression

import src.config as config
from src.data_utils import transform_ts_data_info_features

def get_hopsworks_project() -> hopsworks.project.Project:
    return hopsworks.login(
        project=config.HOPSWORKS_PROJECT_NAME, api_key_value=config.HOPSWORKS_API_KEY
    )

def get_feature_store() -> FeatureStore:
    project = get_hopsworks_project()
    return project.get_feature_store()

# Define BaselineModelPreviousHour class
class BaselineModelPreviousHour:
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        pass

    def predict(self, X_test: pd.DataFrame) -> np.array:
        return X_test["rides_t-1"]

def get_model_predictions(model, features: pd.DataFrame, numeric_features: pd.DataFrame, model_name: str) -> pd.DataFrame:
    # Preprocess features based on the model
    if model_name == "baseline_previous_hour":
        # Baseline model only needs rides_t-1
        if "rides_t-1" not in features.columns:
            raise ValueError("rides_t-1 not found in features for baseline_previous_hour model")
        predictions = model.predict(features)

    elif model_name == "lightgbm_28days_lags":
        # LightGBM 28 Days Lags uses all 672 lagged features
        past_ride_columns = [c for c in numeric_features.columns if c.startswith("rides_")]
        if len(past_ride_columns) != 672:
            raise ValueError(f"Expected 672 lagged features, found {len(past_ride_columns)}")
        features_model = numeric_features[past_ride_columns]
        predictions = model.predict(features_model)

    elif model_name == "lightgbm_top10_features":
        # LightGBM Top 10 Features uses the same 10 features selected during training
        # We need to apply the same feature selection (this is a simplification; ideally, we should save the selector)
        selector = SelectKBest(score_func=f_regression, k=10)
        past_ride_columns = [c for c in numeric_features.columns if c.startswith("rides_")]
        # For simplicity, we assume the same features are selected (in practice, we should save the selector)
        selected_features = ['rides_t-672', 'rides_t-504', 'rides_t-360', 'rides_t-336', 'rides_t-312',
                            'rides_t-192', 'rides_t-168', 'rides_t-144', 'rides_t-24', 'rides_t-1']
        features_model = numeric_features[selected_features]
        predictions = model.predict(features_model)

    elif model_name == "gradient_boosting_temporal_features":
        # Gradient Boosting uses lagged features + temporal features
        past_ride_columns = [c for c in numeric_features.columns if c.startswith("rides_")]
        features_model = numeric_features[past_ride_columns].copy()
        features_model["hour"] = features["pickup_hour"].dt.hour
        features_model["day_of_week"] = features["pickup_hour"].dt.dayofweek
        features_model["month"] = features["pickup_hour"].dt.month
        predictions = model.predict(features_model)

    elif model_name == "lightgbm_enhanced_lags_cyclic_temporal_interactions":
        # LightGBM Enhanced uses 7 days of lags + engineered features
        lag_columns_7_days = [f"rides_t-{i}" for i in range(1, 7*24 + 1)]
        lag_columns_7_days = [col for col in lag_columns_7_days if col in numeric_features.columns]
        features_model = numeric_features[lag_columns_7_days].copy()
        features_model["hour"] = features["pickup_hour"].dt.hour
        features_model["day_of_week"] = features["pickup_hour"].dt.dayofweek
        features_model["month"] = features["pickup_hour"].dt.month
        features_model["hour_sin"] = np.sin(2 * np.pi * features_model["hour"] / 24)
        features_model["hour_cos"] = np.cos(2 * np.pi * features_model["hour"] / 24)
        features_model["day_of_week_sin"] = np.sin(2 * np.pi * features_model["day_of_week"] / 7)
        features_model["day_of_week_cos"] = np.cos(2 * np.pi * features_model["day_of_week"] / 7)
        features_model["month_sin"] = np.sin(2 * np.pi * features_model["month"] / 12)
        features_model["month_cos"] = np.cos(2 * np.pi * features_model["month"] / 12)
        features_model["is_winter"] = features["pickup_hour"].dt.month.isin([12, 1, 2]).astype(int)
        features_model["is_weekend"] = features["pickup_hour"].dt.dayofweek.isin([5, 6]).astype(int)
        holiday_dates = [(12, 25), (12, 31), (1, 1)]
        features_model["is_holiday"] = features["pickup_hour"].apply(
            lambda x: 1 if (x.month, x.day) in holiday_dates else 0
        )
        trend_lags = [f"rides_t-{i}" for i in range(1, 7*24 + 1)]
        trend_lags = [col for col in lag_columns_7_days if col in features_model.columns]
        features_model["trend_7d"] = features_model[trend_lags].mean(axis=1)
        features_model["rides_t1_hour"] = features_model["rides_t-1"] * features_model["hour"]
        features_model["rides_t24_day_of_week"] = features_model["rides_t-24"] * features_model["day_of_week"]
        predictions = model.predict(features_model)

    else:
        raise ValueError(f"Unknown model name: {model_name}")

    results = pd.DataFrame()
    results["start_station_name"] = features["start_station_name"].values
    results["predicted_rides"] = predictions.round(0)

    return results

def load_batch_of_features_from_store(
    current_date: datetime,
) -> pd.DataFrame:
    feature_store = get_feature_store()

    # Retrieve the feature group with recent data
    feature_group = feature_store.get_feature_group(
        name="recent_time_series_hourly_feature_group",
        version=1
    )

    # Create or retrieve the feature view for recent data
    try:
        feature_store.create_feature_view(
            name="citi_bike_recent_hourly_feature_view",
            version=1,
            query=feature_group.select_all(),
        )
        print(f"Feature view 'citi_bike_recent_hourly_feature_view' (version 1) created successfully.")
    except Exception as e:
        print(f"Feature view creation skipped: {e}")

    feature_view = feature_store.get_feature_view(
        name="citi_bike_recent_hourly_feature_view",
        version=1
    )

    # Read time-series data from the feature store
    fetch_data_to = current_date - timedelta(hours=1)
    fetch_data_from = current_date - timedelta(days=29)
    print(f"Fetching data from {fetch_data_from} to {fetch_data_to}")

    ts_data = feature_view.get_batch_data(
        start_time=(fetch_data_from - timedelta(days=1)),
        end_time=(fetch_data_to + timedelta(days=1)),
    )
    ts_data = ts_data[ts_data.pickup_hour.between(fetch_data_from, fetch_data_to)]

    # Sort data by station and time
    ts_data.sort_values(by=["start_station_name", "pickup_hour"], inplace=True)

    features = transform_ts_data_info_features(
        ts_data, window_size=24 * 28, step_size=23
    )

    return features

def load_model_from_registry(model_name=None):
    from pathlib import Path
    import joblib

    project = get_hopsworks_project()
    model_registry = project.get_model_registry()

    if model_name is None:
        raise ValueError("model_name must be specified to load a model from the registry.")

    # Load the specified model
    model = model_registry.get_model(
        name=model_name,
        version=1  # Assuming version 1; we can make this configurable if needed
    )
    model_dir = model.download()
    model = joblib.load(Path(model_dir) / f"{model_name}.pkl")

    return model

def load_metrics_from_registry(model_name=None):
    project = get_hopsworks_project()
    model_registry = project.get_model_registry()

    if model_name is None:
        raise ValueError("model_name must be specified to load metrics from the registry.")

    model = model_registry.get_model(
        name=model_name,
        version=1
    )

    return model.training_metrics

def fetch_next_hour_predictions(feature_group_name):
    # Get current UTC time and round up to next hour
    now = datetime.now(tz=pytz.UTC)
    next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)

    fs = get_feature_store()
    fg = fs.get_feature_group(name=feature_group_name, version=1)
    if fg is None:
        raise ValueError(f"Feature group {feature_group_name} not found in the feature store.")
    df = fg.read()
    if df.empty:
        return df, None
    # Filter for the exact next hour
    df_next = df[df["pickup_hour"] == next_hour]
    if df_next.empty:
        return df_next, None
    return df_next, next_hour

def fetch_predictions(hours, feature_group_name):
    current_hour = (pd.Timestamp.now(tz="Etc/UTC") - timedelta(hours=hours)).floor("h")

    fs = get_feature_store()
    fg = fs.get_feature_group(name=feature_group_name, version=1)
    if fg is None:
        raise ValueError(f"Feature group {feature_group_name} not found in the feature store.")

    df = fg.filter(fg.pickup_hour >= current_hour).read()
    # Sort by pickup_hour in ascending order
    df = df.sort_values("pickup_hour", ascending=True)

    return df

def fetch_hourly_rides(hours):
    current_hour = (pd.Timestamp.now(tz="Etc/UTC") - timedelta(hours=hours)).floor("h")

    fs = get_feature_store()
    fg = fs.get_feature_group(name=config.FEATURE_GROUP_NAME, version=1)

    query = fg.select_all()
    query = query.filter(fg.pickup_hour >= current_hour)

    df = query.read()
    # Sort by pickup_hour in ascending order
    df = df.sort_values("pickup_hour", ascending=True)

    return df

def fetch_days_data(days):
    current_date = pd.to_datetime(datetime.now(timezone.utc))
    fetch_data_from = current_date - timedelta(days=(365 + days))
    fetch_data_to = current_date - timedelta(days=365)
    print(fetch_data_from, fetch_data_to)
    fs = get_feature_store()
    fg = fs.get_feature_group(name=config.FEATURE_GROUP_NAME, version=1)

    query = fg.select_all()
    df = query.read()
    cond = (df["pickup_hour"] >= fetch_data_from) & (df["pickup_hour"] <= fetch_data_to)
    return df[cond]