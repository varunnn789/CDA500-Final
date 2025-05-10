import joblib
from hsml.model_schema import ModelSchema
from hsml.schema import Schema
from sklearn.metrics import mean_absolute_error

import src.config as config
from src.data_utils import transform_ts_data_info_features_and_target
from src.inference import (
    fetch_days_data,
    get_hopsworks_project,
)

# Function to get the appropriate pipeline for each model
def get_pipeline(model_name):
    from sklearn.pipeline import Pipeline
    from lightgbm import LGBMRegressor
    from src.inference import BaselineModelPreviousHour

    if model_name == "baseline_previous_hour":
        return Pipeline([
            ("model", BaselineModelPreviousHour())
        ])
    elif model_name == "lightgbm_28days_lags":
        return Pipeline([
            ("model", LGBMRegressor(objective="regression", random_state=42))
        ])
    elif model_name == "lightgbm_top10_features":
        return Pipeline([
            ("model", LGBMRegressor(objective="regression", random_state=42))
        ])
    elif model_name == "gradient_boosting_temporal_features":
        from sklearn.ensemble import GradientBoostingRegressor
        return Pipeline([
            ("model", GradientBoostingRegressor(random_state=42))
        ])
    elif model_name == "lightgbm_enhanced_lags_cyclic_temporal_interactions":
        return Pipeline([
            ("model", LGBMRegressor(objective="regression", random_state=42))
        ])
    else:
        raise ValueError(f"Unknown model name: {model_name}")

print(f"Fetching data from feature store...")
ts_data = fetch_days_data(180)

print(f"Transforming ts_data into features and targets...")
features, targets = transform_ts_data_info_features_and_target(
    ts_data, window_size=24 * 28, step_size=23
)

# Define the list of models to retrain
models = [
    "baseline_previous_hour",
    "lightgbm_28days_lags",
    "lightgbm_top10_features",
    "gradient_boosting_temporal_features",
    "lightgbm_enhanced_lags_cyclic_temporal_interactions"
]

# Retrain each model and save to the model registry
project = get_hopsworks_project()
model_registry = project.get_model_registry()

for model_name in models:
    print(f"\nTraining model: {model_name}")
    
    # Get the pipeline for this model
    pipeline = get_pipeline(model_name)
    
    # Preprocess features: Drop non-numeric columns for models that require numeric input
    features_for_model = features.copy()
    if model_name != "baseline_previous_hour":  # baseline_previous_hour handles features differently
        features_for_model = features_for_model.drop(columns=["pickup_hour", "start_station_name"])
    
    # Fit the pipeline on the recent data
    pipeline.fit(features_for_model, targets)
    
    # Calculate MAE on the training data
    predictions = pipeline.predict(features_for_model)
    train_mae = mean_absolute_error(targets, predictions)
    print(f"Training MAE for {model_name}: {train_mae:.4f}")
    
    # Save the pipeline locally
    model_path = config.MODELS_DIR / f"{model_name}.pkl"
    joblib.dump(pipeline, model_path)
    
    # Define the model schema
    input_schema = Schema(features)
    output_schema = Schema(targets)
    model_schema = ModelSchema(input_schema=input_schema, output_schema=output_schema)
    
    # Determine the next available version for this model
    try:
        existing_models = model_registry.get_models(name=model_name)
        if existing_models:
            latest_version = max(model.version for model in existing_models)
            new_version = latest_version + 1
        else:
            new_version = 1
    except Exception as e:
        print(f"Error fetching existing models for {model_name}: {e}")
        new_version = 1  # Fallback to version 1 if we can't fetch existing versions
    
    print(f"Registering {model_name} with version {new_version}")
    
    # Save the model to the model registry
    model = model_registry.sklearn.create_model(
        name=model_name,
        version=new_version,
        metrics={"train_mae": train_mae},
        description=f"Retrained {model_name} model on recent 180 days of data",
        input_example=features.sample(),
        model_schema=model_schema,
    )
    model.save(str(model_path))
    print(f"Saved {model_name} to model registry with version {new_version}")