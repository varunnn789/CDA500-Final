import os
import sys
import calendar
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import requests
import zipfile
import io

from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR

def fetch_raw_trip_data(year: int, month: int) -> Path:
    url_patterns = [
        f"https://s3.amazonaws.com/tripdata/{year}{month:02}-citibike-tripdata.csv.zip",
        f"https://s3.amazonaws.com/tripdata/{year}{month:02}-citibike-tripdata.zip"
    ]

    response = None
    for url in url_patterns:
        response = requests.get(url)
        if response.status_code == 200:
            print(f"Successfully accessed URL: {url}")
            break

    if response is None or response.status_code != 200:
        raise Exception(f"None of the URLs for {year}-{month:02} are available: {url_patterns}")

    path = RAW_DATA_DIR / f"rides_{year}_{month:02}.parquet"
    
    expected_columns = [
        'ride_id', 'rideable_type', 'started_at', 'ended_at',
        'start_station_name', 'start_station_id', 'end_station_name', 'end_station_id',
        'start_lat', 'start_lng', 'end_lat', 'end_lng', 'member_casual'
    ]

    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        csv_files = [f for f in z.namelist() if f.endswith('.csv')]
        if not csv_files:
            raise Exception(f"No CSV files found in {url}")

        dfs = []
        for csv_file in csv_files:
            with z.open(csv_file) as f:
                df = pd.read_csv(f, encoding='latin1', on_bad_lines='skip')
                df = df[[col for col in expected_columns if col in df.columns]]
                dfs.append(df)

        combined_df = pd.concat(dfs, ignore_index=True)

    combined_df['start_station_id'] = combined_df['start_station_id'].astype(str).replace('nan', '')
    combined_df['end_station_id'] = combined_df['end_station_id'].astype(str).replace('nan', '')
    combined_df['started_at'] = pd.to_datetime(combined_df['started_at'], errors='coerce')
    combined_df['ended_at'] = pd.to_datetime(combined_df['ended_at'], errors='coerce')
    combined_df['start_lat'] = combined_df['start_lat'].astype(float, errors='ignore')
    combined_df['start_lng'] = combined_df['start_lng'].astype(float, errors='ignore')
    combined_df['end_lat'] = combined_df['end_lat'].astype(float, errors='ignore')
    combined_df['end_lng'] = combined_df['end_lng'].astype(float, errors='ignore')
    combined_df['member_casual'] = combined_df['member_casual'].astype(str)

    combined_df.to_parquet(path, engine="pyarrow", index=False)
    return path

def filter_citi_bike_data(rides: pd.DataFrame, year: int, month: int) -> pd.DataFrame:
    if not (1 <= month <= 12):
        raise ValueError("Month must be between 1 and 12.")
    if not isinstance(year, int) or not isinstance(month, int):
        raise ValueError("Year and month must be integers.")

    print(f"Before filtering - {year}-{month:02}: {rides.shape[0]} rows")

    start_stations = rides[['start_station_name', 'start_station_id', 'start_lat', 'start_lng']].drop_duplicates()
    end_stations = rides[['end_station_name', 'end_station_id', 'end_lat', 'end_lng']].drop_duplicates()
    start_stations.columns = ['station_name', 'station_id', 'latitude', 'longitude']
    end_stations.columns = ['station_name', 'station_id', 'latitude', 'longitude']
    station_mapping = pd.concat([start_stations, end_stations], ignore_index=True).drop_duplicates(subset=['station_name', 'station_id'])
    mapping_path = PROCESSED_DATA_DIR / "mapping.csv"
    station_mapping.to_csv(mapping_path, index=False)

    validated_rides = rides.drop(columns=[
        'ride_id', 'rideable_type', 'end_station_name', 'end_station_id',
        'start_station_id', 'start_lat', 'start_lng', 'end_lat', 'end_lng', 'member_casual'
    ])

    validated_rides = validated_rides.reset_index(drop=True)
    validated_rides["duration"] = validated_rides["ended_at"] - validated_rides["started_at"]

    start_date = pd.Timestamp(year=year, month=month, day=1)
    end_date = pd.Timestamp(year=year + (month // 12), month=(month % 12) + 1, day=1)

    duration_filter = (validated_rides["duration"] >= pd.Timedelta(minutes=1)) & (
        validated_rides["duration"] <= pd.Timedelta(hours=2)
    )
    print(f"After duration filter - {year}-{month:02}: {validated_rides[duration_filter].shape[0]} rows")

    station_filter = validated_rides['start_station_name'].notnull()
    print(f"After station filter - {year}-{month:02}: {validated_rides[station_filter].shape[0]} rows")

    date_range_filter = (validated_rides["started_at"] >= start_date) & (
        validated_rides["started_at"] < end_date
    )
    print(f"After date range filter - {year}-{month:02}: {validated_rides[date_range_filter].shape[0]} rows")

    final_filter = duration_filter & station_filter & date_range_filter

    total_records = len(validated_rides)
    valid_records = final_filter.sum()
    records_dropped = total_records - valid_records
    percent_dropped = (records_dropped / total_records) * 100

    print(f"Total records: {total_records:,}")
    print(f"Valid records: {valid_records:,}")
    print(f"Records dropped: {records_dropped:,} ({percent_dropped:.2f}%)")

    validated_rides = validated_rides[final_filter].copy()
    validated_rides.rename(columns={"started_at": "pickup_datetime"}, inplace=True)

    if validated_rides.empty:
        print(f"Warning: No valid rides found for {year}-{month:02} after filtering.")
        return validated_rides

    return validated_rides

def load_and_process_citi_bike_data(
    year: int, months: Optional[List[int]] = None
) -> pd.DataFrame:
    if months is None:
        months = list(range(1, 13))

    monthly_rides = []
    for month in months:
        file_path = RAW_DATA_DIR / f"rides_{year}_{month:02}.parquet"

        try:
            if not file_path.exists():
                print(f"Downloading data for {year}-{month:02}...")
                fetch_raw_trip_data(year, month)
                print(f"Successfully downloaded data for {year}-{month:02}.")
            else:
                print(f"File already exists for {year}-{month:02}.")

            print(f"Loading data for {year}-{month:02}...")
            rides = pd.read_parquet(file_path, engine="pyarrow")
            rides['started_at'] = pd.to_datetime(rides['started_at'])
            rides['ended_at'] = pd.to_datetime(rides['ended_at'])

            rides = filter_citi_bike_data(rides, year, month)
            print(f"Successfully processed data for {year}-{month:02}.")

            if not rides.empty:
                monthly_rides.append(rides)

        except FileNotFoundError:
            print(f"File not found for {year}-{month:02}. Skipping...")
        except Exception as e:
            print(f"Error processing data for {year}-{month:02}: {str(e)}")
            continue

    if not monthly_rides:
        raise Exception(
            f"No data could be loaded for the year {year} and specified months: {months}"
        )

    print("Combining all monthly data...")
    combined_rides = pd.concat(monthly_rides, ignore_index=True)
    print("Data loading and processing complete!")
    return combined_rides

def fill_missing_rides_full_range(df, hour_col, station_col, rides_col):
    if df.empty:
        return df

    df[hour_col] = pd.to_datetime(df[hour_col])
    full_hours = pd.date_range(start=df[hour_col].min(), end=df[hour_col].max(), freq="h")
    all_stations = df[station_col].unique()

    full_combinations = pd.DataFrame(
        [(hour, station) for hour in full_hours for station in all_stations],
        columns=[hour_col, station_col],
    )

    merged_df = pd.merge(full_combinations, df, on=[hour_col, station_col], how='left')
    merged_df[rides_col] = merged_df[rides_col].fillna(0).astype(int)

    return merged_df.sort_values([station_col, hour_col]).reset_index(drop=True)

def transform_raw_data_into_ts_data(rides: pd.DataFrame) -> pd.DataFrame:
    if rides.empty:
        raise ValueError("Input DataFrame is empty. Cannot transform into time series.")

    print("Inspecting rides DataFrame before transformation:")
    print(rides.head())
    print(f"Number of rows in rides: {rides.shape[0]}")
    print(f"pickup_datetime range: {rides['pickup_datetime'].min()} to {rides['pickup_datetime'].max()}")
    print(f"Unique start_station_name values: {rides['start_station_name'].nunique()}")
    print(f"pickup_datetime dtype: {rides['pickup_datetime'].dtype}")
    print(f"start_station_name dtype: {rides['start_station_name'].dtype}")
    print(f"Sample start_station_name values: {rides['start_station_name'].head().tolist()}")

    rides["pickup_hour"] = rides["pickup_datetime"].dt.floor("h")

    print("After creating pickup_hour:")
    print(f"pickup_hour range: {rides['pickup_hour'].min()} to {rides['pickup_hour'].max()}")
    print(f"Any NaN in pickup_hour: {rides['pickup_hour'].isna().sum()}")
    print(f"Any NaN in start_station_name: {rides['start_station_name'].isna().sum()}")
    print(f"pickup_hour dtype: {rides['pickup_hour'].dtype}")

    rides['start_station_name'] = rides['start_station_name'].astype(str)

    grouped = rides.groupby(["pickup_hour", "start_station_name"])
    agg_rides_counts = grouped.size()
    print("After groupby().size():")
    print(f"Number of groups: {len(agg_rides_counts)}")
    print(f"Sample groups:\n{agg_rides_counts.head()}")

    agg_rides = pd.DataFrame({
        'pickup_hour': agg_rides_counts.index.get_level_values('pickup_hour'),
        'start_station_name': agg_rides_counts.index.get_level_values('start_station_name'),
        'rides': agg_rides_counts.values
    })

    if agg_rides.empty:
        raise ValueError("Aggregated DataFrame is empty after grouping by pickup_hour and start_station_name.")

    agg_rides_all_slots = fill_missing_rides_full_range(
        agg_rides, "pickup_hour", "start_station_name", "rides"
    )

    agg_rides_all_slots = agg_rides_all_slots.astype(
        {"start_station_name": "string", "rides": "int16"}
    )

    # Filter to top 3 stations
    station_rides = agg_rides_all_slots.groupby('start_station_name')['rides'].sum().reset_index()
    top_stations = station_rides.nlargest(3, 'rides')
    top_stations_path = PROCESSED_DATA_DIR / "top_stations.csv"
    top_stations.to_csv(top_stations_path, index=False)

    # Filter the data to include only the top 3 stations
    top_station_names = top_stations['start_station_name'].tolist()
    agg_rides_all_slots = agg_rides_all_slots[
        agg_rides_all_slots['start_station_name'].isin(top_station_names)
    ].reset_index(drop=True)

    print(f"After filtering to top 3 stations:")
    print(f"Number of rows: {agg_rides_all_slots.shape[0]}")
    print(f"Unique start_station_name values: {agg_rides_all_slots['start_station_name'].nunique()}")
    print(f"Stations retained: {agg_rides_all_slots['start_station_name'].unique()}")

    return agg_rides_all_slots

def transform_ts_data_info_features_and_target_loop(
    df, feature_col="rides", window_size=12, step_size=1
):
    if df.empty:
        raise ValueError("Input DataFrame is empty. Cannot transform into features and targets.")

    station_names = df["start_station_name"].unique()
    transformed_data = []

    for station_name in station_names:
        try:
            station_data = df[df["start_station_name"] == station_name].reset_index(drop=True)
            values = station_data[feature_col].values
            times = station_data["pickup_hour"].values

            if len(values) <= window_size:
                raise ValueError("Not enough data to create even one window.")

            rows = []
            for i in range(0, len(values) - window_size, step_size):
                features = values[i : i + window_size]
                target = values[i + window_size]
                target_time = times[i + window_size]
                row = np.append(features, [target, station_name, target_time])
                rows.append(row)

            feature_columns = [f"{feature_col}_t-{window_size - i}" for i in range(window_size)]
            all_columns = feature_columns + ["target", "start_station_name", "pickup_hour"]
            transformed_df = pd.DataFrame(rows, columns=all_columns)
            transformed_data.append(transformed_df)

        except ValueError as e:
            print(f"Skipping station_name {station_name}: {str(e)}")

    if not transformed_data:
        raise ValueError("No data could be transformed.")

    final_df = pd.concat(transformed_data, ignore_index=True)
    features = final_df[feature_columns + ["pickup_hour", "start_station_name"]]
    targets = final_df["target"]
    return features, targets

def transform_ts_data_info_features_and_target(
    df, feature_col="rides", window_size=12, step_size=1
):
    if df.empty:
        raise ValueError("Input DataFrame is empty. Cannot transform into features and targets.")

    station_names = df["start_station_name"].unique()
    transformed_data = []

    for station_name in station_names:
        try:
            station_data = df[df["start_station_name"] == station_name].reset_index(drop=True)
            values = station_data[feature_col].values
            times = station_data["pickup_hour"].values

            if len(values) <= window_size:
                raise ValueError("Not enough data to create even one window.")

            rows = []
            for i in range(0, len(values) - window_size, step_size):
                features = values[i : i + window_size]
                target = values[i + window_size]
                target_time = times[i + window_size]
                row = np.append(features, [target, station_name, target_time])
                rows.append(row)

            feature_columns = [f"{feature_col}_t-{window_size - i}" for i in range(window_size)]
            all_columns = feature_columns + ["target", "start_station_name", "pickup_hour"]
            transformed_df = pd.DataFrame(rows, columns=all_columns)
            transformed_data.append(transformed_df)

        except ValueError as e:
            print(f"Skipping station_name {station_name}: {str(e)}")

    if not transformed_data:
        raise ValueError("No data could be transformed.")

    final_df = pd.concat(transformed_data, ignore_index=True)
    features = final_df[feature_columns + ["pickup_hour", "start_station_name"]]
    targets = final_df["target"]
    return features, targets

def split_time_series_data(
    df: pd.DataFrame,
    cutoff_date: datetime,
    target_column: str,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    if df.empty:
        raise ValueError("Input DataFrame is empty. Cannot split into train/test sets.")

    train_data = df[df["pickup_hour"] < cutoff_date].reset_index(drop=True)
    test_data = df[df["pickup_hour"] >= cutoff_date].reset_index(drop=True)

    X_train = train_data.drop(columns=[target_column])
    y_train = train_data[target_column]
    X_test = test_data.drop(columns=[target_column])
    y_test = test_data[target_column]

    return X_train, y_train, X_test, y_test

def fetch_batch_raw_data(
    from_date: Union[datetime, str], to_date: Union[datetime, str]
) -> pd.DataFrame:
    if isinstance(from_date, str):
        from_date = datetime.fromisoformat(from_date)
    if isinstance(to_date, str):
        to_date = datetime.fromisoformat(to_date)

    if not isinstance(from_date, datetime) or not isinstance(to_date, datetime):
        raise ValueError("Both 'from_date' and 'to_date' must be datetime objects or valid ISO format strings.")
    if from_date >= to_date:
        raise ValueError("'from_date' must be earlier than 'to_date'.")

    # Shift dates back by 16 weeks (instead of 52 weeks)
    historical_from_date = from_date - timedelta(weeks=16)
    historical_to_date = to_date - timedelta(weeks=16)

    rides_from = load_and_process_citi_bike_data(
        year=historical_from_date.year, months=[historical_from_date.month]
    )
    rides_from = rides_from[
        rides_from.pickup_datetime >= historical_from_date.to_datetime64()
    ]

    if historical_to_date.month != historical_from_date.month:
        rides_to = load_and_process_citi_bike_data(
            year=historical_to_date.year, months=[historical_to_date.month]
        )
        rides_to = rides_to[
            rides_to.pickup_datetime < historical_to_date.to_datetime64()
        ]
        rides = pd.concat([rides_from, rides_to], ignore_index=True)
    else:
        rides = rides_from

    if rides.empty:
        raise ValueError("No data available for the specified historical period.")

    # Shift the pickup_datetime forward by 16 weeks to align with the prediction period
    rides["pickup_datetime"] += timedelta(weeks=16)
    rides.sort_values(by=["start_station_name", "pickup_datetime"], inplace=True)
    return rides

def transform_ts_data_info_features(
    df, feature_col="rides", window_size=12, step_size=1
):
    if df.empty:
        raise ValueError("Input DataFrame is empty. Cannot transform into features.")

    station_names = df["start_station_name"].unique()
    transformed_data = []

    for station_name in station_names:
        try:
            station_data = df[df["start_station_name"] == station_name].reset_index(drop=True)
            # Explicitly sort by pickup_hour to ensure chronological order
            station_data.sort_values("pickup_hour", inplace=True)

            values = station_data[feature_col].values
            times = station_data["pickup_hour"].values

            if len(values) <= window_size:
                raise ValueError("Not enough data to create even one window.")

            rows = []
            # Adjust the loop to include the most recent window
            # Start from the last possible index and work backwards
            end_idx = len(values) - window_size - 1
            if end_idx < 0:
                end_idx = 0  # Ensure at least one window if possible
            for i in range(end_idx, -1, -step_size):
                features = values[i : i + window_size]
                target_time = times[i + window_size]
                row = np.append(features, [station_name, target_time])
                rows.append(row)

            feature_columns = [f"{feature_col}_t-{window_size - i}" for i in range(window_size)]
            all_columns = feature_columns + ["start_station_name", "pickup_hour"]
            transformed_df = pd.DataFrame(rows, columns=all_columns)
            transformed_data.append(transformed_df)

        except ValueError as e:
            print(f"Skipping station_name {station_name}: {str(e)}")

    if not transformed_data:
        raise ValueError("No data could be transformed.")

    final_df = pd.concat(transformed_data, ignore_index=True)
    return final_df