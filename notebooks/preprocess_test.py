import pandas as pd
import numpy as np
import joblib
import json
import os

# Paths
TEST_DATA_PATH = "../data/raw/test_FD001.txt"
SCALER_PATH = "../models/scaler.pkl"
FEATURES_PATH = "../models/features.json"
PROCESSED_TEST_PATH = "../data/processed/test_FD001_processed.csv"

# Load expected features
with open(FEATURES_PATH, "r") as f:
    expected_features = json.load(f)

# Column names for test data
col_names = ["unit_number", "cycle"] + [f"setting_{i}" for i in range(1, 4)] + [f"sensor_{i}" for i in range(1, 22)]
sensor_cols = [f"sensor_{i}" for i in [2, 3, 4, 7, 8, 9, 11, 12, 13, 15, 17, 20, 21]]

# Load test data
test_df = pd.read_csv(TEST_DATA_PATH, sep=r"\s+", header=None, names=col_names)
test_df["engine_id"] = test_df["unit_number"]
test_df = test_df[["engine_id", "cycle"] + sensor_cols]

# Compute rolling statistics
window_size = 5
for sensor in sensor_cols:
    test_df[f"{sensor}_rolling_mean"] = test_df.groupby("engine_id")[sensor].transform(lambda x: x.rolling(window=window_size, min_periods=1).mean())
    test_df[f"{sensor}_rolling_std"] = test_df.groupby("engine_id")[sensor].transform(lambda x: x.rolling(window=window_size, min_periods=1).std())
    test_df[f"{sensor}_rolling_std"] = test_df[f"{sensor}_rolling_std"].fillna(0)

# Drop unnecessary
if "unit_number" in test_df.columns:
    test_df = test_df.drop(columns=["unit_number"])
X_test = test_df.drop(columns=["cycle"])

# Align columns with training features
missing = set(expected_features) - set(X_test.columns)
extra = set(X_test.columns) - set(expected_features)

for col in missing:
    X_test[col] = 0.0
X_test = X_test[[col for col in expected_features if col in X_test.columns]]

# Scale using trained scaler
scaler = joblib.load(SCALER_PATH)
X_test_scaled = scaler.transform(X_test)

# Save processed test data
processed_test_df = pd.DataFrame(X_test_scaled, columns=expected_features)
processed_test_df["cycle"] = test_df["cycle"].values
processed_test_df.to_csv(PROCESSED_TEST_PATH, index=False)

print("✅ Test data preprocessed and saved.")
