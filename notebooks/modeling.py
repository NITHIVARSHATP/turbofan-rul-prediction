import pandas as pd
import joblib
import os
import json
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Paths
DATA_PATH = "../data/processed/train_FD001_finalized_Full_scaled.csv"
MODELS_DIR = "../models"
RESULTS_PATH = os.path.join(MODELS_DIR, "results.json")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")

os.makedirs(MODELS_DIR, exist_ok=True)

# Load data
df = pd.read_csv(DATA_PATH)
X = df.drop(columns=["RUL", "cycle"])
y = df["RUL"]

# Train/val split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit scaler and save it
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
joblib.dump(scaler, SCALER_PATH)

# Models to train
models = {
    "LinearRegression": LinearRegression(),
    "DecisionTree": DecisionTreeRegressor(random_state=42),
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42)
}

results = {}

# Train, evaluate and save models
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_val_scaled)

    rmse = mean_squared_error(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)

    print(f"🔍 Evaluation of {name} on Validation Set:"
          f"\n  - RMSE: {rmse:.2f}"
          f"\n  - MAE : {mae:.2f}"
          f"\n  - R²  : {r2:.4f}")
    joblib.dump(model, os.path.join(MODELS_DIR, f"{name}.pkl"))
    results[name] = {"RMSE": rmse, "MAE": mae, "R2": r2}

# Save results
FEATURES_PATH = os.path.join(MODELS_DIR, "features.json")
with open(FEATURES_PATH, "w") as f:
    json.dump(list(X.columns), f)

# Display best model
best_model = min(results.items(), key=lambda x: x[1]['RMSE'])[0]
print(f"✅ Best model: {best_model}")
