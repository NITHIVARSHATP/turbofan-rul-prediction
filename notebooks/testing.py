import pandas as pd
import joblib
import json
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Paths
PROCESSED_TEST_PATH = "../data/processed/test_FD001_processed.csv"
TRUE_RUL_PATH = "../data/raw/RUL_FD001.txt"
MODELS_DIR = "../models"
RESULTS_PATH = os.path.join(MODELS_DIR, "results.json")
FEATURES_PATH = os.path.join(MODELS_DIR, "features.json")

# Load processed test data
df = pd.read_csv(PROCESSED_TEST_PATH)

# ✅ FIX: Extract last cycle for each engine
last_cycles = df.groupby("engine_id").last().reset_index()

# Load true RUL values
true_rul = pd.read_csv(TRUE_RUL_PATH, header=None)[0].values

# Load best model name
with open(RESULTS_PATH, "r") as f:
    results = json.load(f)
best_model_name = min(results.items(), key=lambda x: x[1]['RMSE'])[0]
print(f"✅ Using best model: {best_model_name}")

# Load trained model
model_path = os.path.join(MODELS_DIR, f"{best_model_name}.pkl")
model = joblib.load(model_path)

# Load expected features
with open(FEATURES_PATH, "r") as f:
    expected_features = json.load(f)

# Drop unnecessary columns
last_cycles = last_cycles.drop(columns=["cycle", "engine_id"], errors='ignore')

# Align features
for col in expected_features:
    if col not in last_cycles.columns:
        last_cycles[col] = 0.0
last_cycles = last_cycles[expected_features]

# Predict RUL
preds = model.predict(last_cycles)

# Check shape
print(f"🔢 true_rul shape: {true_rul.shape}")
print(f"🔢 preds shape   : {preds.shape}")

# Evaluate
rmse = mean_squared_error(true_rul, preds)
mae = mean_absolute_error(true_rul, preds)
r2 = r2_score(true_rul, preds)

print(f"\n📊 Evaluation on Test Set:")
print(f"  - RMSE: {rmse:.2f}")
print(f"  - MAE : {mae:.2f}")
print(f"  - R²  : {r2:.4f}")

# Plot predictions vs true RUL
plt.figure(figsize=(10, 5))
plt.plot(true_rul, label="True RUL", marker='o')
plt.plot(preds, label="Predicted RUL", marker='x')
plt.xlabel("Engine")
plt.ylabel("RUL")
plt.title(f"Predicted vs True RUL ({best_model_name})")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("../results/test_rul_predictions.png")
plt.show()








