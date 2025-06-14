import pandas as pd

# 1. Define columns as per dataset
columns = [
    'engine_id', 'cycle',
    'setting_1', 'setting_2', 'setting_3',
    'sensor_1', 'sensor_2', 'sensor_3', 'sensor_4', 'sensor_5',
    'sensor_6', 'sensor_7', 'sensor_8', 'sensor_9', 'sensor_10',
    'sensor_11', 'sensor_12', 'sensor_13', 'sensor_14', 'sensor_15',
    'sensor_16', 'sensor_17', 'sensor_18', 'sensor_19', 'sensor_20',
    'sensor_21'
]

# 2. Load training data
df = pd.read_csv('../data/raw/train_FD001.txt', sep=r'\s+', header=None, names=columns)

# 3. Drop any empty columns if present
df.dropna(axis=1, inplace=True)

# 4. Add Remaining Useful Life (RUL) column
df['RUL'] = df.groupby('engine_id')['cycle'].transform(max) - df['cycle']

# 5. Check variance of all features including operational settings
variance_df = df.std()
print("\nFeature Standard Deviations:")
print(variance_df)

# 6. Identify low variance columns (< threshold), except engine_id, cycle, RUL
threshold = 1e-2  # Variance threshold to consider 'low'
low_variance_cols = [col for col in df.columns if variance_df[col] < threshold and col not in ['engine_id', 'cycle', 'RUL']]

print(f"\nColumns with low variance (< {threshold}): {low_variance_cols}")

# 7. Drop low variance columns, including operational settings if found low variance
df_reduced = df.drop(columns=low_variance_cols)

print(f"\nShape before dropping columns: {df.shape}")
print(f"Shape after dropping low variance columns: {df_reduced.shape}")

# 8. Save the cleaned dataframe to CSV for downstream usage
output_path = '../data/processed/train_FD001_cleaned.csv'
df_reduced.to_csv(output_path, index=False)
print(f"\nCleaned dataset saved to {output_path}")

