import pandas as pd

# Define column names based on the C-MAPSS dataset description
columns = [
    'engine_id', 'cycle',
    'setting_1', 'setting_2', 'setting_3',
    'sensor_1', 'sensor_2', 'sensor_3', 'sensor_4', 'sensor_5',
    'sensor_6', 'sensor_7', 'sensor_8', 'sensor_9', 'sensor_10',
    'sensor_11', 'sensor_12', 'sensor_13', 'sensor_14', 'sensor_15',
    'sensor_16', 'sensor_17', 'sensor_18', 'sensor_19', 'sensor_20',
    'sensor_21'
]

# Load training data
df = pd.read_csv('../data/raw/train_FD001.txt', sep=r'\s+', header=None, names=columns)
df.dropna(axis=1, inplace=True)  # Drop extra empty columns

# Add RUL column to training data
df['RUL'] = df.groupby('engine_id')['cycle'].transform(max) - df['cycle']

# Load test data
test_df = pd.read_csv('../data/raw/test_FD001.txt', sep=r'\s+', header=None, names=columns)
test_df.dropna(axis=1, inplace=True)

# Load RUL labels BEFORE using them
rul_labels = pd.read_csv('../data/raw/RUL_FD001.txt', header=None)

# Extract last cycle of each engine from test set
last_cycles = test_df.groupby('engine_id').tail(1).reset_index(drop=True)

# Assign true RULs from file to last cycles
last_cycles['true_RUL'] = rul_labels

# Preview result
print("\n✅ Last cycles with matched true RULs:")
print(last_cycles[['engine_id', 'cycle', 'true_RUL']].head())

# Additional training set exploration
print("📊 First few rows of the training dataset:")
print(df.head())

print("\nShape of the training dataset:")
print(df.shape)

print("\nInfo about the training dataset:")
print(df.info())

print("\nMissing values in each column (train):")
print(df.isnull().sum())
