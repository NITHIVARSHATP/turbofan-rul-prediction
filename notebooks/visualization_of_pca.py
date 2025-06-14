import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the PCA-transformed data
df_pca = pd.read_csv('../data/processed/train_FD001_pca_standard.csv')

# --- 1. Scatter Plot: PC1 vs PC2 colored by RUL ---
plt.figure(figsize=(10, 6))
scatter = plt.scatter(df_pca['PC1'], df_pca['PC2'], c=df_pca['RUL'], cmap='viridis', alpha=0.7)
plt.colorbar(scatter, label='RUL')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PC1 vs PC2 colored by RUL')
plt.grid(True)
plt.tight_layout()
plt.show()


# --- 2. Scatter Plot: PC1 vs RUL ---
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_pca, x='PC1', y='RUL', alpha=0.5)
plt.title('PC1 vs RUL')
plt.xlabel('PC1')
plt.ylabel('RUL')
plt.grid(True)
plt.tight_layout()
plt.show()


# --- 3. Sensor Trajectories by RUL Buckets ---

# Load the deduplicated original dataset for sensor trends
df_orig = pd.read_csv('../data/processed/train_FD001_deduplicated.csv')

# Create RUL buckets: Early (RUL > 100), Mid (RUL 50-100), Late (RUL < 50)
def bucket_rul(rul):
    if rul > 100:
        return 'Early Life (RUL > 100)'
    elif rul > 50:
        return 'Mid Life (50 < RUL ≤ 100)'
    else:
        return 'Late Life (RUL ≤ 50)'

df_orig['RUL_bucket'] = df_orig['RUL'].apply(bucket_rul)

# Select sensors to plot
sensors_to_plot = ['sensor_2', 'sensor_3']

# Plot each sensor by RUL bucket
for sensor in sensors_to_plot:
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_orig, x='cycle', y=sensor, hue='RUL_bucket', alpha=0.5)
    plt.title(f'{sensor} Trajectory across RUL Buckets')
    plt.xlabel('Cycle')
    plt.ylabel(sensor)
    plt.legend(title='RUL Bucket')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
