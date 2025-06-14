import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

def run_pca_workflow(scaling_method='standard', variance_threshold=0.95):
    # 1. Load the deduplicated dataset
    input_path = '../data/processed/train_FD001_deduplicated.csv'
    df = pd.read_csv(input_path)

    # 2. Separate features and target
    exclude_cols = ['engine_id', 'cycle', 'RUL']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    X = df[feature_cols]
    y = df['RUL']

    # Display before PCA: number of features
    original_count = X.shape[1]
    print(f"Number of original features: {X.shape[1]}")

    # Generate a bar chart for original feature count
    plt.figure(figsize=(6, 4))
    plt.bar(['Original Features'], [original_count])
    plt.title('Feature Count Before PCA')
    plt.ylabel('Number of Features')
    plt.tight_layout()
    plt.savefig('feature_count_before_pca.png')
    plt.show()

    # 3. Scale features
    if scaling_method == 'standard':
        scaler = StandardScaler()
    elif scaling_method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError("scaling_method must be either 'standard' or 'minmax'")

    X_scaled = scaler.fit_transform(X)

    # 4. Apply PCA
    pca = PCA(n_components=variance_threshold)
    X_pca = pca.fit_transform(X_scaled)

    # Display after PCA: number of principal components
    comp_count = pca.n_components_
    print(f"Number of principal components selected: {pca.n_components_}")

    # Generate a bar chart comparing before and after counts
    plt.figure(figsize=(6, 4))
    plt.bar(['Before PCA', 'After PCA'], [original_count, comp_count], color=['skyblue', 'salmon'])
    plt.title('Feature Count Before vs After PCA')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('feature_count_comparison.png')
    plt.show()

    # 5. Plot cumulative explained variance
    explained_var_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_var_ratio)

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(cumulative_variance)+1), cumulative_variance, marker='o')
    plt.title('Cumulative Variance Explained by Principal Components')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('pca_variance_plot.png')
    plt.show()

    print(f"\nNumber of components explaining >= {variance_threshold * 100:.0f}% variance: {pca.n_components_}")

    # 6. Save PCA-transformed data
    df_pca = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(pca.n_components_)])
    df_pca['RUL'] = y.values
    df_pca['engine_id'] = df['engine_id']
    df_pca['cycle'] = df['cycle']
    
    # Reorder columns to move engine_id, cycle, RUL to the front
    front_cols = ['engine_id', 'cycle']
    pca_cols = [col for col in df_pca.columns if col not in front_cols]
    df_pca = df_pca[front_cols + pca_cols]

    output_path = f"../data/processed/train_FD001_pca_{scaling_method}.csv"
    df_pca.to_csv(output_path, index=False)
    print(f"PCA-transformed data saved to: {output_path}")

if __name__ == '__main__':
    run_pca_workflow(scaling_method='standard') 
