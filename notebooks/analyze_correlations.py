import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def analyze_correlations():
    # 1. Load the scaled dataset
    input_path = '../data/processed/train_FD001_scaled.csv'
    df = pd.read_csv(input_path)

    # 2. Define columns to exclude (non-feature columns)
    exclude_cols = ['engine_id', 'cycle', 'RUL']
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    # 3. Compute correlation matrix (absolute values for simplicity)
    corr_matrix = df[feature_cols].corr().abs()

    # 4. Visualize correlation matrix with a heatmap
    plt.figure(figsize=(18, 15))
    sns.heatmap(corr_matrix, cmap='coolwarm', annot=False)
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('feature_correlation_heatmap.png')
    plt.show()

    # 5. Extract upper triangle of correlation matrix to avoid duplicate pairs
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # 6. Find all highly correlated feature pairs (correlation > 0.95)
    high_corr_pairs = [
        (col1, col2, upper_triangle.loc[col1, col2])
        for col1 in upper_triangle.columns
        for col2 in upper_triangle.columns
        if upper_triangle.loc[col1, col2] > 0.95
    ]

    print("\nHighly correlated feature pairs (corr > 0.95):")
    for col1, col2, corr in high_corr_pairs:
        print(f"{col1} ↔ {col2} : {corr:.2f}")

    # 7. Select one feature to drop from each highly correlated pair
    to_drop = set()
    for col1, col2, _ in high_corr_pairs:
        if col2 not in to_drop:
            to_drop.add(col2)

    print(f"\nDropping {len(to_drop)} redundant features due to high correlation.")
    print(f"Features dropped: {sorted(to_drop)}")

    # 8. Drop the redundant features
    df_reduced = df.drop(columns=to_drop)

    # 9. Save the reduced feature set
    output_path = '../data/processed/train_FD001_deduplicated.csv'
    df_reduced.to_csv(output_path, index=False)
    print(f"\nReduced dataset saved to: {output_path}")

if __name__ == '__main__':
    analyze_correlations()
