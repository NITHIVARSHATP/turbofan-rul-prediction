import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def evaluate_model(df, feature_cols, dataset_name=""):
    X = df[feature_cols]
    y = df['RUL']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"{dataset_name} RMSE: {rmse:.2f}")
    return rmse

def main():
    print("🔹 Evaluating PCA-reduced dataset...")
    df_pca = pd.read_csv('../data/processed/train_FD001_pca_standard.csv')
    pca_features = [col for col in df_pca.columns if col.startswith('PC')]
    rmse_pca = evaluate_model(df_pca, pca_features, dataset_name="PCA-reduced")

    print("\n🔹 Preparing full scaled dataset...")
    df_full = pd.read_csv('../data/processed/train_FD001_deduplicated.csv')
    exclude_cols = ['engine_id', 'cycle', 'RUL']
    feature_cols = [col for col in df_full.columns if col not in exclude_cols]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_full[feature_cols])
    df_scaled = pd.DataFrame(X_scaled, columns=feature_cols)

    df_scaled['engine_id'] = df_full['engine_id']
    df_scaled['cycle'] = df_full['cycle']
    df_scaled['RUL'] = df_full['RUL']

    front_cols = ['engine_id', 'cycle', 'RUL']
    df_scaled = df_scaled[front_cols + feature_cols]

    print("🔹 Evaluating full feature (scaled) dataset...")
    rmse_full = evaluate_model(df_scaled, feature_cols, dataset_name="Full scaled")

    print("\n📊 RMSE Comparison:")
    print(f"  - PCA-reduced RMSE: {rmse_pca:.2f}")
    print(f"  - Full scaled RMSE: {rmse_full:.2f}")

    if rmse_full < rmse_pca:
        print("✅ Full scaled dataset performs better. Use it for modeling.")
        output_path = '../data/processed/train_FD001_finalized_Full_scaled.csv'
        df_scaled.to_csv(output_path, index=False)
    else:
        print("✅ PCA-reduced dataset performs well. Use it for compact modeling.")
        output_path = '../data/processed/train_FD001_finalized_PCA_reduced.csv'
        df_pca.to_csv(output_path, index=False)

    print(f"\n📁 Finalized training dataset saved to: {output_path}")

if __name__ == '__main__':
    main()
