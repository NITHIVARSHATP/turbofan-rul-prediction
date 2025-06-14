import pandas as pd
from sklearn.preprocessing import StandardScaler

def scale_features():
    input_path = '../data/processed/train_FD001_feature_engineered.csv'
    df = pd.read_csv(input_path)
    
    # Identify features to scale (all except 'engine_id', 'cycle', 'RUL')
    exclude_cols = ['engine_id', 'cycle', 'RUL']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    scaler = StandardScaler()
    df_scaled = df.copy()
    
    df_scaled[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    output_path = '../data/processed/train_FD001_scaled.csv'
    df_scaled.to_csv(output_path, index=False)
    print(f"Scaled dataset saved to: {output_path}")

if __name__ == '__main__':
    scale_features()
