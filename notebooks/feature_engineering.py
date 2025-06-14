import pandas as pd

def main():
    # 1. Load cleaned dataset (output from clean_data.py)
    input_path = '../data/processed/train_FD001_cleaned.csv'
    df = pd.read_csv(input_path)

    # 2. Identify sensor columns (all columns starting with 'sensor_')
    sensor_cols = [col for col in df.columns if col.startswith('sensor_')]

    # 3. Define rolling window size (number of cycles)
    window_size = 5

    # 4. Generate rolling mean and rolling std features for each sensor column
    for sensor in sensor_cols:
        # Group data by engine_id and calculate rolling mean for each group
        rolling_mean = df.groupby('engine_id')[sensor].transform(
            lambda x: x.rolling(window=window_size, min_periods=1).mean()
        )
        df[f'{sensor}_rolling_mean'] = rolling_mean

        # Calculate rolling std deviation for each group and replace NaNs with 0
        rolling_std = df.groupby('engine_id')[sensor].transform(
            lambda x: x.rolling(window=window_size, min_periods=1).std()
        ).fillna(0)
        df[f'{sensor}_rolling_std'] = rolling_std

    # 5. Save the feature-engineered dataframe to a new CSV file
    output_path = '../data/processed/train_FD001_feature_engineered.csv'
    df.to_csv(output_path, index=False)
    print(f"Feature engineered data saved to: {output_path}")

if __name__ == '__main__':
    main()
