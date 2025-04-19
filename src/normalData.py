import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import time

def normalize_features(input_file):
    try:
        df = pd.read_csv(input_file)
        print(f"Loaded features for {len(df)} trials")
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        return
    except Exception as e:
        print(f"An error occurred while reading the input file: {e}")
        return

    metadata_cols = ['trial_number', 'label']
    feature_cols = [col for col in df.columns if col not in metadata_cols]
    features_data = df[feature_cols]
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(features_data)
    normalized_df = pd.DataFrame(normalized_data, columns=feature_cols)
    for col in metadata_cols:
        normalized_df[col] = df[col]
    final_cols = metadata_cols + feature_cols
    normalized_df = normalized_df[final_cols]
    output_filename = f"normalized_features_{time.strftime('%Y%m%d-%H%M%S')}.csv"
    
    try:
        normalized_df.to_csv(output_filename, index=False)
        print(f"Normalized features saved to: {output_filename}")
    except Exception as e:
        print(f"An error occurred while saving the output file: {e}")
        return

    return normalized_df

if __name__ == "__main__":
    features_file = '/Users/joaomachado/Desktop/IC_V3/post_rfe/rfe_selected_features_20250417-012314.csv'
    normalized_data = normalize_features(features_file)
    if normalized_data is not None:
        print("Feature normalization completed successfully!")