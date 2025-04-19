import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import time

def apply_rfe(input_file, n_features_to_select=10):
    try:
        data = pd.read_csv(input_file)
        print(f"Loaded data with {len(data)} trials")
        
        metadata_cols = ['trial_number', 'label']
        channel3_cols = [col for col in data.columns if col.startswith('channel3_')]
        channel4_cols = [col for col in data.columns if col.startswith('channel4_')]
        
        print(f"Channel 3 features: {len(channel3_cols)}")
        print(f"Channel 4 features: {len(channel4_cols)}")
        
        X = data[channel3_cols + channel4_cols]
        y = data['label']
        
        estimator = SVC(kernel="linear")
        selector = RFE(estimator=estimator, 
                      n_features_to_select=n_features_to_select * 2,
                      step=1)
        
        selector = selector.fit(X, y)
        
        selected_features = []
        features_ranking = []
        
        for channel_cols in [channel3_cols, channel4_cols]:
            channel_rankings = [
                (feature, rank) 
                for feature, rank in zip(channel_cols, selector.ranking_)
            ]
            channel_rankings.sort(key=lambda x: x[1])
            selected_features.extend([f for f, _ in channel_rankings[:n_features_to_select]])
            features_ranking.extend(channel_rankings)
        
        feature_ranking = pd.DataFrame(features_ranking, columns=['Feature', 'Ranking'])
        feature_ranking = feature_ranking.sort_values('Ranking')
        
        print("\nSelected features:")
        print("\nChannel 3 features:")
        for f in selected_features[:n_features_to_select]:
            print(f"- {f}")
        print("\nChannel 4 features:")
        for f in selected_features[n_features_to_select:]:
            print(f"- {f}")
            
        output_data = data[metadata_cols + selected_features].copy()
        
        output_filename = f"rfe_selected_features_{time.strftime('%Y%m%d-%H%M%S')}.csv"
        output_data.to_csv(output_filename, index=False)
        print(f"\nSelected features saved to: {output_filename}")
        
        ranking_filename = f"feature_rankings_{time.strftime('%Y%m%d-%H%M%S')}.csv"
        feature_ranking.to_csv(ranking_filename, index=False)
        print(f"Feature rankings saved to: {ranking_filename}")
        
        return output_data, feature_ranking
        
    except Exception as e:
        print(f"Error during RFE: {e}")
        return None, None

if __name__ == "__main__":
    normalized_file = '/Users/joaomachado/Desktop/IC_V3/pre_rfe_normal/normalized_features_20250417-011201.csv'
    selected_features, rankings = apply_rfe(normalized_file, n_features_to_select=10)
    
    if selected_features is not None:
        print("\nRFE completed successfully!")