import pandas as pd
import numpy as np
from scipy import stats, signal
import time

def calculate_power_bands(data, fs=512):
    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 12),
        'mu': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 45)
    }
    
    try:
        windowed_data = data * signal.windows.hann(len(data))
        fft_data = np.abs(np.fft.rfft(windowed_data))
        freqs = np.fft.rfftfreq(len(data), 1/fs)
        
        powers = {}
        for band_name, (low, high) in bands.items():
            band_mask = (freqs >= low) & (freqs <= high)
            powers[band_name] = np.mean(fft_data[band_mask]**2) if np.any(band_mask) else 0
        
        return powers
    except Exception as e:
        print(f"Error calculating power bands: {e}")
        return {band: 0 for band in bands.keys()}

def aggregate_data_by_trial(data, feature_cols):
    # Group data by trial_number and aggregate features by mean, std, etc.
    aggregated_data = data.groupby('trial_number')[feature_cols].agg(['mean', 'std', 'var', 'min', 'max']).reset_index()
    aggregated_data.columns = ['_'.join(col).strip() for col in aggregated_data.columns.values]  # Flatten multi-index columns
    return aggregated_data

def extract_features(filtered_file):
    try:
        df = pd.read_csv(filtered_file)
        print(f"Loaded {len(df)} samples")
        print(f"Found {len(df['trial_number'].unique())} trials")
        
        reorganized_features = []
        
        for trial_num in sorted(df['trial_number'].unique()):
            trial_data = df[df['trial_number'] == trial_num]
            
            channel3_data = trial_data['channel_3'].values
            channel3_powers = calculate_power_bands(channel3_data)
            
            channel4_data = trial_data['channel_4'].values
            channel4_powers = calculate_power_bands(channel4_data)
            
            feature_dict = {
                'trial_number': trial_num,
                'label': trial_data['label'].iloc[0],
                'channel3_mean': np.mean(channel3_data),
                'channel3_std': np.std(channel3_data),
                'channel3_var': np.var(channel3_data),
                'channel3_min': np.min(channel3_data),
                'channel3_max': np.max(channel3_data),
                'channel3_range': np.ptp(channel3_data),
                'channel3_skewness': stats.skew(channel3_data),
                'channel3_kurtosis': stats.kurtosis(channel3_data),
                'channel3_zero_crossing_rate': len(np.where(np.diff(np.signbit(channel3_data)))[0])/len(channel3_data),
                'channel3_signal_energy': np.sum(channel3_data**2),
                'channel3_delta_power': channel3_powers['delta'],
                'channel3_theta_power': channel3_powers['theta'],
                'channel3_alpha_power': channel3_powers['alpha'],
                'channel3_mu_power': channel3_powers['mu'],
                'channel3_beta_power': channel3_powers['beta'],
                'channel3_gamma_power': channel3_powers['gamma'],
                'channel4_mean': np.mean(channel4_data),
                'channel4_std': np.std(channel4_data),
                'channel4_var': np.var(channel4_data),
                'channel4_min': np.min(channel4_data),
                'channel4_max': np.max(channel4_data),
                'channel4_range': np.ptp(channel4_data),
                'channel4_skewness': stats.skew(channel4_data),
                'channel4_kurtosis': stats.kurtosis(channel4_data),
                'channel4_zero_crossing_rate': len(np.where(np.diff(np.signbit(channel4_data)))[0])/len(channel4_data),
                'channel4_signal_energy': np.sum(channel4_data**2),
                'channel4_delta_power': channel4_powers['delta'],
                'channel4_theta_power': channel4_powers['theta'],
                'channel4_alpha_power': channel4_powers['alpha'],
                'channel4_mu_power': channel4_powers['mu'],
                'channel4_beta_power': channel4_powers['beta'],
                'channel4_gamma_power': channel4_powers['gamma']
            }
            
            reorganized_features.append(feature_dict)
            print(f"Processed trial {trial_num}")
        
        reorganized_df = pd.DataFrame(reorganized_features)
        
        output_filename = f"eeg_features_reorganized_{time.strftime('%Y%m%d-%H%M%S')}.csv"
        reorganized_df.to_csv(output_filename, index=False)
        print(f"\nFeatures saved to: {output_filename}")
        
        return reorganized_df
        
    except Exception as e:
        print(f"Error processing file: {e}")
        return None

if __name__ == "__main__":
    filtered_file = '/Users/joaomachado/Desktop/IC_V3_BKP/filtered_segmented_data_20250422-185058.csv'
    features_df = extract_features(filtered_file)
    if features_df is not None:
        print("Feature extraction completed successfully!")
        print("\nFeature summary:")
        print(features_df.describe())
