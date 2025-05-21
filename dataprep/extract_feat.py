# extract_feat.py (Comprehensive Feature Set Version)

import mne
import pandas as pd
import numpy as np
from scipy import stats, signal # For stats and signal.welch
import time
import os

# --- Configuration Parameters for Feature Extraction ---

# Input Filtered Epochs File (output from filter_epochs.py)
# !!! UPDATE THIS PATH to your saved filtered epochs file !!!
FILTERED_EPOCHS_FILENAME = "/Users/joaomachado/Desktop/pipeline/eeg_processed_data/processed_epochs_20250515-235629_filtered-epo.fif" # Example from your file

# Feature Extraction Parameters
# Define ALL power bands you originally had.
COMPREHENSIVE_POWER_BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 12),
    'mu': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 45)
}
# We will use all these bands for feature extraction in this version.

# Output Settings
DATA_FOLDER_OUT = "eeg_features_comprehensive" # New folder for these features
TIMESTAMP_SUFFIX_OUT = time.strftime('%Y%m%d-%H%M%S')
if 'your_filtered_epochs_file-epo.fif' in FILTERED_EPOCHS_FILENAME or FILTERED_EPOCHS_FILENAME.endswith('filtered-epo.fif') == False : # check if placeholder
    FEATURES_FILENAME = os.path.join(DATA_FOLDER_OUT, f"comprehensive_features_{TIMESTAMP_SUFFIX_OUT}.csv")
else:
    basename = os.path.basename(FILTERED_EPOCHS_FILENAME).replace('_filtered-epo.fif', '').replace('-epo.fif', '')
    FEATURES_FILENAME = os.path.join(DATA_FOLDER_OUT, f"{basename}_comprehensive_features.csv")

# --- End Configuration Parameters ---


def calculate_band_powers_for_channel(channel_data_one_epoch, fs, target_bands):
    """
    Calculates average power in specified frequency bands for a single channel's data
    from one epoch using Welch's method. Returns regular power (not log).
    """
    powers = {band_name: 0.0 for band_name in target_bands.keys()}
    try:
        nperseg = min(len(channel_data_one_epoch), int(fs * 2)) # e.g., 2-second window
        if nperseg < 1 or len(channel_data_one_epoch) < nperseg :
            return powers

        freqs, psd = signal.welch(channel_data_one_epoch, fs=fs, nperseg=nperseg)

        for band_name, (low, high) in target_bands.items():
            band_mask = (freqs >= low) & (freqs <= high)
            if np.any(band_mask) and np.any(psd[band_mask]):
                powers[band_name] = np.mean(psd[band_mask])
        return powers
    except Exception as e:
        # print(f"Error calculating band powers with Welch for a channel: {e}")
        return powers


def extract_comprehensive_features_from_mne_epochs(epochs_object, power_bands_to_calculate_dict):
    """
    Extracts a comprehensive set of statistical and band power features 
    from an MNE Epochs object, similar to your original script.
    """
    data_array = epochs_object.get_data(copy=False) 
    labels = epochs_object.events[:, -1] 
    ch_names = epochs_object.ch_names
    sfreq = epochs_object.info['sfreq']
    all_trial_features = []

    print(f"\n[INFO] Extracting comprehensive features for {len(data_array)} epochs...")
    for i in range(data_array.shape[0]): 
        trial_feature_dict = {}
        trial_feature_dict['epoch_index'] = i 
        trial_feature_dict['label'] = labels[i] 

        for ch_idx, ch_name in enumerate(ch_names): 
            channel_data = data_array[i, ch_idx, :]
            len_ch_data = len(channel_data)

            # === Statistical Features (as in your original feat_extraction.py) ===
            trial_feature_dict[f'{ch_name}_mean'] = np.mean(channel_data)
            trial_feature_dict[f'{ch_name}_std'] = np.std(channel_data)
            trial_feature_dict[f'{ch_name}_var'] = np.var(channel_data)
            trial_feature_dict[f'{ch_name}_min'] = np.min(channel_data)
            trial_feature_dict[f'{ch_name}_max'] = np.max(channel_data)
            trial_feature_dict[f'{ch_name}_range'] = np.ptp(channel_data)
            trial_feature_dict[f'{ch_name}_skewness'] = stats.skew(channel_data)
            trial_feature_dict[f'{ch_name}_kurtosis'] = stats.kurtosis(channel_data)
            
            if len_ch_data > 0:
                trial_feature_dict[f'{ch_name}_zero_crossing_rate'] = \
                    (len(np.where(np.diff(np.signbit(channel_data)))[0]) / len_ch_data)
            else:
                trial_feature_dict[f'{ch_name}_zero_crossing_rate'] = 0.0
            
            trial_feature_dict[f'{ch_name}_signal_energy'] = np.sum(channel_data**2)
            # === End of statistical features ===

            # === Band power features (regular power, all original bands) ===
            channel_powers = calculate_band_powers_for_channel(
                channel_data, sfreq, power_bands_to_calculate_dict
            )
            for band_name, power_val in channel_powers.items():
                # Using a consistent naming like your original for the CSV column
                trial_feature_dict[f'{ch_name}_{band_name}_power'] = power_val
        
        all_trial_features.append(trial_feature_dict)

    if not all_trial_features:
        print("WARNING: No features were extracted.")
        return pd.DataFrame()

    features_df = pd.DataFrame(all_trial_features)
    return features_df


# --- Main Script Execution ---
if __name__ == "__main__":
    output_dir_feat = os.path.dirname(FEATURES_FILENAME)
    if not os.path.exists(output_dir_feat):
        os.makedirs(output_dir_feat)
        print(f"[INFO] Created output data folder for features: {output_dir_feat}")

    print(f"--- Starting Comprehensive Feature Extraction Process ---")
    mne.set_log_level('WARNING') 

    print(f"\n[INFO] Loading filtered epochs from: {FILTERED_EPOCHS_FILENAME}")
    if not os.path.exists(FILTERED_EPOCHS_FILENAME):
        print(f"ERROR: Filtered epochs file not found: {FILTERED_EPOCHS_FILENAME}")
        print("Please run the filter_epochs.py script first and ensure the path is correct.")
        exit()

    try:
        filtered_epochs = mne.read_epochs(FILTERED_EPOCHS_FILENAME, preload=True)
        print("\n--- Filtered Epochs Information (for feature extraction) ---")
        print(filtered_epochs)
        if len(filtered_epochs) == 0:
            print("ERROR: No epochs found in the loaded filtered file. Cannot proceed.")
            exit()
    except Exception as e:
        print(f"ERROR: Could not load filtered epochs from {FILTERED_EPOCHS_FILENAME}. Error: {e}")
        exit()

    # Extract features using the defined COMPREHENSIVE_POWER_BANDS
    features_dataframe = extract_comprehensive_features_from_mne_epochs(
        filtered_epochs, 
        COMPREHENSIVE_POWER_BANDS
    )

    if not features_dataframe.empty:
        try:
            features_dataframe.to_csv(FEATURES_FILENAME, index=False)
            print(f"\n--- Comprehensive features extracted and saved successfully ---")
            print(f"Features saved to: {FEATURES_FILENAME}")
            print(f"Features DataFrame head:\n{features_dataframe.head()}")
            print(f"\nFeature columns (Total: {len(features_dataframe.columns)-2}): {features_dataframe.columns.tolist()}") # -2 for epoch_index, label
        except Exception as e:
            print(f"ERROR: Could not save features to {FEATURES_FILENAME}. Error: {e}")
    else:
        print("WARNING: Feature extraction resulted in an empty DataFrame. Nothing saved.")

    print("\n--- Comprehensive Feature Extraction Script Finished ---")