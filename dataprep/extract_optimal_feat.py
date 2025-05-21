import mne
import pandas as pd
import numpy as np
from scipy import signal # For signal.welch
import time
import os

# --- Configuration Parameters for Feature Extraction ---

# Input Filtered Epochs File (output from filter_epochs.py)
# !!! UPDATE THIS PATH to your saved filtered epochs file !!!
FILTERED_EPOCHS_FILENAME = "/Users/joaomachado/Desktop/pipeline/eeg_processed_data/processed_epochs_20250515-235629_filtered-epo.fif" # Example from your file path

# Feature Extraction Parameters - Focusing on Log Mu and Log Beta bands
TARGET_POWER_BANDS = {
    'mu': (8, 13),     # Standard Mu band
    'beta': (13, 30)   # Standard Beta band
}
# A small constant to add before taking log to avoid log(0) if power is exactly zero
LOG_EPSILON = 1e-10 

# Output Settings
DATA_FOLDER_OUT = "eeg_features_optimal" # Folder to save these specific features
TIMESTAMP_SUFFIX_OUT = time.strftime('%Y%m%d-%H%M%S')
# Try to base output filename on input filename
if 'your_filtered_epochs_file-epo.fif' in FILTERED_EPOCHS_FILENAME or FILTERED_EPOCHS_FILENAME.endswith('_filtered-epo.fif') == False:
    FEATURES_FILENAME = os.path.join(DATA_FOLDER_OUT, f"optimal_logbandpower_features_{TIMESTAMP_SUFFIX_OUT}.csv")
else:
    basename = os.path.basename(FILTERED_EPOCHS_FILENAME).replace('_filtered-epo.fif', '').replace('-epo.fif', '')
    FEATURES_FILENAME = os.path.join(DATA_FOLDER_OUT, f"{basename}_optimal_logbandpower_features.csv")

# --- End Configuration Parameters ---


def calculate_log_band_powers_for_channel(channel_data_one_epoch, fs, target_bands):
    """
    Calculates the logarithm of average power in specified frequency bands for a 
    single channel's data from one epoch using Welch's method.
    """
    log_powers = {band_name: 0.0 for band_name in target_bands.keys()}
    try:
        nperseg = min(len(channel_data_one_epoch), int(fs)) # Use 1 second of data for Welch window if possible
        if nperseg < 1 or len(channel_data_one_epoch) < nperseg :
            # print(f"Warning: Channel data length ({len(channel_data_one_epoch)}) is too short for nperseg ({nperseg}). Skipping PSD.")
            return log_powers 

        freqs, psd = signal.welch(channel_data_one_epoch, fs=fs, nperseg=nperseg)

        for band_name, (low, high) in target_bands.items():
            band_mask = (freqs >= low) & (freqs <= high)
            if np.any(band_mask) and np.any(psd[band_mask]):
                avg_power = np.mean(psd[band_mask])
                log_powers[band_name] = np.log(avg_power + LOG_EPSILON) 
            # else: log_powers[band_name] remains 0.0 (effectively log(LOG_EPSILON) if avg_power was 0)
        return log_powers
    except Exception as e:
        # print(f"Error calculating log band powers with Welch for a channel: {e}")
        return log_powers


def extract_focused_features_from_mne_epochs(epochs_object, power_bands_to_calculate_dict):
    """
    Extracts log-bandpower features (Mu, Beta) from an MNE Epochs object.
    """
    data_array = epochs_object.get_data(copy=False) 
    labels = epochs_object.events[:, -1] 
    ch_names = epochs_object.ch_names # Should be ['C3', 'C4']
    sfreq = epochs_object.info['sfreq']
    all_trial_features = []

    print(f"\n[INFO] Extracting focused log-bandpower features for {len(data_array)} epochs...")
    for i in range(data_array.shape[0]): 
        trial_feature_dict = {}
        trial_feature_dict['epoch_index'] = i 
        trial_feature_dict['label'] = labels[i] 

        for ch_idx, ch_name in enumerate(ch_names): 
            channel_data = data_array[i, ch_idx, :]

            channel_log_powers = calculate_log_band_powers_for_channel(
                channel_data, sfreq, power_bands_to_calculate_dict
            )
            for band_name, log_power_val in channel_log_powers.items():
                trial_feature_dict[f'{ch_name}_log_{band_name}_power'] = log_power_val # e.g., C3_log_mu_power
        
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

    print(f"--- Starting Focused Feature Extraction Process ---")
    mne.set_log_level('WARNING') 

    print(f"\n[INFO] Loading filtered epochs from: {FILTERED_EPOCHS_FILENAME}")
    if not os.path.exists(FILTERED_EPOCHS_FILENAME):
        print(f"ERROR: Filtered epochs file not found: {FILTERED_EPOCHS_FILENAME}")
        print("Please run the filter_epochs.py script first and ensure the path is correct.")
        exit()

    try:
        # It's assumed that filterEpochs.py has already applied an 8-30Hz bandpass
        filtered_epochs = mne.read_epochs(FILTERED_EPOCHS_FILENAME, preload=True)
        print("\n--- Filtered Epochs Information (for feature extraction) ---")
        print(filtered_epochs)
        if len(filtered_epochs) == 0:
            print("ERROR: No epochs found in the loaded filtered file. Cannot proceed.")
            exit()
    except Exception as e:
        print(f"ERROR: Could not load filtered epochs from {FILTERED_EPOCHS_FILENAME}. Error: {e}")
        exit()

    # Extract features using the defined TARGET_POWER_BANDS
    features_dataframe = extract_focused_features_from_mne_epochs(
        filtered_epochs, 
        TARGET_POWER_BANDS 
    )

    if not features_dataframe.empty:
        try:
            features_dataframe.to_csv(FEATURES_FILENAME, index=False)
            print(f"\n--- Focused features extracted and saved successfully ---")
            print(f"Features saved to: {FEATURES_FILENAME}")
            print(f"Features DataFrame head:\n{features_dataframe.head()}")
            print(f"\nFeature columns (Total: {len(features_dataframe.columns)-2}): {features_dataframe.columns.tolist()}")
        except Exception as e:
            print(f"ERROR: Could not save features to {FEATURES_FILENAME}. Error: {e}")
    else:
        print("WARNING: Feature extraction resulted in an empty DataFrame. Nothing saved.")

    print("\n--- Focused Feature Extraction Script Finished ---")