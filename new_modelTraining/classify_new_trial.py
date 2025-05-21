import mne
import pandas as pd
import numpy as np
from scipy import stats, signal
import joblib # For loading saved model, scaler, selector
import os
import time # Only if generating a new timestamp for any output, not strictly needed here

# --- Configuration for Classification ---

# Paths to your saved model artifacts (UPDATE THESE TIMESTAMPS/FILENAMES)
MODEL_ARTIFACTS_FOLDER = "model_artifacts" 
# Replace with the actual timestamp/name of your saved artifacts
MODEL_FILENAME = os.path.join(MODEL_ARTIFACTS_FOLDER, "svm_model_final_YYYYMMDD-HHMMSS.joblib")
SCALER_FILENAME = os.path.join(MODEL_ARTIFACTS_FOLDER, "svm_scaler_final_YYYYMMDD-HHMMSS.joblib")
SELECTOR_FILENAME = os.path.join(MODEL_ARTIFACTS_FOLDER, "svm_selector_final_YYYYMMDD-HHMMSS.joblib") # May not exist if RFE wasn't used for reduction
# This file stores the feature names array as it was input to the scaler/selector during training
PIPELINE_INPUT_FEATURES_NAMES_FILENAME = os.path.join(MODEL_ARTIFACTS_FOLDER, "PIPELINE_INPUT_feature_names_YYYYMMDD-HHMMSS.joblib")


# EEG Parameters (must match training data processing)
SAMPLING_RATE = 125  # Hz
IMAGERY_DURATION_SAMPLES = int(SAMPLING_RATE * 4.0) # Assuming 4-second trials
CH_NAMES = ['C3', 'C4'] # Must match the channels your model was trained on
CH_TYPES = ['eeg', 'eeg']

# Filtering Parameters (must match training data processing)
NOTCH_FREQ = 60.0  # Or 50.0, or None if not used
FILTER_L_FREQ = 8.0
FILTER_H_FREQ = 30.0

# Feature Extraction Parameters (must match training data processing)
# Choose which set of features your model was trained on:
# Option 1: Optimal 4 log-bandpower features
# TARGET_POWER_BANDS_CLASSIFY = {'mu': (8, 13), 'beta': (13, 30)}
# USE_ALL_STATISTICAL_FEATURES = False

# Option 2: The 10 features your last training run used (mean,std,var,mu,beta for C3/C4)
TARGET_POWER_BANDS_CLASSIFY = {'mu': (8, 13), 'beta': (13, 30)}
EXTRACT_MEAN_STD_VAR = True # Set to True if your model used these
# If using ALL original features, set EXTRACT_ALL_STAT_FEATURES to True
# and ensure TARGET_POWER_BANDS_CLASSIFY includes all delta, theta, alpha, mu, beta, gamma.
# For now, assuming the 10 features from your last successful training output
# (mean, std, var, mu_power, beta_power for C3 & C4)

LOG_EPSILON = 1e-10 # For log power calculation

# Class labels (must match training)
CLASS_MAP = {1: "IMAGERY_LEFT", 2: "IMAGERY_RIGHT"} # Or however your labels were numerically encoded

# --- Helper Functions (adapted from your feature extraction and filtering) ---

def apply_filters_to_single_trial_data(raw_trial_data_np, sfreq, ch_names, ch_types,
                                       notch_f, l_f, h_f):
    """Applies filtering to a single trial's raw EEG data (NumPy array)."""
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    # Create a RawArray for the single trial
    # Data needs to be scaled to Volts if not already (e.g. if from BrainFlow in uV)
    raw_mne = mne.io.RawArray(raw_trial_data_np * 1e-6, info, verbose=False)

    if notch_f is not None and notch_f > 0:
        try:
            raw_mne.notch_filter(freqs=notch_f, fir_design='firwin', verbose=False)
        except AttributeError: # Fallback for older MNE that might not have notch_filter on Raw
            print("Warning: Raw.notch_filter() not found (older MNE?). Trying mne.filter.notch_filter on data array.")
            # This is more complex as it requires converting back and forth or careful handling
            # For simplicity, this example will assume modern MNE for Raw.notch_filter
            # Or, ensure your MNE is updated / notch was done before this stage if using older MNE.
            pass # Or implement manual notch filtering if really needed here
        except Exception as e:
            print(f"Error during notch filtering single trial: {e}")


    if l_f is not None and h_f is not None:
        raw_mne.filter(l_freq=l_f, h_freq=h_f, fir_design='firwin', verbose=False)
    
    return raw_mne.get_data() # Returns (n_channels, n_samples)

# Adapted calculate_power_bands_for_epoch_channel
def calculate_log_band_powers_for_channel(channel_data_one_epoch, fs, target_bands):
    log_powers = {band_name: 0.0 for band_name in target_bands.keys()}
    try:
        nperseg = min(len(channel_data_one_epoch), int(fs))
        if nperseg < 1 or len(channel_data_one_epoch) < nperseg :
            return log_powers
        freqs, psd = signal.welch(channel_data_one_epoch, fs=fs, nperseg=nperseg)
        for band_name, (low, high) in target_bands.items():
            band_mask = (freqs >= low) & (freqs <= high)
            if np.any(band_mask) and np.any(psd[band_mask]):
                avg_power = np.mean(psd[band_mask])
                log_powers[band_name] = np.log(avg_power + LOG_EPSILON)
        return log_powers
    except Exception:
        return log_powers

def extract_features_for_single_trial(filtered_trial_data_np, sfreq, ch_names_list, 
                                      target_power_bands_dict, extract_mean_std_var=False):
    """Extracts features from a single trial's filtered EEG data."""
    trial_features = {} # Using a dictionary to store features with names
    
    for ch_idx, ch_name in enumerate(ch_names_list):
        channel_data = filtered_trial_data_np[ch_idx, :]
        
        if extract_mean_std_var:
            trial_features[f'{ch_name}_mean'] = np.mean(channel_data)
            trial_features[f'{ch_name}_std'] = np.std(channel_data)
            trial_features[f'{ch_name}_var'] = np.var(channel_data)
            # Add other statistical features here if your model was trained with them
            # e.g., min, max, skewness, kurtosis, etc.

        channel_log_powers = calculate_log_band_powers_for_channel(
            channel_data, sfreq, target_power_bands_dict
        )
        for band_name, log_power_val in channel_log_powers.items():
            # Naming convention consistent with "optimal" features if only these are used,
            # or "_power" if combined with mean/std/var to match your 10-feature set.
            # Let's match the 10-feature set for now based on your last training output.
            feature_name_key = f'{ch_name}_{band_name}_power' # Matches your 10-feature set names
            if not extract_mean_std_var: # If only doing log_band_power
                 feature_name_key = f'{ch_name}_log_{band_name}_power'
            trial_features[feature_name_key] = log_power_val
            
    return trial_features


def classify_new_eeg_data(raw_eeg_trial_data_np, # Shape: (n_channels, n_samples)
                           sfreq, ch_names_list, ch_types_list,
                           filter_notch_f, filter_l_f, filter_h_f,
                           feature_target_bands, feature_extract_mean_std_var,
                           loaded_model, loaded_scaler, loaded_selector, 
                           pipeline_input_feature_names):
    """
    Full pipeline to classify a new single trial of EEG data.
    """
    print("\n--- Classifying New EEG Trial ---")

    # 1. Preprocess: Filter the new raw trial data
    print("Step 1: Filtering new trial data...")
    filtered_data = apply_filters_to_single_trial_data(
        raw_eeg_trial_data_np, sfreq, ch_names_list, ch_types_list,
        filter_notch_f, filter_l_f, filter_h_f
    )
    print(f"Filtering complete. Filtered data shape: {filtered_data.shape}")

    # 2. Preprocess: Extract features from the filtered trial data
    print("\nStep 2: Extracting features from new trial data...")
    new_trial_features_dict = extract_features_for_single_trial(
        filtered_data, sfreq, ch_names_list, 
        feature_target_bands, feature_extract_mean_std_var
    )
    print(f"Extracted features: {new_trial_features_dict}")

    # Ensure features are in the same order as during training
    # The `pipeline_input_feature_names` were saved during training.
    # This list defines the order of columns expected by the scaler and selector.
    
    # Create a 1-row DataFrame or 2D NumPy array for the new trial's features
    # in the correct order.
    # If a feature is missing from new_trial_features_dict (should not happen if extraction is consistent),
    # it would cause an error or need default handling.
    try:
        new_trial_features_ordered = [new_trial_features_dict[fname] for fname in pipeline_input_feature_names]
    except KeyError as e:
        print(f"ERROR: Feature mismatch! A feature expected by the pipeline ('{e}') was not found in the extracted features.")
        print(f"Expected features by pipeline: {pipeline_input_feature_names}")
        print(f"Actually extracted features: {list(new_trial_features_dict.keys())}")
        return None, None

    X_new_trial = np.array([new_trial_features_ordered]) # Shape: (1, n_features)
    print(f"Feature vector for classification (shape {X_new_trial.shape}):\n{X_new_trial}")


    # 3. Apply Scaler
    print("\nStep 3: Applying loaded scaler...")
    X_new_trial_scaled = loaded_scaler.transform(X_new_trial)
    print(f"Scaled feature vector:\n{X_new_trial_scaled}")

    # 4. Apply Feature Selector (if one was used and saved)
    X_for_model = X_new_trial_scaled
    if loaded_selector is not None:
        print("\nStep 4: Applying loaded RFE feature selector...")
        try:
            X_for_model = loaded_selector.transform(X_new_trial_scaled)
            print(f"Feature vector after RFE selection (shape {X_for_model.shape}):\n{X_for_model}")
        except Exception as e:
            print(f"Error applying RFE selector: {e}. Check if features match those selector was trained on.")
            return None, None

    else:
        print("\nStep 4: RFE feature selector was not loaded or not used. Using all scaled features.")


    # 5. Make Prediction
    print("\nStep 5: Making prediction with loaded model...")
    try:
        prediction_code = loaded_model.predict(X_for_model)
        predicted_label = CLASS_MAP.get(prediction_code[0], f"Unknown_Code_{prediction_code[0]}")
        
        probabilities = None
        if hasattr(loaded_model, "predict_proba"):
            probabilities = loaded_model.predict_proba(X_for_model)
            print(f"Prediction probabilities: {probabilities[0]}")

        print(f"\n==> Predicted Class: {predicted_label} (Code: {prediction_code[0]}) <==")
        return predicted_label, probabilities
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None, None


# --- Main Execution ---
if __name__ == "__main__":
    print("--- EEG Real-Time Classification Simulation ---")

    # Load all necessary artifacts
    print("\nLoading saved model and preprocessing artifacts...")
    try:
        model = joblib.load(MODEL_FILENAME)
        scaler = joblib.load(SCALER_FILENAME)
        pipeline_feature_names = joblib.load(PIPELINE_INPUT_FEATURES_NAMES_FILENAME)
        
        # Try to load selector, it might not exist if RFE wasn't effectively used
        # or if n_features_to_select was equal to total features in training
        selector = None
        if os.path.exists(SELECTOR_FILENAME):
            try:
                selector = joblib.load(SELECTOR_FILENAME)
                print(f"Loaded RFE Selector: {selector}")
            except Exception as e_sel:
                print(f"Warning: Could not load selector from {SELECTOR_FILENAME}: {e_sel}. Assuming no RFE selection needed beyond pipeline_feature_names.")
        else:
            print(f"Info: Selector file {SELECTOR_FILENAME} not found. Assuming RFE was not used for feature reduction or all features were selected.")

        print("All necessary artifacts loaded successfully.")
        print(f"Model will expect features in this order initially: {pipeline_feature_names}")
        if selector:
             print(f"RFE selector will then potentially transform these further based on its training.")


    except FileNotFoundError as e:
        print(f"ERROR: Could not load one or more artifact files: {e}")
        print("Please ensure MODEL_FILENAME, SCALER_FILENAME, and PIPELINE_INPUT_FEATURES_NAMES_FILENAME point to correct saved files from training.")
        exit()
    except Exception as e:
        print(f"ERROR loading artifacts: {e}")
        exit()

    # --- Simulate receiving new EEG data for ONE trial ---
    # This would eventually come from your BrainFlow headset in a real application.
    # For now, let's create some dummy data with the correct shape.
    # (2 channels, 4 seconds at 125 Hz = 500 samples)
    print("\nSimulating new raw EEG trial data (C3, C4)...")
    # Replace this with actual data acquisition for a new trial
    # Example: new_trial_eeg_data_uV = my_brainflow_get_4_seconds_of_C3_C4_data()
    # Ensure it's in microvolts if the training data was processed assuming uV input to scaling.
    # The apply_filters_to_single_trial_data function currently multiplies by 1e-6, assuming input is like uV.
    
    # Dummy data: (n_channels, n_samples)
    # This data should be similar in magnitude to what your board outputs (e.g., microvolts)
    # before the * 1e-6 scaling for MNE.
    dummy_c3_data = np.random.randn(IMAGERY_DURATION_SAMPLES) * 10 # Simulating some uV range data
    dummy_c4_data = np.random.randn(IMAGERY_DURATION_SAMPLES) * 10
    new_raw_trial_data = np.array([dummy_c3_data, dummy_c4_data])
    print(f"Shape of simulated new trial data: {new_raw_trial_data.shape}")

    # Classify the new trial
    predicted_class, prediction_probs = classify_new_eeg_data(
        new_raw_trial_data,
        SAMPLING_RATE, CH_NAMES, CH_TYPES,
        NOTCH_FREQ, FILTER_L_FREQ, FILTER_H_FREQ,
        TARGET_POWER_BANDS_CLASSIFY, EXTRACT_MEAN_STD_VAR,
        model, scaler, selector,
        pipeline_feature_names
    )

    if predicted_class:
        print(f"\nFINAL PREDICTION FOR NEW TRIAL: {predicted_class}")
        if prediction_probs is not None:
            # Assuming binary classification (class 1 and class 2 from your labels)
            # Probabilities might be [prob_class_1, prob_class_2]
            prob_left = prediction_probs[0] if CLASS_MAP.get(model.classes_[0]) == "IMAGERY_LEFT" else prediction_probs[1]
            prob_right = prediction_probs[1] if CLASS_MAP.get(model.classes_[1]) == "IMAGERY_RIGHT" else prediction_probs[0]
            # This mapping needs to be robust if model.classes_ order isn't guaranteed [1, 2]
            # A safer way:
            idx_left = np.where(model.classes_ == 1)[0] # Assuming 1 was LEFT
            idx_right = np.where(model.classes_ == 2)[0] # Assuming 2 was RIGHT
            
            if len(idx_left) > 0 and len(idx_right) > 0 :
                 print(f"  Confidence: P(Left)={prediction_probs[0, idx_left[0]]:.2f}, P(Right)={prediction_probs[0, idx_right[0]]:.2f}")


    # --- How you would use this for real-time (conceptual) ---
    # In a real-time loop:
    # 1. Collect ~4 seconds of C3, C4 data from BrainFlow.
    # 2. Pass it to classify_new_eeg_data().
    # 3. Use the predicted_class to control something.
    # 4. Repeat.