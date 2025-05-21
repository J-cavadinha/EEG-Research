import pandas as pd
import numpy as np
import mne # Make sure MNE is installed: pip install mne
import os   # For os.path.join and os.makedirs
import time # For time.strftime

# --- Configuration Parameters ---
# !! These are for the MNE Segmentation Script !!

# Input file paths (UPDATE THESE TO POINT TO YOUR COLLECTED DATA)
# You need to replace 'your_timestamp' with the actual timestamp from your data files
EEG_FILE_PATH = '/Users/joaomachado/Desktop/pipeline/eeg_data/eeg_c3_c4_ts_20250515-221354.csv' # Example: 'eeg_data/eeg_c3_c4_ts_20250515-203000.csv'
EVENTS_FILE_PATH = '/Users/joaomachado/Desktop/pipeline/eeg_data/events_20250515-221354.csv'   # Example: 'eeg_data/events_20250515-203000.csv'

# EEG Setup (should match your data collection)
SAMPLING_RATE = 125  # Hz (from your Cyton+Daisy board info)

# Trial Timings (must match how annotations are created from events)
IMAGERY_DURATION = 4.0 # Seconds, from your collection script config
BASELINE_DURATION = 2.5 # Seconds, from your FIXATION_DURATION in collection script

# Output settings
DATA_FOLDER = "eeg_processed_data" # Folder to save processed epochs
TIMESTAMP_SUFFIX = time.strftime('%Y%m%d-%H%M%S') # Timestamp for when this script is run

# Event ID mapping for epoching
EVENT_ID_MAP = {
    'IMAGERY_LEFT': 1,
    'IMAGERY_RIGHT': 2
    # You can add baseline IDs here if you epoch them separately:
    # 'BASELINE_LEFT': 11,
    # 'BASELINE_RIGHT': 12
}
# --- End Configuration Parameters ---

def segment_eeg_data():
    print("--- Starting MNE Segmentation Script ---")
    mne.set_log_level('INFO')

    # Create output data folder if it doesn't exist
    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)
        print(f"[INFO] Created output data folder: {DATA_FOLDER}")

    # --- 1. Load Your Data ---
    print(f"\n[INFO] Attempting to load EEG data from: {EEG_FILE_PATH}")
    try:
        eeg_df = pd.read_csv(EEG_FILE_PATH)
        print(f"[INFO] EEG data loaded. Shape: {eeg_df.shape}")
    except FileNotFoundError:
        print(f"ERROR: EEG file not found at {EEG_FILE_PATH}. Please check the path and filename.")
        return
    except Exception as e:
        print(f"ERROR: Could not load EEG data. Error: {e}")
        return

    print(f"\n[INFO] Attempting to load events data from: {EVENTS_FILE_PATH}")
    try:
        events_df = pd.read_csv(EVENTS_FILE_PATH)
        print(f"[INFO] Events data loaded. Shape: {events_df.shape}")
    except FileNotFoundError:
        print(f"ERROR: Events file not found at {EVENTS_FILE_PATH}. Please check the path and filename.")
        return
    except Exception as e:
        print(f"ERROR: Could not load events data. Error: {e}")
        return

    try:
        c3_data = eeg_df['C3'].values
        c4_data = eeg_df['C4'].values
        # board_timestamps = eeg_df['Timestamp'].values # Used for alignment logic if needed
    except KeyError as e:
        print(f"ERROR: Missing one of 'C3', 'C4' columns in EEG CSV. Error: {e}")
        return

    data_for_mne = np.array([c3_data, c4_data])

    # --- 2. Prepare MNE Raw Object ---
    ch_names = ['C3', 'C4']
    ch_types = ['eeg', 'eeg']
    info = mne.create_info(ch_names=ch_names, sfreq=SAMPLING_RATE, ch_types=ch_types)
    
    # Data should be in Volts for MNE. Assuming BrainFlow output is in microvolts.
    raw = mne.io.RawArray(data_for_mne * 1e-6, info) # Scaled to Volts
    print(f"[INFO] MNE RawArray created. Duration: {raw.times[-1]:.2f} seconds")

    # --- 3. Creating MNE Annotations ---
    print("\n[INFO] Preparing MNE Annotations...")
    if events_df.empty:
        print("ERROR: Events DataFrame is empty.")
        return

    first_event_sw_time = events_df['software_timestamp_s'].iloc[0]

    onsets = []
    durations = []
    descriptions = []

    for _, row in events_df.iterrows():
        onset_seconds = row['software_timestamp_s'] - first_event_sw_time
        event_type = row['event_type']
        condition = row['condition']

        if event_type == 'IMAGERY_START':
            onsets.append(onset_seconds)
            durations.append(IMAGERY_DURATION)
            descriptions.append(f"IMAGERY_{condition}")
        elif event_type == 'FIXATION_START':
            onsets.append(onset_seconds)
            durations.append(BASELINE_DURATION)
            descriptions.append(f"BASELINE_{condition}")

    if not onsets:
        print("WARNING: No relevant events found in events_df to create annotations.")
        return
    
    try:
        annotations = mne.Annotations(onset=onsets, duration=durations, description=descriptions, orig_time=None)
        raw.set_annotations(annotations)
        print(f"[INFO] Annotations set. Found descriptions: {np.unique(raw.annotations.description)}")
    except Exception as e:
        print(f"ERROR: Failed to create or set MNE Annotations. Error: {e}")
        return

    # --- 4. Epoching ---
    print("\n[INFO] Starting Epoching process...")
    tmin_imagery, tmax_imagery = 0.0, IMAGERY_DURATION # Relative to IMAGERY_START marker

    try:
        if raw.annotations is None or len(raw.annotations) == 0:
            print("ERROR: No annotations found in raw data for epoching.")
            return

        events_from_annot, event_dict_from_annot = mne.events_from_annotations(raw, event_id=EVENT_ID_MAP)
        
        if events_from_annot.shape[0] == 0:
            print("WARNING: No events matching your EVENT_ID_MAP were found from annotations. Cannot create epochs.")
            print(f"  Available event descriptions in annotations were: {np.unique(raw.annotations.description if raw.annotations else [])}")
            print(f"  Event dictionary derived from these was: {event_dict_from_annot}")
            print(f"  Your script is looking for event_ids corresponding to: {list(EVENT_ID_MAP.keys())}")
            return
            
        epochs_imagery = mne.Epochs(raw, events=events_from_annot,
                                    event_id=EVENT_ID_MAP,
                                    tmin=tmin_imagery, tmax=tmax_imagery,
                                    baseline=None, preload=True,
                                    event_repeated='drop',
                                    on_missing='warning')

        print("\n--- Motor Imagery Epochs Information ---")
        print(epochs_imagery)

        if len(epochs_imagery) > 0:
            print(f"Successfully created {len(epochs_imagery)} motor imagery epochs.")

            # --- MODIFICATION: SAVING THE EPOCHS ---
            output_epo_filename = os.path.join(DATA_FOLDER, f"processed_epochs_{TIMESTAMP_SUFFIX}-epo.fif")
            try:
                epochs_imagery.save(output_epo_filename, overwrite=True)
                print(f"\n--- Processed epochs saved successfully ---")
                print(f"Saved to: {output_epo_filename}")
            except Exception as e:
                print(f"ERROR: Could not save epochs. Error: {e}")
            # --- END OF MODIFICATION ---
        else:
            print("WARNING: No motor imagery epochs were created, so nothing to save.")

    except Exception as e:
        print(f"ERROR: An error occurred during epoching: {e}")
        if raw.annotations:
            print(f"Unique annotation descriptions found: {np.unique(raw.annotations.description)}")

    print("\n--- MNE Segmentation Script Finished ---")

if __name__ == "__main__":
    # Before running, make sure EEG_FILE_PATH and EVENTS_FILE_PATH point to your actual files!
    # Example:
    # EEG_FILE_PATH = 'eeg_data/eeg_c3_c4_ts_20231026-153000.csv' # Replace with your actual filename
    # EVENTS_FILE_PATH = 'eeg_data/events_20231026-153000.csv'   # Replace with your actual filename
    if 'your_timestamp' in EEG_FILE_PATH or 'your_timestamp' in EVENTS_FILE_PATH:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! PLEASE UPDATE EEG_FILE_PATH and EVENTS_FILE_PATH in the script !!!")
        print("!!! with the actual paths to your collected data files.          !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    else:
        segment_eeg_data()