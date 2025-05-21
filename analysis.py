import mne
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time # For timestamp in output plots if needed

# --- Configuration Parameters ---

# Input file paths (FROM YOUR newGetData.py SCRIPT OUTPUT)
# !!! UPDATE THESE TO YOUR ACTUAL DATA FILES !!!
EEG_DATA_CSV_PATH = "/Users/joaomachado/Desktop/pipeline/eeg_data/eeg_c3_c4_ts_20250521-091722.csv" # Example from your newSegmentation.py
EVENTS_CSV_PATH = "/Users/joaomachado/Desktop/pipeline/eeg_data/events_20250521-091722.csv"     # Example from your newSegmentation.py

# EEG Parameters (must match your data collection)
SAMPLING_RATE = 125  # Hz (from your Cyton+Daisy board)
CH_NAMES = ['C3', 'C4']
CH_TYPES = ['eeg', 'eeg']

# Filtering Parameters (apply to continuous raw data for this analysis)
NOTCH_FREQ = 60.0  # Hz (e.g., 50.0 for Brazil/Europe, 60.0 for US). Set to None or 0 if no notch.
FILTER_L_FREQ = 8.0  # Lower cutoff for bandpass filter
FILTER_H_FREQ = 30.0 # Upper cutoff for bandpass filter

# Epoching Parameters for ERD/ERS analysis
TMIN_EPOCH = -2.5  # Start epochs 2.5 seconds before imagery onset
TMAX_EPOCH = 4.0   # End epochs 4.0 seconds after imagery onset (assuming 4s imagery)
BASELINE_PERIOD_FOR_TFR = (-2.0, -0.5) # Baseline from -2.0s to -0.5s relative to imagery onset

# Time-Frequency Analysis Parameters
TFR_FREQUENCIES = np.arange(7, 31)  # Frequencies of interest (e.g., 7 Hz to 30 Hz)
TFR_N_CYCLES = TFR_FREQUENCIES / 2.  # Number of cycles for Morlet wavelets (can be adjusted)

# Event ID mapping (must match your event descriptions and segmentation script)
EVENT_ID_MAP = {
    'IMAGERY_LEFT': 1,
    'IMAGERY_RIGHT': 2
}
# --- End Configuration Parameters ---

def analyze_erd_ers():
    """
    Loads EEG data, preprocesses, epochs with baseline, performs TFR analysis,
    and plots ERD/ERS patterns for motor imagery.
    """
    print("--- Starting ERD/ERS Analysis Script ---")
    mne.set_log_level('INFO')

    # --- 1. Load Continuous EEG Data and Events ---
    print(f"\n[INFO] Loading continuous EEG data from: {EEG_DATA_CSV_PATH}")
    if not os.path.exists(EEG_DATA_CSV_PATH):
        print(f"ERROR: EEG data CSV file not found: {EEG_DATA_CSV_PATH}")
        return
    try:
        eeg_df = pd.read_csv(EEG_DATA_CSV_PATH)
        data_for_mne = eeg_df[['C3', 'C4']].values.T * 1e-6 # Transpose and scale to Volts
        print(f"[INFO] Continuous EEG data loaded. Shape for MNE: {data_for_mne.shape}")
    except Exception as e:
        print(f"ERROR: Could not load or process EEG data CSV. Error: {e}")
        return

    print(f"\n[INFO] Loading events data from: {EVENTS_CSV_PATH}")
    if not os.path.exists(EVENTS_CSV_PATH):
        print(f"ERROR: Events CSV file not found: {EVENTS_CSV_PATH}")
        return
    try:
        events_df = pd.read_csv(EVENTS_CSV_PATH)
        print(f"[INFO] Events data loaded. Shape: {events_df.shape}")
        if events_df.empty:
            print("ERROR: Events DataFrame is empty. Cannot proceed.")
            return
    except Exception as e:
        print(f"ERROR: Could not load events data CSV. Error: {e}")
        return

    # --- 2. Create MNE RawArray Object ---
    info = mne.create_info(ch_names=CH_NAMES, sfreq=SAMPLING_RATE, ch_types=CH_TYPES)
    raw = mne.io.RawArray(data_for_mne, info, verbose=False)
    print(f"[INFO] MNE RawArray created. Duration: {raw.times[-1]:.2f} seconds")

    # --- 3. Apply Filtering to Continuous Raw Data ---
    if NOTCH_FREQ is not None and NOTCH_FREQ > 0:
        print(f"\n[INFO] Applying Notch filter to RAW data at {NOTCH_FREQ} Hz...")
        try:
            raw.notch_filter(freqs=NOTCH_FREQ, fir_design='firwin', verbose=False)
            print("Notch filtering on RAW data complete.")
        except Exception as e:
            print(f"Warning: Error during notch filtering on RAW data: {e}. Continuing without it.")

    if FILTER_L_FREQ is not None and FILTER_H_FREQ is not None:
        print(f"\n[INFO] Applying band-pass filter ({FILTER_L_FREQ}-{FILTER_H_FREQ} Hz) to RAW data...")
        try:
            raw.filter(l_freq=FILTER_L_FREQ, h_freq=FILTER_H_FREQ, fir_design='firwin', verbose=False)
            print("Band-pass filtering on RAW data complete.")
        except Exception as e:
            print(f"ERROR: Error during band-pass filtering on RAW data: {e}")
            return
    
    # --- 4. Create MNE Annotations from Events ---
    print("\n[INFO] Preparing MNE Annotations...")
    first_event_sw_time = events_df['software_timestamp_s'].iloc[0]
    onsets = []
    durations = []
    descriptions = []
    imagery_duration_from_collection = 4.0 # From your newGetData.py IMAGERY_DURATION

    for _, row in events_df.iterrows():
        onset_seconds = row['software_timestamp_s'] - first_event_sw_time
        event_type = row['event_type']
        condition = row['condition']

        if event_type == 'IMAGERY_START': 
            onsets.append(onset_seconds)
            durations.append(imagery_duration_from_collection) 
            descriptions.append(f"IMAGERY_{condition}")

    if not onsets:
        print("WARNING: No 'IMAGERY_START' events found in events_df to create annotations for epoching.")
        return
    
    try:
        annotations = mne.Annotations(onset=onsets, duration=durations, description=descriptions, orig_time=None)
        raw.set_annotations(annotations)
        print(f"[INFO] Annotations set. Found descriptions: {np.unique(raw.annotations.description)}")
    except Exception as e:
        print(f"ERROR: Failed to create or set MNE Annotations. Error: {e}")
        return

    # --- 5. Epoch Data with Baseline ---
    print(f"\n[INFO] Epoching data from t={TMIN_EPOCH}s to t={TMAX_EPOCH}s relative to imagery onset...")
    try:
        events_from_annot, event_dict_from_annot = mne.events_from_annotations(raw, event_id=EVENT_ID_MAP)
        
        if events_from_annot.shape[0] == 0:
            print("WARNING: No events matching your EVENT_ID_MAP were found from annotations. Cannot create epochs.")
            return
            
        epochs = mne.Epochs(raw, events=events_from_annot,
                            event_id=EVENT_ID_MAP,
                            tmin=TMIN_EPOCH, tmax=TMAX_EPOCH,
                            baseline=None, 
                            preload=True,
                            event_repeated='drop',
                            on_missing='warning',
                            verbose=False)
        print("\n--- Epoched Data Information (for TFR) ---")
        print(epochs)
        if len(epochs) == 0:
            print("ERROR: No epochs were created. Cannot proceed with TFR analysis.")
            return
    except Exception as e:
        print(f"ERROR: An error occurred during epoching: {e}")
        return

    # --- 6. Time-Frequency Analysis (TFR) ---
    print(f"\n[INFO] Computing Time-Frequency Representations (TFRs) using Morlet wavelets...")
    print(f"   Frequencies: {TFR_FREQUENCIES.min()}-{TFR_FREQUENCIES.max()} Hz")

    power_left = None
    if 'IMAGERY_LEFT' in epochs.event_id:
        try:
            power_left = mne.time_frequency.tfr_morlet(epochs['IMAGERY_LEFT'],
                                                       freqs=TFR_FREQUENCIES,
                                                       n_cycles=TFR_N_CYCLES,
                                                       use_fft=True, return_itc=False,
                                                       decim=3, n_jobs=1, verbose=False)
            print("[INFO] TFR computed for IMAGERY_LEFT.")
        except Exception as e:
            print(f"ERROR computing TFR for IMAGERY_LEFT: {e}")
    else:
        print("WARNING: No epochs found for 'IMAGERY_LEFT'. Skipping TFR calculation for LEFT.")

    power_right = None
    if 'IMAGERY_RIGHT' in epochs.event_id:
        try:
            power_right = mne.time_frequency.tfr_morlet(epochs['IMAGERY_RIGHT'],
                                                        freqs=TFR_FREQUENCIES,
                                                        n_cycles=TFR_N_CYCLES,
                                                        use_fft=True, return_itc=False,
                                                        decim=3, n_jobs=1, verbose=False)
            print("[INFO] TFR computed for IMAGERY_RIGHT.")
        except Exception as e:
            print(f"ERROR computing TFR for IMAGERY_RIGHT: {e}")
    else:
        print("WARNING: No epochs found for 'IMAGERY_RIGHT'. Skipping TFR calculation for RIGHT.")


    # --- 7. Plot TFRs with Baseline Correction ---
    print("\n[INFO] Plotting TFRs with baseline correction...")
    plots_dir = "erd_ers_plots"
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    plot_timestamp = time.strftime('%Y%m%d-%H%M%S')

    if power_left is not None:
        print("Plotting for IMAGERY_LEFT...")
        # --- MODIFICATION HERE ---
        figures_left_c3 = power_left.plot(picks=['C3'], baseline=BASELINE_PERIOD_FOR_TFR, mode='logratio',
                                          title='IMAGERY LEFT - C3 Power (Log Ratio vs Baseline)', show=False)
        if isinstance(figures_left_c3, list): # If plot returns a list of figures
            figures_left_c3[0].savefig(os.path.join(plots_dir, f"tfr_left_c3_{plot_timestamp}.png"))
            plt.close(figures_left_c3[0])
        else: # If plot returns a single figure object
            figures_left_c3.savefig(os.path.join(plots_dir, f"tfr_left_c3_{plot_timestamp}.png"))
            plt.close(figures_left_c3)

        figures_left_c4 = power_left.plot(picks=['C4'], baseline=BASELINE_PERIOD_FOR_TFR, mode='logratio',
                                          title='IMAGERY LEFT - C4 Power (Log Ratio vs Baseline)', show=False)
        if isinstance(figures_left_c4, list):
            figures_left_c4[0].savefig(os.path.join(plots_dir, f"tfr_left_c4_{plot_timestamp}.png"))
            plt.close(figures_left_c4[0])
        else:
            figures_left_c4.savefig(os.path.join(plots_dir, f"tfr_left_c4_{plot_timestamp}.png"))
            plt.close(figures_left_c4)
        # --- END MODIFICATION ---
        print(f"IMAGERY_LEFT plots saved to '{plots_dir}' directory.")

    if power_right is not None:
        print("Plotting for IMAGERY_RIGHT...")
        # --- MODIFICATION HERE ---
        figures_right_c3 = power_right.plot(picks=['C3'], baseline=BASELINE_PERIOD_FOR_TFR, mode='logratio',
                                            title='IMAGERY RIGHT - C3 Power (Log Ratio vs Baseline)', show=False)
        if isinstance(figures_right_c3, list):
            figures_right_c3[0].savefig(os.path.join(plots_dir, f"tfr_right_c3_{plot_timestamp}.png"))
            plt.close(figures_right_c3[0])
        else:
            figures_right_c3.savefig(os.path.join(plots_dir, f"tfr_right_c3_{plot_timestamp}.png"))
            plt.close(figures_right_c3)

        figures_right_c4 = power_right.plot(picks=['C4'], baseline=BASELINE_PERIOD_FOR_TFR, mode='logratio',
                                            title='IMAGERY RIGHT - C4 Power (Log Ratio vs Baseline)', show=False)
        if isinstance(figures_right_c4, list):
            figures_right_c4[0].savefig(os.path.join(plots_dir, f"tfr_right_c4_{plot_timestamp}.png"))
            plt.close(figures_right_c4[0])
        else:
            figures_right_c4.savefig(os.path.join(plots_dir, f"tfr_right_c4_{plot_timestamp}.png"))
            plt.close(figures_right_c4)
        # --- END MODIFICATION ---
        print(f"IMAGERY_RIGHT plots saved to '{plots_dir}' directory.")

    if power_left is None and power_right is None:
        print("No TFR data was computed, so no plots generated.")
    else:
        print("\nAll TFR plots generated. Please check the 'erd_ers_plots' directory.")
        print("Look for RED areas (ERS - power increase) and BLUE areas (ERD - power decrease) relative to baseline.")
        print("For LEFT hand imagery, expect ERD (blue) over C4 (right hemisphere).")
        print("For RIGHT hand imagery, expect ERD (blue) over C3 (left hemisphere).")

    print("\n--- ERD/ERS Analysis Script Finished ---")

if __name__ == "__main__":
    #if '20250515-221354' not in EEG_DATA_CSV_PATH or '20250515-221354' not in EVENTS_CSV_PATH: 
    #    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    #    print("!!! PLEASE UPDATE EEG_DATA_CSV_PATH and EVENTS_CSV_PATH in the   !!!")
    #    print("!!! script with the actual paths to your collected data files.   !!!")
    #    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    #else:
    analyze_erd_ers()
