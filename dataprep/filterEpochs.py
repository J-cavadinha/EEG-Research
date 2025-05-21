import mne
import numpy as np
import os
import time

# --- Configuration Parameters for Filtering ---

# Input Epochs File (from your segmentation script)
# !!! UPDATE THIS PATH to your saved segmented epochs file !!!
INPUT_EPOCHS_FILENAME = "/Users/joaomachado/Desktop/pipeline/eeg_processed_data/processed_epochs_20250515-235629-epo.fif" 

# Filtering Parameters
NOTCH_FREQ = 60.0  # Hz (Set to None or 0 if no notch filter needed, e.g., 50.0 for Brazil/Europe)
FILTER_L_FREQ = 8.0  # Lower cutoff for bandpass filter (e.g., start of Mu band)
FILTER_H_FREQ = 30.0 # Upper cutoff for bandpass filter (e.g., end of Beta band)

# Output Settings
# Folder to save the filtered epochs
DATA_FOLDER_OUT = "eeg_processed_data" 
# Filename for the filtered epochs. We'll add a suffix to the input name or use a new timestamp.
# Let's try to base it on the input filename to keep track.
if 'your_segmented_epochs_file-epo.fif' in INPUT_EPOCHS_FILENAME:
    # Default name if placeholder is not changed
    OUTPUT_FILTERED_EPOCHS_FILENAME = os.path.join(DATA_FOLDER_OUT, f"filtered_epochs_{time.strftime('%Y%m%d-%H%M%S')}-epo.fif")
else:
    basename = os.path.basename(INPUT_EPOCHS_FILENAME).replace('-epo.fif', '')
    OUTPUT_FILTERED_EPOCHS_FILENAME = os.path.join(DATA_FOLDER_OUT, f"{basename}_filtered-epo.fif")

# --- End Configuration Parameters ---

def filter_mne_epochs(input_epochs_file, output_filtered_file, 
                        notch_f, l_f, h_f):
    """
    Loads MNE Epochs, applies notch and bandpass filters, and saves the filtered Epochs.
    """
    print(f"--- Starting Epoch Filtering Process ---")
    mne.set_log_level('INFO') # Set MNE to be verbose

    # Create output data folder if it doesn't exist
    output_dir = os.path.dirname(output_filtered_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"[INFO] Created output directory: {output_dir}")

    # 1. Load Epochs
    print(f"\n[INFO] Loading epochs from: {input_epochs_file}")
    if not os.path.exists(input_epochs_file):
        print(f"ERROR: Epochs file not found: {input_epochs_file}")
        return False
    
    try:
        epochs = mne.read_epochs(input_epochs_file, preload=True)
        print("\n--- Input Epochs Information ---")
        print(epochs)
        if len(epochs) == 0:
            print("ERROR: No epochs found in the loaded file. Cannot proceed.")
            return False
        # Assuming data was already scaled to Volts when creating the RawArray if needed
    except Exception as e:
        print(f"ERROR: Could not load epochs from {input_epochs_file}. Error: {e}")
        return False

    # 2. Apply Filters
    epochs_filtered = epochs.copy() # Work on a copy

    if notch_f is not None and notch_f > 0:
        print(f"\n[INFO] Applying Notch filter at {notch_f} Hz...")
        try:
            epochs_filtered.notch_filter(freqs=notch_f, fir_design='firwin', verbose=False)
            print("Notch filtering complete.")
        except Exception as e:
            print(f"Error during notch filtering: {e}")
            # Decide if you want to proceed without notch or stop
            # return False 

    if l_f is not None and h_f is not None:
        print(f"\n[INFO] Applying band-pass filter ({l_f}-{h_f} Hz)...")
        try:
            epochs_filtered.filter(l_freq=l_f, h_freq=h_f, fir_design='firwin', verbose=False)
            print("Band-pass filtering complete.")
        except Exception as e:
            print(f"Error during band-pass filtering: {e}")
            return False # Typically stop if bandpass fails
    
    print("\n--- Filtered Epochs Information ---")
    print(epochs_filtered)

    # 3. Save Filtered Epochs
    try:
        epochs_filtered.save(output_filtered_file, overwrite=True)
        print(f"\n--- Filtered epochs saved successfully ---")
        print(f"Saved to: {output_filtered_file}")
        return True
    except Exception as e:
        print(f"ERROR: Could not save filtered epochs to {output_filtered_file}. Error: {e}")
        return False

# --- Main Script Execution ---
if __name__ == "__main__":
    if 'your_segmented_epochs_file-epo.fif' in INPUT_EPOCHS_FILENAME:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! PLEASE UPDATE INPUT_EPOCHS_FILENAME in the script              !!!")
        print("!!! with the actual path to your segmented (unfiltered) epochs file. !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    else:
        success = filter_mne_epochs(
            input_epochs_file=INPUT_EPOCHS_FILENAME,
            output_filtered_file=OUTPUT_FILTERED_EPOCHS_FILENAME,
            notch_f=NOTCH_FREQ,
            l_f=FILTER_L_FREQ,
            h_f=FILTER_H_FREQ
        )
        if success:
            print("\nFiltering process completed.")
        else:
            print("\nFiltering process encountered errors.")