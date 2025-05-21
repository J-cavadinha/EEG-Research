import time
import csv
import random
import os
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
import numpy as np

# --- Configuration Parameters ---
SERIAL_PORT = "/dev/cu.usbserial-DQ007TQ0"  # !!! UPDATE THIS to your serial port !!!
BOARD_ID = BoardIds.CYTON_DAISY_BOARD.value

# Experiment Setup
RUNS = 1
TRIALS_PER_CLASS_PER_RUN = 4
CLASSES = {"LEFT": 0, "RIGHT": 1}
CLASS_NAMES = {v: k for k, v in CLASSES.items()}

# Trial Timings (in seconds)
FIXATION_DURATION = 2.5
CUE_DURATION = 1.5
IMAGERY_DURATION = 4.0
ITI_DURATION = 3.0
BREAK_BETWEEN_RUNS_DURATION = 60 # Consider increasing for longer sessions (e.g., 60-120s)

DATA_FOLDER = "eeg_data"
TIMESTAMP_SUFFIX = time.strftime('%Y%m%d-%H%M%S')
# Filename will now reflect that it's C3, C4, and Timestamps
EEG_DATA_FILENAME = os.path.join(DATA_FOLDER, f"eeg_c3_c4_ts_{TIMESTAMP_SUFFIX}.csv")
EVENTS_FILENAME = os.path.join(DATA_FOLDER, f"events_{TIMESTAMP_SUFFIX}.csv")

C3_EEG_CHANNEL_INDEX_IN_EEG_LIST = 2 # Example: 3rd channel in the EEG list
C4_EEG_CHANNEL_INDEX_IN_EEG_LIST = 3 # Example: 4th channel in the EEG list

def collect_eeg_session():
    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)

    params = BrainFlowInputParams()
    params.serial_port = SERIAL_PORT
    board = BoardShim(BOARD_ID, params)

    print("Preparing BrainFlow session...")
    try:
        board.prepare_session()
        board_descr = BoardShim.get_board_descr(BOARD_ID)
        eeg_channel_indices_from_board = board_descr['eeg_channels'] # Actual indices in the data array for EEG
        timestamp_channel_from_board = board_descr['timestamp_channel']

        print(f"Board: {BoardIds(BOARD_ID).name}")
        print(f"Sampling Rate: {board_descr['sampling_rate']} Hz")
        print(f"Total EEG Channels available: {len(eeg_channel_indices_from_board)}")
        print(f"Timestamp channel index in data array: {timestamp_channel_from_board}")

        # Determine the actual data array indices for C3 and C4
        # This assumes C3_EEG_CHANNEL_INDEX_IN_EEG_LIST and C4_EEG_CHANNEL_INDEX_IN_EEG_LIST
        # are correct indices *within the list* of EEG channels.
        if C3_EEG_CHANNEL_INDEX_IN_EEG_LIST >= len(eeg_channel_indices_from_board) or \
           C4_EEG_CHANNEL_INDEX_IN_EEG_LIST >= len(eeg_channel_indices_from_board):
            print("ERROR: C3_EEG_CHANNEL_INDEX_IN_EEG_LIST or C4_EEG_CHANNEL_INDEX_IN_EEG_LIST is out of bounds for the available EEG channels.")
            print(f"Available EEG channel indices in data array: {eeg_channel_indices_from_board}")
            return

        c3_actual_data_array_index = eeg_channel_indices_from_board[C3_EEG_CHANNEL_INDEX_IN_EEG_LIST]
        c4_actual_data_array_index = eeg_channel_indices_from_board[C4_EEG_CHANNEL_INDEX_IN_EEG_LIST]
        print(f"Data will be saved for: ")
        print(f"  C3 (using overall data array index: {c3_actual_data_array_index})")
        print(f"  C4 (using overall data array index: {c4_actual_data_array_index})")
        print(f"  Timestamp (using overall data array index: {timestamp_channel_from_board})")

    except Exception as e:
        print(f"Error preparing session: {e}")
        return

    all_events = []
    overall_trial_count = 0
    data_to_save = None # Initialize

    try:
        print("\n--- Starting Data Collection Session ---")
        input("Press Enter to begin the first run...")
        board.start_stream(450000) # Start streaming

        for run_num in range(1, RUNS + 1):
            print(f"\n--- Starting Run {run_num}/{RUNS} ---")
            run_labels = ([CLASSES["LEFT"]] * TRIALS_PER_CLASS_PER_RUN +
                          [CLASSES["RIGHT"]] * TRIALS_PER_CLASS_PER_RUN)
            random.shuffle(run_labels)

            for trial_in_run_num, label_code in enumerate(run_labels):
                overall_trial_count += 1
                condition_name = CLASS_NAMES[label_code]
                print(f"\nRun {run_num}, Trial {trial_in_run_num + 1}/{len(run_labels)} (Overall: {overall_trial_count})")

                # 1. Fixation
                print(f"  {time.strftime('%H:%M:%S')}: Fixation (+)")
                all_events.append([run_num, overall_trial_count, condition_name, "FIXATION_START", time.time()])
                time.sleep(FIXATION_DURATION)
                all_events.append([run_num, overall_trial_count, condition_name, "FIXATION_END", time.time()])

                # 2. Cue
                cue_text = f"Cue: Imagine {condition_name} hand"
                print(f"  {time.strftime('%H:%M:%S')}: {cue_text}")
                all_events.append([run_num, overall_trial_count, condition_name, f"CUE_{condition_name}_START", time.time()])
                time.sleep(CUE_DURATION)
                all_events.append([run_num, overall_trial_count, condition_name, f"CUE_{condition_name}_END", time.time()])

                # 3. Motor Imagery
                print(f"  {time.strftime('%H:%M:%S')}: Imagery ({condition_name}) - START (Imagine repeatedly!)")
                all_events.append([run_num, overall_trial_count, condition_name, "IMAGERY_START", time.time()])
                time.sleep(IMAGERY_DURATION)
                all_events.append([run_num, overall_trial_count, condition_name, "IMAGERY_END", time.time()])
                print(f"  {time.strftime('%H:%M:%S')}: Imagery - END")

                # 4. ITI
                print(f"  {time.strftime('%H:%M:%S')}: Rest (ITI)")
                all_events.append([run_num, overall_trial_count, condition_name, "ITI_START", time.time()])
                time.sleep(ITI_DURATION)
                all_events.append([run_num, overall_trial_count, condition_name, "ITI_END", time.time()])

            if run_num < RUNS:
                print(f"\n--- End of Run {run_num} ---")
                print(f"Taking a break for {BREAK_BETWEEN_RUNS_DURATION} seconds...")
                all_events.append([run_num, 0, "BREAK", "BREAK_START", time.time()]) # Trial number 0 for break
                time.sleep(BREAK_BETWEEN_RUNS_DURATION)
                all_events.append([run_num, 0, "BREAK", "BREAK_END", time.time()])
                input(f"Press Enter to begin Run {run_num + 1}...")

        print("\n--- Data Collection Session Finished ---")
        print("Retrieving data from board buffer...")
        # board_data will be (total_channels_including_non_eeg, num_samples)
        board_data = board.get_board_data()
        print(f"Retrieved full data shape: {board_data.shape}")

        # Select only C3, C4, and Timestamp data
        # board_data is (channels, samples). We want to select specific rows (channels).
        selected_channel_indices = [
            c3_actual_data_array_index,
            c4_actual_data_array_index,
            timestamp_channel_from_board
        ]
        data_to_save = board_data[selected_channel_indices, :]
        # Transpose so that samples are rows and these 3 selected channels are columns
        data_to_save = data_to_save.T
        print(f"Shape of data to save (C3, C4, Timestamp) - (samples, channels): {data_to_save.shape}")


    except KeyboardInterrupt:
        print("\nSession interrupted by user.")
        if board.is_prepared() and board.get_board_data_count() > 0: # Check if data exists
             board_data = board.get_board_data()
             if board_data is not None and board_data.shape[1] > 0:
                selected_channel_indices = [c3_actual_data_array_index, c4_actual_data_array_index, timestamp_channel_from_board]
                data_to_save = board_data[selected_channel_indices, :].T
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print("Stopping stream and releasing session...")
        if board.is_prepared():
            board.stop_stream()
            board.release_session()
        print("Session ended.")

    if data_to_save is not None and data_to_save.shape[0] > 0:
        print(f"\nSaving C3, C4, and Timestamp data to: {EEG_DATA_FILENAME}")
        # Header for the CSV file containing C3, C4, and Timestamp
        header = ['C3', 'C4', 'Timestamp']
        with open(EEG_DATA_FILENAME, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)
            writer.writerows(data_to_save)
        print("C3, C4, and Timestamp EEG data saved.")
    else:
        print("No specific C3/C4/Timestamp EEG data to save (or an error occurred).")

    if all_events:
        print(f"Saving events to: {EVENTS_FILENAME}")
        with open(EVENTS_FILENAME, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['run_number', 'trial_number_overall', 'condition',
                             'event_type', 'software_timestamp_s'])
            writer.writerows(all_events)
        print("Events saved.")
    else:
        print("No events to save.")

if __name__ == "__main__":
    collect_eeg_session()