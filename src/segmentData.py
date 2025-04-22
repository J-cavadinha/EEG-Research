import pandas as pd
import numpy as np
import time

def combine_eeg_files(file1, file2):
    try:
        data1 = pd.read_csv(file1)
        data2 = pd.read_csv(file2)
        combined_data = pd.concat([data1, data2], ignore_index=True)
        combined_data = combined_data.sort_values('timestamp').reset_index(drop=True)
        return combined_data
    except FileNotFoundError as e:
        print(f"Error: One or both files not found - {e}")
        return None
    except Exception as e:
        print(f"Error combining files: {e}")
        return None

def segment_trials(csv_files, min_duration=1.5, max_duration=1.5):
    if len(csv_files) == 2:
        df = combine_eeg_files(csv_files[0], csv_files[1])
    else:
        print("Error: Please provide exactly two CSV files")
        return None

    if df is None:
        return None

    df['timestamp'] = pd.to_numeric(df['timestamp'])
    print(f"Total samples: {len(df)}")
    print(f"Labels found: {df['label'].unique()}")
    
    trials = []
    trial_number = 0
    
    for label in df['label'].unique():
        label_data = df[df['label'] == label].copy()
        print(f"\nProcessing data for label {label}")
        
        # Calculate sampling rate more robustly
        time_diff = np.diff(label_data['timestamp'].values)
        sampling_rate = 1 / np.median(time_diff)  # Using median instead of mean
        samples_per_segment = max(1, int(max_duration * sampling_rate))  # Ensure at least 1 sample
        
        print(f"Time differences stats:")
        print(f"Mean: {np.mean(time_diff):.4f}s")
        print(f"Median: {np.median(time_diff):.4f}s")
        print(f"Samples per segment: {samples_per_segment}")
        
        if samples_per_segment == 0:
            print("Warning: Sample rate too low for the specified duration")
            continue
        
        total_duration = label_data['timestamp'].max() - label_data['timestamp'].min()
        print(f"Total duration for label {label}: {total_duration:.2f}s")
        print(f"Sampling rate: {sampling_rate:.2f} Hz")
        
        for i in range(0, len(label_data) - samples_per_segment, samples_per_segment):
            trial = label_data.iloc[i:i + samples_per_segment].copy()
            
            duration = trial['timestamp'].iloc[-1] - trial['timestamp'].iloc[0]
            if abs(duration - max_duration) < 0.1:
                trial['trial_number'] = trial_number
                trials.append(trial)
                print(f"Added trial {trial_number} (duration: {duration:.2f}s, label: {label})")
                trial_number += 1

    if trials:
        trials_df = pd.concat(trials, ignore_index=True)
        output_filename = f"segmented_data_{time.strftime('%Y%m%d-%H%M%S')}.csv"
        trials_df.to_csv(output_filename, index=False)
        print(f"\nTotal trials found: {trial_number}")
        print(f"Segmented data saved to: {output_filename}")
        return trials_df
    else:
        print("\nNo trials found matching the duration criteria:")
        print(f"Min duration: {min_duration}s")
        print(f"Max duration: {max_duration}s")
        return pd.DataFrame()

if __name__ == "__main__":
    csv_files = [
        '/Users/joaomachado/Desktop/IC_V3_BKP/merged_output_md.csv',
        '/Users/joaomachado/Desktop/IC_V3_BKP/merged_output_me.csv'
    ]
    segmented_data = segment_trials(csv_files, min_duration=1.5, max_duration=1.5)
    if segmented_data is not None and not segmented_data.empty:
        print("Segmentation completed successfully!")