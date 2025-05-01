import pandas as pd
import mne
from mne.preprocessing import compute_current_source_density
import numpy as np
import time

def filter_eeg_data(csv_file):
    try:
        df = pd.read_csv(csv_file)
        print(f"Loaded {len(df)} samples from segmented data")
        print(f"Found {len(df['trial_number'].unique())} trials")
    except FileNotFoundError:
        print(f"Error: File not found - {csv_file}")
        return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

    eeg_data = df[['channel_3', 'channel_4']].values.T
    ch_names = ['channel_3', 'channel_4']
    ch_types = ['eeg', 'eeg']
    info = mne.create_info(ch_names=ch_names, sfreq=512, ch_types=ch_types)
    raw_data = mne.io.RawArray(eeg_data, info)
    raw_data.rename_channels({'channel_3': 'C3', 'channel_4': 'C4'})
    montage = mne.channels.make_standard_montage('standard_1020')
    raw_data.set_montage(montage)
    raw_data.notch_filter(freqs=60, picks=['C3', 'C4'])
    raw_data.filter(l_freq=0.5, h_freq=70, picks=['C3', 'C4'])
    filtered_data = raw_data.get_data(picks=['C3', 'C4']).T
    filtered_df = pd.DataFrame(filtered_data, columns=['channel_3', 'channel_4'])
    filtered_df['timestamp'] = df['timestamp'].values
    filtered_df['label'] = df['label'].values
    filtered_df['trial_number'] = df['trial_number'].values
    filtered_df = filtered_df.iloc[1:].reset_index(drop=True)
    output_filename = f"filtered_segmented_data_{time.strftime('%Y%m%d-%H%M%S')}.csv"
    filtered_df.to_csv(output_filename, index=False)
    print(f"Filtered data saved to: {output_filename}")
    print(f"First row of data excluded from output")
    return filtered_df

if __name__ == "__main__":
    segmented_file = '/Users/joaomachado/Desktop/IC_V3_BKP/segmented_data_20250422-184733.csv'
    filtered_data = filter_eeg_data(segmented_file)
    if filtered_data is not None:
        print("Filtering completed successfully!")