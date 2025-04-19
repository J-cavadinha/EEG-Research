import time
import csv
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter

def get_eeg_data(board_id, serial_port, label):
    params = BrainFlowInputParams()
    params.serial_port = serial_port
    board = BoardShim(board_id, params)
    board.prepare_session()
    board.start_stream()

    timestamp_channel = BoardShim.get_timestamp_channel(board_id)
    eeg_channels = BoardShim.get_eeg_channels(board_id)
    channel_3_index = eeg_channels[2]
    channel_4_index = eeg_channels[3]

    file_name = f"eeg_data_{time.strftime('%Y%m%d-%H%M%S')}.csv"
    start_time = time.time()

    with open(file_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['timestamp', 'channel_3', 'channel_4', 'label'])

        try:
            while True:
                data = board.get_board_data()
                if data.shape[1] > 0:
                    timestamp = data[timestamp_channel, -1]
                    channel_3_data = data[channel_3_index, -1]
                    channel_4_data = data[channel_4_index, -1]
                    writer.writerow([timestamp, channel_3_data, channel_4_data, label])
                    elapsed_time = int(time.time() - start_time)
                    print(f"\rRecording time: {elapsed_time} seconds", end="", flush=True)
        except KeyboardInterrupt:
            final_time = int(time.time() - start_time)
            print(f"\nData collection stopped by user after {final_time} seconds.")
        finally:
            board.stop_stream()
            board.release_session()

if __name__ == "__main__":
    board_id = BoardIds.CYTON_BOARD.value
    serial_port = "/dev/cu.usbserial-DQ007TQ0"
    label = input("Enter label for this data: ")
    get_eeg_data(board_id, serial_port, label)