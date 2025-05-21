from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

params = BrainFlowInputParams()
params.serial_port = "/dev/cu.usbserial-DQ007TQ0" # or 'COM3' etc.

# Specify CYTON_DAISY_BOARD
board_id = BoardIds.CYTON_DAISY_BOARD.value
board = BoardShim(board_id, params)

board.prepare_session()
board.start_stream()

# To get sampling rate and channel count:
sampling_rate = BoardShim.get_sampling_rate(board_id)
eeg_channels = BoardShim.get_eeg_channels(board_id) # List of EEG channel indices

print(f"Board: {BoardIds.CYTON_DAISY_BOARD.name}")
print(f"Sampling Rate: {sampling_rate} Hz") # Should be 125 Hz
print(f"EEG Channels count: {len(eeg_channels)}") # Should be 16

board.stop_stream()
board.release_session()