import asyncio
import websockets
import json
import logging
import numpy as np
from scipy import signal
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, DetrendOperations

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Constants from OpenBCI documentation
CYTON_GAIN = 24.0
SCALE_FACTOR_EEG = (4.5 / CYTON_GAIN) / (2**23 - 1)
SAMPLE_RATE = 250

# Frequency bands
FREQ_BANDS = {
    'Delta': (0.5, 4),
    'Theta': (4, 8),
    'Alpha': (8, 13),
    'Beta': (13, 30),
    'Gamma': (30, 100)
}

def extract_band(data, band_range):
    """Extract frequency band from EEG data"""
    # Convert to microvolts and remove DC offset
    data_uv = data * SCALE_FACTOR_EEG * 1e6
    
    # Design bandpass filter
    nyquist = SAMPLE_RATE / 2.0
    b, a = signal.butter(4, [band_range[0]/nyquist, band_range[1]/nyquist], btype='bandpass')
    
    # Apply filter
    filtered = signal.filtfilt(b, a, data_uv)
    return filtered

async def stream_data(websocket):
    logger.info("New client connected")
    logger.debug(f"Client connected from {websocket.remote_address}")
    try:
        params = BrainFlowInputParams()
        params.serial_port = "/dev/cu.usbserial-DP04WG7H"
        params.timeout = 15

        board = BoardShim(BoardIds.CYTON_BOARD, params)
        logger.info("Preparing session...")
        board.prepare_session()
        logger.info("Starting stream...")
        board.start_stream()

        while True:
            data = board.get_current_board_data(250)  # Get 1 second of data
            if data.size > 0:
                eeg_data = data[:8]  # Get first 8 channels
                band_data = {band: np.zeros(eeg_data.shape[1]) for band in FREQ_BANDS}
                band_powers = {band: 0 for band in FREQ_BANDS}
                
                # Process each channel
                for channel in eeg_data:
                    # Remove DC offset
                    DataFilter.detrend(channel, DetrendOperations.CONSTANT.value)
                    
                    # Process each frequency band
                    for band_name, band_range in FREQ_BANDS.items():
                        # Extract band
                        band = extract_band(channel, band_range)
                        
                        # Add to combined band data
                        band_data[band_name] += band
                        
                        # Add to band power
                        band_powers[band_name] += np.mean(np.square(band))
                
                # Average across channels
                for band_name in FREQ_BANDS:
                    band_data[band_name] /= len(eeg_data)
                    band_powers[band_name] /= len(eeg_data)
                
                # Prepare data for sending
                message = {
                    "band_data": {
                        band: data.tolist() 
                        for band, data in band_data.items()
                    },
                    "band_powers": band_powers,
                    "timestamp": int(asyncio.get_event_loop().time() * 1000)  # milliseconds
                }
                
                # Log debug info
                logger.debug(f"Band Powers: {', '.join([f'{band}: {power:.2f} μV²' for band, power in band_powers.items()])}")
                
                # Send data
                await websocket.send(json.dumps(message))
            
            await asyncio.sleep(0.05)  # 50ms interval
            
    except Exception as e:
        logger.error("Error in stream_data", exc_info=True)
    finally:
        logger.info("Cleaning up board connection")
        board.stop_stream()
        board.release_session()
        logger.info("Client disconnected")

async def main():
    logger.info("Starting server...")
    async with websockets.serve(stream_data, "0.0.0.0", 8765):
        logger.info("Server running on ws://0.0.0.0:8765")
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())