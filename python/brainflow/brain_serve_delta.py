import asyncio
import websockets
import json
import logging
import numpy as np
from scipy import signal
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowError
from brainflow.data_filter import DataFilter, DetrendOperations
from collections import deque

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Constants from OpenBCI documentation
CYTON_GAIN = 24.0
SCALE_FACTOR_EEG = (4.5 / CYTON_GAIN) / (2**23 - 1)
SAMPLE_RATE = 250

# Delta band frequency range
DELTA_LOW = 0.5
DELTA_HIGH = 4.0

class DataBuffer:
    def __init__(self, max_size=1000):
        self.buffer = deque(maxlen=max_size)
        self.min_samples = 100  # Minimum samples needed for filtering

    def add_data(self, data):
        if isinstance(data, np.ndarray):
            self.buffer.extend(data)
        else:
            self.buffer.append(data)

    def get_data(self):
        return np.array(self.buffer)

    def has_enough_data(self):
        return len(self.buffer) >= self.min_samples

def extract_delta_power(data, data_buffer):
    """Extract delta band power from EEG data"""
    # Add new data to buffer
    data_buffer.add_data(data)
    
    # Check if we have enough data
    if not data_buffer.has_enough_data():
        return 0  # Return zero if not enough data
        
    # Get buffered data
    data_to_filter = data_buffer.get_data()
    
    # Convert to microvolts and remove DC offset
    data_uv = data_to_filter * SCALE_FACTOR_EEG * 1e6
    
    # Design bandpass filter for delta band
    nyquist = SAMPLE_RATE / 2.0
    b, a = signal.butter(4, [DELTA_LOW/nyquist, DELTA_HIGH/nyquist], btype='bandpass')
    
    # Apply filter to buffered data
    filtered = signal.filtfilt(b, a, data_uv)
    
    # Calculate power (mean of squared values)
    power = np.mean(np.square(filtered[-len(data):]))
    
    # Scale power to better fit 7-40 Hz range
    # First normalize to 0-1 range with typical values, focusing on lower range
    normalized_power = min(1.0, power / 40.0)  # Changed from 100.0 to 40.0
    
    # Apply exponential scaling to spread out lower values
    scaled_power = int(20000 * (normalized_power ** 0.5))
    
    return scaled_power

async def stream_data(websocket):
    logger.info("New client connected")
    logger.debug(f"Client connected from {websocket.remote_address}")
    board = None
    data_buffer = DataBuffer()  # Single buffer for channel 1
    
    try:
        params = BrainFlowInputParams()
        params.serial_port = "/dev/cu.usbserial-DP04WG7H"
        params.timeout = 15

        logger.info(f"Attempting to connect to board at {params.serial_port}")
        board = BoardShim(BoardIds.CYTON_BOARD, params)
        logger.info("Preparing session...")
        
        await asyncio.sleep(1)
        
        # Add retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.info(f"Attempt {attempt + 1} of {max_retries} to prepare session")
                board.prepare_session()
                logger.info("Session prepared successfully")
                break
            except BrainFlowError as e:
                logger.error(f"BrainFlow error during attempt {attempt + 1}: {str(e)}")
                if attempt == max_retries - 1:
                    raise
                logger.warning(f"Retrying in 2 seconds...")
                await asyncio.sleep(2)
        
        logger.info("Starting stream...")
        board.start_stream()
        logger.info("Stream started successfully")

        while True:
            try:
                data = board.get_current_board_data(250)  # Get 1 second of data
                if data.size == 0:
                    logger.warning("Received empty data from board")
                    await asyncio.sleep(0.1)
                    continue
                    
                # Get channel 1 data
                channel_data = data[0]  # First channel
                
                # Process delta power
                DataFilter.detrend(channel_data, DetrendOperations.CONSTANT.value)
                delta_power = extract_delta_power(channel_data, data_buffer)
                
                # Create message with just delta power
                message = {
                    "delta_power": delta_power,
                    "timestamp": int(asyncio.get_event_loop().time() * 1000)
                }
                
                logger.debug(f"Delta power: {delta_power}")
                await websocket.send(json.dumps(message))
                
                await asyncio.sleep(0.05)  # 50ms interval
                
            except Exception as e:
                logger.error(f"Error in data processing: {str(e)}", exc_info=True)
                break
            
    except Exception as e:
        logger.error(f"Error in stream_data: {str(e)}", exc_info=True)
    finally:
        if board is not None:
            try:
                logger.info("Cleaning up board connection")
                if board.is_prepared():
                    logger.info("Stopping stream...")
                    board.stop_stream()
                    logger.info("Releasing session...")
                    board.release_session()
                    logger.info("Cleanup completed successfully")
            except Exception as e:
                logger.error(f"Error during cleanup: {str(e)}", exc_info=True)
        logger.info("Client disconnected")

async def main():
    logger.info("Starting server...")
    async with websockets.serve(stream_data, "0.0.0.0", 8765):
        logger.info("Server running on ws://0.0.0.0:8765")
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())