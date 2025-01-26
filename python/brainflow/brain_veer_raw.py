import asyncio
import websockets
import json
import logging
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def format_channel_value(value):
    """Format channel value to be more readable"""
    if abs(value) > 1000:
        return f"{value/1000:.2f}mV"
    else:
        return f"{value:.2f}μV"

async def main():
    uri = 'ws://localhost:8765'  # Connect to localhost
    try:
        async with websockets.connect(uri) as websocket:
            logger.info("Connected to WebSocket server")
            while True:
                try:
                    data = await websocket.recv()
                    eeg_data = json.loads(data)['eeg_data']
                    
                    # Validate data structure
                    if not eeg_data or not all(isinstance(ch, list) for ch in eeg_data):
                        logger.warning("Invalid data structure received")
                        continue
                    
                    logger.info("Received EEG data")
                    print("\nEEG Data Channels:")
                    print("-" * 50)
                    
                    # Calculate statistics for each channel
                    for i, channel in enumerate(eeg_data[:8]):
                        if channel:  # Check if channel has data
                            channel_data = np.array(channel)
                            mean_val = np.mean(channel_data)
                            min_val = np.min(channel_data)
                            max_val = np.max(channel_data)
                            
                            print(f"Channel {i+1}:")
                            print(f"  Current: {format_channel_value(channel[-1])}")
                            print(f"  Mean: {format_channel_value(mean_val)}")
                            print(f"  Range: {format_channel_value(min_val)} to {format_channel_value(max_val)}")
                    
                    print("-" * 50)
                    
                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode error: {str(e)}")
                except KeyError as e:
                    logger.error(f"Missing key in data: {str(e)}")
                except IndexError as e:
                    logger.error(f"Index error processing data: {str(e)}")
                except Exception as e:
                    logger.error(f"Unexpected error: {str(e)}")
                    break
                
                await asyncio.sleep(0.1)  # Add small delay to prevent CPU overload
                
    except Exception as e:
        logger.error(f"Connection error: {str(e)}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProgram terminated by user")