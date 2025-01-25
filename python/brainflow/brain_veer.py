import asyncio
import websockets
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    uri = 'ws://localhost:8765'  # Connect to localhost
    try:
        async with websockets.connect(uri) as websocket:
            logger.info("Connected to WebSocket server")
            while True:
                try:
                    data = await websocket.recv()
                    eeg_data = json.loads(data)['eeg_data']
                    logger.info("Received EEG data")
                    print("EEG Data Channels:")
                    for i, channel in enumerate(eeg_data[:8]):
                        print(f"Ch{i+1}: {channel[-1]}")  # Print latest value
                    print("-" * 50)
                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode error: {str(e)}")
                except Exception as e:
                    logger.error(f"Unexpected error: {str(e)}")
                    break
    except Exception as e:
        logger.error(f"Connection error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())