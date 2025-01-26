import asyncio
import websockets
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    uri = 'ws://localhost:8765'
    try:
        async with websockets.connect(uri) as websocket:
            logger.info("Connected to WebSocket server")
            while True:
                try:
                    data = json.loads(await websocket.recv())
                    band_powers = data['band_powers']
                    timestamp = data['timestamp']
                    
                    print("\nBrain Wave Power Levels (μV²):")
                    print("-" * 50)
                    for band, power in band_powers.items():
                        # Format power to 2 decimal places
                        print(f"{band}: {power:.2f} μV²")
                    print(f"Timestamp: {timestamp}")
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