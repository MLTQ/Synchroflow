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
                    gamma_power = data['gamma_power']
                    channel_names = data['channel_names']
                    
                    print("\nGamma Power Levels (μV²):")
                    print("-" * 50)
                    for name, power in zip(channel_names, gamma_power):
                        # Format power to 2 decimal places
                        print(f"{name}: {power:.2f} μV²")
                    print(f"Sampling Rate: {data['units']['gamma_power']}")
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