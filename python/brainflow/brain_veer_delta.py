import asyncio
import websockets
import json
import logging
from datetime import datetime
from websockets.exceptions import ConnectionClosed
import os
import sys

# Set up logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

def clear_screen():
    """Clear the terminal screen for better visualization"""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_delta_power(delta_power):
    """Pretty print the delta power with enhanced formatting"""
    clear_screen()
    
    print("\n🧠 Delta Wave Power 🧠")
    print("=" * 50)
    
    # Create a visual bar representation
    bar_length = int((delta_power / 200) * 40)  # Scale to 40 characters max
    bar = "█" * bar_length
    
    # Print power value and bar
    print(f"Delta Power: {delta_power:3d} {bar}")
    
    # Print timestamp
    print("=" * 50)
    print(f"Time: {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")
    print("\nPress Ctrl+C to exit")

async def connect_and_monitor():
    uri = 'ws://localhost:8765'
    reconnect_delay = 3  # Initial reconnect delay in seconds
    max_reconnect_delay = 30  # Maximum reconnect delay
    
    while True:  # Keep trying to reconnect
        try:
            async with websockets.connect(uri) as websocket:
                logger.info("Connected to WebSocket server")
                print("\nMonitoring delta wave power in real-time...")
                
                # Reset reconnect delay on successful connection
                reconnect_delay = 3
                
                while True:
                    try:
                        # Receive and parse data
                        message = await websocket.recv()
                        data = json.loads(message)
                        
                        # Validate data structure
                        if 'delta_power' not in data:
                            logger.warning("Received incomplete data structure")
                            continue
                        
                        # Print delta power
                        print_delta_power(data['delta_power'])
                        
                    except ConnectionClosed:
                        logger.warning("Connection closed by server, attempting to reconnect...")
                        break
                    except json.JSONDecodeError as e:
                        logger.error(f"JSON decode error: {str(e)}")
                        continue
                    except KeyError as e:
                        logger.error(f"Missing expected data key: {str(e)}")
                        continue
                    except Exception as e:
                        logger.error(f"Unexpected error: {str(e)}")
                        break
                    
        except ConnectionRefusedError:
            logger.error(f"Could not connect to server at {uri}. Is it running?")
        except Exception as e:
            logger.error(f"Connection error: {str(e)}")
        
        # Exponential backoff for reconnection attempts
        logger.info(f"Retrying connection in {reconnect_delay} seconds...")
        await asyncio.sleep(reconnect_delay)
        reconnect_delay = min(reconnect_delay * 1.5, max_reconnect_delay)

async def main():
    try:
        # Print startup banner
        clear_screen()
        print("\n🧠 Delta Wave Monitor 🧠")
        print("=" * 50)
        print("Connecting to delta wave server...")
        
        await connect_and_monitor()
    except KeyboardInterrupt:
        logger.info("\nMonitoring stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
    finally:
        print("\nThank you for using Delta Wave Monitor!")

if __name__ == "__main__":
    try:
        if sys.platform == 'win32':
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProgram terminated by user")