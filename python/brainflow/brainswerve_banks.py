import asyncio
import websockets
import json
import matplotlib
matplotlib.use('TkAgg')  # Make sure this is set before importing pyplot
import matplotlib.pyplot as plt
import numpy as np
import threading
import logging
from queue import Queue, Empty
import tkinter as tk
from matplotlib.animation import FuncAnimation

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

BAND_COLORS = {
    'Delta': 'blue',
    'Theta': 'green',
    'Alpha': 'red',
    'Beta': 'orange',
    'Gamma': 'purple'
}

class DataPlotter:
    def __init__(self):
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 8))
        self.lines = {}
        
        # Initialize lines for band data
        for band, color in BAND_COLORS.items():
            self.lines[band], = self.ax1.plot([], [], color=color, label=band)
        
        self.ax1.set_title('Real-time Brain Wave Bands')
        self.ax1.set_xlabel('Sample')
        self.ax1.set_ylabel('Amplitude (μV)')
        self.ax1.grid(True)
        self.ax1.legend()
        
        # Initialize bar plot for powers
        self.bar_container = self.ax2.bar(BAND_COLORS.keys(), [0] * len(BAND_COLORS), 
                                        color=BAND_COLORS.values())
        self.ax2.set_title('Band Powers')
        self.ax2.set_ylabel('Power (μV²)')
        
        self.data_queue = Queue()
        plt.tight_layout()
        
        # Setup animation
        self.ani = FuncAnimation(
            self.fig, self.update, interval=50,
            blit=False, cache_frame_data=False)

    def update(self, frame):
        try:
            data = self.data_queue.get_nowait()
            
            # Update band data plots
            for band, values in data['band_data'].items():
                self.lines[band].set_data(range(len(values)), values)
            
            # Update power bars
            for idx, (band, power) in enumerate(data['band_powers'].items()):
                self.bar_container[idx].set_height(power)
            
            # Adjust axes limits
            self.ax1.relim()
            self.ax1.autoscale_view()
            self.ax2.relim()
            self.ax2.autoscale_view()
            
        except Empty:
            pass
        except Exception as e:
            logger.error(f"Plot update exception: {str(e)}", exc_info=True)

async def handle_websocket(plotter):
    uri = "ws://localhost:8765"
    while True:
        try:
            async with websockets.connect(uri) as websocket:
                logger.info("Connected to WebSocket server")
                
                while True:
                    try:
                        message = await websocket.recv()
                        logger.debug(f"Received raw message: {message[:100]}...")
                        
                        data = json.loads(message)
                        logger.debug(f"Parsed data keys: {data.keys()}")
                        
                        if 'band_data' not in data or 'band_powers' not in data:
                            logger.error(f"Missing required data fields. Available keys: {data.keys()}")
                            continue
                            
                        plotter.data_queue.put(data)
                        
                    except websockets.ConnectionClosed:
                        logger.warning("Connection closed by server, attempting to reconnect...")
                        break
                    except json.JSONDecodeError as e:
                        logger.error(f"JSON decode error: {e}", exc_info=True)
                        continue
                    except Exception as e:
                        logger.error(f"Error processing message: {e}", exc_info=True)
                        continue
                        
        except Exception as e:
            logger.error(f"Connection error: {e}", exc_info=True)
            await asyncio.sleep(5)
            logger.info("Attempting to reconnect...")
            continue

async def main_async():
    # Create the plotter
    plotter = DataPlotter()
    
    # Create tasks
    websocket_task = asyncio.create_task(handle_websocket(plotter))
    
    try:
        # Show the plot (non-blocking)
        plt.show(block=False)
        
        # Keep the program running
        while plt.get_fignums():  # While there are still figures open
            await asyncio.sleep(0.1)
            plt.pause(0.1)  # Allow matplotlib to update
            
    except KeyboardInterrupt:
        logger.info("Program terminated by user")
    finally:
        # Clean up
        websocket_task.cancel()
        try:
            await websocket_task
        except asyncio.CancelledError:
            pass
        plt.close('all')

def main():
    try:
        # Create and run the asyncio event loop
        asyncio.run(main_async())
    except KeyboardInterrupt:
        logger.info("Program terminated by user")

if __name__ == "__main__":
    main()