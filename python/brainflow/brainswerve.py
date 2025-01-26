import asyncio
import websockets
import json
import matplotlib
matplotlib.use('Agg')  # Change to non-GUI backend
import matplotlib.pyplot as plt
import numpy as np
import logging
from websockets.exceptions import ConnectionClosed

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EEGPlotter:
    def __init__(self):
        self.fig, self.axes = plt.subplots(8, 1, figsize=(12, 8))
        plt.subplots_adjust(hspace=0.5)
        self.lines = []
        self.x = np.arange(100)

        for idx, ax in enumerate(self.axes):
            line, = ax.plot(self.x, np.zeros(100))
            ax.set_ylim(-200000, 200000)
            ax.set_title(f'Channel {idx + 1}')
            ax.grid(True)
            self.lines.append(line)
        plt.ion()
        self.fig.show()

    def update(self, eeg_data):
        for idx, line in enumerate(self.lines):
            y_data = np.array(eeg_data[idx])[:100]  # Ensure consistent length
            if len(y_data) < 100:
                y_data = np.pad(y_data, (0, 100 - len(y_data)))
            line.set_ydata(y_data)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()


async def connect_with_retry(uri, max_retries=5, delay=2):
    for attempt in range(max_retries):
        try:
            logger.info(f"Attempting to connect to {uri} (attempt {attempt + 1}/{max_retries})")
            return await websockets.connect(uri)
        except Exception as e:
            logger.error(f"Connection attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                await asyncio.sleep(delay)
            else:
                raise

async def main():
    plotter = EEGPlotter()
    uri = 'ws://localhost:8765'  # Connect to localhost
    
    while True:
        try:
            async with await connect_with_retry(uri) as websocket:
                logger.info("Connected to WebSocket server")
                while True:
                    try:
                        data = await websocket.recv()
                        eeg_data = json.loads(data)['eeg_data']
                        plotter.update(eeg_data)
                    except ConnectionClosed as e:
                        logger.error(f"WebSocket connection closed: {str(e)}")
                        break
                    except json.JSONDecodeError as e:
                        logger.error(f"JSON decode error: {str(e)}")
                        continue
                    except Exception as e:
                        logger.error(f"Unexpected error: {str(e)}")
                        break
        except Exception as e:
            logger.error(f"Connection error: {str(e)}")
            await asyncio.sleep(5)  # Wait before retrying
            continue

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Program terminated by user")
    except Exception as e:
        logger.error(f"Program terminated due to error: {str(e)}")