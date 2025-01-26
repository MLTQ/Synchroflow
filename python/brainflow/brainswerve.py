import asyncio
import websockets
import json
import matplotlib
matplotlib.use('TkAgg')  # Changed to TkAgg for better interactive support
import matplotlib.pyplot as plt
import numpy as np
import logging
from websockets.exceptions import ConnectionClosed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GammaPlotter:
    def __init__(self):
        self.fig = plt.figure(figsize=(15, 10))
        
        # Subplot for gamma waves
        self.ax_waves = self.fig.add_subplot(211)
        self.gamma_lines = []
        self.x = np.arange(100)
        
        # Initialize 8 lines for gamma waves
        for i in range(8):
            line, = self.ax_waves.plot(self.x, np.zeros(100), label=f'Channel {i+1}')
            self.gamma_lines.append(line)
        
        self.ax_waves.set_title('Gamma Waves (30-100 Hz)')
        self.ax_waves.set_ylabel('Voltage (V)')
        self.ax_waves.grid(True)
        self.ax_waves.legend()
        
        # Subplot for gamma power
        self.ax_power = self.fig.add_subplot(212)
        self.power_bars = self.ax_power.bar(range(8), np.zeros(8))
        self.ax_power.set_title('Gamma Power by Channel')
        self.ax_power.set_xlabel('Channel')
        self.ax_power.set_ylabel('Power (μV²)')
        
        plt.tight_layout()
        plt.ion()
        self.fig.show()

    def update(self, gamma_data, gamma_power, channel_names):
        # Update gamma waves
        for idx, line in enumerate(self.gamma_lines):
            y_data = np.array(gamma_data[idx])[:100]
            if len(y_data) < 100:
                y_data = np.pad(y_data, (0, 100 - len(y_data)))
            line.set_ydata(y_data)
            line.set_label(f'{channel_names[idx]}: {gamma_power[idx]:.1f} μV²')
        
        # Update power bars
        for bar, power in zip(self.power_bars, gamma_power):
            bar.set_height(power)
        
        # Update axis limits for power plot
        self.ax_power.set_ylim(0, max(gamma_power) * 1.1)
        
        # Update legends and draw
        self.ax_waves.legend()
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
    plotter = GammaPlotter()
    uri = 'ws://localhost:8765'
    
    while True:
        try:
            async with await connect_with_retry(uri) as websocket:
                logger.info("Connected to WebSocket server")
                while True:
                    try:
                        data = json.loads(await websocket.recv())
                        gamma_data = data['gamma_data']
                        gamma_power = data['gamma_power']
                        channel_names = data['channel_names']
                        plotter.update(gamma_data, gamma_power, channel_names)
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
            await asyncio.sleep(5)
            continue

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Program terminated by user")
        plt.close('all')
    except Exception as e:
        logger.error(f"Program terminated due to error: {str(e)}")
        plt.close('all')