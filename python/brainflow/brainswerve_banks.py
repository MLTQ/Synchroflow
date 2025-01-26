import asyncio
import websockets
import json
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import logging
from websockets.exceptions import ConnectionClosed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BAND_COLORS = {
    'Delta': 'blue',
    'Theta': 'green',
    'Alpha': 'red',
    'Beta': 'orange',
    'Gamma': 'purple'
}

class BrainWavePlotter:
    def __init__(self):
        plt.ion()  # Enable interactive mode
        self.fig = plt.figure(figsize=(15, 10))
        
        # Subplot for brain waves
        self.ax_waves = self.fig.add_subplot(211)
        self.wave_lines = {}
        self.x = np.arange(250)
        
        # Initialize lines for each frequency band
        for band, color in BAND_COLORS.items():
            line, = self.ax_waves.plot(self.x, np.zeros(250), label=band, 
                                     color=color, lw=2)
            self.wave_lines[band] = line
        
        self.ax_waves.set_title('Brain Wave Bands')
        self.ax_waves.set_ylabel('Amplitude (μV)')
        self.ax_waves.set_ylim(-50, 50)
        self.ax_waves.grid(True)
        self.ax_waves.legend(loc='upper right')
        
        # Subplot for band power
        self.ax_power = self.fig.add_subplot(212)
        x_pos = np.arange(len(BAND_COLORS))
        self.power_bars = self.ax_power.bar(x_pos, np.zeros(len(BAND_COLORS)),
                                          color=list(BAND_COLORS.values()))
        self.ax_power.set_title('Band Powers')
        self.ax_power.set_xlabel('Frequency Band')
        self.ax_power.set_ylabel('Power (μV²)')
        self.ax_power.set_xticks(x_pos)
        self.ax_power.set_xticklabels(list(BAND_COLORS.keys()))
        
        # Add text for power values
        self.power_text = []
        for i in range(len(BAND_COLORS)):
            txt = self.ax_power.text(i, 0, '', ha='center', va='bottom')
            self.power_text.append(txt)
        
        plt.tight_layout()
        self.fig.canvas.draw()
        plt.pause(0.1)  # Small pause to ensure window is shown

    def update(self, band_data, band_powers):
        try:
            # Update wave plots
            for band_name, line in self.wave_lines.items():
                y_data = np.array(band_data[band_name])
                line.set_data(self.x[:len(y_data)], y_data)
            
            # Update power bars and text
            for idx, (band_name, power) in enumerate(band_powers.items()):
                self.power_bars[idx].set_height(power)
                self.power_text[idx].set_text(f'{power:.1f}')
                self.power_text[idx].set_position((idx, power))
            
            # Update power axis limits
            max_power = max(band_powers.values())
            self.ax_power.set_ylim(0, max_power * 1.1)
            
            # Redraw
            self.fig.canvas.draw_idle()
            plt.pause(0.001)  # Small pause to allow GUI to update
            
        except Exception as e:
            logger.error(f"Error updating plot: {e}")

async def data_receiver(websocket, plotter):
    try:
        while True:
            try:
                message = await websocket.recv()
                data = json.loads(message)
                plotter.update(data['band_data'], data['band_powers'])
            except ConnectionClosed:
                logger.info("Connection closed by server")
                break
            except Exception as e:
                logger.error(f"Error processing data: {e}")
                continue
    except Exception as e:
        logger.error(f"Error in data receiver: {e}")

async def main():
    uri = "ws://localhost:8765"
    plotter = BrainWavePlotter()
    
    while True:
        try:
            async with websockets.connect(uri) as websocket:
                logger.info("Connected to WebSocket server")
                await data_receiver(websocket, plotter)
        except Exception as e:
            logger.error(f"Connection error: {e}")
            await asyncio.sleep(5)
        finally:
            logger.info("Attempting to reconnect...")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Program terminated by user")
        plt.close('all')
    except Exception as e:
        logger.error(f"Program terminated due to error: {e}")
        plt.close('all')