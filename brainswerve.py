import asyncio
import websockets
import json
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np


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


async def main():
    plotter = EEGPlotter()
    async with websockets.connect('ws://localhost:8765') as websocket:
        while True:
            try:
                data = await websocket.recv()
                eeg_data = json.loads(data)['eeg_data']
                plotter.update(eeg_data)
            except Exception as e:
                print(f"Error: {e}")
                break


if __name__ == "__main__":
    asyncio.run(main())