import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
import matplotlib
import logging

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def update(frame, board, lines):
    try:
        data = board.get_current_board_data(250)
        if data.size > 0:
            eeg_data = data[:8]
            x = np.arange(eeg_data.shape[1])
            for idx, line in enumerate(lines):
                line.set_data(x, eeg_data[idx])
                line.axes.relim()  # Recompute the data limits
                line.axes.autoscale_view()  # Autoscale the view
        return lines
    except Exception as e:
        logger.error(f"Error in update function: {e}")
        return lines

def main():
    try:
        BoardShim.enable_dev_board_logger()
        params = BrainFlowInputParams()
        params.serial_port = "/dev/cu.usbserial-DP04WG7H"
        params.timeout = 15

        board = BoardShim(BoardIds.CYTON_BOARD, params)
        logger.info("Preparing session...")
        board.prepare_session()
        logger.info("Starting stream...")
        board.start_stream()

        fig, axes = plt.subplots(8, 1, figsize=(12, 8))
        plt.subplots_adjust(hspace=0.5)
        lines = []

        for idx, ax in enumerate(axes):
            line, = ax.plot([], [])
            ax.set_xlim(0, 250)
            ax.set_title(f'Channel {idx + 1}')
            ax.grid(True)
            lines.append(line)

        ani = FuncAnimation(fig, update, fargs=(board, lines),
                            interval=50, blit=True)
        plt.show()
    except Exception as e:
        logger.error(f"Error in main function: {e}")
    finally:
        logger.info("Cleaning up board connection")
        board.stop_stream()
        board.release_session()

if __name__ == "__main__":
    main()