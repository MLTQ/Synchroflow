import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def update(frame, board, lines):
    data = board.get_current_board_data(250)
    if data.size > 0:
        eeg_data = data[:8]
        x = np.arange(eeg_data.shape[1])
        for idx, line in enumerate(lines):
            line.set_data(x, eeg_data[idx])
    return lines


def main():
    BoardShim.enable_dev_board_logger()
    params = BrainFlowInputParams()
    params.serial_port = "/dev/cu.usbserial-DP05IOKG"
    params.timeout = 15

    board = BoardShim(BoardIds.CYTON_BOARD, params)
    board.prepare_session()
    board.start_stream()

    fig, axes = plt.subplots(8, 1, figsize=(12, 8))
    plt.subplots_adjust(hspace=0.5)
    lines = []

    for idx, ax in enumerate(axes):
        line, = ax.plot([], [])
        ax.set_xlim(0, 250)
        ax.set_ylim(-200000, 200000)
        ax.set_title(f'Channel {idx + 1}')
        ax.grid(True)
        lines.append(line)

    ani = FuncAnimation(fig, update, fargs=(board, lines),
                        interval=50, blit=True)
    plt.show()


if __name__ == "__main__":
    main()