import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
import matplotlib
import logging
from scipy import signal
from brainflow.data_filter import DataFilter, DetrendOperations

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Constants from OpenBCI documentation
CYTON_GAIN = 24.0
SCALE_FACTOR_EEG = (4.5 / CYTON_GAIN) / (2**23 - 1)
SAMPLE_RATE = 250

def extract_gamma(data):
    """Extract gamma band (30-100 Hz) from EEG data"""
    # Convert to microvolts and remove DC offset
    data_uv = data * SCALE_FACTOR_EEG * 1e6
    
    # Design bandpass filter for gamma (30-100 Hz)
    nyquist = SAMPLE_RATE / 2.0
    b, a = signal.butter(4, [30/nyquist, 100/nyquist], btype='bandpass')
    
    # Apply filter
    gamma = signal.filtfilt(b, a, data_uv)
    return gamma

def update(frame, board, lines, power_bars, power_text):
    try:
        data = board.get_current_board_data(250)
        if data.size > 0:
            eeg_data = data[:8]  # Get first 8 channels
            gamma_data = []
            gamma_powers = []
            
            # Process each channel
            for channel in eeg_data:
                # Remove DC offset
                DataFilter.detrend(channel, DetrendOperations.CONSTANT.value)
                
                # Extract gamma
                gamma = extract_gamma(channel)
                gamma_data.append(gamma)
                
                # Calculate power
                power = np.mean(np.square(gamma))
                gamma_powers.append(power)
                
            # Update wave plots
            x = np.arange(len(gamma_data[0]))
            for idx, (line, gamma) in enumerate(zip(lines, gamma_data)):
                line.set_data(x, gamma)
            
            # Update power bars and text
            for idx, (power, bar, txt) in enumerate(zip(gamma_powers, power_bars, power_text)):
                bar.set_height(power)
                txt.set_text(f'Ch{idx+1}: {power:.1f} μV²')
                
                # Log some debug info for first channel
                if idx == 0:
                    logger.debug(f"Channel 1 - Power: {power:.2f} μV², Range: {np.min(gamma_data[0]):.2f} to {np.max(gamma_data[0]):.2f} μV")

        return lines + list(power_bars) + power_text
    except Exception as e:
        logger.error(f"Error in update function: {e}")
        return lines + list(power_bars) + power_text

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

        # Create figure with two subplots
        fig = plt.figure(figsize=(15, 10))
        
        # Top subplot for gamma waves
        ax_waves = plt.subplot(211)
        lines = []
        for i in range(8):
            line, = ax_waves.plot([], [], label=f'Channel {i+1}', lw=1)
            lines.append(line)
        ax_waves.set_xlim(0, 250)
        ax_waves.set_ylim(-50, 50)  # Set reasonable μV range for gamma
        ax_waves.set_title('Gamma Waves (30-100 Hz)')
        ax_waves.set_ylabel('Amplitude (μV)')
        ax_waves.grid(True)
        ax_waves.legend(loc='upper right')

        # Bottom subplot for gamma power
        ax_power = plt.subplot(212)
        x_pos = np.arange(8)
        power_bars = ax_power.bar(x_pos, np.zeros(8))
        ax_power.set_ylim(0, 1000)  # Set reasonable μV² range for power
        power_text = []
        for i in range(8):
            txt = ax_power.text(i, 0, '', ha='center', va='bottom')
            power_text.append(txt)
        ax_power.set_title('Gamma Power by Channel')
        ax_power.set_xlabel('Channel')
        ax_power.set_ylabel('Power (μV²)')
        ax_power.set_xticks(x_pos)
        ax_power.set_xticklabels([f'Ch{i+1}' for i in range(8)])

        plt.tight_layout()

        ani = FuncAnimation(fig, update, fargs=(board, lines, power_bars, power_text),
                          interval=50, blit=True, cache_frame_data=False)
        plt.show()

    except Exception as e:
        logger.error(f"Error in main function: {e}")
    finally:
        logger.info("Cleaning up board connection")
        board.stop_stream()
        board.release_session()

if __name__ == "__main__":
    main()