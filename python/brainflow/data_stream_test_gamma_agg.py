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

def update(frame, board, line, power_text):
    try:
        data = board.get_current_board_data(250)
        if data.size > 0:
            eeg_data = data[:8]  # Get first 8 channels
            combined_gamma = np.zeros(eeg_data.shape[1])
            channel_powers = []
            
            # Process each channel
            for channel in eeg_data:
                # Remove DC offset
                DataFilter.detrend(channel, DetrendOperations.CONSTANT.value)
                
                # Extract gamma
                gamma = extract_gamma(channel)
                
                # Add to combined gamma
                combined_gamma += gamma
                
                # Calculate individual channel power
                power = np.mean(np.square(gamma))
                channel_powers.append(power)
            
            # Calculate average gamma (divide by number of channels)
            combined_gamma /= len(eeg_data)
            
            # Update wave plot
            x = np.arange(len(combined_gamma))
            line.set_data(x, combined_gamma)
            
            # Calculate and update total power
            total_power = np.mean(channel_powers)
            power_text.set_text(f'Total Gamma Power: {total_power:.1f} μV²\n' + 
                              f'Average Channel Power: {total_power/8:.1f} μV²')
            
            # Log debug info
            logger.debug(f"Combined Gamma - Power: {total_power:.2f} μV², " +
                        f"Range: {np.min(combined_gamma):.2f} to {np.max(combined_gamma):.2f} μV")

        return [line, power_text]
    except Exception as e:
        logger.error(f"Error in update function: {e}")
        return [line, power_text]

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

        # Create figure with single plot
        fig = plt.figure(figsize=(15, 8))
        
        # Plot for combined gamma wave
        ax_wave = plt.subplot(111)
        line, = ax_wave.plot([], [], label='Combined Gamma', lw=2, color='purple')
        ax_wave.set_xlim(0, 250)
        ax_wave.set_ylim(-50, 50)  # Set reasonable μV range for gamma
        ax_wave.set_title('Combined Gamma Wave Activity (30-100 Hz)')
        ax_wave.set_ylabel('Amplitude (μV)')
        ax_wave.set_xlabel('Sample')
        ax_wave.grid(True)
        
        # Add text for power display
        power_text = ax_wave.text(0.02, 0.95, '', transform=ax_wave.transAxes,
                                verticalalignment='top', fontsize=10,
                                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()

        ani = FuncAnimation(fig, update, fargs=(board, line, power_text),
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