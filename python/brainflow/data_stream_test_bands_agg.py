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

# Frequency bands
FREQ_BANDS = {
    'Delta': (0.5, 4),
    'Theta': (4, 8),
    'Alpha': (8, 13),
    'Beta': (13, 30),
    'Gamma': (30, 100)
}

BAND_COLORS = {
    'Delta': 'blue',
    'Theta': 'green',
    'Alpha': 'red',
    'Beta': 'orange',
    'Gamma': 'purple'
}

def extract_band(data, band_range):
    """Extract frequency band from EEG data"""
    # Convert to microvolts and remove DC offset
    data_uv = data * SCALE_FACTOR_EEG * 1e6
    
    # Design bandpass filter
    nyquist = SAMPLE_RATE / 2.0
    b, a = signal.butter(4, [band_range[0]/nyquist, band_range[1]/nyquist], btype='bandpass')
    
    # Apply filter
    filtered = signal.filtfilt(b, a, data_uv)
    return filtered

def update(frame, board, lines, power_text):
    try:
        data = board.get_current_board_data(250)
        if data.size > 0:
            eeg_data = data[:8]  # Get first 8 channels
            band_data = {band: np.zeros(eeg_data.shape[1]) for band in FREQ_BANDS}
            band_powers = {band: 0 for band in FREQ_BANDS}
            
            # Process each channel
            for channel in eeg_data:
                # Remove DC offset
                DataFilter.detrend(channel, DetrendOperations.CONSTANT.value)
                
                # Process each frequency band
                for band_name, band_range in FREQ_BANDS.items():
                    # Extract band
                    band = extract_band(channel, band_range)
                    
                    # Add to combined band data
                    band_data[band_name] += band
                    
                    # Add to band power
                    band_powers[band_name] += np.mean(np.square(band))
            
            # Average across channels
            for band_name in FREQ_BANDS:
                band_data[band_name] /= len(eeg_data)
                band_powers[band_name] /= len(eeg_data)
            
            # Update wave plots
            x = np.arange(len(band_data['Alpha']))  # All bands have same length
            for (band_name, line) in zip(FREQ_BANDS, lines):
                line.set_data(x, band_data[band_name])
            
            # Update power text
            power_str = '\n'.join([f'{band}: {power:.1f} μV²' 
                                 for band, power in band_powers.items()])
            power_text.set_text(power_str)
            
            # Log debug info
            logger.debug(f"Band Powers - {', '.join([f'{band}: {power:.2f} μV²' for band, power in band_powers.items()])}")

        return lines + [power_text]
    except Exception as e:
        logger.error(f"Error in update function: {e}")
        return lines + [power_text]

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

        # Create figure
        fig = plt.figure(figsize=(15, 8))
        
        # Plot for all frequency bands
        ax = plt.subplot(111)
        lines = []
        for band_name in FREQ_BANDS:
            line, = ax.plot([], [], label=band_name, lw=2, 
                          color=BAND_COLORS[band_name])
            lines.append(line)
        
        ax.set_xlim(0, 250)
        ax.set_ylim(-50, 50)  # Set reasonable μV range
        ax.set_title('Combined EEG Frequency Bands')
        ax.set_ylabel('Amplitude (μV)')
        ax.set_xlabel('Sample')
        ax.grid(True)
        ax.legend(loc='upper right')
        
        # Add text for power display
        power_text = ax.text(0.02, 0.95, '', transform=ax.transAxes,
                           verticalalignment='top', fontsize=10,
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()

        ani = FuncAnimation(fig, update, fargs=(board, lines, power_text),
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