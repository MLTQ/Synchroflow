#!/usr/bin/env python3
"""
Generate a simple tone using ALSA
"""

import math
import numpy as np
import alsaaudio
import time
import sys
from io import StringIO

class OutputCapture:
    def __init__(self):
        self.output = StringIO()
        self.stdout = sys.stdout

    def write(self, text):
        self.output.write(text)
        self.stdout.write(text)

    def flush(self):
        self.stdout.flush()

    def get_output(self):
        return self.output.getvalue()

def generate_sine_wave(frequency, duration, sample_rate=44100):
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    tone = np.sin(2 * np.pi * frequency * t)
    # Normalize and convert to 16-bit integer
    samples = (tone * 32767).astype(np.int16)
    return samples

def play_tone(frequency=440, duration=3):
    try:
        # Generate audio data
        samples = generate_sine_wave(frequency, duration)
        
        # Setup audio output
        device = alsaaudio.PCM(alsaaudio.PCM_PLAYBACK)
        device.setchannels(1)  # Mono
        device.setrate(44100)  # CD quality audio
        device.setformat(alsaaudio.PCM_FORMAT_S16_LE)  # 16-bit little-endian
        device.setperiodsize(1024)
        
        # Play the tone
        print(f"Playing {frequency}Hz tone for {duration} seconds...")
        device.write(samples.tobytes())
        time.sleep(duration)
        
        return True
        
    except Exception as e:
        print(f"Error playing tone: {e}")
        return False

if __name__ == "__main__":
    output_capture = OutputCapture()
    sys.stdout = output_capture

    print("\nTrying to play tone...")
    if play_tone():
        print("Successfully played tone")
    else:
        print("Failed to play tone")

    sys.stdout = output_capture.stdout
    
    with open('sound_output.txt', 'w') as f:
        f.write(output_capture.get_output())
    
    print("\nOutput has been saved to sound_output.txt")