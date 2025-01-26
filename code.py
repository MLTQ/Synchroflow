import board
import busio
import wifi
import socketpool
import time
import json
from adafruit_minimqtt.adafruit_minimqtt import MQTT

# Configure your WiFi credentials
WIFI_SSID = "Bob2"
WIFI_PASSWORD = "theaustralianfederation"

# Configure the server address
SERVER_ADDRESS = "ws://192.168.1.100:8765"

# Setup LED for visual feedback
led = digitalio.DigitalInOut(board.LED)
led.direction = digitalio.Direction.OUTPUT

def connect_wifi():
    print("Connecting to WiFi...")
    wifi.radio.connect(WIFI_SSID, WIFI_PASSWORD)
    print("Connected to WiFi!")
    
def blink_led(times=1):
    for _ in range(times):
        led.value = True
        time.sleep(0.1)
        led.value = False
        time.sleep(0.1)

def handle_eeg_data(client, data):
    try:
        eeg_data = json.loads(data)['eeg_data']
        # Visual feedback for received data
        blink_led()
        # Print first channel value for debugging
        print(f"Ch1: {eeg_data[0][-1]}")
    except Exception as e:
        print(f"Error processing data: {e}")

def main():
    connect_wifi()
    
    # Create a socket pool
    pool = socketpool.SocketPool(wifi.radio)
    
    # Set up MQTT client
    mqtt_client = MQTT(
        broker=SERVER_ADDRESS,
        port=8765,
        socket_pool=pool,
    )
    
    # Set up callback
    mqtt_client.on_message = handle_eeg_data
    
    print("Connecting to server...")
    mqtt_client.connect()

    while True:
        try:
            mqtt_client.loop()
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(1)
            continue

if __name__ == "__main__":
    main() 