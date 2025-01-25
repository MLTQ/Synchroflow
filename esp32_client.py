import network
import websockets
import json
import machine
import asyncio
from time import sleep

# WiFi credentials
SSID = "Bob2"
PASSWORD = "theaustralianfederation"

# WebSocket server details
SERVER_URI = "ws://192.168.1.38:8765"  # Replace with your server's IP

# Setup LED (most ESP32 dev boards use GPIO 2 for the onboard LED)
led = machine.Pin(2, machine.Pin.OUT)

def connect_wifi():
    """Connect to WiFi network"""
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    
    print(f"Connecting to {SSID}...")
    if not wlan.isconnected():
        wlan.connect(SSID, PASSWORD)
        # Wait for connection with timeout
        max_wait = 10
        while max_wait > 0:
            if wlan.isconnected():
                break
            max_wait -= 1
            print("Waiting for connection...")
            sleep(1)
    
    if wlan.isconnected():
        print("WiFi connected")
        print(f"Network config: {wlan.ifconfig()}")
    else:
        print("WiFi connection failed")
        machine.reset()

def blink_led():
    """Blink the LED once"""
    led.on()
    sleep(0.1)
    led.off()

async def receive_data():
    """Connect to WebSocket server and handle incoming data"""
    while True:
        try:
            async with websockets.connect(SERVER_URI) as websocket:
                print("Connected to WebSocket server")
                while True:
                    try:
                        data = await websocket.recv()
                        # Parse the JSON data
                        eeg_data = json.loads(data)['eeg_data']
                        # Blink LED to indicate data received
                        blink_led()
                        # Optional: print first channel's latest value
                        print(f"Ch1: {eeg_data[0][-1]}")
                    except Exception as e:
                        print(f"Error receiving data: {e}")
                        break
        except Exception as e:
            print(f"Connection error: {e}")
            await asyncio.sleep(5)  # Wait before retrying
            continue

async def main():
    # Connect to WiFi first
    connect_wifi()
    
    # Start receiving data
    print("Starting WebSocket client...")
    await receive_data()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Program terminated by user")
    except Exception as e:
        print(f"Program terminated due to error: {e}")
        machine.reset() 