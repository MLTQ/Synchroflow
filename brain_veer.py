import asyncio
import websockets
import json

async def main():
    async with websockets.connect('ws://localhost:8765') as websocket:
        while True:
            try:
                data = await websocket.recv()
                eeg_data = json.loads(data)['eeg_data']
                print("EEG Data Channels:")
                for i, channel in enumerate(eeg_data[:8]):
                    print(f"Ch{i+1}: {channel[-1]}")  # Print latest value
                print("-" * 50)
            except Exception as e:
                print(f"Error: {e}")
                break

if __name__ == "__main__":
    asyncio.run(main())