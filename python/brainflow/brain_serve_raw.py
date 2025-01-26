import asyncio
import websockets
import json
import logging
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

logging.basicConfig(level=logging.DEBUG)  # Set to DEBUG level for detailed logs
logger = logging.getLogger(__name__)


async def stream_data(websocket):  # Remove path argument for newer websockets version
    logger.info("New client connected")
    logger.debug(f"Client connected from {websocket.remote_address}")
    try:
        params = BrainFlowInputParams()
        params.serial_port = "/dev/cu.usbserial-DP04WG7H"
        params.timeout = 15

        board = BoardShim(BoardIds.CYTON_BOARD, params)
        logger.info("Preparing session...")
        board.prepare_session()
        logger.info("Starting stream...")
        board.start_stream()

        while True:
            data = board.get_current_board_data(100)
            logger.debug(f"Retrieved data size: {data.size}")
            if data.size > 0:
                eeg_data = data[:8].tolist()
                logger.debug(f"EEG data to send: {eeg_data}")
                await websocket.send(json.dumps({"eeg_data": eeg_data}))
            await asyncio.sleep(0.1)
    except Exception as e:
        logger.error("Error in stream_data", exc_info=True)
    finally:
        logger.info("Cleaning up board connection")
        board.stop_stream()
        board.release_session()
        logger.info("Client disconnected")


async def main():
    logger.info("Starting server...")
    async with websockets.serve(stream_data, "0.0.0.0", 8765):
        logger.info("Server running on ws://0.0.0.0:8765")
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())