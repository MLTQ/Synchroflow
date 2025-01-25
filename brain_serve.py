import asyncio
import websockets
import json
import logging
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def stream_data(websocket):
    logger.info("New client connected")
    try:
        params = BrainFlowInputParams()
        params.serial_port = "/dev/cu.usbserial-DP05IOKG"
        params.timeout = 15

        board = BoardShim(BoardIds.CYTON_BOARD, params)
        logger.info("Preparing session...")
        board.prepare_session()
        logger.info("Starting stream...")
        board.start_stream()

        while True:
            data = board.get_current_board_data(100)
            if data.size > 0:
                eeg_data = data[:8].tolist()
                await websocket.send(json.dumps({"eeg_data": eeg_data}))
            await asyncio.sleep(0.1)
    except Exception as e:
        logger.error(f"Error in stream_data: {e}")
    finally:
        logger.info("Cleaning up board connection")
        board.stop_stream()
        board.release_session()


async def main():
    logger.info("Starting server...")
    async with websockets.serve(stream_data, "localhost", 8765):
        logger.info("Server running on ws://localhost:8765")
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())