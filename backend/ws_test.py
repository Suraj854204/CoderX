import asyncio
import websockets

async def main():
    async with websockets.connect("ws://localhost:8765") as ws:
        print("CONNECTED to CoderX WebSocket")
        try:
            message = await asyncio.wait_for(ws.recv(), timeout=5)
            print("MESSAGE:", message)
        except asyncio.TimeoutError:
            print("CONNECTED, but no message received yet.")

asyncio.run(main())