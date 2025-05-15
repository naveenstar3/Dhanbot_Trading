
import asyncio
import websockets
import struct
import datetime

# Dhan WebSocket Feed Constants
WS_URL = "wss://api-feed.dhan.co"
CLIENT_ID = "1106857359"
ACCESS_TOKEN = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiJ9.eyJpc3MiOiJkaGFuIiwicGFydG5lcklkIjoiIiwiZXhwIjoxNzQ4MDcyMDEzLCJ0b2tlbkNvbnN1bWVyVHlwZSI6IlNFTEYiLCJ3ZWJob29rVXJsIjoiIiwiZGhhbkNsaWVudElkIjoiMTEwNjg1NzM1OSJ9.ISl7D5ixliWbjnpWQwSXOXJToLpJ8FEGCIIwZTCKPCk6pOGnrO74jQa1SvZpsHhAm7tC1vjwnK1tH8vXaqoQaQ"

# RELIANCE NSE: Segment ID = 1, Security ID = 2885
SEGMENT = 1  # NSE
SECURITY_ID = 2885
SUBSCRIPTION_MODE = 1  # 1 = Ticker Feed

# Auth packet is 2 + 30 + 500 bytes: '2P' + client_id + access_token
def build_auth_packet():
    auth_type = b"2P"
    client_id_bytes = CLIENT_ID.encode().ljust(30, b" ")
    access_token_bytes = ACCESS_TOKEN.encode().ljust(500, b" ")
    return auth_type + client_id_bytes + access_token_bytes

# Subscribe packet for a single instrument
def build_subscribe_packet():
    packet = struct.pack(">B", SUBSCRIPTION_MODE)              # Subscription type: Ticker
    packet += struct.pack(">H", 1)                              # Number of instruments: 1
    packet += struct.pack(">B", SEGMENT)                        # Exchange segment
    packet += struct.pack(">I", SECURITY_ID)                    # Security ID
    return packet

async def run_websocket():
    async with websockets.connect(WS_URL) as ws:
        print("🔌 Connected to Dhan WebSocket Feed")

        # Step 1: Send authentication packet
        auth_packet = build_auth_packet()
        await ws.send(auth_packet)
        print("✅ Sent auth packet")

        # Step 2: Wait briefly and send subscription
        await asyncio.sleep(1)
        sub_packet = build_subscribe_packet()
        await ws.send(sub_packet)
        print(f"📡 Subscribed to tick data for Security ID: {SECURITY_ID}")

        # Step 3: Listen for incoming messages
        while True:
            try:
                msg = await ws.recv()
                now = datetime.datetime.now().strftime('%H:%M:%S')
                if isinstance(msg, bytes):
                    print(f"📥 [{now}] Tick Data Received: {len(msg)} bytes")
                    print("🔍 Raw Preview:", msg[:24].hex())
                else:
                    print(f"📩 [{now}] Text Message:", msg)
            except Exception as e:
                print("❌ Error receiving tick:", e)
                break

if __name__ == "__main__":
    asyncio.run(run_websocket())
