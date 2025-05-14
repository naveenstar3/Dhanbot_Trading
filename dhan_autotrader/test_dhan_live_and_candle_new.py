
import asyncio
from dhanhq import marketfeed

# Replace with your credentials
client_id = "1106857359"
access_token = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiJ9.eyJpc3MiOiJkaGFuIiwicGFydG5lcklkIjoiIiwiZXhwIjoxNzQ4MDcyMDEzLCJ0b2tlbkNvbnN1bWVyVHlwZSI6IlNFTEYiLCJ3ZWJob29rVXJsIjoiIiwiZGhhbkNsaWVudElkIjoiMTEwNjg1NzM1OSJ9.ISl7D5ixliWbjnpWQwSXOXJToLpJ8FEGCIIwZTCKPCk6pOGnrO74jQa1SvZpsHhAm7tC1vjwnK1tH8vXaqoQaQ"

# NSE = 1, RELIANCE securityId = 2885
instruments = [(1, "2885")]
subscription_code = 1  # 1 = Ticker

async def on_connect(instance):
    print("✅ Connected to Dhan WebSocket Feed")

async def on_message(instance, message):
    print("📥 Live Tick Message:", message)

# Initialize and run the feed
feed = marketfeed.DhanFeed(
    client_id,
    access_token,
    instruments,
    subscription_code,
    on_connect=on_connect,
    on_message=on_message
)

feed.run_forever()
