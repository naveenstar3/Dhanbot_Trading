from dhanhq import DhanContext, MarketFeed
import time

# Load credentials from config (avoid hardcoding)
CONFIG_PATH = "D:/Downloads/Dhanbot/dhan_autotrader/config.json"
import json
with open(CONFIG_PATH, 'r') as f:
    config = json.load(f)

client_id = config["client_id"]
access_token = config["access_token"]

# Step 1: Create context
dhan_context = DhanContext(client_id, access_token)

# Step 2: Subscribe to instruments (ExchangeSegment, SecurityID, Mode)
# Example: ITC = 1660, GRASIM = 1232
instruments = [
    (MarketFeed.NSE, "1660", MarketFeed.Ticker),  # ITC
    (MarketFeed.NSE, "1232", MarketFeed.Ticker)   # GRASIM
]

# Step 3: Start MarketFeed WebSocket
print("📡 Connecting to Dhan WebSocket...")
market_feed = MarketFeed(dhan_context, instruments, version="v2")

try:
    while True:
        market_feed.run_forever()
        data = market_feed.get_data()
        if data:
            print(f"📈 Live Feed: {data}")
        time.sleep(1)

except KeyboardInterrupt:
    print("🛑 Terminated by user.")
    market_feed.disconnect()

except Exception as e:
    print(f"❌ Error: {e}")
