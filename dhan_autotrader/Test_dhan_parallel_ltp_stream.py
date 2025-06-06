import json
from dhanhq.marketfeed import MarketFeed

# 🔧 Patch: Simulated Dhan context with required methods
class DummyDhanContext:
    def __init__(self, client_id, access_token):
        self._client_id = client_id
        self._access_token = access_token

    def get_client_id(self):
        return self._client_id

    def get_access_token(self):
        return self._access_token

# 🛠️ Load config
CONFIG_PATH = "config.json"
print(f"🛠️ Loading config from: {CONFIG_PATH}")
with open(CONFIG_PATH) as f:
    config = json.load(f)

ACCESS_TOKEN = config.get("access_token")
CLIENT_ID = config.get("client_id")

print(f"🔐 CLIENT_ID: {CLIENT_ID}")
print(f"🔐 ACCESS_TOKEN: {ACCESS_TOKEN[:5]}**********")

# 🧠 Inject dummy dhan_context
dhan_context = DummyDhanContext(client_id=CLIENT_ID, access_token=ACCESS_TOKEN)

# 📈 Instruments list (exchange, security_id, packet_type)
instruments = [
    ("NSE", "1333", 15),    # HDFC Bank
    ("NSE", "11536", 15),   # Reliance
    ("NSE", "7229", 15),    # Infosys
]
print(f"📦 Subscribing to instruments: {instruments}")

# 📊 Tick handler
def print_tick(data):
    print("📈 Tick Received:", data)

# 🚀 Run the WebSocket feed
def run_market_feed():
    try:
        print("🧪 Initializing MarketFeed...")
        feed = MarketFeed(dhan_context=dhan_context, instruments=instruments, version="v2")
        print("✅ MarketFeed initialized.")
        feed.on_ticks = print_tick
        print("🔌 Connecting to Dhan WebSocket feed...")
        feed.run_forever()
        print("🔁 This line should not execute unless feed ends.")
    except Exception as e:
        print(f"❌ Connection error: {e}")

if __name__ == "__main__":
    print("🚀 Starting LTP Feed Debug Mode...")
    run_market_feed()
    print("🛑 Script execution complete.")
