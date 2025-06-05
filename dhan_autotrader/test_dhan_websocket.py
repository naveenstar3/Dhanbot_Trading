import json
import pandas as pd
import websockets
import asyncio
import ssl
import platform

CONFIG_PATH = "D:/Downloads/Dhanbot/dhan_autotrader/config.json"
MASTER_PATH = "D:/Downloads/Dhanbot/dhan_autotrader/dhan_master.csv"
WATCHLIST = ["RELIANCE", "INFY", "SBIN", "TATAMOTORS", "HDFCBANK"]

print(f"🐍 websockets module version: {websockets.__version__}")

# 🔧 Load config
print(f"🛠️ Loading config from: {CONFIG_PATH}")
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)
access_token = config["access_token"]
client_id = config["client_id"]

print(f"🔑 Loaded access_token: {access_token[:10]}... (length: {len(access_token)}) ✅")
print(f"🆔 Loaded client_id: {client_id} ✅")

# 📂 Load Dhan master
print(f"📂 Loading dhan_master from: {MASTER_PATH}")
df = pd.read_csv(MASTER_PATH)

print(f"✅ Loaded dhan_master with {len(df)} rows")

# 👁️ Preview columns
print("\n🧪 Available dhan_master column names:")
for col in df.columns:
    print(f" - {col.strip().lower()}")

# ✅ Use correct column to match symbols
symbol_col = "sem_trading_symbol"
security_id_col = "sem_smst_security_id"

# Normalize
df.columns = df.columns.str.strip().str.lower()
df[symbol_col.lower()] = df[symbol_col.lower()].astype(str).str.upper().str.strip()

print("\n🔍 Sample normalized symbols from dhan_master:")
print(df[symbol_col.lower()].dropna().unique()[:20])

# ✅ Lookup security IDs
security_ids = []
for sym in WATCHLIST:
    row = df[df[symbol_col.lower()] == sym]
    if not row.empty:
        sec_id = int(row[security_id_col.lower()].values[0])
        print(f"✅ Found {sym} with Security ID: {sec_id}")
        security_ids.append(sec_id)
    else:
        print(f"❌ Symbol not found in dhan_master: {sym}")

if not security_ids:
    print("❌ No valid security IDs found. Exiting.")
    exit()

# 🌐 Construct WebSocket URL
socket_url = f"wss://streamapi.dhan.co/marketdata?client_id={client_id}"
print("🧠 Initializing Dhan WebSocket context...")
print("▶️ Starting script...")
print("🚀 Connecting to Dhan WebSocket for live LTP stream...")

async def connect_ws():
    try:
        headers = {
            "access-token": access_token,
            "client-id": client_id
        }

        ssl_ctx = ssl.create_default_context()

        # 🧪 Show Python platform info
        print(f"🔎 Platform: {platform.system()} {platform.release()}")

        async with websockets.connect(
            socket_url,
            ssl=ssl_ctx,
            extra_headers=headers
        ) as ws:
            # 🧠 Subscribe to symbols
            for sec_id in security_ids:
                sub_msg = json.dumps({
                    "subscription_mode": "LTP",
                    "instrument_token": str(sec_id)
                })
                await ws.send(sub_msg)
                print(f"📩 Subscribed to Security ID: {sec_id}")

            # 🔁 Listen for messages
            while True:
                msg = await ws.recv()
                print(f"📨 Received: {msg}")

    except Exception as e:
        print(f"❌ WebSocket connection error: {e}")

# Start WebSocket client
asyncio.run(connect_ws())
