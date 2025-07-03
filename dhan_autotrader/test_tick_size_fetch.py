import requests
import json

# Load config and headers
CONFIG_PATH = "D:/Downloads/Dhanbot/dhan_autotrader/config.json"
with open(CONFIG_PATH, 'r') as f:
    config = json.load(f)

HEADERS = {
    "access-token": config["access_token"],
    "client-id": config["client_id"]
}

# HDFC NSE EQ security_id (from dhan_master.csv)
security_id = "1330"
url = f"https://api.dhan.co/instruments-details?security_id={security_id}"

try:
    print(f"🔍 Fetching tick size for HDFC (security_id: {security_id})...")
    resp = requests.get(url, headers=HEADERS, timeout=10)

    if resp.status_code == 200:
        data = resp.json()
        tick_size = float(data.get("tickSize", 0))
        prev_close = float(data.get("prevClose", 0))
        print(f"✅ tickSize: {tick_size}")
        print(f"ℹ️ prevClose: ₹{prev_close}")
    else:
        print(f"❌ Failed: Status {resp.status_code} - {resp.text}")
except Exception as e:
    print(f"❌ Exception: {e}")
