import requests
import json
from datetime import datetime

# === Load Config ===
try:
    config_path = "D:/Downloads/Dhanbot/dhan_autotrader/dhan_config.json"
    with open(config_path) as f:
        config = json.load(f)
    ACCESS_TOKEN = config["access_token"]
except Exception as e:
    print(f"❌ Failed to load config file: {e}")
    exit()

# === Constants ===
security_id = "2885"  # RELIANCE
exchange_segment = "NSE_EQ"
instrument_type = "EQUITY"
today = datetime.now().strftime("%Y-%m-%d")

headers = {
    "access-token": ACCESS_TOKEN,
    "Content-Type": "application/json"
}

# === Print Input Verification ===
print("\n🔍 INPUT VERIFICATION:")
print(f"✅ Access Token: {ACCESS_TOKEN[:10]}... (truncated)")
print(f"✅ Security ID: {security_id}")
print(f"✅ Exchange Segment: {exchange_segment}")
print(f"✅ Instrument Type: {instrument_type}")
print(f"✅ Date: {today}")
print(f"✅ Headers: {headers}")
print("--------------------------------------------------\n")

# === Step 1: V1 LIVE PRICE ===
def test_v1_live_price():
    url = f"https://api.dhan.co/market-feed/quotes/{security_id}?exchangeSegment={exchange_segment}"
    print(f"\n▶️ Requesting V1 Live Price from: {url}")
    try:
        response = requests.get(url, headers=headers)
        print(f"📥 Response [{response.status_code}]:", response.text)
        data = response.json().get("data", {})
        return float(data.get("lastTradedPrice", 0)) / 100
    except Exception as e:
        print(f"❌ Exception in V1 Live Price: {e}")
        return None

# === Step 2: V1 HISTORICAL ===
def test_v1_historical():
    url = f"https://api.dhan.co/chart/intraday/{security_id}?exchangeSegment={exchange_segment}&instrumentId={security_id}&interval=5m&limit=5"
    print(f"\n▶️ Requesting V1 Historical from: {url}")
    try:
        response = requests.get(url, headers=headers)
        print(f"📥 Response [{response.status_code}]:", response.text)
        return response.json().get("data", [])
    except Exception as e:
        print(f"❌ Exception in V1 Historical: {e}")
        return []

# === Step 3: V2 LIVE PRICE ===
def test_v2_live_price():
    url = "https://api.dhan.co/v2/market-feed/quotes"
    payload = {
        "security_id": security_id,
        "exchange_segment": exchange_segment,
        "instrument_type": instrument_type
    }
    print(f"\n▶️ Requesting V2 Live Price (POST): {url}")
    print("📝 Payload:", json.dumps(payload))
    try:
        response = requests.post(url, headers=headers, json=payload)
        print(f"📥 Response [{response.status_code}]:", response.text)
        return response.json().get("data", {})
    except Exception as e:
        print(f"❌ Exception in V2 Live Price: {e}")
        return {}

# === Step 4: V2 HISTORICAL ===
def test_v2_historical():
    url = "https://api.dhan.co/v2/charts/intraday"
    payload = {
        "security_id": security_id,
        "exchange_segment": exchange_segment,
        "instrument_type": instrument_type,
        "interval": "5m",
        "from_date": today,
        "to_date": today
    }
    print(f"\n▶️ Requesting V2 Historical (POST): {url}")
    print("📝 Payload:", json.dumps(payload))
    try:
        response = requests.post(url, headers=headers, json=payload)
        print(f"📥 Response [{response.status_code}]:", response.text)
        return response.json().get("data", [])
    except Exception as e:
        print(f"❌ Exception in V2 Historical: {e}")
        return []

# === Run All Tests ===
if __name__ == "__main__":
    print("🧪 Running Dhan API Diagnostics...\n")

    test_v1_live_price()
    test_v1_historical()
    test_v2_live_price()
    test_v2_historical()

    print("\n✅ Diagnostics Complete.")
