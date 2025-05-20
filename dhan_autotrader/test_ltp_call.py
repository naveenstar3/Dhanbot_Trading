import requests
import json

# ✅ Load Dhan credentials from config
with open("D:/Downloads/Dhanbot/dhan_autotrader/dhan_config.json", "r") as f:
    config = json.load(f)

ACCESS_TOKEN = config["access_token"]
CLIENT_ID = config["client_id"]

# ✅ RELIANCE NSE securityId
symbol = "RELIANCE"
security_id = "2885"

# ✅ Correct API endpoint
url = "https://api.dhan.co/quotes/market-feed/quote"
headers = {
    "access-token": ACCESS_TOKEN,
    "client-id": CLIENT_ID,
    "Content-Type": "application/json"
}
payload = {
    "securityId": security_id,
    "exchangeSegment": "NSE_EQ"
}

response = requests.post(url, headers=headers, json=payload)

print("🔗 Status Code:", response.status_code)
try:
    data = response.json()
    print("📦 Raw JSON:", data)
    if "data" in data and "lastTradedPrice" in data["data"]:
        ltp = float(data["data"]["lastTradedPrice"]) / 100
        print(f"✅ LTP for {symbol}: ₹{ltp}")
    else:
        print(f"⚠️ No lastTradedPrice found. Full data: {data}")
except Exception as e:
    print(f"❌ JSON Parse Error: {e}")
