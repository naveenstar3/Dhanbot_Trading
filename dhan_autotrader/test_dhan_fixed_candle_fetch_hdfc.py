
import requests
import datetime

# Constants
ACCESS_TOKEN = "************"
CLIENT_ID = "1106857359"

HEADERS = {
    "access-token": ACCESS_TOKEN,
    "client-id": CLIENT_ID,
    "Content-Type": "application/json"
}

# RELIANCE = 2885, you can replace this with 1333 as per support example
payload = {
    "securityId": "1333",
    "exchangeSegment": "NSE_EQ",
    "instrument": "EQUITY",
    "interval": "5",  # 5-min candle
    "oi": False,
    "fromDate": "2025-03-01",
    "toDate": "2025-05-13"
}

def fetch_intraday_data():
    url = "https://api.dhan.co/market/intraday"
    print("🔍 Fetching intraday 5-min candle data...")
    response = requests.post(url, headers=HEADERS, json=payload)
    
    if response.status_code == 200:
        data = response.json()
        candles = data.get("data", [])
        print(f"✅ Received {len(candles)} candles.")
        for row in candles[:5]:
            print(row)
    else:
        print("❌ Error:", response.status_code, response.text)

if __name__ == "__main__":
    fetch_intraday_data()
