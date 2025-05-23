import requests
import datetime
import logging
import pandas as pd
import json
import os

# ✅ Load config.json
config_path = "D:/Downloads/Dhanbot/dhan_autotrader/config.json"
with open(config_path, 'r') as f:
    config = json.load(f)

ACCESS_TOKEN = config["access_token"]
CLIENT_ID = config["client_id"]

HEADERS = {
    "access-token": ACCESS_TOKEN,
    "client-id": CLIENT_ID,
    "Content-Type": "application/json"
}

# ✅ Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def get_now_window(minutes=5):
    now = datetime.datetime.now()
    to_time = now.replace(second=0, microsecond=0)
    from_time = to_time - datetime.timedelta(minutes=minutes)
    return from_time.strftime("%Y-%m-%d %H:%M:%S"), to_time.strftime("%Y-%m-%d %H:%M:%S")

def fetch_hdfcbank_5min():
    from_time, to_time = get_now_window(5)
    payload = {
        "securityId": "1333",  # HDFCBANK
        "exchangeSegment": "NSE_EQ",
        "instrument": "EQUITY",
        "interval": "5",  # 5-min candles
        "oi": "false",
        "fromDate": from_time,
        "toDate": to_time
    }

    logging.info(f"🔍 Fetching 5-min candle for HDFCBANK from {from_time} → {to_time}")
    try:
        url = "https://api.dhan.co/v2/charts/intraday"
        response = requests.post(url, headers=HEADERS, json=payload)

        if response.status_code == 200:
            data = response.json()
            if "close" in data and data["close"]:
                df = pd.DataFrame({
                    "timestamp": pd.to_datetime(data["timestamp"], unit="s", utc=True).tz_convert("Asia/Kolkata"),
                    "open": data["open"],
                    "high": data["high"],
                    "low": data["low"],
                    "close": data["close"],
                    "volume": data["volume"]
                })
                print(df)
            else:
                print("⚠️ No candle data found.")
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
    except Exception as e:
        print(f"❌ Exception occurred: {str(e)}")

if __name__ == "__main__":
    fetch_hdfcbank_5min()
