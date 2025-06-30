import requests
import datetime
import logging
import pandas as pd
import json

# ✅ Setup logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

CONFIG_PATH = "D:/Downloads/Dhanbot/dhan_autotrader/config.json"

with open(CONFIG_PATH, 'r') as f:
    config = json.load(f)
ACCESS_TOKEN = config["access_token"]
CLIENT_ID = config["client_id"]

HEADERS = {
    "access-token": ACCESS_TOKEN,
    "client-id": CLIENT_ID,
    "Content-Type": "application/json"
}

payload = {
    "securityId": "1333",  # HDFC
    "exchangeSegment": "NSE_EQ",
    "instrument": "EQUITY",
    "interval": "1",  # 1-min candles
    "oi": "false",
    "fromDate": "2025-05-19 09:30:00",
    "toDate": "2025-05-21 15:30:00"
}

def fetch_and_save_intraday_csv():
    url = "https://api.dhan.co/v2/charts/intraday"
    logging.info("🔍 Fetching intraday candle data...")

    try:
        response = requests.post(url, headers=HEADERS, json=payload)
        logging.debug(f"Response Status: {response.status_code}")
        logging.debug(f"Response Body: {response.text}")

        if response.status_code == 200:
            data = response.json()

            # Validate keys
            required_keys = {"open", "high", "low", "close", "volume", "timestamp"}
            if not required_keys.issubset(data.keys()):
                logging.error("❌ Missing one or more required keys in the response.")
                return

            # Build DataFrame
            df = pd.DataFrame({
                "timestamp": pd.to_datetime(data["timestamp"], unit="s", utc=True).tz_convert("Asia/Kolkata"),
                "open": data["open"],
                "high": data["high"],
                "low": data["low"],
                "close": data["close"],
                "volume": data["volume"]
            })

            df.to_csv("hdfc_intraday_candles.csv", index=False)
            print("✅ Data saved to hdfc_intraday_candles.csv")
        else:
            logging.error(f"❌ Error {response.status_code}: {response.text}")
    except Exception as e:
        logging.exception("❌ Exception while fetching candle data:")

if __name__ == "__main__":
    fetch_and_save_intraday_csv()
