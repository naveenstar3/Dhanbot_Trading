import requests
import json
import pandas as pd
from datetime import datetime, timedelta
import logging

# === Logging Setup ===
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# === Load config.json for credentials ===
with open("D:/Downloads/Dhanbot/dhan_autotrader/config.json", "r") as f:
    config = json.load(f)

ACCESS_TOKEN = config["access_token"]
CLIENT_ID = config["client_id"]

HEADERS = {
    "access-token": ACCESS_TOKEN,
    "client-id": CLIENT_ID,
    "Content-Type": "application/json"
}

# === Chart Parameters ===
SECURITY_ID = "1333"  # HDFCBANK
EXCHANGE_SEGMENT = "NSE_EQ"
INSTRUMENT = "EQUITY"
INTERVAL = "1"  # 1-minute candles

# === Expand range to 8 calendar days (to get ~5 trading days) ===
to_date_dt = datetime.now().replace(hour=15, minute=30, second=0, microsecond=0)
from_date_dt = to_date_dt - timedelta(days=8)

from_date = from_date_dt.strftime('%Y-%m-%d %H:%M:%S')
to_date = to_date_dt.strftime('%Y-%m-%d %H:%M:%S')

payload = {
    "securityId": SECURITY_ID,
    "exchangeSegment": EXCHANGE_SEGMENT,
    "instrument": INSTRUMENT,
    "interval": INTERVAL,
    "oi": False,  # Correct boolean
    "fromDate": from_date,
    "toDate": to_date
}

# === Volume Fetch Logic ===
try:
    logging.info(f"📈 Fetching up to 5 trading days of 1-min volume data for securityId={SECURITY_ID}")
    response = requests.post("https://api.dhan.co/v2/charts/intraday", headers=HEADERS, json=payload)

    if response.status_code == 200:
        data = response.json()

        if "timestamp" in data and "volume" in data:
            df = pd.DataFrame({
                "timestamp": pd.to_datetime(data["timestamp"], unit="s"),
                "volume": data["volume"]
            })

            df["date"] = df["timestamp"].dt.date
            volume_by_day = df.groupby("date")["volume"].sum()

            print("\n📊 Last 5 Trading Day Volume Summary:")
            print(volume_by_day.tail(5))
        else:
            logging.error("❌ API response missing 'timestamp' or 'volume'.")
    elif response.status_code == 401:
        logging.error("❌ Authentication failed. Token may be expired.")
    else:
        logging.error(f"❌ Error {response.status_code}: {response.text}")

except Exception as e:
    logging.exception("❌ Exception occurred during volume fetch:")
