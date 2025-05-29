import requests
import json
import pandas as pd
from datetime import datetime, timedelta

# === Load Dhan config ===
CONFIG_PATH = "D:/Downloads/Dhanbot/dhan_autotrader/config.json"
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

ACCESS_TOKEN = config["access_token"]
CLIENT_ID = config["client_id"]

# === Target details ===
symbol = "HDFCBANK"
security_id = "1333"
from_datetime = (datetime.now() - timedelta(days=15)).strftime('%Y-%m-%d 09:30:00')
to_datetime = datetime.now().strftime('%Y-%m-%d 15:30:00')

headers = {
    "access-token": ACCESS_TOKEN,
    "client-id": CLIENT_ID,
    "Content-Type": "application/json"
}

payload = {
    "securityId": security_id,
    "exchangeSegment": "NSE_EQ",
    "instrument": "EQUITY",
    "interval": "1",
    "oi": False,
    "fromDate": from_datetime,
    "toDate": to_datetime
}

print(f"🔍 Fetching 1-min candles for {symbol} from {from_datetime} to {to_datetime}...")

url = "https://api.dhan.co/v2/charts/intraday"
resp = requests.post(url, headers=headers, json=payload)

if resp.status_code == 200:
    data = resp.json()
    if all(k in data for k in ["high", "low", "timestamp"]):
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(data["timestamp"], unit="s"),
            "high": data["high"],
            "low": data["low"]
        })
        df["date"] = df["timestamp"].dt.date
        df["range"] = df["high"] - df["low"]

        # Group by date and compute max daily range
        daily_ranges = df.groupby("date")["range"].max().dropna().tail(5)
        if len(daily_ranges) >= 5:
            atr_proxy = round(daily_ranges.mean(), 2)
            print(f"✅ Average daily movement (ATR-proxy) over last 5 days: ₹{atr_proxy}")
        else:
            print("❌ Not enough valid trading days for ATR calculation.")
    else:
        print("❌ Missing expected keys in response.")
else:
    print(f"❌ Request failed. Status: {resp.status_code}, Message: {resp.text}")
