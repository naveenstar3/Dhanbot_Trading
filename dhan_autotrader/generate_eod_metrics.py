import requests
import pandas as pd
import json
from datetime import datetime, timedelta

# === Credentials ===
with open("D:/Downloads/Dhanbot/dhan_autotrader/config.json", "r") as f:
    config = json.load(f)

ACCESS_TOKEN = config["access_token"]
CLIENT_ID = config["client_id"]
HEADERS = {
    "access-token": ACCESS_TOKEN,
    "client-id": CLIENT_ID,
    "Content-Type": "application/json"
}

print("📊 Fetching 1-min intraday data (last 5d)...")

# === Load symbol list ===
stock_df = pd.read_csv("dynamic_stock_list.csv")
stock_df.columns = [c.strip().lower() for c in stock_df.columns]

# === Time range ===
end = datetime.now()
start = end - timedelta(days=5)
from_date = start.strftime('%Y-%m-%d') + " 09:30:00"
to_date = end.strftime('%Y-%m-%d') + " 15:30:00"

results = []

for _, row in stock_df.iterrows():
    symbol = row["symbol"]
    security_id = str(row["security_id"])

    payload = {
        "securityId": security_id,
        "exchangeSegment": "NSE_EQ",
        "instrument": "EQUITY",
        "interval": "1",
        "oi": "false",
        "fromDate": from_date,
        "toDate": to_date
    }

    try:
        response = requests.post("https://api.dhan.co/v2/charts/intraday", headers=HEADERS, json=payload)
        data = response.json()

        if not all(k in data for k in ["volume", "timestamp"]):
            print(f"[{symbol}] ❌ No volume/timestamp data.")
            continue

        # Build DataFrame
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(data["timestamp"], unit="s"),
            "volume": data["volume"]
        })

        df["date"] = df["timestamp"].dt.date
        volume_by_day = df.groupby("date")["volume"].sum()
        avg_volume = int(volume_by_day.tail(5).mean())

        results.append({
            "symbol": symbol,
            "security_id": security_id,
            "AvgVolume_5d": avg_volume
        })

        print(f"[{symbol}] ✅ Avg Vol = {avg_volume}")

    except Exception as e:
        print(f"[{symbol}] ❌ Error: {e}")

# === Output ===
if results:
    pd.DataFrame(results).to_csv("eod_metrics.csv", index=False)
    print("✅ Saved to eod_metrics.csv")
else:
    print("⚠️ No data collected. Check API access or token.")
