import pandas as pd
import requests
import json
import time as systime
from datetime import datetime, timedelta
from dhan_api import get_current_capital, get_historical_price
from utils_logger import log_bot_action

with open('D:/Downloads/Dhanbot/dhan_autotrader/config.json', 'r') as f:
    config = json.load(f)

HEADERS = {
    "access-token": config["access_token"],
    "client-id": config["client_id"],
    "Content-Type": "application/json"
}

MASTER_PATH = "D:/Downloads/Dhanbot/dhan_autotrader/dhan_master.csv"
UNIVERSE_PATH = "D:/Downloads/Dhanbot/dhan_autotrader/core_universe.csv"
CAPITAL = get_current_capital()
FINAL_VOL_THRESHOLD = 200000  # 5-day average
MAX_CHECK = 2500  # Limit to control scan time

def fetch_avg_volume(symbol, secid):
    end = datetime.now()
    start = end - timedelta(days=5)
    from_date = start.strftime('%Y-%m-%d') + " 09:30:00"
    to_date = end.strftime('%Y-%m-%d') + " 15:30:00"

    payload = {
        "securityId": secid,
        "exchangeSegment": "NSE_EQ",
        "instrument": "EQUITY",
        "interval": "1",
        "oi": "false",
        "fromDate": from_date,
        "toDate": to_date
    }

    try:
        resp = requests.post("https://api.dhan.co/v2/charts/intraday", headers=HEADERS, json=payload)
        data = resp.json()
        if not all(k in data for k in ["volume", "timestamp"]):
            return 0

        df = pd.DataFrame({
            "timestamp": pd.to_datetime(data["timestamp"], unit="s"),
            "volume": data["volume"]
        })
        df["date"] = df["timestamp"].dt.date
        volume_by_day = df.groupby("date")["volume"].sum()
        avg_vol = int(volume_by_day.tail(5).mean())
        return avg_vol
    except:
        return 0

def run_core_universe_builder():
    print("🚀 Building core_universe.csv from full stock universe...")
    reader = pd.read_csv(MASTER_PATH)
    final_universe = []

    for idx, row in reader.iterrows():
        if idx > MAX_CHECK:
            break

        symbol = str(row['SEM_TRADING_SYMBOL']).strip().upper()
        secid = str(row['SEM_SMST_SECURITY_ID']).strip()

        if not secid.isdigit():
            continue

        try:
            # Price filter
            eod_data = get_historical_price(secid, interval="EOD")
            if not eod_data or "close" not in eod_data or not eod_data["close"]:
                continue
            price = eod_data["close"][-1]
            
            if price is None or price > CAPITAL:
                continue

            # Volume filter
            avg_vol = fetch_avg_volume(symbol, secid)
            systime.sleep(0.4)

            if avg_vol >= FINAL_VOL_THRESHOLD:
                final_universe.append((symbol, secid, price, avg_vol))
                print(f"✅ {symbol} (₹{price}, Vol={avg_vol})")
        except Exception as e:
            print(f"⚠️ {symbol} skipped: {e}")
            continue

    # Save to CSV
    df = pd.DataFrame(final_universe, columns=["symbol", "security_id", "ltp", "avg_volume"])
    df.to_csv(UNIVERSE_PATH, index=False)
    print(f"✅ Saved {len(df)} core stocks to {UNIVERSE_PATH}")

    log_bot_action("core_universe_manager.py", "build_universe", "✅ COMPLETE", f"Saved {len(df)} stocks")

if __name__ == "__main__":
    run_core_universe_builder()
