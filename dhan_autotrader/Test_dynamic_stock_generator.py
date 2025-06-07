# dynamic_stock_generator.py — Final Full Version (with Retry Patch + Safe Dates)

import os
import sys
import json
import pandas as pd
import requests
from datetime import datetime, timedelta
from dhan_api import get_live_price
import pytz
import time
from nsepython import nsefetch

# ========== CONFIGURATION ==========
start_time = datetime.now()
CAPITAL_PATH = "D:/Downloads/Dhanbot/dhan_autotrader/current_capital.csv"
CONFIG_PATH = "D:/Downloads/Dhanbot/dhan_autotrader/config.json"
OUTPUT_CSV = "D:/Downloads/Dhanbot/dhan_autotrader/dynamic_stock_list.csv"
NIFTY100_CACHE = "D:/Downloads/Dhanbot/dhan_autotrader/nifty100_constituents.csv"
MASTER_CSV = "D:/Downloads/Dhanbot/dhan_autotrader/dhan_master.csv"
MIN_VOLUME = 500000
MIN_ATR = 2.0

# ========== TRADING DAY CHECK ==========
def is_trading_day():
    today = datetime.now(pytz.timezone("Asia/Kolkata")).date()
    return today.weekday() < 5

if not is_trading_day():
    if len(sys.argv) > 1 and sys.argv[1].strip().upper() == "NO":
        print("⚠️ Running in TEST MODE (Non-Trading Day Bypass)")
    else:
        print("⛔ Non-trading day. Exiting.")
        sys.exit(0)

# ========== LOAD CONFIG ==========
with open(CONFIG_PATH, 'r') as f:
    config = json.load(f)
ACCESS_TOKEN = config["access_token"]
CLIENT_ID = config["client_id"]
HEADERS = {
    "access-token": ACCESS_TOKEN,
    "client-id": CLIENT_ID,
    "Content-Type": "application/json"
}

# ========== LOAD CAPITAL ==========
try:
    CAPITAL = float(pd.read_csv(CAPITAL_PATH, header=None).iloc[0, 0])
    print(f"💰 Capital Loaded: ₹{CAPITAL:,.2f}")
except Exception as e:
    print(f"❌ Capital loading failed: {e}")
    sys.exit(1)

# ========== Step A: Fetch Nifty 100 ==========
def get_nifty100_constituents():
    try:
        url = 'https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20100'
        data = nsefetch(url)
        symbols = [item["symbol"].strip().upper() for item in data["data"]]
        pd.DataFrame(symbols, columns=["symbol"]).to_csv(NIFTY100_CACHE, index=False)
        print(f"✅ Fetched {len(symbols)} Nifty 100 symbols")
        return symbols
    except Exception as e:
        print(f"⚠️ NSE fetch failed: {e}. Using cached list.")
        if os.path.exists(NIFTY100_CACHE):
            return pd.read_csv(NIFTY100_CACHE)["symbol"].tolist()
        return []

nifty100_symbols = get_nifty100_constituents()
if not nifty100_symbols:
    print("❌ No Nifty 100 symbols available")
    sys.exit(1)

# ========== Step B: Load dhan_master.csv ==========
try:
    master_df = pd.read_csv(MASTER_CSV)
    master_df["base_symbol"] = master_df["SEM_TRADING_SYMBOL"].str.replace("-EQ", "").str.strip().str.upper()
    master_df = master_df[master_df["base_symbol"].isin(nifty100_symbols)]
    if master_df.empty:
        print("❌ No Nifty100 stocks found in master")
        sys.exit(1)
except Exception as e:
    print(f"❌ Master load error: {e}")
    sys.exit(1)

# ========== Step C: Get Top Performing Sectors ==========
try:
    sector_data = nsefetch("https://www.nseindia.com/api/allIndices")
    sector_df = pd.DataFrame(sector_data["data"])
    valid = sector_df[~sector_df["index"].str.contains("PSU|SME|Infra|Gilt", case=False)]
    top_sectors = valid.sort_values("percentChange", ascending=False).head(2)["index"].tolist()
    print(f"✅ Top Sectors Today: {top_sectors}")
except Exception as e:
    print(f"⚠️ Failed to fetch sector performance: {e}")
    top_sectors = []

# ========== Step D: Filter by Sector ==========
sector_col = None
for col in ["SECTOR", "SECTOR_NAME", "Industry"]:
    if col in master_df.columns:
        sector_col = col
        break

if sector_col and top_sectors:
    master_df = master_df[master_df[sector_col].isin(top_sectors)]
    print(f"🎯 Sector-filtered stock count: {len(master_df)}")

# ========== Step E: Full Scan ==========
def get_valid_chart_payload(secid, symbol):
    for offset in range(1, 8):
        test_day = datetime.now() - timedelta(days=offset)
        if test_day.weekday() >= 5:
            continue  # Skip weekends

        from_date = test_day.replace(hour=9, minute=15, second=0, microsecond=0).strftime('%Y-%m-%d %H:%M:%S')
        to_date = test_day.replace(hour=15, minute=25, second=0, microsecond=0).strftime('%Y-%m-%d %H:%M:%S')

        print(f"📅 Trying {symbol} with {from_date} → {to_date}")

        payload = {
            "securityId": secid,
            "exchangeSegment": "NSE_EQ",
            "instrument": "EQUITY",
            "interval": "1",
            "oi": False,
            "fromDate": from_date,
            "toDate": to_date
        }

        r = requests.post("https://api.dhan.co/v2/charts/intraday", headers=HEADERS, json=payload, timeout=10)
        candles = r.json().get("timestamp", [])
        print(f"📊 {symbol}: Got {len(candles)} candles for {from_date} → {to_date}")
        if r.status_code == 200 and len(candles) > 0:
            return r.json()
        if "DH-905" not in r.text:
            print(f"❌ Non-date error for {symbol}: {r.status_code} | {r.text}")
            break

    return None

results = []
print("\n🚀 Starting scan...")
for idx, row in master_df.iterrows():
    symbol = row["base_symbol"]
    secid = str(row["SEM_SMST_SECURITY_ID"])
    print(f"\n🔍 [{idx+1}] Scanning {symbol}")

    try:
        # Step 1: Validate chart dates first
        data = get_valid_chart_payload(secid, symbol)
        if not data:
            print(f"❌ Chart data fetch failed for {symbol}")
            continue
        
        # Step 2: Get pre-market LTP after date confirmed
        ltp = get_live_price(symbol, secid, premarket=True)
        if ltp is None:
            print("⚠️ LTP is None. Skipping.")
            continue
        if ltp > CAPITAL:
            print(f"⛔ Skip: LTP ₹{ltp:.2f} > ₹{CAPITAL}")
            continue        

        # Step 3: Volume & ATR check (retrying if toDate/fromDate invalid)
        data = get_valid_chart_payload(secid, symbol)
        if not data:
            print(f"❌ Failed to fetch chart data for {symbol}")
            continue

        df = pd.DataFrame({
            "timestamp": pd.to_datetime(data["timestamp"], unit="s"),
            "volume": data["volume"],
            "high": data["high"],
            "low": data["low"]
        })
        df["date"] = df["timestamp"].dt.date
        df["range"] = df["high"] - df["low"]
        avg_vol = df.groupby("date")["volume"].sum().tail(5).mean()
        avg_range = df.groupby("date")["range"].max().tail(5).mean()

        if avg_vol < MIN_VOLUME:
            print(f"⛔ Low Volume: {avg_vol:,.0f}")
            continue
        if avg_range < MIN_ATR:
            print(f"⛔ Low ATR: ₹{avg_range:.2f}")
            continue

        qty = int(CAPITAL // ltp)
        results.append({
            "symbol": symbol,
            "security_id": secid,
            "ltp": ltp,
            "quantity": qty,
            "capital_used": round(qty * ltp, 2),
            "avg_volume": int(avg_vol),
            "avg_range": round(avg_range, 2),
            "score": avg_vol * avg_range
        })
        print(f"✅ SELECTED | Vol: {avg_vol:,.0f} | ATR: ₹{avg_range:.2f}")

    except Exception as e:
        print(f"⚠️ Error: {e}")
    time.sleep(0.3)

# ========== Step F: Save Output ==========
if results:
    df_out = pd.DataFrame(results)
    df_out = df_out.sort_values("score", ascending=False)
    df_out[["symbol", "security_id"]].to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Saved {len(df_out)} stocks to {OUTPUT_CSV}")
else:
    print("\n❌ No valid stocks found")

elapsed = datetime.now() - start_time
print(f"\n⏱️ Total scan time: {elapsed.total_seconds():.1f} sec | Final count: {len(results)}")