import os
import sys
import json
import time
import math
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from dateutil.parser import parse

# Step 0: Load config and capital
with open("config.json") as f:
    config = json.load(f)
TOKEN = config["access_token"]
DHAN_CLIENT = config["client_id"]
try:
    capital_df = pd.read_csv("current_capital.csv", header=None)
    CAPITAL = float(capital_df.iloc[0, 0])
except Exception as e:
    print(f"❌ Capital Load Error: {e}")
    sys.exit(1)
print(f"💰 Capital Loaded: ₹{CAPITAL:,.2f}")

# Step 1: Check for test mode
IS_TEST = len(sys.argv) > 1 and sys.argv[1].strip().upper() == "NO"
if not IS_TEST:
    day = datetime.now().strftime("%A")
    if day in ["Saturday", "Sunday"]:
        print("⛔ Non-trading day. Exiting.")
        sys.exit()
    else:
        print("⚠️ Running in TEST MODE (Non-Trading Day Bypass)")

# Step 2: Load master & Nifty100
master_df = pd.read_csv("dhan_master.csv")
master_df.columns = master_df.columns.str.strip()  # Remove trailing spaces
nifty100 = pd.read_csv("nifty100_constituents.csv") 
symbols = list(nifty100[nifty100.columns[0]].dropna().astype(str).str.upper().unique())
print(f"✅ Fetched {len(symbols)} Nifty 100 symbols")
print("Master Columns:", master_df.columns.tolist())
        

# Step 3: Helper - get_chart_data
def get_chart_data(symbol, from_date, to_date, interval):
    url = "https://api.dhan.co/charting/v1/getHistoricalCharts"
    match = master_df[master_df["SEM_TRADING_SYMBOL"].str.upper() == symbol]
    if match.empty:
        print(f"❌ Symbol not found in master: {symbol}")
        return []
    security_id = str(match["SEM_SMST_SECURITY_ID"].values[0])
    payload = {
        "securityId": security_id,
        "exchangeSegment": "NSE_EQ",
        "instrumentType": "EQUITY",
        "interval": interval,
        "fromDate": from_date.isoformat(),
        "toDate": to_date.isoformat()
    }
    
    headers = {
        "access-token": TOKEN,
        "client-id": DHAN_CLIENT
    }
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200 and "candles" in response.json():
        return response.json()["candles"]
    else:
        print(f"❌ Chart fetch failed for {symbol}: {response.status_code} | {response.text}")
        return []

# Step 4: Detect market sentiment using NIFTY index
def get_market_sentiment():
    index_symbol = "NIFTY"
    today = datetime.now()
    from_dt = today.replace(hour=9, minute=15, second=0)
    to_dt = from_dt + timedelta(minutes=15)

    candles = get_chart_data(index_symbol, from_dt, to_dt, interval="15m")
    if not candles:
        print("⚠️ Failed to fetch Nifty index data for sentiment analysis.")
        return "Unknown"

    o, h, l, c = candles[0][1:5]
    if c > o * 1.0015:
        print("📈 Market Sentiment: Bullish")
        return "Bullish"
    elif c < o * 0.9985:
        print("📉 Market Sentiment: Bearish")
        return "Bearish"
    else:
        print("😐 Market Sentiment: Sideways")
        return "Sideways"

market_direction = get_market_sentiment()

# Step 5: Get Top Sectors from live index API
def get_top_sectors():
    url = "https://api.dhan.co/market/v1/index/live"
    headers = {
        "access-token": TOKEN,
        "client-id": DHAN_CLIENT
    }
    try:
        res = requests.get(url, headers=headers)
        if res.status_code != 200:
            return []
        data = res.json()["indices"]
        top_sectors = sorted(data, key=lambda x: x["percentChange"], reverse=True)
        selected = [s["indexName"] for s in top_sectors if "NIFTY" in s["indexName"] and "SME" not in s["indexName"]][:3]
        return selected
    except Exception as e:
        print(f"❌ Gainer fetch failed: {e}")
        return []

top_sectors = get_top_sectors()
print(f"✅ Top Sectors Today: {top_sectors}")

# Step 6: Dynamic Stock Filtering
final_stocks = []
start_time = time.time()

print("\n🚀 Starting scan...\n")
for idx, symbol in enumerate(symbols):
    print(f"🔍 [{idx}] Scanning {symbol}")

    if symbol not in list(master_df["SEM_TRADING_SYMBOL"].str.upper()):
        continue

    sec_id = master_df[master_df["SEM_TRADING_SYMBOL"].str.upper() == symbol]["SEM_SMST_SECURITY_ID"].values[0]
    sector = master_df[master_df["SEM_TRADING_SYMBOL"].str.upper() == symbol]["SEM_SEGMENT"].values[0]

    # 💡 Filter out stocks not in today's top sectors
    if not any(s in sector.upper() for s in top_sectors):
        continue

    from_time = datetime.now().replace(hour=9, minute=15, second=0)
    to_time = datetime.now().replace(hour=15, minute=25, second=0)

    # Step 7: Candle fetch & validations
    candles = get_chart_data(symbol, from_time, to_time, "1m")
    print(f"📊 {symbol}: Got {len(candles)} candles for {from_time.strftime('%Y-%m-%d')}")

    if not candles:
        continue

    ltp = candles[-1][4]
    if ltp is None or ltp > CAPITAL:
        print(f"⛔ Skip: LTP ₹{ltp} > ₹{CAPITAL:.2f}")
        continue

    # Step 8: Volume average (last 5 days)
    vol_avg = 0
    for d in range(1, 6):
        day_from = from_time - timedelta(days=d)
        day_to = to_time - timedelta(days=d)
        past_candles = get_chart_data(symbol, day_from, day_to, "1m")
        vol_day = sum([c[5] for c in past_candles])
        vol_avg += vol_day / 5 if vol_day else 0

    today_volume = sum([c[5] for c in candles])
    if today_volume < vol_avg * 1.2:
        print(f"⛔ Low volume: {today_volume} < {vol_avg:.0f}")
        continue

    # Step 9: ATR proxy (Volatility)
    closes = [c[4] for c in candles]
    atr = np.mean([abs(closes[i] - closes[i - 1]) for i in range(1, len(closes))])
    if atr < 1.5:
        print(f"⛔ Low ATR: ₹{atr:.2f}")
        continue

    # Step 10: Match sentiment
    bullish = closes[-1] > closes[0]
    if market_direction == "Bullish" and not bullish:
        print("⛔ Mismatch: Market Bullish but stock not trending up")
        continue
    elif market_direction == "Bearish" and bullish:
        print("⛔ Mismatch: Market Bearish but stock not falling")
        continue

    print(f"✅ SELECTED | Vol: {today_volume:,} | ATR: ₹{atr:.2f}")
    final_stocks.append({
        "Symbol": symbol,
        "Security Id": sec_id,
        "Sector": sector,
        "LTP": ltp,
        "Volume": today_volume,
        "ATR": round(atr, 2)
    })

# Step 11: Save Final CSV
if final_stocks:
    df_final = pd.DataFrame(final_stocks)
    df_final.to_csv("dynamic_stock_list.csv", index=False)
    print(f"\n✅ Final stock list saved ➝ dynamic_stock_list.csv ({len(df_final)} stocks)\n")
else:
    print("\n❌ No valid stocks found")

print(f"\n⏱️ Total scan time: {time.time() - start_time:.1f} sec")

# Step 5: Get Top Sectors from live index API
def get_top_sectors():
    url = "https://api.dhan.co/market/v1/index/live"
    headers = {
        "access-token": TOKEN,
        "client-id": DHAN_CLIENT
    }
    try:
        res = requests.get(url, headers=headers)
        if res.status_code != 200:
            return []
        data = res.json()["indices"]
        top_sectors = sorted(data, key=lambda x: x["percentChange"], reverse=True)
        selected = [s["indexName"] for s in top_sectors if "NIFTY" in s["indexName"] and "SME" not in s["indexName"]][:3]
        return selected
    except Exception as e:
        print(f"❌ Gainer fetch failed: {e}")
        return []

top_sectors = get_top_sectors()
print(f"✅ Top Sectors Today: {top_sectors}")

# Step 6: Dynamic Stock Filtering
final_stocks = []
start_time = time.time()

print("\n🚀 Starting scan...\n")
for idx, symbol in enumerate(symbols):
    print(f"🔍 [{idx}] Scanning {symbol}")

    if symbol not in list(master_df[master_df.columns[0]].str.upper()):
        continue

    sec_id = master_df[master_df[master_df.columns[0]].str.upper() == symbol]["Security Id"].values[0]
    sector = master_df[master_df[master_df.columns[0]].str.upper() == symbol]["Sector"].values[0]  

    # 💡 Filter out stocks not in today's top sectors
    if not any(s in sector.upper() for s in top_sectors):
        continue

    from_time = datetime.now().replace(hour=9, minute=15, second=0)
    to_time = datetime.now().replace(hour=15, minute=25, second=0)

    # Step 7: Candle fetch & validations
    candles = get_chart_data(symbol, from_time, to_time, "1m")
    print(f"📊 {symbol}: Got {len(candles)} candles for {from_time.strftime('%Y-%m-%d')}")

    if not candles:
        continue

    ltp = candles[-1][4]
    if ltp is None or ltp > CAPITAL:
        print(f"⛔ Skip: LTP ₹{ltp} > ₹{CAPITAL:.2f}")
        continue

    # Step 8: Volume average (last 5 days)
    vol_avg = 0
    for d in range(1, 6):
        day_from = from_time - timedelta(days=d)
        day_to = to_time - timedelta(days=d)
        past_candles = get_chart_data(symbol, day_from, day_to, "1m")
        vol_day = sum([c[5] for c in past_candles])
        vol_avg += vol_day / 5 if vol_day else 0

    today_volume = sum([c[5] for c in candles])
    if today_volume < vol_avg * 1.2:
        print(f"⛔ Low volume: {today_volume} < {vol_avg:.0f}")
        continue

    # Step 9: ATR proxy (Volatility)
    closes = [c[4] for c in candles]
    atr = np.mean([abs(closes[i] - closes[i - 1]) for i in range(1, len(closes))])
    if atr < 1.5:
        print(f"⛔ Low ATR: ₹{atr:.2f}")
        continue

    # Step 10: Match sentiment
    bullish = closes[-1] > closes[0]
    if market_direction == "Bullish" and not bullish:
        print("⛔ Mismatch: Market Bullish but stock not trending up")
        continue
    elif market_direction == "Bearish" and bullish:
        print("⛔ Mismatch: Market Bearish but stock not falling")
        continue

    print(f"✅ SELECTED | Vol: {today_volume:,} | ATR: ₹{atr:.2f}")
    final_stocks.append({
        "Symbol": symbol,
        "Security Id": sec_id,
        "Sector": sector,
        "LTP": ltp,
        "Volume": today_volume,
        "ATR": round(atr, 2)
    })

# Step 11: Save Final CSV
if final_stocks:
    df_final = pd.DataFrame(final_stocks)
    df_final.to_csv("dynamic_stock_list.csv", index=False)
    print(f"\n✅ Final stock list saved ➝ dynamic_stock_list.csv ({len(df_final)} stocks)\n")
else:
    print("\n❌ No valid stocks found")

print(f"\n⏱️ Total scan time: {time.time() - start_time:.1f} sec")
