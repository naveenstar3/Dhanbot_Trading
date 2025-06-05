# premarket_nifty100_scanner.py
import os
import sys
import json
import pandas as pd
import requests
from datetime import datetime, timedelta
from dhan_api import get_live_price
import pytz
import time
from nsepython import nsefetch  # Using your working library

# ======== CONFIGURATION ========
start_time = datetime.now()
CAPITAL_PATH = "D:/Downloads/Dhanbot/dhan_autotrader/current_capital.csv"
CONFIG_PATH = "D:/Downloads/Dhanbot/dhan_autotrader/config.json"
OUTPUT_CSV = "D:/Downloads/Dhanbot/dhan_autotrader/dynamic_stock_list.csv"
NIFTY100_CACHE = "D:/Downloads/Dhanbot/dhan_autotrader/nifty100_constituents.csv"
MASTER_CSV = "D:/Downloads/Dhanbot/dhan_autotrader/dhan_master.csv"

# ======== TRADING DAY CHECK ========
def is_trading_day():
    today = datetime.now(pytz.timezone("Asia/Kolkata")).date()
    if today.weekday() >= 5:  # Weekend check
        return False
    return True

if not is_trading_day():
    print("⛔ Non-trading day. Exiting.")
    sys.exit(0)

# ======== LOAD CREDENTIALS & CAPITAL ========
with open(CONFIG_PATH, 'r') as f:
    config = json.load(f)
ACCESS_TOKEN = config["access_token"]
CLIENT_ID = config["client_id"]
HEADERS = {
    "access-token": ACCESS_TOKEN,
    "client-id": CLIENT_ID,
    "Content-Type": "application/json"
}

try:
    CAPITAL = float(pd.read_csv(CAPITAL_PATH, header=None).iloc[0, 0])
    print(f"💰 Capital Loaded: ₹{CAPITAL:,.2f}")
except Exception as e:
    print(f"❌ Capital loading failed: {e}")
    sys.exit(1)

# ======== RELIABLE NIFTY 100 FETCH ========
def get_nifty100_constituents():
    """Fetch Nifty 100 using nsepython with robust error handling"""
    try:
        # Using the same method as your Top_Losers_Strategy
        url = 'https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20100'
        data = nsefetch(url)
        symbols = [item["symbol"].strip().upper() for item in data["data"]]
        print(f"✅ Fetched {len(symbols)} current Nifty 100 constituents")
        return symbols
    except Exception as e:
        print(f"⚠️ NSE fetch failed: {e}. Using cached list")
        if os.path.exists(NIFTY100_CACHE):
            return pd.read_csv(NIFTY100_CACHE)["symbol"].tolist()
        return []  # Fallback

nifty100_symbols = get_nifty100_constituents()
if not nifty100_symbols:
    print("❌ No Nifty 100 symbols available")
    sys.exit(1)

# Save/update cache
pd.DataFrame(nifty100_symbols, columns=["symbol"]).to_csv(NIFTY100_CACHE, index=False)

# ======== LOAD MASTER SECURITY LIST ========
try:
    master_df = pd.read_csv(MASTER_CSV)
    # Create clean symbol for matching
    master_df["base_symbol"] = master_df["SEM_TRADING_SYMBOL"].str.replace("-EQ", "").str.strip().str.upper()
    nifty100_df = master_df[master_df["base_symbol"].isin(nifty100_symbols)]
    print(f"📊 Master list filtered to {len(nifty100_df)} Nifty 100 stocks")
    
    if len(nifty100_df) == 0:
        print("❌ No matching securities found in master list")
        sys.exit(1)
except Exception as e:
    print(f"❌ Master CSV load failed: {e}")
    sys.exit(1)

# ======== PRE-MARKET SCAN ========
results = []
MIN_VOLUME = 500000  # 5 lakh shares
MIN_ATR = 2.0  # ₹2 minimum daily range

print("\n🚀 Starting pre-market scan...")
for idx, row in nifty100_df.iterrows():
    symbol = row["base_symbol"]
    secid = str(row["SEM_SMST_SECURITY_ID"])
    print(f"\n🔍 [{len(results)+1}/{len(nifty100_df)}] Scanning {symbol}")

    try:
        # Step 1: Get pre-market LTP
        ltp = get_live_price(symbol, secid, premarket=True)
        if not ltp or ltp > CAPITAL:
            print(f"⛔ Unaffordable: ₹{ltp or 0:,.2f} > ₹{CAPITAL:,.2f}")
            continue
        
        # Step 2: Volume check (last 5 trading days)
        payload = {
            "securityId": secid,
            "exchangeSegment": "NSE_EQ",
            "instrument": "EQUITY",  # ADD THIS MISSING FIELD
            "interval": "1",
            "oi": False,  # ADD THIS MISSING FIELD
            "fromDate": (datetime.now() - timedelta(days=10)).strftime('%Y-%m-%d %H:%M:%S'),  # ADD TIME
            "toDate": (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d %H:%M:%S')  # USE YESTERDAY
        }
        vol_response = requests.post(
            "https://api.dhan.co/v2/charts/intraday", 
            headers=HEADERS, 
            json=payload,
            timeout=10
        )
        
        if vol_response.status_code != 200:
            print(f"❌ Volume API error: {vol_response.status_code}")
            continue
            
        vol_data = vol_response.json()
        df_vol = pd.DataFrame({
            "timestamp": pd.to_datetime(vol_data["timestamp"], unit="s"),
            "volume": vol_data["volume"]
        })
        df_vol["date"] = df_vol["timestamp"].dt.date
        daily_vol = df_vol.groupby("date")["volume"].sum()
        avg_volume = daily_vol.tail(5).mean()
        
        if pd.isna(avg_volume) or avg_volume < MIN_VOLUME:
            print(f"⛔ Low volume: {avg_volume:,.0f} < {MIN_VOLUME:,.0f}")
            continue

        # Step 3: Volatility check (ATR proxy)
        if "high" in vol_data and "low" in vol_data:
            df_vol["high"] = vol_data["high"]
            df_vol["low"] = vol_data["low"]
            df_vol["range"] = df_vol["high"] - df_vol["low"]
            daily_range = df_vol.groupby("date")["range"].max()
            atr = daily_range.tail(5).mean()
        else:
            print(f"⛔ Missing high/low data for volatility check")
            continue
        
        if pd.isna(atr) or atr < MIN_ATR:
            print(f"⛔ Low volatility: ₹{atr:.2f} < ₹{MIN_ATR:.2f}")
            continue

        # Step 4: Calculate position size
        quantity = int(CAPITAL // ltp)
        capital_used = quantity * ltp
        
        results.append({
            "symbol": symbol,
            "security_id": secid,
            "ltp": ltp,
            "quantity": quantity,
            "capital_used": capital_used,
            "avg_volume": int(avg_volume),
            "avg_range": round(atr, 2),
            "potential_profit": round(quantity * atr, 2)
        })
        print(f"✅ SELECTED: ₹{ltp:,.2f} | Vol: {avg_volume:,.0f} | Range: ₹{atr:.2f}")

    except Exception as e:
        print(f"⚠️ Processing error: {str(e)[:70]}")
    finally:
        time.sleep(0.3)  # Rate limit protection

# ======== SAVE RESULTS ========
if results:
    results_df = pd.DataFrame(results)
    # Prioritize high volatility and volume
    results_df["priority_score"] = results_df["avg_range"] * results_df["avg_volume"]
    results_df = results_df.sort_values("priority_score", ascending=False)
    results_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Saved {len(results_df)} stocks to {OUTPUT_CSV}")
    print(f"📊 Top 5 opportunities:")
    print(results_df[["symbol", "ltp", "quantity", "potential_profit"]].head().to_string(index=False))
else:
    print("\n❌ No stocks passed all filters")

# ======== PERFORMANCE METRICS ========
elapsed = datetime.now() - start_time
print(f"\n⏱️ Total scan time: {elapsed.total_seconds():.1f} seconds")
print(f"💵 Capital available: ₹{CAPITAL:,.2f}")
print(f"📈 Potential positions: {len(results)}")